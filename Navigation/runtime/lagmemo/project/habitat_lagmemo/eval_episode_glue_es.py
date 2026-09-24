#!/usr/bin/env python3
"""
eval_episode_glue_es.py  — Phase 4 end-to-end nav with ES online mapper.

Adds --enable_timing flag: records per-step latency breakdown and
reports online-navigation compliance summary at the end.

Timing columns saved to results_dir/timing_log.csv:
  episode_id, step, snap, dt_total_ms, dt_push_ms, dt_goals_ms, dt_act_ms, is_upgrade_step

Online standard thresholds (printed at end):
  ≥10 FPS → 100 ms/step   (real-time robot, e.g. 10 Hz)
  ≥ 5 FPS → 200 ms/step   (high-performance)
  ≥ 1 FPS → 1000 ms/step  (quasi-real-time)
  ≥ 0.2 FPS → 5000 ms/step (acceptable simulation overhead)
"""

import warnings
warnings.filterwarnings("ignore")
import os, sys, csv

os.environ['MAGNUM_LOG'] = 'quiet'
os.environ['HABITAT_SIM_LOG'] = 'quiet'

import argparse
import json
import time
from pathlib import Path
from pprint import pprint

import numpy as np
from tqdm import tqdm

_THIS_FILE = Path(__file__).resolve()
_LAGMEMO_ROOT = _THIS_FILE.parents[2]
_REPO_ROOT = _LAGMEMO_ROOT.parent
_INTEG_SCRIPTS = _REPO_ROOT / 'ES_LagMemo_Integration' / 'scripts'
if str(_INTEG_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_INTEG_SCRIPTS))
_LAGMEMO_SRC = _LAGMEMO_ROOT / 'src' / 'lagmemo'
if str(_LAGMEMO_SRC) not in sys.path:
    sys.path.insert(0, str(_LAGMEMO_SRC))

from config_utils import get_config
from habitat.core.env import Env

from lagmemo.agent.lagmemo_agent.glue_agent import GLUEAgent
from lagmemo.core.interfaces import DiscreteNavigationAction
from lagmemo.env.habitat_lagmemo_env import HabitatGoatEnv

from online_es_provider import OnlineESProvider


# ── Timing helpers ────────────────────────────────────────────────────────────

_THRESHOLDS = [
    ('≥10 FPS', 100),
    ('≥ 5 FPS', 200),
    ('≥ 1 FPS', 1000),
    ('≥0.2 FPS', 5000),
]


def _project_goals_to_navigable(goals, pathfinder):
    max_distance = float(
        os.environ.get('ES_NAV_PROJECTION_MAX_DISTANCE', '1.5')
    )
    projected = {}
    for task_idx, candidates in (goals or {}).items():
        accepted = []
        for candidate in candidates:
            source = np.asarray(candidate, dtype=np.float32)
            try:
                snapped = np.asarray(pathfinder.snap_point(source), dtype=np.float32)
            except Exception as exc:
                print(
                    f'[CANDIDATE_PROJECTION] task={task_idx} failed={exc}',
                    flush=True,
                )
                continue
            if snapped.shape != (3,) or not np.isfinite(snapped).all():
                continue
            distance = float(np.linalg.norm(snapped[[0, 2]] - source[[0, 2]]))
            if distance > max_distance:
                continue
            if any(
                np.linalg.norm(snapped[[0, 2]] - np.asarray(old)[[0, 2]]) < 0.25
                for old in accepted
            ):
                continue
            accepted.append(snapped.tolist())
        projected[int(task_idx)] = accepted
        print(
            f'[CANDIDATE_PROJECTION] task={task_idx} '
            f'input={len(candidates)} accepted={len(accepted)}',
            flush=True,
        )
    return projected


def _sample_navmesh_goal(pathfinder, origin, mode):
    origin = np.asarray(origin, dtype=np.float32)
    max_vertical_delta = float(
        os.environ.get('ES_NAVMESH_MAX_VERTICAL_DELTA', '0.35')
    )
    sample_count = int(os.environ.get('ES_NAVMESH_SAMPLE_COUNT', '32'))
    if mode == 'escape':
        radius = float(os.environ.get('ES_NAVMESH_ESCAPE_RADIUS', '3.0'))
        min_distance = float(
            os.environ.get('ES_NAVMESH_ESCAPE_MIN_DISTANCE', '1.25')
        )
    elif mode == 'random':
        radius = None
        min_distance = float(
            os.environ.get('ES_NAVMESH_RANDOM_MIN_DISTANCE', '2.0')
        )
    else:
        raise ValueError(f'unsupported navmesh sampling mode: {mode}')

    try:
        snapped_origin = np.asarray(pathfinder.snap_point(origin), dtype=np.float32)
        island = int(pathfinder.get_island(snapped_origin))
    except Exception as exc:
        return None, {'error': f'origin_snap_failed:{exc}'}
    if snapped_origin.shape != (3,) or not np.isfinite(snapped_origin).all():
        return None, {'error': 'origin_not_navigable'}

    candidates = []
    vertical_rejections = 0
    island_rejections = 0
    for _ in range(sample_count):
        try:
            if mode == 'escape':
                point = pathfinder.get_random_navigable_point_near(
                    snapped_origin, radius, 100, island
                )
            else:
                point = pathfinder.get_random_navigable_point(100, island)
            point = np.asarray(point, dtype=np.float32)
            if point.shape != (3,) or not np.isfinite(point).all():
                continue
            if int(pathfinder.get_island(point)) != island:
                island_rejections += 1
                continue
        except Exception:
            continue
        if abs(float(point[1] - snapped_origin[1])) > max_vertical_delta:
            vertical_rejections += 1
            continue
        distance = float(
            np.linalg.norm(point[[0, 2]] - snapped_origin[[0, 2]])
        )
        if distance < min_distance:
            continue
        candidates.append((distance, point))

    details = {
        'mode': mode,
        'sample_count': sample_count,
        'accepted': len(candidates),
        'vertical_rejections': vertical_rejections,
        'island_rejections': island_rejections,
        'max_vertical_delta': max_vertical_delta,
        'stairs_suspected': vertical_rejections >= max(3, sample_count // 4),
    }
    if not candidates:
        details['error'] = 'no_same_floor_navigable_sample'
        return None, details
    _, goal = max(candidates, key=lambda item: item[0])
    details['goal'] = goal.tolist()
    return goal, details


def _print_timing_summary(rows, results_dir, save_csv=True):
    """Print per-step timing summary and write CSV."""
    if not rows:
        return

    # Save CSV
    if save_csv:
        csv_path = os.path.join(results_dir, 'timing_log.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f'\n[Timing] Saved {len(rows)} rows → {csv_path}')

    totals   = np.array([r['dt_total_ms']  for r in rows])
    push_ms  = np.array([r['dt_push_ms']   for r in rows])
    goals_ms = np.array([r['dt_goals_ms']  for r in rows])
    act_ms   = np.array([r['dt_act_ms']    for r in rows])
    upgrades = np.array([r['is_upgrade_step'] for r in rows], dtype=bool)

    print('\n' + '='*62)
    print(' Online Navigation Timing Report (Phase 4 OnlineESProvider)')
    print('='*62)

    total_wall = totals.sum() / 1000.0
    n = len(totals)
    print(f'  Total steps      : {n}')
    print(f'  Total wall time  : {total_wall:.1f} s ({total_wall/60:.1f} min)')
    print(f'  Overall FPS      : {n / total_wall:.3f}')
    print(f'  Mean step latency: {totals.mean():.0f} ms')
    print(f'  Median step lat  : {np.median(totals):.0f} ms')
    print(f'  Max  step lat    : {totals.max():.0f} ms')
    print()

    n_upg = upgrades.sum()
    n_norm = (~upgrades).sum()
    print(f'  Upgrade steps  : {n_upg} ({100*n_upg/n:.0f}%)  '
          f'[every 10 steps — snapshot reload]')
    if n_upg:
        upg_t = totals[upgrades]
        print(f'    Mean: {upg_t.mean():.0f} ms   Median: {np.median(upg_t):.0f} ms'
              f'   Max: {upg_t.max():.0f} ms')
    print(f'  Normal  steps  : {n_norm} ({100*n_norm/n:.0f}%)')
    if n_norm:
        norm_t = totals[~upgrades]
        print(f'    Mean: {norm_t.mean():.0f} ms   Median: {np.median(norm_t):.0f} ms'
              f'   Max: {norm_t.max():.0f} ms')
    print()

    # Component breakdown
    print('  Step latency breakdown (mean across ALL steps):')
    print(f'    push_frame IPC   : {push_ms.mean():.1f} ms')
    print(f'    get_all_goals    : {goals_ms.mean():.1f} ms  '
          f'(non-zero only at upgrade steps)')
    print(f'    agent.act        : {act_ms.mean():.1f} ms')
    print(f'    other (obs/misc) : '
          f'{(totals - push_ms - goals_ms - act_ms).mean():.1f} ms')
    print()

    # Compliance table
    print('  Online standard compliance:')
    print(f'    {"Threshold":<20} {"All steps":>10} {"Normal only":>12}')
    print(f'    {"-"*20} {"-"*10} {"-"*12}')
    for label, budget_ms in _THRESHOLDS:
        pct_all  = 100.0 * (totals <= budget_ms).sum() / n
        pct_norm = 100.0 * ((totals[~upgrades] <= budget_ms).sum() / n_norm
                             if n_norm else 0)
        flag = '✓' if pct_all > 90 else ('△' if pct_all > 50 else '✗')
        print(f'    {label:<20} {pct_all:>9.1f}%  {pct_norm:>11.1f}%  {flag}')
    print()

    # Verdict
    norm_ok = (totals[~upgrades] <= 1000).sum() / n_norm * 100 if n_norm else 0
    upg_ok  = (totals[upgrades]  <= 1000).sum() / n_upg  * 100 if n_upg  else 0
    print('  Verdict:')
    if n_upg and upg_ok < 10:
        print('  ✗ Snapshot upgrade steps (~{:.0f}s avg) VIOLATE all real-time'.format(
              totals[upgrades].mean()/1000))
        print('    thresholds. Async preloading needed for real deployment.')
    if norm_ok > 90:
        print('  ✓ Normal (non-upgrade) steps meet ≥1 FPS standard '
              f'({norm_ok:.0f}% ≤ 1000 ms).')
    print('='*62 + '\n')


if __name__ == '__main__':
    print('Start time:', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--habitat_config_path', type=str,
                        default='project/config/habitat/lagmemo_hm3d.yaml')
    parser.add_argument('--baseline_config_path', type=str,
                        default='project/config/agent/hm3d_eval.yaml')
    parser.add_argument('--scene_idx', type=int, default=0)
    parser.add_argument('--scenes', type=str, default='5cdEh9F2hJL')
    parser.add_argument('--seem_ckpt_path', type=str,
                        default='./src/third_party/seem/checkpoints/seem_focall_v0.pt')
    parser.add_argument('--seem_yaml_path', type=str,
                        default='./src/third_party/seem/configs/seem/focall_unicl_lang_demo.yaml')
    parser.add_argument('--mobileclip_ckpt_path', type=str,
                        default='./src/third_party/ml-mobileclip/checkpoints/mobileclip_s0.pt')
    parser.add_argument('--output_path', type=str,
                        default=str(_REPO_ROOT / 'ES_LagMemo_Integration' / 'runs' / 'p4_demo'))
    parser.add_argument('--input_data', type=str, default='3_episode_data')
    parser.add_argument(
        '--dataset_root',
        type=str,
        default=None,
        help='Absolute GOAT dataset root containing val_seen.json.gz and content/.',
    )
    parser.add_argument('--socket_path', type=str, default='/tmp/es_mapper.sock')
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--live_mode', action='store_true',
                        help='Phase 5: stream live obs to ES server')
    parser.add_argument('--enable_timing', action='store_true',
                        help='Record per-step timing and print compliance report')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    print(json.dumps(vars(args), indent=4))
    print('-' * 80)

    dataset_root = (
        Path(args.dataset_root).expanduser().resolve()
        if args.dataset_root
        else _LAGMEMO_ROOT / 'data' / 'datasets' / 'goat' / 'hm3d' / args.input_data
    )
    dataset_index = dataset_root / 'val_seen.json.gz'
    dataset_content = dataset_root / 'content'
    if not dataset_index.is_file() or not dataset_content.is_dir():
        raise FileNotFoundError(
            f'dataset_root must contain val_seen.json.gz and content/: {dataset_root}'
        )
    print(f'[ES eval] dataset_root={dataset_root}', flush=True)

    timing_rows = []   # filled when --enable_timing

    # ── Provider ────────────────────────────────────────────────────────
    print('[P4] Connecting to ES Mapper Server...', flush=True)
    candidate_provider = OnlineESProvider(
        socket_path=args.socket_path,
        top_k=args.top_k,
        connect_timeout=120.0,
        content_root=str(dataset_content),
    )
    if args.live_mode:
        candidate_provider._live_mode = True
        print('[Phase 5] Live mode ON: streaming frames to ES server', flush=True)
    print('[P4] ES Mapper Server connected.', flush=True)

    # ── LagMemo setup ───────────────────────────────────────────────────
    seem_cfg = {
        'conf_path': args.seem_yaml_path,
        'ckpt_path': args.seem_ckpt_path,
    }

    config = get_config(args.habitat_config_path, args.baseline_config_path)
    # ES online candidates are indexed directly by Habitat subtask id.
    # LagMemo's EXP_FIRST mutates Habitat episodes by inserting a duplicate
    # first subtask, which shifts candidate/query indices and corrupts SR.
    config.EXP_FIRST = False
    try:
        config.habitat.environment.iterator_options.shuffle = False
    except Exception:
        pass
    print('[ES eval] EXP_FIRST=False; episode iterator shuffle=False', flush=True)
    if args.output_path:
        config.DUMP_LOCATION = args.output_path
    config['habitat']['dataset']['data_path'] = str(dataset_index)
    all_scenes = sorted([
        x.name.removesuffix('.json.gz')
        for x in dataset_content.glob('*.json.gz')
    ])
    config.habitat.dataset.content_scenes = \
        all_scenes if args.scenes == 'all' else [args.scenes]
    config.NUM_ENVIRONMENTS = 1
    config.PRINT_IMAGES = 1
    config.EXP_NAME = f'{config.EXP_NAME}_{args.scene_idx}'

    habitat_env = Env(config)
    env = HabitatGoatEnv(habitat_env, config=config)
    agent = GLUEAgent(config=config, seem_cfg=seem_cfg,
                      clip_cfg=args.mobileclip_ckpt_path)

    results_dir = os.path.join(config.DUMP_LOCATION,
                               f'results_{args.scenes}', config.EXP_NAME)
    os.makedirs(results_dir, exist_ok=True)

    metrics = {}
    es_status_log = []

    # ── Episode loop ─────────────────────────────────────────────────────
    for ep_i in range(len(env.habitat_env.episodes)):
        env.reset()
        scene_id = env.habitat_env.current_episode.scene_id.split('/')[-1].split('.')[0]
        episode  = env.habitat_env.current_episode
        episode_id = episode.episode_id

        candidate_provider.reset_episode(scene_id, episode_id)
        candidate_provider._goals_cache = {}

        agent.reset(start_position=episode.start_position,
                    start_rotation=episode.start_rotation,
                    candidate_provider=candidate_provider)
        agent.set_runtime_context(scene_id, episode_id)
        initial_goals = _project_goals_to_navigable(
            agent.gs_poses, env.habitat_env.sim.pathfinder
        )
        agent.refresh_gs_poses(initial_goals)
        # ES: disable exploration-first at task_idx=0 (legacy LagMemo behavior)
        # test_gs=3 → no exploration phase, use gs_poses for all tasks directly
        agent._es_test_gs_override = 3

        t = 0

        if os.path.exists(os.path.join(results_dir, 'per_episode_metrics.json')):
            with open(os.path.join(results_dir, 'per_episode_metrics.json')) as fp:
                metrics = json.load(fp)
        if f'{scene_id}_{episode_id}' in metrics:
            continue

        agent.planner.set_vis_dir(scene_id,
            f'{episode_id}_{env.habitat_env.task.current_task_idx}')
        agent.imagenav_visualizer.set_vis_dir(
            f'{scene_id}_{episode_id}_{env.habitat_env.task.current_task_idx}')
        agent.matching.set_vis_dir(
            f'{scene_id}_{episode_id}_{env.habitat_env.task.current_task_idx}')
        env.visualizer.set_vis_dir(scene_id,
            f'{episode_id}_{env.habitat_env.task.current_task_idx}')
        agent.set_lightglue_vis_dir(
            f'{scene_id}_{episode_id}_{env.habitat_env.task.current_task_idx}')

        all_subtask_metrics = []
        all_stop_reasons = []
        pbar = tqdm(total=config.AGENT.max_steps)

        n_subtasks = None   # P6_FALLBACK: set at t==1; used by episode-boundary guard
        while not env.episode_over:
            current_task_idx = env.habitat_env.task.current_task_idx
            # P6_FALLBACK (PROGRESS §4.3): habitat advances current_task_idx past the
            # last sub-task before flipping episode_over, so agent.act() gets an
            # out-of-range idx and KeyErrors (frontier_flag/gs_poses keyed 0..N-1),
            # which previously killed the WHOLE run at every episode boundary.
            # End the episode cleanly instead.
            if n_subtasks is not None and current_task_idx >= n_subtasks:
                print(f'[P6 boundary] current_task_idx={current_task_idx} >= '
                      f'n_subtasks={n_subtasks}; ending episode cleanly', flush=True)
                break
            t += 1

            t_step_start = time.perf_counter()

            obs = env.get_observation()
            if t == 1:
                obs_tasks = []
                for task in obs.task_observations['tasks']:
                    obs_task = {k: v for k, v in task.items() if k != 'image'}
                    obs_tasks.append(obs_task)
                n_subtasks = len(obs_tasks)   # P6_FALLBACK: sub-task count for boundary guard
                pprint(obs_tasks)

            # ── push_live_frame (Phase 5) or push_frame (Phase 4) ──────────
            dt_push = 0.0
            try:
                t0 = time.perf_counter()
                if getattr(candidate_provider, '_live_mode', False):
                    # Phase 5: send actual Habitat obs to ES server
                    _rgb_raw = obs.rgb                                  # (H,W,3) uint8
                    _dep_raw = obs.depth                                # (H,W) float32 m
                    _gp_raw  = np.array(obs.globalpose, dtype=np.float32)  # (7,)
                    candidate_provider.push_live_frame(_rgb_raw, _dep_raw, _gp_raw)
                else:
                    # Phase 4: snapshot-based, send frame index only
                    candidate_provider.push_frame(t - 1)
                dt_push = (time.perf_counter() - t0) * 1000
            except Exception as e:
                print(f'[P5] push_frame warning: {e}', flush=True)

            # ── periodic goal refresh ───────────────────────────────────
            dt_goals = 0.0
            is_upgrade = False
            if t % 10 == 0:
                candidate_provider._goals_cache = {}
                t0 = time.perf_counter()
                try:
                    new_goals = candidate_provider.get_all_goals()
                    if new_goals:
                        projected_goals = _project_goals_to_navigable(
                            new_goals, env.habitat_env.sim.pathfinder
                        )
                        agent.refresh_gs_poses(projected_goals)
                except Exception as e:
                    print(f'[P4] goal refresh warning: {e}', flush=True)
                dt_goals = (time.perf_counter() - t0) * 1000
                is_upgrade = True

            # ── agent act ───────────────────────────────────────────────
            t0 = time.perf_counter()
            action, info = agent.act(obs)
            dt_act = (time.perf_counter() - t0) * 1000
            action_metadata = getattr(agent, 'last_action_metadata', {})
            stop_reason = action_metadata.get('stop_reason')
            planner_error = action_metadata.get('planner_error')
            pathfinder = env.habitat_env.sim.pathfinder
            trapped_event = action_metadata.get('trapped_event')
            random_exploration_requested = action_metadata.get(
                'random_exploration_requested', False
            )
            if trapped_event and action != DiscreteNavigationAction.STOP:
                escape_goal, escape_details = _sample_navmesh_goal(
                    pathfinder, np.asarray(obs.globalpose)[4:], 'escape'
                )
                trapped_event['escape'] = escape_details
                if escape_goal is not None:
                    agent.set_navmesh_override_goal(escape_goal, 'escape')
                    action = DiscreteNavigationAction.TURN_RIGHT
                else:
                    agent.clear_local_navigation_state()
                print(
                    f'[TRAPPED_RECOVERY] {json.dumps(trapped_event, ensure_ascii=False)}',
                    flush=True,
                )
            elif (
                random_exploration_requested
                and action != DiscreteNavigationAction.STOP
            ):
                random_goal, random_details = _sample_navmesh_goal(
                    pathfinder, np.asarray(obs.globalpose)[4:], 'random'
                )
                if random_goal is not None:
                    agent.set_navmesh_override_goal(random_goal, 'random')
                    action = DiscreteNavigationAction.TURN_RIGHT
                print(
                    '[RANDOM_EXPLORATION_GOAL] '
                    f'{json.dumps(random_details, ensure_ascii=False)}',
                    flush=True,
                )
            if (
                action == DiscreteNavigationAction.STOP
                and stop_reason not in {'verified_target', 'max_steps'}
            ):
                raise RuntimeError(
                    f'illegal STOP reason={stop_reason!r} '
                    f'planner_error={planner_error!r}'
                )

            env.apply_action(action, info=info)

            dt_total = (time.perf_counter() - t_step_start) * 1000

            # ── record timing ───────────────────────────────────────────
            if args.enable_timing:
                timing_rows.append({
                    'episode_id':      episode_id,
                    'step':            t,
                    'snap':            candidate_provider.current_snap,
                    'dt_total_ms':     round(dt_total,  1),
                    'dt_push_ms':      round(dt_push,   1),
                    'dt_goals_ms':     round(dt_goals,  1),
                    'dt_act_ms':       round(dt_act,    1),
                    'is_upgrade_step': int(is_upgrade),
                })

            pbar.set_description(f'{scene_id}_{episode_id}_{current_task_idx}')
            pbar.update(1)

            if action == DiscreteNavigationAction.STOP:
                all_stop_reasons.append(stop_reason)
                print(
                    f'[STOP_REASON] scene={scene_id} episode={episode_id} '
                    f'task={current_task_idx} reason={stop_reason}',
                    flush=True,
                )
                ep_metrics = env.get_episode_metrics()
                env.reset_subtask()
                ep_metrics.pop('goat_top_down_map', None)
                print('----')
                print(f'{scene_id}_{episode_id}_{current_task_idx}', ep_metrics)
                print('----')
                all_subtask_metrics.append(ep_metrics)
                if not env.episode_over:
                    agent.imagenav_visualizer.set_vis_dir(
                        f'{scene_id}_{episode_id}_{env.habitat_env.task.current_task_idx}')
                    agent.matching.set_vis_dir(
                        f'{scene_id}_{episode_id}_{env.habitat_env.task.current_task_idx}')
                    agent.planner.set_vis_dir(scene_id,
                        f'{episode_id}_{env.habitat_env.task.current_task_idx}')
                    agent.set_lightglue_vis_dir(
                        f'{scene_id}_{episode_id}_{env.habitat_env.task.current_task_idx}')
                    env.visualizer.set_vis_dir(scene_id,
                        f'{episode_id}_{env.habitat_env.task.current_task_idx}')
                    pbar.reset()

        pbar.close()

        # Log ES status
        try:
            status = candidate_provider.status()
            es_status_log.append({'episode_id': episode_id, 't': t, 'status': status})
        except Exception:
            pass

        scene_ep_id = f'{scene_id}_{episode_id}'
        metrics[scene_ep_id] = {'metrics': all_subtask_metrics}  # ES: no pre-explore, keep all subtasks
        metrics[scene_ep_id]['total_num_steps'] = t
        metrics[scene_ep_id]['sub_task_timesteps'] = agent.sub_task_timesteps[0]  # ES: all subtasks
        metrics[scene_ep_id]['tasks'] = obs_tasks  # ES: all tasks
        metrics[scene_ep_id]['stop_reasons'] = all_stop_reasons
        metrics[scene_ep_id]['trapped_events'] = agent.trapped_events

        try:
            for metric in list(metrics.values())[0]['metrics'][0].keys():
                vals = np.array([y[metric] for y in metrics[scene_ep_id]['metrics']])
                metrics[scene_ep_id][f'{metric}_mean'] = round(float(np.nanmean(vals)), 4)
                metrics[scene_ep_id][f'{metric}_median'] = round(float(np.nanmedian(vals)), 4)
                print(f'{scene_ep_id} {metric}_mean: {metrics[scene_ep_id][f"{metric}_mean"]}')
        except Exception as e:
            print(e)

        print('-' * 60)

        with open(os.path.join(results_dir, 'per_episode_metrics.json'), 'w') as fp:
            json.dump(metrics, fp, indent=4)

        stats = {}
        for metric in list(metrics.values())[0]['metrics'][0].keys():
            vals = [y[metric] for ep in metrics.values() for y in ep['metrics']]
            stats[f'{metric}_mean'] = round(float(np.nanmean(vals)), 4)
            stats[f'{metric}_median'] = round(float(np.nanmedian(vals)), 4)
        with open(os.path.join(results_dir, 'cumulative_metrics.json'), 'w') as fp:
            json.dump(stats, fp, indent=4)

    # ── Save ES status log ──────────────────────────────────────────────
    with open(os.path.join(results_dir, 'es_status.json'), 'w') as fp:
        json.dump(es_status_log, fp, indent=4)

    # ── Timing summary ──────────────────────────────────────────────────
    if args.enable_timing and timing_rows:
        _print_timing_summary(timing_rows, results_dir, save_csv=True)

    candidate_provider.close()
    print('End time:', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    print('Total time:', round(time.time() - start_time, 1), 's')
