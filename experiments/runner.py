"""Run complete ordered episodes with injected environment, policy and services."""

# Portions adapted from LagMemo, Copyright (c) Meta Platforms, Inc. and affiliates.
# Upstream MIT terms and provenance are retained in THIRD_PARTY_NOTICES.md.

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from pprint import pprint
import time
from typing import Mapping, Union

import numpy as np

from experiments.dataset import task_category as _task_category
from experiments.metrics import (
    DISTANCE_TO_GOAL_KEY, SPL_KEY, SUCCESS_KEY, build_subtask_telemetry,
    summarize_episode, summarize_records, write_json_strict,
)
from experiments.recorder import (
    finalize_dispatch_event as _finalize_scsp_dispatch_event,
    finite_metric_summary as _finite_metric_summary,
    record_provider_event as _record_scsp_provider_event,
    write_timing_rows as _write_timing_rows,
)
from pbsp.navigation.cooldown import cooldown_for_category
from pbsp.navigation.planning import (
    active_candidate as _active_candidate,
    apply_runtime_order as _apply_scsp_runtime_order,
    fallback_value as _fallback_value,
    raw_rank_for_position as _scsp_raw_rank_for_position,
    reset_task_planning as _reset_scsp_task_state,
)
from pbsp.types import FallbackReason


@dataclass(frozen=True)
class RunSettings:
    episode_start: int
    episode_end: int
    max_steps_per_task: int
    enable_timing: bool
    query_refresh_cadence_steps: int
    record_decisions: bool = True

    def __post_init__(self):
        if not 0 <= self.episode_start < self.episode_end:
            raise ValueError("Expected a nonempty episode range")
        if min(self.max_steps_per_task, self.query_refresh_cadence_steps) < 1:
            raise ValueError("Step budget and query cadence must be positive")


@dataclass(frozen=True)
class PlanningRunSettings:
    mode: str
    arm: str
    ordered_subtasks: int
    candidate_cap: int
    suffix_weight: float
    idea1_config: Mapping
    provenance: Mapping
    episode_identities: Mapping

    def __post_init__(self):
        if self.mode not in {"off", "shadow", "idea1_only", "ordered_suffix_mvp"}:
            raise ValueError("Unknown planning mode")
        if self.mode != "off" and self.ordered_subtasks < 1:
            raise ValueError("Active planning requires an ordered task count")


@dataclass
class RunArtifacts:
    directory: Union[str, Path]
    metrics: dict = field(default_factory=dict)
    timing_rows: list = field(default_factory=list)
    es_status_log: list = field(default_factory=list)
    distance_cache: dict = field(default_factory=dict)


def run_episodes(
    *, env, evaluator, agent, candidate_provider, category_cooldown, reranker,
    planning, sample_goal, actions, settings: RunSettings,
    planning_settings: PlanningRunSettings, artifacts: RunArtifacts,
    progress_factory=None, clock=time,
):
    """Execute the full episode/task loop; resources belong to the caller.

    The evaluator is read after an action and before measurement reset. Map
    state persists across tasks and is reset once per episode by the provider.
    """
    if progress_factory is None:
        from tqdm import tqdm as progress_factory
    tqdm = progress_factory
    time = clock
    DiscreteNavigationAction = actions
    episode_start, episode_end = settings.episode_start, settings.episode_end
    results_dir = str(artifacts.directory)
    os.makedirs(results_dir, exist_ok=True)
    existing_outputs = [
        name for name in (
            'per_episode_metrics.json', 'timing_log.csv',
            'scsp_provider_events.jsonl', 'scsp_decisions.jsonl',
        ) if os.path.exists(os.path.join(results_dir, name))
    ]
    if existing_outputs:
        raise RuntimeError(
            'MON standard-metric runs require a fresh results directory; '
            f'in-place resume is unsupported: {existing_outputs}'
        )
    scsp_provider_events_path = Path(results_dir) / 'scsp_provider_events.jsonl' if settings.record_decisions else None
    scsp_decisions_path = Path(results_dir) / 'scsp_decisions.jsonl' if settings.record_decisions else None
    metrics = artifacts.metrics
    timing_rows = artifacts.timing_rows
    es_status_log = artifacts.es_status_log
    scsp_distance_cache = artifacts.distance_cache
    scsp_runtime_mode = planning_settings.mode
    scsp_enabled = scsp_runtime_mode != 'off'
    scsp_arm = planning_settings.arm
    scsp_ordered_subtasks = planning_settings.ordered_subtasks
    scsp_candidate_cap = planning_settings.candidate_cap
    scsp_suffix_weight = planning_settings.suffix_weight
    scsp_idea1_config = planning_settings.idea1_config
    scsp_provenance = planning_settings.provenance
    scsp_episode_identities = planning_settings.episode_identities
    scsp_reranker = reranker
    _project_goals_to_navigable = planning.project_goals
    _prepare_scsp_refresh = planning.prepare_refresh
    _record_scsp_shadow_decision = planning.record_decision
    _sample_navmesh_goal = sample_goal

    for ep_i in range(episode_start, episode_end):
        env.reset()
        scene_id = env.habitat_env.current_episode.scene_id.split('/')[-1].split('.')[0]
        episode  = env.habitat_env.current_episode
        episode_id = episode.episode_id
        scsp_episode_identity = scsp_episode_identities.get(
            (str(scene_id), str(episode_id))
        )
        if scsp_enabled and scsp_episode_identity is None:
            raise RuntimeError(
                'SCSP active/shadow execution requires a complete '
                'identity-only episode record'
            )

        candidate_provider.reset_episode(scene_id, episode_id)
        candidate_provider.invalidate_cache()

        agent.reset(start_position=episode.start_position,
                    start_rotation=episode.start_rotation,
                    candidate_provider=candidate_provider)
        agent.set_runtime_context(scene_id, episode_id)
        category_cooldown.reset()
        scsp_reranker.reset()
        scsp_distance_cache.clear()
        category_cooldown_events = []
        scsp_decisions = []
        initial_provider_event = _record_scsp_provider_event(
            candidate_provider,
            phase='initial_query',
            current_task=0,
            scene_id=scene_id,
            episode_id=episode_id,
            output_path=scsp_provider_events_path,
        )
        scsp_provider_events = [initial_provider_event]
        initial_raw_goals = agent.gs_poses
        initial_refresh_pending = scsp_enabled
        if not scsp_enabled:
            initial_goals = _project_goals_to_navigable(
                agent.gs_poses, env.habitat_env.sim.pathfinder
            )
            agent.refresh_gs_poses(initial_goals)
        # Candidate task indices follow the dataset's ordered tasks.
        agent._es_test_gs_override = 3

        t = 0
        scsp_last_task_idx = 0
        scsp_subtask_reset_count = 0
        scsp_boundary_replan_pending = False

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
        episode_frame_times = []
        subtask_frame_times = {}
        pbar = tqdm(total=settings.max_steps_per_task)

        n_subtasks = None  # Read from the first observation.
        while not env.episode_over:
            current_task_idx = env.habitat_env.task.current_task_idx
            # Habitat can advance the task index before episode_over becomes true.
            if n_subtasks is not None and current_task_idx >= n_subtasks:
                print(f'[P6 boundary] current_task_idx={current_task_idx} >= '
                      f'n_subtasks={n_subtasks}; ending episode cleanly', flush=True)
                break
            if scsp_enabled and current_task_idx != scsp_last_task_idx:
                _reset_scsp_task_state(scsp_reranker, scsp_distance_cache)
                scsp_last_task_idx = int(current_task_idx)
                scsp_subtask_reset_count += 1
                scsp_boundary_replan_pending = scsp_runtime_mode in {
                    'idea1_only', 'ordered_suffix_mvp'
                }
                print(
                    '[SCSP_RUNTIME] cleared plan state for '
                    f'current_task={current_task_idx}',
                    flush=True,
                )
            t += 1

            t_step_start = time.perf_counter()

            obs = env.get_observation()
            if t == 1:
                obs_tasks = []
                for task in obs.task_observations['tasks']:
                    obs_task = {k: v for k, v in task.items() if k != 'image'}
                    obs_tasks.append(obs_task)
                n_subtasks = len(obs_tasks)
                if scsp_enabled and n_subtasks != scsp_ordered_subtasks:
                    raise RuntimeError(
                        'SCSP ordered_subtasks does not match runtime tasks'
                    )
                if category_cooldown.enabled:
                    for task in obs_tasks:
                        cooldown_for_category(
                            category_cooldown.cooldown_by_category,
                            _task_category(task),
                        )
                pprint(obs_tasks)
                runtime_categories = tuple(
                    _task_category(task) for task in obs_tasks
                )
                if scsp_episode_identity is not None and (
                    scsp_episode_identity['scene_id'] != str(scene_id)
                    or scsp_episode_identity['runtime_episode_id']
                    != str(episode_id)
                    or tuple(scsp_episode_identity['ordered_categories'])
                    != runtime_categories
                ):
                    raise RuntimeError(
                        'SCSP episode identity does not match runtime tasks'
                    )
                runtime_context_event = _record_scsp_provider_event(
                    candidate_provider,
                    phase='runtime_context',
                    current_task=current_task_idx,
                    scene_id=scene_id,
                    episode_id=episode_id,
                    ordered_categories=runtime_categories,
                    output_path=scsp_provider_events_path,
                )
                scsp_provider_events.append(runtime_context_event)
                if initial_refresh_pending:
                    (
                        initial_goals,
                        initial_decision,
                        retry_events,
                    ) = _prepare_scsp_refresh(
                        phase='initial_runtime_dispatch',
                        runtime_mode=scsp_runtime_mode,
                        arm=scsp_arm,
                        provider=candidate_provider,
                        provider_event=runtime_context_event,
                        raw_goals=initial_raw_goals,
                        current_task=current_task_idx,
                        origin=np.asarray(
                            episode.start_position, dtype=np.float32
                        ),
                        pathfinder=env.habitat_env.sim.pathfinder,
                        agent=agent,
                        reranker=scsp_reranker,
                        distance_cache=scsp_distance_cache,
                        candidate_cap=scsp_candidate_cap,
                        suffix_weight=scsp_suffix_weight,
                        idea1_config=scsp_idea1_config,
                        provenance=scsp_provenance,
                        episode_identity=scsp_episode_identity,
                        scene_id=scene_id,
                        episode_id=episode_id,
                        ordered_categories=runtime_categories,
                        provider_events_path=scsp_provider_events_path,
                    )
                    scsp_provider_events.extend(retry_events)
                    initial_goals, initial_dispatch_metadata = (
                        _apply_scsp_runtime_order(
                            initial_goals,
                            initial_decision,
                            scsp_runtime_mode,
                        )
                    )
                    initial_dispatch = agent.dispatch_initial_candidates(
                        initial_goals, initial_dispatch_metadata,
                        task_idx=current_task_idx,
                        candidate_batch=initial_decision.get('candidate_batch'),
                    )
                    initial_refresh_applied = initial_dispatch.refresh_applied
                    initial_refresh_error = initial_dispatch.refresh_error
                    initial_realized = initial_dispatch.realized_active_position
                    initial_realized_rank = initial_dispatch.realized_raw_rank
                    initial_decision = _finalize_scsp_dispatch_event(
                        initial_decision,
                        initial_dispatch_metadata,
                        cooldown_event=None,
                        refresh_applied=initial_refresh_applied,
                        realized_active_position=initial_realized,
                        realized_raw_rank=initial_realized_rank,
                        refresh_error=initial_refresh_error,
                    )
                    scsp_decisions.append(
                        _record_scsp_shadow_decision(
                            decision_event=initial_decision,
                            output_path=scsp_decisions_path,
                        )
                    )
                    if not initial_refresh_applied:
                        raise RuntimeError(
                            'SCSP initial candidate refresh failed closed: '
                            f'{initial_refresh_error}'
                        )
                    initial_refresh_pending = False
                    initial_raw_goals = None

            local_step = int(
                agent.sub_task_timesteps[0][current_task_idx]
            )
            if category_cooldown.enabled and local_step > 0:
                realized_active = _active_candidate(agent, current_task_idx)
                external_switch = category_cooldown.observe_active(
                    current_task_idx,
                    realized_active,
                    local_step,
                )
                if external_switch:
                    external_event = {
                        'task_idx': int(current_task_idx),
                        'category': _task_category(
                            obs_tasks[current_task_idx]
                        ),
                        'subtask_step': local_step,
                        'cooldown_steps': cooldown_for_category(
                            category_cooldown.cooldown_by_category,
                            _task_category(obs_tasks[current_task_idx]),
                        ),
                        'reason': 'EXTERNAL_ACTIVE_SWITCH',
                        'realized_switch': 1,
                        'realized_active_position': (
                            None
                            if realized_active is None
                            else [float(value) for value in realized_active]
                        ),
                    }
                    category_cooldown_events.append(external_event)
                    print(
                        '[CATEGORY_COOLDOWN] '
                        f'{json.dumps(external_event, sort_keys=True)}',
                        flush=True,
                    )

            # Stream the current observation or advance the snapshot replay.
            dt_push = 0.0
            try:
                t0 = time.perf_counter()
                if getattr(candidate_provider, '_live_mode', False):
                    _rgb_raw = obs.rgb                                  # (H,W,3) uint8
                    _dep_raw = obs.depth                                # (H,W) float32 m
                    _gp_raw  = np.array(obs.globalpose, dtype=np.float32)  # (7,)
                    candidate_provider.push_live_frame(_rgb_raw, _dep_raw, _gp_raw)
                else:
                    candidate_provider.push_frame(t - 1)
                dt_push = (time.perf_counter() - t0) * 1000
            except Exception as e:
                print(f'[P5] push_frame warning: {e}', flush=True)

            # ── periodic goal refresh ───────────────────────────────────
            dt_goals = 0.0
            is_upgrade = False
            periodic_refresh_due = t % settings.query_refresh_cadence_steps == 0
            if periodic_refresh_due or scsp_boundary_replan_pending:
                refresh_phase = (
                    'periodic_refresh'
                    if periodic_refresh_due
                    else 'subtask_boundary'
                )
                t0 = time.perf_counter()
                new_goals = None
                query_error = None
                if scsp_boundary_replan_pending and not periodic_refresh_due:
                    try:
                        cached_batch = candidate_provider.get_candidate_batch()
                        if cached_batch is not None:
                            new_goals = {
                                task_idx: [list(position) for position in positions]
                                for task_idx, positions
                                in cached_batch.positions_by_task
                            }
                    except Exception as e:
                        query_error = f'{type(e).__name__}: {e}'
                else:
                    candidate_provider.invalidate_cache()
                    try:
                        new_goals = candidate_provider.get_all_goals()
                    except Exception as e:
                        query_error = f'{type(e).__name__}: {e}'
                        print(f'[P4] goal refresh warning: {e}', flush=True)
                periodic_provider_event = _record_scsp_provider_event(
                    candidate_provider,
                    phase=refresh_phase,
                    current_task=current_task_idx,
                    scene_id=scene_id,
                    episode_id=episode_id,
                    ordered_categories=tuple(
                        _task_category(task) for task in obs_tasks
                    ),
                    output_path=scsp_provider_events_path,
                    query_error=query_error,
                )
                scsp_provider_events.append(periodic_provider_event)
                if scsp_enabled or new_goals:
                    projected_goals = None
                    periodic_decision = None
                    dispatch_metadata = None
                    try:
                        if scsp_enabled:
                            (
                                projected_goals,
                                periodic_decision,
                                retry_events,
                            ) = _prepare_scsp_refresh(
                                phase=refresh_phase,
                                runtime_mode=scsp_runtime_mode,
                                arm=scsp_arm,
                                provider=candidate_provider,
                                provider_event=(
                                    periodic_provider_event or {}
                                ),
                                raw_goals=new_goals or {},
                                current_task=current_task_idx,
                                origin=np.asarray(
                                    obs.globalpose[4:], dtype=np.float32
                                ),
                                pathfinder=(
                                    env.habitat_env.sim.pathfinder
                                ),
                                agent=agent,
                                reranker=scsp_reranker,
                                distance_cache=scsp_distance_cache,
                                candidate_cap=scsp_candidate_cap,
                                suffix_weight=scsp_suffix_weight,
                                idea1_config=scsp_idea1_config,
                                provenance=scsp_provenance,
                                episode_identity=scsp_episode_identity,
                                scene_id=scene_id,
                                episode_id=episode_id,
                                ordered_categories=tuple(
                                    _task_category(task)
                                    for task in obs_tasks
                                ),
                                provider_events_path=(
                                    scsp_provider_events_path
                                ),
                                preserve_existing_on_missing=True,
                            )
                            scsp_provider_events.extend(retry_events)
                            periodic_decision['subtask_boundary_replan'] = bool(
                                scsp_boundary_replan_pending
                            )
                            periodic_decision['periodic_refresh_due'] = bool(
                                periodic_refresh_due
                            )
                            projected_goals, dispatch_metadata = (
                                _apply_scsp_runtime_order(
                                    projected_goals,
                                    periodic_decision,
                                    scsp_runtime_mode,
                                )
                            )
                        else:
                            projected_goals = _project_goals_to_navigable(
                                new_goals, env.habitat_env.sim.pathfinder
                            )
                    except Exception as e:
                        print(f'[P4] goal refresh warning: {e}', flush=True)
                    dispatch = agent.dispatch_candidates(
                        projected_goals, dispatch_metadata,
                        task_idx=current_task_idx,
                        category=_task_category(obs_tasks[current_task_idx]),
                        local_step=local_step,
                        category_cooldown=category_cooldown,
                        candidate_batch=(
                            periodic_decision.get('candidate_batch')
                            if periodic_decision is not None else None
                        ),
                    )
                    projected_goals = dispatch.projected_goals
                    dispatch_metadata = dispatch.metadata
                    cooldown_event = dispatch.cooldown_event
                    refresh_applied = dispatch.refresh_applied
                    refresh_error = dispatch.refresh_error
                    if cooldown_event is not None:
                        category_cooldown_events.append(cooldown_event)
                        print(
                            '[CATEGORY_COOLDOWN] '
                            f'{json.dumps(cooldown_event, sort_keys=True)}',
                            flush=True,
                        )
                    if periodic_decision is not None:
                        realized_active = _active_candidate(
                            agent, current_task_idx
                        )
                        realized_raw_rank = _scsp_raw_rank_for_position(
                            periodic_decision.get('candidate_batch'),
                            current_task_idx,
                            realized_active,
                        )
                        periodic_decision = _finalize_scsp_dispatch_event(
                            periodic_decision,
                            dispatch_metadata,
                            cooldown_event=cooldown_event,
                            refresh_applied=refresh_applied,
                            realized_active_position=realized_active,
                            realized_raw_rank=realized_raw_rank,
                            refresh_error=refresh_error,
                        )
                        scsp_decisions.append(
                            _record_scsp_shadow_decision(
                                decision_event=periodic_decision,
                                output_path=scsp_decisions_path,
                            )
                        )
                dt_goals = (time.perf_counter() - t0) * 1000
                is_upgrade = True
                scsp_boundary_replan_pending = False

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
            episode_frame_times.append(dt_total)
            subtask_frame_times.setdefault(current_task_idx, []).append(dt_total)

            # ── record timing ───────────────────────────────────────────
            if settings.enable_timing:
                timing_rows.append({
                    'scene_id':       scene_id,
                    'episode_id':      episode_id,
                    'subtask_index':   current_task_idx,
                    'frame_index':     t,
                    'snap':            candidate_provider.current_snap,
                    'dt_total_ms':     dt_total,
                    'dt_push_ms':      dt_push,
                    'dt_goals_ms':     dt_goals,
                    'dt_act_ms':       dt_act,
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
                ep_metrics = evaluator.get_episode_metrics()
                ep_metrics.pop('goat_top_down_map', None)
                path_lengths = evaluator.get_subtask_path_lengths()
                ep_metrics.update(build_subtask_telemetry(
                    ep_metrics,
                    actual_path_length=path_lengths['actual_path_length'],
                    shortest_path_length=path_lengths['shortest_path_length'],
                    steps=len(subtask_frame_times.get(current_task_idx, [])),
                    frame_times_ms=subtask_frame_times.get(current_task_idx, []),
                ))
                evaluator.reset_subtask()
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
        if n_subtasks is None or len(all_subtask_metrics) != n_subtasks:
            raise RuntimeError(
                f'incomplete MON episode {scene_ep_id}: '
                f'captured={len(all_subtask_metrics)} expected={n_subtasks}'
            )
        metrics[scene_ep_id] = {'metrics': all_subtask_metrics}
        metrics[scene_ep_id]['total_num_steps'] = t
        metrics[scene_ep_id]['sub_task_timesteps'] = agent.sub_task_timesteps[0]
        metrics[scene_ep_id]['tasks'] = obs_tasks
        metrics[scene_ep_id]['stop_reasons'] = all_stop_reasons
        metrics[scene_ep_id]['trapped_events'] = agent.trapped_events
        metrics[scene_ep_id]['scsp_provider_event_summary'] = {
            'event_count': len(scsp_provider_events),
            'invalid_batch_count': sum(
                event['batch_validation_status'] != 'valid'
                for event in scsp_provider_events
            ),
            'would_invalidate_count': sum(
                bool(event['version_gate'].get('would_invalidate', True))
                for event in scsp_provider_events
            ),
            'jsonl_recorded_count': sum(
                bool(event.get('recorded_to_jsonl', False))
                for event in scsp_provider_events
            ),
            'recording_error_count': sum(
                'recording_error' in event
                for event in scsp_provider_events
            ),
        }
        if scsp_enabled:
            planned_decisions = [
                event
                for event in scsp_decisions
                if event.get('status') == 'planned'
            ]
            decision_summary = {
                'runtime_mode': scsp_runtime_mode,
                'event_count': len(scsp_decisions),
                'planned_count': len(planned_decisions),
                'structured_fallback_count': sum(
                    event.get('status') == 'fallback'
                    for event in scsp_decisions
                ),
                'planner_fallback_count': sum(
                    _fallback_value(
                        event['suffix_plan'].fallback_reason
                    ) != FallbackReason.NONE.value
                    for event in planned_decisions
                    if event.get('suffix_plan') is not None
                ),
                'idea1_selected_changed_count': sum(
                    bool(event['changed']['idea1_from_legacy'])
                    for event in planned_decisions
                ),
                'suffix_selected_changed_count': sum(
                    bool(event['changed']['suffix_from_idea1'])
                    for event in planned_decisions
                ),
                'suffix_eligible_count': sum(
                    bool(event.get('dispatch', {}).get('suffix_eligible'))
                    for event in scsp_decisions
                ),
                'suffix_proposal_count': sum(
                    event.get('proposed_order_source') == 'suffix'
                    for event in scsp_decisions
                ),
                'suffix_dispatch_count': sum(
                    event.get('dispatch_order_source') == 'suffix'
                    and bool(event.get('refresh_applied'))
                    for event in scsp_decisions
                ),
                'realized_suffix_intervention_count': sum(
                    event.get('dispatch_order_source') == 'suffix'
                    and bool(event.get('dispatch', {}).get(
                        'suffix_intervention'
                    ))
                    and not bool(event.get('dispatch', {}).get(
                        'cooldown_vetoed'
                    ))
                    and event.get('selected_raw_ranks', {}).get('dispatched')
                    is not None
                    and event.get('selected_raw_ranks', {}).get('dispatched')
                    == event.get('dispatch', {}).get(
                        'post_cooldown_selected_raw_rank'
                    )
                    and event.get('selected_raw_ranks', {}).get('dispatched')
                    != event.get('selected_raw_ranks', {}).get('idea1')
                    for event in scsp_decisions
                ),
                'idea1_dispatch_count': sum(
                    event.get('dispatch_order_source')
                    in {'idea1', 'idea1_fallback'}
                    and bool(event.get('refresh_applied'))
                    for event in scsp_decisions
                ),
                'cooldown_veto_count': sum(
                    bool(event.get('dispatch', {}).get('cooldown_vetoed'))
                    for event in scsp_decisions
                ),
                'synchronous_requery_count': sum(
                    int(event.get('synchronous_requery_count', 0))
                    for event in scsp_decisions
                ),
                'synchronous_requery_success_count': sum(
                    bool(event.get('synchronous_requery_succeeded'))
                    for event in scsp_decisions
                ),
                'stale_suffix_dispatch_count': sum(
                    int(event.get('dispatch', {}).get(
                        'stale_suffix_dispatch', 0
                    ))
                    for event in scsp_decisions
                ),
                'refresh_applied_count': sum(
                    bool(event.get('refresh_applied'))
                    for event in scsp_decisions
                ),
                'recording_error_count': sum(
                    'recording_error' in event
                    for event in scsp_decisions
                ),
                'distance_cache_entries': len(scsp_distance_cache),
                'subtask_state_reset_count': scsp_subtask_reset_count,
                'subtask_boundary_replan_count': sum(
                    bool(event.get('subtask_boundary_replan'))
                    for event in scsp_decisions
                ),
                'shadow_only': scsp_runtime_mode == 'shadow',
            }
            metrics[scene_ep_id]['scsp_decision_summary'] = decision_summary
            if scsp_runtime_mode == 'shadow':
                metrics[scene_ep_id]['scsp_shadow_decision_summary'] = dict(
                    decision_summary
                )
        metrics[scene_ep_id].update(summarize_episode(
            all_subtask_metrics,
            total_steps=t,
            stop_reasons=all_stop_reasons,
            frame_times_ms=episode_frame_times,
            categories=[_task_category(task) for task in obs_tasks],
        ))
        if category_cooldown.enabled:
            metrics[scene_ep_id]['query_switch_cooldown_by_category'] = dict(
                category_cooldown.cooldown_by_category
            )
            metrics[scene_ep_id]['category_cooldown_events'] = (
                category_cooldown_events
            )

        for metric in (
            DISTANCE_TO_GOAL_KEY,
            SUCCESS_KEY,
            SPL_KEY,
        ):
            metric_summary = _finite_metric_summary(
                metrics[scene_ep_id]['metrics'], metric
            )
            metrics[scene_ep_id][f'{metric}_mean'] = metric_summary['mean']
            metrics[scene_ep_id][f'{metric}_median'] = metric_summary['median']
            if metric == DISTANCE_TO_GOAL_KEY:
                metrics[scene_ep_id][f'{metric}_finite_count'] = (
                    metric_summary['finite_count']
                )
                metrics[scene_ep_id][f'{metric}_unreachable_count'] = (
                    metric_summary['unreachable_count']
                )
            print(
                f'{scene_ep_id} {metric}_mean: '
                f'{metrics[scene_ep_id][f"{metric}_mean"]}'
            )

        print('-' * 60)

        write_json_strict(
            Path(results_dir) / 'per_episode_metrics.json', metrics
        )

        stats = {}
        for metric in (
            DISTANCE_TO_GOAL_KEY,
            SUCCESS_KEY,
            SPL_KEY,
        ):
            metric_summary = _finite_metric_summary(
                [y for ep in metrics.values() for y in ep['metrics']], metric
            )
            stats[f'{metric}_mean'] = metric_summary['mean']
            stats[f'{metric}_median'] = metric_summary['median']
            if metric == DISTANCE_TO_GOAL_KEY:
                stats[f'{metric}_finite_count'] = metric_summary['finite_count']
                stats[f'{metric}_unreachable_count'] = (
                    metric_summary['unreachable_count']
                )
        stats.update(summarize_records(
            metrics,
            frame_times_ms=(
                [row['dt_total_ms'] for row in timing_rows]
                if settings.enable_timing
                else None
            ),
        ))
        write_json_strict(Path(results_dir) / 'cumulative_metrics.json', stats)
        if settings.enable_timing:
            _write_timing_rows(timing_rows, results_dir)

    artifacts.metrics = metrics
    return artifacts
