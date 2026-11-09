"""Counterfactual logging/analysis checks; never edits raw data or fits parameters."""
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import base_candidate_identities as base_ids
import proposed_candidate_identities as new_ids
from action_path_metrics import action_paths, xyz
from native_candidate_gt import NativeCandidateGT

REPO = Path('/nvme/khy/ForesightNav_exp4_depthfix')
ROOT = REPO / 'results/exp4'
ISOLATED = Path('/nvme/khy/ForesightNav_exp4_identityfix')


def load(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def main():
    old_path = REPO / 'experiments/blp_observer.py'
    new_path = ISOLATED / 'experiments/blp_observer.py'
    assert sha(old_path) == 'f3587f64f88b5470f6917258c35c0924eb77dd4f8de22ba57de962b4467dddf1'
    assert sha(new_path) == '12dd680b3f4d102fd70277a193d1c85ad5deb219dfb04846e3060df9304af9fb'
    original, proposed = module('original_observer_xb', old_path), module('proposed_observer_xb', new_path)
    audit = json.loads((ROOT / 'meta/completed-calibration-scene-audits.json').read_text())
    scenes = audit['completed_scenes']
    scenes = list(scenes.values()) if isinstance(scenes, dict) else scenes
    entry = next(x for x in scenes if x['scene'] == 'XB4GS9ShBRE')
    run = Path(entry['run_directory'])
    decisions_file = next(run.rglob('scsp_decisions.jsonl'))
    snapshots_file = next(run.rglob('blp_snapshots.jsonl'))
    events_file = next(run.rglob('blp_events.jsonl'))
    metrics_file = next(run.rglob('per_episode_metrics.json'))
    decisions, snapshots, all_events = load(decisions_file), load(snapshots_file), load(events_file)
    metrics = json.loads(metrics_file.read_text())
    truth = NativeCandidateGT(run)
    cases = []
    for episode, cutoff, expected_frames in [('2', 30, list(range(20, 30))),
                                             ('11', 28, list(range(20, 28)))]:
        trace_file = next(run.rglob('episodes/XB4GS9ShBRE/' + episode + '/trajectory.jsonl'))
        trace = load(trace_file)
        events = [x for x in all_events if str(x.get('episode_id')) == episode]
        d = [x for x in decisions if str(x.get('runtime_episode_id')) == episode]
        s = [x for x in snapshots if str(x.get('episode_id')) == episode]
        assert len(d) == len(s)
        refreshes = {a['t']: b for a, b in zip(s, d) if a['t'] < cutoff}
        assert not any(x['event'] == 'candidate_checked' and x['t'] < cutoff for x in events)
        contexts = [SimpleNamespace(blp_context={}), SimpleNamespace(blp_context={})]
        writes = [[], []]
        original.append = lambda p, f, r: writes[0].append((f, copy.deepcopy(r)))
        proposed.append = lambda p, f, r: writes[1].append((f, copy.deepcopy(r)))
        derived, changed, drift = copy.deepcopy(trace), [], []
        for row in trace:
            t = row['frame_index']
            if t >= cutoff:
                break
            if t in refreshes:
                for mod, context in zip((original, proposed), contexts):
                    context.blp_context.update(t=t, batch=SimpleNamespace(scene_id='XB4GS9ShBRE', episode_id=episode))
                    for task, records in refreshes[t]['candidate_batch']['records_by_task']:
                        for record in records:
                            mod.register_candidate(context, task, SimpleNamespace(**record))
            target = {k: v for k, v in row['blp_action_target'].items()
                      if k not in {'frame_index', 'candidate_id', 'query_identity_status', 'query_identity_resolution'}}
            before = original.bind_action_target(contexts[0], {'blp_action_target': target}, row['subtask_index'], t)
            after = proposed.bind_action_target(contexts[1], {'blp_action_target': target}, row['subtask_index'], t)
            assert before == row['blp_action_target'], (episode, t)
            if before != after:
                assert before['query_identity_status'] == 'unresolved'
                assert after['query_identity_resolution'] == 'exact_registered_position'
                derived[t - 1]['blp_action_target'] = after
                changed.append({'frame': t, 'before': before, 'after': after})
                if t == 20:
                    point = target['position']
                    registry = contexts[0].blp_context['candidate_registry'][row['subtask_index']]
                    drift.append(min(math.hypot(x['position'][0]-point[0], x['position'][2]-point[2]) for x in registry))
        assert [x['frame'] for x in changed] == expected_frames
        expected = [x for x in events if x['event'] == 'candidate_first_appearance' and x['first_t'] < cutoff]
        assert [r for _, r in writes[0]] == expected and writes[0] == writes[1]
        native_paths = {('XB4GS9ShBRE', episode, k): v['goat_sub-task_path_length_m']
                        for k, v in enumerate(metrics['XB4GS9ShBRE_' + episode]['metrics'])}
        pair = ('XB4GS9ShBRE', episode)
        old_events, old_trace, old_stats = base_ids.canonicalize(events, {pair: trace})
        raw_paths = action_paths(old_events, old_trace, truth, native_paths)
        logged_events, logged_trace, _ = base_ids.canonicalize(events, {pair: derived})
        logged_paths = action_paths(logged_events, logged_trace, truth, native_paths)
        new_events, new_trace, new_stats = new_ids.canonicalize(events, {pair: derived})
        corrected_paths = action_paths(new_events, new_trace, truth, native_paths)
        assert not raw_paths['invalid_path_complete']
        assert logged_paths['invalid_path_complete'] and corrected_paths['invalid_path_complete']
        assert old_stats['canonical_checked_identities'] == new_stats['canonical_checked_identities']
        relevant = lambda values: [x for x in values if x['event'] in ('verification_final', 'empty_activation')]
        assert relevant(old_events) == relevant(new_events)
        assert raw_paths['accepted_action_path_m'] == logged_paths['accepted_action_path_m'] == corrected_paths['accepted_action_path_m']
        invalid_ids = {(x['k'], x['candidate_id']) for x in new_events if x['event'] == 'verification_final' and x['y'] == 0}
        expected_added = 0.
        for offset, (before, after) in enumerate(zip(logged_trace[pair], new_trace[pair])):
            a, b = before['blp_action_target'], after['blp_action_target']
            if a.get('candidate_id') == b.get('candidate_id') or (b['task_idx'], b.get('candidate_id')) not in invalid_ids:
                continue
            if truth(pair[0], pair[1], b['task_idx'], b['position'])['y'] == 0:
                end = xyz(after['world_pose_after']) if 'world_pose_after' in after else xyz(new_trace[pair][offset+1]['world_pose']) if offset+1 < len(new_trace[pair]) else xyz(after['world_pose'])
                expected_added += math.dist(xyz(after['world_pose']), end)
        delta = corrected_paths['invalid_visit_path_m'] - logged_paths['invalid_visit_path_m']
        assert math.isclose(delta, expected_added, abs_tol=1e-8)
        cases.append({'episode': episode, 'scope': 'counterfactual log reconstruction; not a native rerun or formal score',
                      'replayed_frames': cutoff-1, 'changed': changed, 'registry_min_distance_at_gap_m': drift,
                      'first_appearance_events_exact_match': len(expected),
                      'raw_gap_count': len(raw_paths['invalid_path_association_gaps']),
                      'logger_only_derived_path_complete': logged_paths['invalid_path_complete'],
                      'logger_and_alias_derived_path_complete': corrected_paths['invalid_path_complete'],
                      'invalid_visit_path_logger_only_m': logged_paths['invalid_visit_path_m'],
                      'invalid_visit_path_with_alias_m': corrected_paths['invalid_visit_path_m'],
                      'invalid_path_added_m': delta, 'expected_native_GT_invalid_alias_displacement_m': expected_added,
                      'accepted_physical_path_unchanged': True, 'fit_input_rows_identical': True,
                      'native_stage_checks': corrected_paths['native_stage_path_check_count'],
                      'native_stage_path_max_abs_error_m': corrected_paths['native_stage_path_max_abs_error_m'],
                      'unchecked_aliases': new_stats['unchecked_first_appearance_aliases'],
                      'input_sha256': {str(x.relative_to(run)): sha(x) for x in [decisions_file, snapshots_file, events_file, metrics_file, trace_file]}})
    total_rows, aliases, group_before, group_after = 0, 0, 0, 0
    input_hashes = []
    for scene in scenes:
        source = next(Path(scene['run_directory']).rglob('blp_events.jsonl'))
        values = load(source)
        before, _, bs = base_ids.canonicalize(values)
        after, _, ns = new_ids.canonicalize(values)
        relevant = lambda rows: [x for x in rows if x['event'] in ('verification_final', 'empty_activation')]
        left, right = relevant(before), relevant(after)
        assert left == right
        assert bs['canonical_checked_identities'] == ns['canonical_checked_identities']
        total_rows += len(left)
        aliases += ns['unchecked_first_appearance_aliases']
        group_before += bs['canonical_checked_identities']
        group_after += ns['canonical_checked_identities']
        input_hashes.append({'scene': scene['scene'], 'raw_events_sha256': sha(source),
                             'fit_input_sha256': hashlib.sha256(json.dumps(left, sort_keys=True).encode()).hexdigest()})
    out = {'checked_at_utc': datetime.now(timezone.utc).isoformat(), 'runtime_applied': False,
           'native_runs_launched': False, 'parameters_fitted': False, 'CPU_waiter_modified': False,
           'source_sha256': {'original_observer': sha(old_path), 'pending_observer': sha(new_path),
                             'base_identity_analysis': base_ids.SOURCE_SHA256,
                             'proposed_identity_analysis': new_ids.SOURCE_SHA256, 'validation': sha(Path(__file__))},
           'cases': cases, 'all_closed_calibration': {'episodes': audit['complete_episodes'],
                'scenes': len(scenes), 'fit_rows_identical': total_rows,
                'checked_identities_before': group_before, 'checked_identities_after': group_after,
                'unchecked_aliases': aliases, 'q_pi_Cexp_inputs_identical': True,
                'input_sha256': input_hashes},
           'raw_trace_events_modified': False, 'derived_data_used_for_calibration_or_scores': False}
    destination = ROOT / 'diagnostics/unchecked-alias-XB-validation.json'
    destination.write_text(json.dumps(out, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'cases': [{k:v for k,v in x.items() if k not in ['changed', 'input_sha256']} for x in cases],
                      'all_closed_calibration': {k:v for k,v in out['all_closed_calibration'].items() if k != 'input_sha256'},
                      'runtime_applied': False}, indent=2), flush=True)


if __name__ == '__main__':
    main()
