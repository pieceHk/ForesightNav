"""Validate an unapplied identity logging change against original functions and raw Qa18."""
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import random
from types import SimpleNamespace

import experiments.blp_observer as proposed
from action_path_metrics import action_paths
from pbsp.navigation.blp import BeliefParameters

WORK = Path(__file__).resolve().parents[3]
OUT = WORK / 'results/exp4'
PENDING = OUT / 'code/pending_identity_binding'
RUNTIME = Path('/nvme/khy/ForesightNav_exp4_depthfix')
RUN = RUNTIME / 'results/exp4/runs/20261109_160731_469662_exp4_calibration_calibration-a3-fold0-QaLdnwvtxbs'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return [json.loads(line) for line in path.open()]


def main():
    spec = importlib.util.spec_from_file_location('original_blp_observer', PENDING / 'base_blp_observer.py')
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    original_writes, proposed_writes = [], []
    original.append = lambda p, f, r: original_writes.append((f, copy.deepcopy(r)))
    proposed.append = lambda p, f, r: proposed_writes.append((f, copy.deepcopy(r)))

    # Registration and verification outputs are compared independently of the
    # action-target read path. This is a function regression, not a native run.
    contexts = [SimpleNamespace(blp_context={}, blp_parameters=BeliefParameters()) for _ in range(2)]
    rng = random.Random(4817)
    operations = 0
    for episode in range(6):
        for p in contexts:
            p.blp_context.clear()
        for step in range(50):
            task = step % 2
            positions = [[round(rng.uniform(-2., 2.), 2), 0., round(rng.uniform(-2., 2.), 2)]
                         for _ in range(4)]
            records = [SimpleNamespace(position=x, semantic_score=.15 + .2 * rng.random()) for x in positions]
            batch = SimpleNamespace(scene_id='regression', episode_id=str(episode),
                                    records_for_task=lambda _, records=records: records)
            agent = SimpleNamespace(gs_poses={task: [positions[0]]}, frontier_flag={task: 0},
                       candidate_evidence=SimpleNamespace(coverage_views=lambda *args: 3))
            evaluator = SimpleNamespace(get_episode_metrics=lambda: {'goat_sub-task_success': 0},
                       label_candidate=lambda *args: {'y': 0, 'criterion': 'function_regression_only'})
            for module, p in zip((original, proposed), contexts):
                p.blp_context.update(t=step + 1, k=task, path_m=step * .25, batch=batch)
                for record in records:
                    module.register_candidate(p, task, record)
                module.prepare_action(p, agent, task)
                module.observe_verification(p, agent, evaluator, {'verification_event': {
                    'task_idx': task, 'candidate_position': positions[0],
                    'confirmed': step % 7 == 0, 'candidate_source': 'query_candidate'}})
            operations += len(records) + 1
            assert original_writes == proposed_writes
            clean = {k: v for k, v in contexts[1].blp_context.items() if k != 'candidate_position_identities'}
            assert contexts[0].blp_context == clean
        for module, p in zip((original, proposed), contexts):
            module.finalize_visit(p, 'diagnostic_end')
        assert original_writes == proposed_writes
    event_preservation = {'scope': 'paired function regression using synthetic query and verification inputs; no native navigation',
                          'operations': operations, 'logged_rows': len(original_writes),
                          'all_registration_and_verification_rows_equal': True,
                          'all_existing_context_fields_equal': True,
                          'event_sha256': hashlib.sha256(json.dumps(original_writes, sort_keys=True).encode()).hexdigest()}

    # Before any verification occurs, the native Qa18 records allow an exact
    # replay of all query registrations and bindings through frame 29.
    decisions_path = next(RUN.rglob('scsp_decisions.jsonl'))
    snapshots_path = next(RUN.rglob('blp_snapshots.jsonl'))
    events_path = next(RUN.rglob('blp_events.jsonl'))
    trace_path = next(RUN.rglob('episodes/QaLdnwvtxbs/18/trajectory.jsonl'))
    decisions = [r for r in load(decisions_path) if str(r.get('runtime_episode_id')) == '18']
    snapshots = [r for r in load(snapshots_path) if str(r.get('episode_id')) == '18']
    events = [r for r in load(events_path) if str(r.get('episode_id')) == '18']
    trace = load(trace_path)
    assert len(decisions) == len(snapshots)
    refreshes = {s['t']: d for s, d in zip(snapshots, decisions) if s['t'] < 30}
    contexts = [SimpleNamespace(blp_context={}) for _ in range(2)]
    original_writes.clear()
    proposed_writes.clear()
    reconstructed = copy.deepcopy(trace)
    changed = []
    for row in trace[:29]:
        t = row['frame_index']
        for module, p in zip((original, proposed), contexts):
            if t in refreshes:
                p.blp_context.update(t=t, batch=SimpleNamespace(scene_id='QaLdnwvtxbs', episode_id='18'))
                for task, records in refreshes[t]['candidate_batch']['records_by_task']:
                    for record in records:
                        module.register_candidate(p, task, SimpleNamespace(**record))
        target = {k: v for k, v in row['blp_action_target'].items()
                  if k not in {'frame_index', 'candidate_id', 'query_identity_status', 'query_identity_resolution'}}
        before = original.bind_action_target(contexts[0], {'blp_action_target': target}, row['subtask_index'], t)
        after = proposed.bind_action_target(contexts[1], {'blp_action_target': target}, row['subtask_index'], t)
        assert before == row['blp_action_target']
        if before != after:
            assert before['query_identity_status'] == 'unresolved'
            assert after['query_identity_resolution'] == 'exact_registered_position'
            reconstructed[t - 1]['blp_action_target'] = after
            changed.append({'frame': t, 'before': before, 'after': after})
    expected_appearance = [r for r in events if r['event'] == 'candidate_first_appearance' and r['first_t'] < 30]
    assert [r for _, r in original_writes] == expected_appearance
    assert original_writes == proposed_writes
    assert [r['frame'] for r in changed] == list(range(10, 20))
    metrics_path = next(RUN.rglob('per_episode_metrics.json'))
    stages = json.loads(metrics_path.read_text())['QaLdnwvtxbs_18']['metrics']
    stage_paths = {('QaLdnwvtxbs', '18', k): s['goat_sub-task_path_length_m']
                   for k, s in enumerate(stages)}
    assert not any(r['y'] == 0 for r in events if r['event'] == 'verification_final')
    raw_paths = action_paths(events, {('QaLdnwvtxbs', '18'): trace}, None, stage_paths)
    derived_paths = action_paths(events, {('QaLdnwvtxbs', '18'): reconstructed}, None, stage_paths)
    assert not raw_paths['invalid_path_complete'] and len(raw_paths['invalid_path_association_gaps']) == 10
    assert derived_paths['invalid_path_complete']
    assert derived_paths['accepted_action_path_m'] == raw_paths['accepted_action_path_m']

    out = {'checked_at_utc': datetime.now(timezone.utc).isoformat(),
           'runtime_applied': False, 'native_run_launched': False, 'fits_parameters': False,
           'source_sha256': {'base': sha(PENDING / 'base_blp_observer.py'),
                             'proposed': sha(WORK / 'experiments/blp_observer.py'),
                             'validation': sha(Path(__file__))},
           'function_regression': event_preservation,
           'real_Qa18': {'scope': 'counterfactual log reconstruction from raw native query/trace data, not a rerun',
                        'run_directory': str(RUN), 'first_appearance_rows_exact_match': len(expected_appearance),
                        'accepted_trace_rows': len(trace), 'replayed_initial_frames': 29, 'changed': changed,
                        'raw_path_complete': raw_paths['invalid_path_complete'],
                        'raw_gap_count': len(raw_paths['invalid_path_association_gaps']),
                        'derived_path_complete': derived_paths['invalid_path_complete'],
                        'native_stage_checks': derived_paths['native_stage_path_check_count'],
                        'native_stage_path_max_abs_error_m': derived_paths['native_stage_path_max_abs_error_m'],
                        'physical_path_unchanged': True,
                        'input_sha256': {str(p.relative_to(RUN)): sha(p) for p in
                                         [decisions_path, snapshots_path, events_path, trace_path, metrics_path]},
                        'raw_trace_and_calibration_events_modified': False,
                        'derived_output_not_used_for_calibration_or_formal_scores': True}}
    destination = OUT / 'diagnostics/identity-binding-validation.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(out, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'function_regression': event_preservation,
                      'real_changed_frames': [r['frame'] for r in changed],
                      'real_first_appearance_match': len(expected_appearance),
                      'raw_path_complete': False, 'derived_path_complete': True,
                      'native_stage_path_max_abs_error_m': derived_paths['native_stage_path_max_abs_error_m'],
                      'runtime_applied': False}, indent=2))


if __name__ == '__main__':
    main()

