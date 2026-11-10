"""Audit exported complete episodes without finalizing a running main stage."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import report_variants as analysis
from audit_first_full_episode import rows, load
from candidate_identities import canonicalize
from experiments.exp4_collect import source_fingerprint
from native_candidate_gt import NativeCandidateGT
from pbsp.navigation.blp_parameters import CalibrationDocument

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('full', 'nearest', 'myopic', 'suffix_only'), default='full')
    args = parser.parse_args()
    ledger = load(ROOT / f'meta/evaluation-{args.stage}-runs.json')
    cp2b = load(ROOT / 'meta/CP2b_acceptance.json')
    parameter = ROOT / 'meta/calibration.json'
    assert ledger['source_fingerprint'] == source_fingerprint() == cp2b['source_fingerprint']
    assert digest(parameter) == ledger['parameter_sha256'] == cp2b['parameter_sha256']
    document = CalibrationDocument.from_mapping(load(parameter))
    versions = {'report_variants.py': analysis.SOURCE_SHA256,
                'action_path_metrics.py': analysis.ACTION_SOURCE_SHA256,
                'native_candidate_gt.py': analysis.GT_SOURCE_SHA256,
                'candidate_identities.py': analysis.IDENTITY_SOURCE_SHA256,
                'blp_parameters.py': analysis.PARAMETER_SOURCE_SHA256}
    assert versions == load(ROOT / 'diagnostics/CP3/summary/full.json')['analysis_sources_sha256']
    destination = ROOT / 'diagnostics/completed-prefix' / args.stage
    bindings = {'parameter_sha256': digest(parameter), 'analysis_sources_sha256': versions,
                'prefix_auditor_sha256': digest(Path(__file__)),
                'reader_source_sha256': digest(ROOT / 'code/audit_first_full_episode.py')}
    checked, fresh, scene_counts = [], [], {}
    started = datetime.now(timezone.utc).isoformat()
    for job in ledger['jobs']:
        if not job.get('run_directory'):
            continue
        run = Path(job['run_directory'])
        exported = analysis.merge_episode_metric_files(run.glob('results_*/**/per_episode_metrics.json'))
        if not exported:
            continue
        scene = job['scene']
        expected = {scene + '_' + str(identity['runtime_episode_id']): identity for identity in job['identities']}
        assert set(exported) <= set(expected), 'Native export outside assigned identities'
        scene_counts[scene] = len(exported)
        targets, missing = {}, []
        for key in sorted(exported, key=lambda value: int(value.rsplit('_', 1)[1])):
            identity, raw = expected[key], exported[key]
            episode = str(identity['runtime_episode_id'])
            assert len(raw['metrics']) == 3 and [task['category'] for task in raw['tasks']] == identity['ordered_categories']
            raw_sha = hashlib.sha256(json.dumps(raw, sort_keys=True, allow_nan=False).encode()).hexdigest()
            cache = destination / scene / (episode + '.json')
            if cache.exists():
                old = load(cache)
                assert old['bindings'] == bindings and old['raw_native_record_sha256'] == raw_sha, 'Cached completed record or analysis changed'
                assert old['process']['invalid_path_complete'] and old['process']['native_stage_path_check_count'] == 3
                checked.append(old)
            else:
                targets[scene, episode] = (identity, raw, raw_sha, cache)
                missing.append(episode)
        if not missing:
            continue
        traces = {}
        for path in run.rglob('trajectory.jsonl'):
            identity = load(path.with_name('identity.json'))['identity']
            key = str(identity['scene_id']), str(identity['runtime_episode_id'])
            if key in targets:
                assert key not in traces
                traces[key] = rows(path)
        assert set(traces) == set(targets)
        grouped = {}
        for name in ('blp_events.jsonl', 'blp_decisions.jsonl', 'blp_snapshots.jsonl', 'scsp_decisions.jsonl'):
            grouped[name] = defaultdict(list)
            for path in run.rglob(name):
                for row in rows(path):
                    identity = row['episode_identity'] if name == 'scsp_decisions.jsonl' else row
                    key = str(identity['scene_id']), str(identity['runtime_episode_id'] if name == 'scsp_decisions.jsonl' else identity['episode_id'])
                    if key in targets:
                        grouped[name][key].append(row)
        gt = NativeCandidateGT(run)
        for key, (identity, raw, raw_sha, cache) in targets.items():
            trace, native_paths = traces[key], {}
            assert len(trace) == raw['total_num_steps']
            for k, metric in enumerate(raw['metrics']):
                track = [row for row in trace if row['subtask_index'] == k]
                assert track and track[-1]['is_stop'] and len(track) == metric['goat_sub-task_num_steps']
                native_paths[key[0], key[1], k] = metric['goat_sub-task_path_length_m']
            events = grouped['blp_events.jsonl'][key]
            verified = gt.verify_events(events)
            analysis.validate_snapshot_parameters(grouped['blp_snapshots.jsonl'][key], document,
                                                    ledger['parameter_sha256'], {key: identity})
            events, canonical_traces, identity_audit = canonicalize(events, {key: trace})
            process = analysis.process_metrics(events, grouped['blp_decisions.jsonl'][key],
                grouped['scsp_decisions.jsonl'][key], canonical_traces, {key}, gt_label=gt, native_stage_paths=native_paths)
            assert process['invalid_path_complete'] and process['native_stage_path_check_count'] == 3
            result = {'checked_at_utc': datetime.now(timezone.utc).isoformat(), 'diagnostic_only': True,
                'scope': 'One authoritatively exported complete native episode inside a possibly running scene; not a main-stage summary',
                'scene_id': key[0], 'episode_id': key[1], 'steps': raw['total_num_steps'],
                'success_vector': [int(metric['goat_sub-task_success']) for metric in raw['metrics']],
                'goals': identity['ordered_categories'], 'source_episode_ids': identity['source_episode_ids'],
                'raw_native_record_sha256': raw_sha, 'bindings': bindings, 'process': process,
                'candidate_event_GT_verification': verified, 'candidate_identity_audit': identity_audit}
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
            checked.append(result)
            fresh.append({'scene': key[0], 'episode': key[1], 'steps': result['steps'],
                          'success_vector': result['success_vector']})
    assert source_fingerprint() == ledger['source_fingerprint'] and digest(parameter) == ledger['parameter_sha256']
    summary = {'started_at_utc': started, 'completed_at_utc': datetime.now(timezone.utc).isoformat(),
        'stage': args.stage, 'diagnostic_only': True, 'formal400_complete': False,
        'completed_native_metric_records_checked': len(checked), 'newly_audited': fresh,
        'scene_exported_records': scene_counts, 'native_stage_path_checks': 3 * len(checked),
        'all_paths_complete': all(row['process']['invalid_path_complete'] for row in checked),
        'max_path_error_m': max((row['process']['native_stage_path_max_abs_error_m'] for row in checked), default=None),
        'source_and_parameters_unchanged': True, 'native_run_statuses_modified': False, 'bindings': bindings}
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
