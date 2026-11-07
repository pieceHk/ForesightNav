"""Audit completed native calibration scenes; never fits or launches workers.

Run before changing the source frozen by the collection ledger. Incomplete
scene assignments are left untouched. Diagnostic path gaps remain explicit.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from experiments.exp4_collect import audit_run, source_fingerprint, write
from report_variants import normalize

OUT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    ledger = json.loads((OUT / 'meta/calibration_runs.json').read_text())
    expected_all = json.loads((OUT / 'meta/2mon-identities.json').read_text())
    audits = []
    key = lambda row: (str(row['scene_id']), str(row['runtime_episode_id']))
    for job in ledger['jobs']:
        if job['status'] != 'completed':
            continue
        require(job['returncode'] == 0, 'Completed scene has a nonzero native exit code')
        run = Path(job['run_directory'])
        expected = [r for r in expected_all if r['scene_id'] == job['scene']]
        wanted = {key(r): r for r in expected}
        observed = []
        for path in run.rglob('identity.json'):
            identity = json.loads(path.read_text())['identity']
            require(key(identity) in wanted, f'Unexpected actual identity: {path}')
            row = wanted[key(identity)]
            for field in ('dataset_manifest_sha256', 'source_episode_ids', 'ordered_categories'):
                require(identity[field] == row[field], f'Actual identity mismatch: {path}, {field}')
            observed.append(key(identity))
        require(len(observed) == 30 and set(observed) == set(wanted),
                'Scene must contain all thirty actual identities without duplicates')
        native = audit_run(run, job['scene'])
        rows, _, summary = normalize('full', [run], expected=expected, allow_diagnostic=True)
        require(all(len(r['stages']) == 2 for r in rows), 'Calibration requires two ordered targets')
        events = [json.loads(line) for p in run.rglob('blp_events.jsonl')
                  for line in p.read_text().splitlines() if line]
        counts = Counter(r['event'] for r in events)
        require(counts['candidate_checked'] == counts['verification_final'],
                'Completed scene has candidate checks without final labels')
        visit_key = lambda e: (str(e['scene_id']), str(e['episode_id']), int(e['k']),
                               str(e['candidate_id']), int(e['t']))
        require(Counter(visit_key(e) for e in events if e['event'] == 'candidate_checked') ==
                Counter(visit_key(e) for e in events if e['event'] == 'verification_final'),
                'Candidate checks and final labels do not match by visit identity')
        audits.append({
            'scene': job['scene'], 'run_directory': str(run), 'native_exit_code': 0,
            'complete_episodes': len(rows), 'targets_per_episode': 2,
            'actual_identity_files_checked': len(observed),
            'frozen_manifest_source_ids_ordered_categories_match': True,
            'complete_trajectory_and_terminal_STOP_checks_passed': True,
            'candidate_checks': counts['candidate_checked'],
            'final_labels': counts['verification_final'],
            'check_to_final_event_keys_match': True,
            'shadow_snapshots': native['snapshots'], 'shadow_errors': 0,
            'process_path_complete_under_legacy_logs': summary['process']['invalid_path_complete'],
            'path_association_gaps': summary['process']['invalid_path_association_gaps'],
            'artifact_sha256': {name: hashlib.sha256((run / name).read_bytes()).hexdigest()
                                for name in ('status.json', 'launch.json', 'mon_metrics.json')},
        })
    require(ledger['source_fingerprint'] == source_fingerprint(), 'Frozen runtime source changed')
    result = {
        'checked_at_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'completed 2-MON calibration integrity only; no fitting, no 3-MON table',
        'frozen_runtime_unchanged': True,
        'complete_episodes': sum(r['complete_episodes'] for r in audits),
        'completed_scenes': audits,
    }
    write(OUT / 'meta/completed-calibration-scene-audits.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
