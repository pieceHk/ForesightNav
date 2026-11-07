"""Reproduce native GT and action-path checks on closed diagnostic artifacts.

This entry point runs only CPU postprocessing. It neither launches a policy
nor fits parameters, and it never writes a formal variant table.
"""
import json
from pathlib import Path

from experiments.exp4_collect import source_fingerprint, write
from native_candidate_gt import NativeCandidateGT, SOURCE_SHA256 as GT_SOURCE_SHA256
from report_variants import normalize

OUT = Path(__file__).resolve().parents[1]


def main():
    ledger = json.loads((OUT / 'meta/calibration_runs.json').read_text())
    if source_fingerprint() != ledger['source_fingerprint']:
        raise ValueError('Audit this collection before changing its frozen source')
    diagnostic_runs = {}
    for name in ('action-trace-plumbing', 'action-trace-confirmation-switch'):
        control = json.loads((OUT / 'diagnostics' / name / 'controller.json').read_text())
        if control['status'] != 'finished' or control['returncode'] != 0:
            raise ValueError('Excluded logging diagnostic did not finish')
        diagnostic_runs[name] = Path(control['run_directory'])
    calibration_runs = [Path(j['run_directory']) for j in ledger['jobs'] if j['status'] == 'completed']
    truth_checks = []
    for run in calibration_runs + list(diagnostic_runs.values()):
        if json.loads((run / 'status.json').read_text())['status'] != 'completed':
            raise ValueError('Only closed native scene runs may enter this check')
        native = NativeCandidateGT(run)
        events = [json.loads(s) for p in run.rglob('blp_events.jsonl')
                  for s in p.read_text().splitlines() if s]
        verified = native.verify_events(events)
        truth_checks.append({'run_directory': str(run), **verified,
                             'native_gt_provenance': native.provenance()})
    truth = {'scope': 'native offline GT reproducibility, no fitting or policy execution',
             'analysis_source_sha256': GT_SOURCE_SHA256,
             'candidate_verifications': sum(r['candidate_verifications'] for r in truth_checks),
             'recovery_verifications': sum(r['recovery_verifications'] for r in truth_checks),
             'max_distance_error_m': max(r['max_distance_error_m'] for r in truth_checks),
             'checks': truth_checks}
    write(OUT / 'diagnostics/native-offline-gt-check.json', truth)
    path_checks = []
    for name, run in diagnostic_runs.items():
        expected = json.loads((run / 'launch.json').read_text())['identities']
        rows, _, summary = normalize('full', [run], expected=expected, allow_diagnostic=True)
        process = summary['process']
        if not process['invalid_path_complete'] or process['native_stage_path_check_count'] != 2:
            raise ValueError('Excluded logging diagnostic has incomplete action paths')
        path_checks.append({'name': name, 'scope': 'excluded 2-MON path reconstruction, no Full score',
                            'run_directory': str(run), 'n_episodes': len(rows),
                            'accepted_steps': sum(r['total_steps'] for r in rows),
                            'analysis_sources_sha256': summary['analysis_sources_sha256'],
                            'process': process})
    paths = {'checks': path_checks, 'all_action_paths_complete': True,
             'native_stage_checks': sum(r['process']['native_stage_path_check_count'] for r in path_checks)}
    write(OUT / 'diagnostics/action-target-path-check.json', paths)
    print(json.dumps({'native_GT_candidate_checks': truth['candidate_verifications'],
                      'native_GT_recovery_checks': truth['recovery_verifications'],
                      'native_GT_max_distance_error_m': truth['max_distance_error_m'],
                      'native_path_stage_checks': paths['native_stage_checks'],
                      'native_path_max_error_m': max(r['process']['native_stage_path_max_abs_error_m'] for r in path_checks),
                      'accepted_steps': sum(r['accepted_steps'] for r in path_checks)}, indent=2))


if __name__ == '__main__':
    main()
