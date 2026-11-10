"""Run final CP3 evidence checks and reporting; never accept or launch Full."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from experiments.exp4_collect import source_fingerprint

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ledger_path = ROOT / 'meta/evaluation-smoke-runs.json'
    ledger = json.loads(ledger_path.read_text())
    assert ledger['status'] in ('completed', 'completed_requires_score_review')
    assert len(ledger['jobs']) == 20 and all(job['status'] == 'completed' for job in ledger['jobs'])
    assert ledger['n_episodes'] == 20 and ledger['n_error'] == 0
    fingerprint, parameter = source_fingerprint(), digest(ROOT / 'meta/calibration.json')
    assert fingerprint == ledger['source_fingerprint'] and parameter == ledger['parameter_sha256']
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    evidence = {'started_at_utc': datetime.now(timezone.utc).isoformat(),
                'source_fingerprint': fingerprint, 'parameter_sha256': parameter,
                'launches_workers': False, 'acceptance_created': False, 'steps': []}
    destination = ROOT / 'diagnostics/CP3/complete-evidence-execution.json'
    commands = [('completed-cases', 'audit_completed_smoke.py', []),
                ('coverage', 'audit_smoke_coverage.py', []),
                ('environment', 'audit_smoke_environment.py', []),
                ('mechanisms', 'audit_smoke_mechanisms.py', []),
                ('raw-archive', 'archive_evaluation_logs.py', ['--stage', 'smoke']),
                ('report-check', 'report_cp3.py', ['--check']),
                ('report-render', 'report_cp3.py', [])]
    for name, script, arguments in commands:
        path = ROOT / 'code' / script
        stdout = ROOT / 'logs/CP3' / (stamp + '-' + name + '.stdout.log')
        stderr = stdout.with_name(stdout.name.replace('.stdout.log', '.stderr.log'))
        stdout.parent.mkdir(parents=True, exist_ok=True)
        started = datetime.now(timezone.utc).isoformat()
        with stdout.open('wb') as out, stderr.open('wb') as err:
            result = subprocess.run([sys.executable, str(path)] + arguments,
                                    cwd=ROOT.parents[1], stdout=out, stderr=err)
        step = {'name': name, 'script': script, 'arguments': arguments,
                'script_sha256': digest(path), 'started_at_utc': started,
                'finished_at_utc': datetime.now(timezone.utc).isoformat(), 'exit_code': result.returncode,
                'stdout': stdout.relative_to(ROOT).as_posix(), 'stdout_sha256': digest(stdout),
                'stderr': stderr.relative_to(ROOT).as_posix(), 'stderr_sha256': digest(stderr)}
        evidence['steps'].append(step)
        destination.write_text(json.dumps(evidence, indent=2) + '\n')
        print(json.dumps({'step': name, 'exit_code': result.returncode}), flush=True)
        if result.returncode:
            raise RuntimeError(name + ' requires review; see recorded execution logs')
        if name == 'mechanisms':
            auxiliary = {'checked_at_utc': datetime.now(timezone.utc).isoformat(),
                         'executions': [{'script': item['script'], 'exit_code': item['exit_code'],
                                         'sha256': item['script_sha256']} for item in evidence['steps']
                                        if item['name'] in ('coverage', 'environment', 'mechanisms')],
                         'native_source_or_parameters_changed': False}
            (ROOT / 'diagnostics/CP3/auxiliary-audit-execution.json').write_text(json.dumps(auxiliary, indent=2) + '\n')
        if name == 'report-check':
            assert json.loads(stdout.read_text().splitlines()[-1])['ready'] is True
        if name == 'report-render':
            receipt = json.loads(stdout.read_text().splitlines()[-1])
            assert receipt['stage_accepted'] is False and receipt['launches_workers'] is False
            (ROOT / 'diagnostics/CP3/report-generation.json').write_text(json.dumps(receipt, indent=2) + '\n')
    assert source_fingerprint() == fingerprint and digest(ROOT / 'meta/calibration.json') == parameter
    evidence.update(completed_at_utc=datetime.now(timezone.utc).isoformat(), source_and_parameter_unchanged=True,
                    report_sha256=digest(ROOT / 'reports/CP3.md'))
    destination.write_text(json.dumps(evidence, indent=2) + '\n')
    summary = json.loads((ROOT / 'diagnostics/CP3/summary/full.json').read_text())
    print(json.dumps({'completed': True, 'metrics': {key: summary[key] for key in
        ('SR', 'SPL', 'PR', 'PPL', 'P_C1', 'P_C2_given_C1', 'P_C3_given_C1C2', 'Joint')},
        'source_and_parameters_unchanged': True, 'launches_workers': False, 'acceptance_created': False}), flush=True)


if __name__ == '__main__':
    main()
