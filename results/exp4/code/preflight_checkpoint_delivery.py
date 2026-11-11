"""Exercise incomplete-stage rejection of the new CPU delivery tools."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from experiments.exp4_collect import source_fingerprint

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    cp2 = json.loads((ROOT / 'meta/CP2b_acceptance.json').read_text())
    fingerprint = source_fingerprint()
    parameter_sha = digest(ROOT / 'meta/calibration.json')
    assert fingerprint == cp2['source_fingerprint'] and parameter_sha == cp2['parameter_sha256']
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    full = json.loads((ROOT / 'meta/evaluation-full-runs.json').read_text())
    assert full['status'] == 'running', 'This preflight is only for the incomplete Full phase'
    guarded = ['reports/CP4.md', 'reports/CP5.md', 'meta/CP4_report_prepared.json',
        'meta/CP5_report_prepared.json', 'meta/CP4_reported.json', 'figures/CP5', 'summary/paired',
        'diagnostics/CP4/complete-evidence-execution.json', 'diagnostics/CP5/complete-evidence-execution.json']
    before = {relative: (ROOT / relative).exists() for relative in guarded}
    assert not any(before.values()), 'Refuse to treat existing complete-stage outputs as an incomplete-state test'
    checks = []
    commands = [('complete_checkpoint_evidence.py', ['--checkpoint', 'CP4', '--check']),
        ('complete_checkpoint_evidence.py', ['--checkpoint', 'CP5', '--check']),
        ('export_paired_episodes.py', ['--check'])]
    for name, arguments in commands:
        path = ROOT / 'code' / name
        result = subprocess.run([sys.executable, str(path), *arguments], cwd=ROOT.parents[1],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert result.returncode == 0 and not result.stderr, result.stderr
        value = json.loads(result.stdout)
        assert value['ready'] is False and value['writes_outputs'] is False
        assert value['native_states_modified'] is False and value['launches_workers'] is False
        checks.append({'source': 'code/' + name, 'source_sha256': digest(path),
            'arguments': arguments, 'exit_code': result.returncode, 'actual_readiness': value})
    after = {relative: (ROOT / relative).exists() for relative in guarded}
    assert before == after and not any(after.values())
    assert source_fingerprint() == fingerprint and digest(ROOT / 'meta/calibration.json') == parameter_sha
    now = datetime.now(timezone.utc)
    relative = 'diagnostics/checkpoint-delivery-preflight/' + now.strftime('%Y%m%dT%H%M%S%fZ') + '.json'
    record = {'checked_at_utc': now.isoformat(), 'stage': 'full', 'diagnostic_only': True,
        'formal400_complete': False, 'scope': 'Three actual CPU CLI incomplete-stage checks; not unit/native tests or complete-stage positive execution.',
        'source': 'code/preflight_checkpoint_delivery.py', 'source_sha256': digest(Path(__file__)),
        'checks': checks, 'artifacts_before': before, 'artifacts_after': after,
        'native_source_entries': len(fingerprint), 'parameter_sha256': parameter_sha,
        'source_and_parameters_unchanged': True, 'native_states_modified': False,
        'launches_workers': False, 'positive_complete_stage_execution_pending': True}
    destination = ROOT / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x') as handle:
        handle.write(json.dumps(record, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'proof': relative, 'proof_sha256': digest(destination), **record}))


if __name__ == '__main__':
    main()
