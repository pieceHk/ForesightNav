from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import subprocess
import sys

from report_cp3 import verify
from experiments.exp4_collect import source_fingerprint

repo = Path('/nvme/khy/ForesightNav_exp4_depthfix')
root = repo / 'results/exp4'
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
ledger, summary, episodes, decisions, figures = verify()
assert ledger['status'] == 'completed' and len(episodes) == 20 and len(decisions) == 1999
assert len(source_fingerprint()) == 161 and source_fingerprint() == ledger['source_fingerprint']
assert digest(root / 'meta/calibration.json') == 'e1d75f3c9503c7439467748fc97be6ae2ad450c429f40dc7235b1564ff7852ae'
assert summary['source_commits'] == [subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()]
sync = json.loads((root / 'meta/smoke-a3-CP3-sync-files.json').read_text())
assert sync['remote_verified'] and sync['publication_commit'] == '7369d04f91aa5d569902760bf8bcaef65edeed28'
review_files = ['reports/CP3.md', 'reports/CP3_review.md', 'reports/dataset_scope.md',
    'diagnostics/CP3/summary/full.json', 'diagnostics/CP3/logs/full_episodes.jsonl',
    'diagnostics/CP3/logs/full_decisions.jsonl', 'diagnostics/CP3/complete-evidence-execution.json',
    'diagnostics/CP3/figure-review.json', 'diagnostics/CP3/final-figure-review.json',
    'diagnostics/CP3/confirmation-failure-review.json', 'diagnostics/CP3/latency-tail-review.json',
    'meta/environment-smoke.json', 'meta/evaluation-smoke-log-archive.json']
for relative in review_files:
    if relative in sync['sha256']:
        assert digest(root / relative) == sync['sha256'][relative], relative
evidence = {relative: digest(root / relative) for relative in review_files}
acceptance = {'schema': 'exp4_CP3_acceptance_v1', 'accepted': True, 'behavior_reviewed': True,
    'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
    'approval': {'source': 'explicit user chat message', 'text': '通过 CP3，继续全量',
                 'reviewed_publication_commit': sync['publication_commit']},
    'source_fingerprint': ledger['source_fingerprint'], 'parameter_sha256': ledger['parameter_sha256'],
    'smoke_ledger_sha256': digest(root / 'meta/evaluation-smoke-runs.json'), 'evidence_sha256': evidence,
    'scope': 'Accept current fixed20 Full smoke and continue Full400 then Nearest400, Myopic400, Suffix-only400 under the existing protocol; no evaluation fitting or selection'}
target = root / 'meta/CP3_acceptance.json'
if target.exists():
    old = json.loads(target.read_text())
    for key in ('accepted', 'behavior_reviewed', 'source_fingerprint', 'parameter_sha256', 'smoke_ledger_sha256', 'evidence_sha256'):
        assert old[key] == acceptance[key], 'Existing acceptance differs: ' + key
else:
    with target.open('x') as output:
        output.write(json.dumps(acceptance, ensure_ascii=False, indent=2) + '\n')
result = subprocess.run([sys.executable, str(root / 'code/evaluate_sequence.py'), '--stage', 'full', '--check'],
    cwd=repo, capture_output=True, text=True, check=True)
check = json.loads(result.stdout)
assert check['ready'] and check['episodes'] == 400 and not check['launches_workers']
proof = {'checked_at_utc': datetime.now(timezone.utc).isoformat(), 'acceptance_sha256': digest(target),
    'source_and_parameters_unchanged': source_fingerprint() == ledger['source_fingerprint'] and
        digest(root / 'meta/calibration.json') == ledger['parameter_sha256'],
    'full_gate_check': check, 'launches_workers': False}
assert proof['source_and_parameters_unchanged']
(root / 'diagnostics/CP3/approval-binding.json').write_text(json.dumps(proof, indent=2) + '\n')
print(json.dumps(proof))
