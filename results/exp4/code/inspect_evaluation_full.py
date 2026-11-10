from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import subprocess
from watch_evaluation import process
from report_variants import merge_episode_metric_files

repo = Path('/nvme/khy/ForesightNav_exp4_depthfix')
root = repo / 'results/exp4'
ledger = json.loads((root / 'meta/evaluation-full-runs.json').read_text())
start = json.loads((root / 'meta/full-start-a3.json').read_text())
observer = json.loads((root / 'meta/full-observer-a3.json').read_text())
native, watching = process(start['pid']), process(observer['observer_pid'])
boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
rows = [json.loads(line) for line in Path(observer['log']).read_bytes().split(b'\n')[:-1] if line]
now = datetime.now(timezone.utc)
progress = []
for job in ledger['jobs']:
    count = None
    if job.get('run_directory'):
        metrics = list(Path(job['run_directory']).glob('results_*/**/per_episode_metrics.json'))
        count = len(merge_episode_metric_files(metrics)) if metrics else 0
    progress.append({'scene': job['scene'], 'gpu': job['gpu'], 'status': job['status'],
                     'completed_native_metric_records': count})
publication = Path('/nvme/khy/ForesightNav_exp4_evalprep')
print(json.dumps({'checked_at_utc': now.isoformat(), 'ledger_status': ledger['status'],
    'controller_live_matching': native['live'] and native.get('start_ticks') == start['start_ticks'] and
        '--stage full' in native.get('command', '') and boot == start['boot_id'],
    'observer_live_matching': watching['live'] and watching.get('start_ticks') == observer['start_ticks'] and boot == observer['boot_id'],
    'snapshot_age_seconds': (now - datetime.fromisoformat(rows[-1]['checked_at_utc'])).total_seconds() if rows else None,
    'latest_observation': rows[-1] if rows else None, 'episode_progress': progress,
    'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
    'parameter_sha256': hashlib.sha256((root / 'meta/calibration.json').read_bytes()).hexdigest(),
    'CP3_acceptance_sha256': hashlib.sha256((root / 'meta/CP3_acceptance.json').read_bytes()).hexdigest(),
    'publication_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=publication, text=True).strip()}))
