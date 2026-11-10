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
    count, active_episodes = None, None
    if job.get('run_directory'):
        run = Path(job['run_directory'])
        metrics = list(run.glob('results_*/**/per_episode_metrics.json'))
        completed = merge_episode_metric_files(metrics) if metrics else {}
        count, active_episodes = len(completed), []
        expected = {str(row['runtime_episode_id']): row for row in job['identities']}
        for trajectory in run.rglob('trajectory.jsonl'):
            identity = json.loads(trajectory.with_name('identity.json').read_text())['identity']
            episode = str(identity['runtime_episode_id'])
            assert identity['scene_id'] == job['scene'] and episode in expected
            for key in ('ordered_categories', 'source_episode_ids'):
                assert identity[key] == expected[episode][key]
            if identity['scene_id'] + '_' + episode in completed:
                continue
            track = [json.loads(line) for line in trajectory.read_bytes().split(b'\n')[:-1] if line]
            stopped = sorted({row['subtask_index'] for row in track if row['is_stop']})
            active_episodes.append({'episode_id': episode, 'action_count': len(track),
                'current_stage': track[-1]['subtask_index'] if track else None,
                'last_frame_index': track[-1]['frame_index'] if track else None,
                'observed_stop_stages': stopped,
                'state': 'terminal_stop_waiting_metric_export' if stopped == [0, 1, 2] else 'active_trajectory',
                'trajectory_modified_at_utc': datetime.fromtimestamp(trajectory.stat().st_mtime, timezone.utc).isoformat()})
        active_episodes.sort(key=lambda row: int(row['episode_id']))
    progress.append({'scene': job['scene'], 'gpu': job['gpu'], 'status': job['status'],
                     'completed_native_metric_records': count, 'active_native_episodes': active_episodes})
publication = Path('/nvme/khy/ForesightNav_exp4_evalprep')
print(json.dumps({'checked_at_utc': now.isoformat(), 'ledger_status': ledger['status'],
    'controller_live_matching': native['live'] and native.get('start_ticks') == start['start_ticks'] and
        '--stage full' in native.get('command', '') and boot == start['boot_id'],
    'observer_live_matching': watching['live'] and watching.get('start_ticks') == observer['start_ticks'] and boot == observer['boot_id'],
    'snapshot_age_seconds': (now - datetime.fromisoformat(rows[-1]['checked_at_utc'])).total_seconds() if rows else None,
    'progress_semantics': 'Current episode/stage comes from nonexported identity-bound trajectories; legacy observer current_stage may refer to an older completed episode',
    'latest_observation': rows[-1] if rows else None, 'episode_progress': progress,
    'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
    'parameter_sha256': hashlib.sha256((root / 'meta/calibration.json').read_bytes()).hexdigest(),
    'CP3_acceptance_sha256': hashlib.sha256((root / 'meta/CP3_acceptance.json').read_bytes()).hexdigest(),
    'publication_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=publication, text=True).strip()}))
