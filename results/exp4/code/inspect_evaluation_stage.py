"""Read one existing evaluation stage using episode-bound native trajectories."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

from evaluate_sequence import OUT, REPO
from report_variants import merge_episode_metric_files
from watch_evaluation import process


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('full', 'nearest', 'myopic', 'suffix_only'), required=True)
    args = parser.parse_args()
    stage = args.stage
    ledger = read(OUT / f'meta/evaluation-{stage}-runs.json')
    start = read(OUT / f'meta/{stage}-start-a3.json')
    observer = read(OUT / f'meta/{stage}-observer-a3.json')
    assert ledger['stage'] == start['stage'] == observer['stage'] == stage
    native, watching = process(start['pid']), process(observer['observer_pid'])
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    rows = [json.loads(line) for line in Path(observer['log']).read_bytes().split(b'\n')[:-1] if line]
    now, progress = datetime.now(timezone.utc), []
    for job in ledger['jobs']:
        count, active = None, None
        if job.get('run_directory'):
            run = Path(job['run_directory'])
            metrics = list(run.glob('results_*/**/per_episode_metrics.json'))
            completed = merge_episode_metric_files(metrics) if metrics else {}
            count, active = len(completed), []
            expected = {str(row['runtime_episode_id']): row for row in job['identities']}
            for path in run.rglob('trajectory.jsonl'):
                identity = read(path.with_name('identity.json'))['identity']
                episode = str(identity['runtime_episode_id'])
                assert identity['scene_id'] == job['scene'] and episode in expected
                for key in ('ordered_categories', 'source_episode_ids'):
                    assert identity[key] == expected[episode][key]
                if identity['scene_id'] + '_' + episode in completed:
                    continue
                track = [json.loads(line) for line in path.read_bytes().split(b'\n')[:-1] if line]
                stops = sorted({row['subtask_index'] for row in track if row['is_stop']})
                active.append({'episode_id': episode, 'action_count': len(track),
                    'current_stage': track[-1]['subtask_index'] if track else None,
                    'last_frame_index': track[-1]['frame_index'] if track else None,
                    'observed_stop_stages': stops,
                    'state': 'terminal_stop_waiting_metric_export' if stops == [0, 1, 2] else 'active_trajectory',
                    'trajectory_modified_at_utc': datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()})
            active.sort(key=lambda row: int(row['episode_id']))
        progress.append({'scene': job['scene'], 'gpu': job['gpu'], 'status': job['status'],
                        'completed_native_metric_records': count, 'active_native_episodes': active})
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    parameters = sha(OUT / 'meta/calibration.json')
    acceptance = sha(OUT / 'meta/CP3_acceptance.json')
    print(json.dumps({'checked_at_utc': now.isoformat(), 'stage': stage, 'ledger_status': ledger['status'],
        'controller_live_matching': native['live'] and native.get('start_ticks') == start['start_ticks']
            and f'--stage {stage}' in native.get('command', '') and boot == start['boot_id'],
        'observer_live_matching': watching['live'] and watching.get('start_ticks') == observer['start_ticks']
            and f'--stage {stage}' in watching.get('command', '') and boot == observer['boot_id'],
        'snapshot_age_seconds': (now - datetime.fromisoformat(rows[-1]['checked_at_utc'])).total_seconds() if rows else None,
        'progress_semantics': 'Current episode/stage comes from nonexported identity-bound trajectories; legacy observer current_stage may refer to an older completed episode',
        'latest_observation': rows[-1] if rows else None, 'episode_progress': progress,
        'source_commit': source, 'source_commit_matches_start': source == start['source_commit'],
        'parameter_sha256': parameters, 'parameters_match_start': parameters == start['parameter_sha256'],
        'CP3_acceptance_sha256': acceptance, 'acceptance_matches_start': acceptance == start['CP3_acceptance_sha256'],
        'publication_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'],
            cwd=Path('/nvme/khy/ForesightNav_exp4_evalprep'), text=True).strip()}))


if __name__ == '__main__':
    main()
