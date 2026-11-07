"""Read-only supervision of the existing controller and assigned run handles."""
from collections import Counter
import datetime
import json
import math
from pathlib import Path

from experiments.exp4_collect import source_fingerprint, write
from pbsp.navigation.blp import BeliefParameters, solve_snapshot

ROOT = Path(__file__).resolve().parents[1]


def process_matches(pid, required):
    path = Path('/proc') / str(pid) / 'cmdline'
    try:
        command = path.read_bytes().replace(b'\0', b' ').decode()
    except FileNotFoundError:
        return False
    return all(part in command for part in required)


def read_complete_lines(path):
    lines = path.read_bytes().splitlines(keepends=True)
    return [json.loads(line) for line in lines if line.endswith(b'\n')]


def main():
    ledger = json.loads((ROOT/'meta/calibration_runs.json').read_text())
    pid = ledger['controller_pid']
    result = {'checked_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'controller_status': ledger['status'], 'controller_pid': pid,
              'controller_live_and_matching': process_matches(pid, ['experiments.exp4_collect']),
              'boot_id_matches': ledger['boot_id'] == Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
              'frozen_files_unchanged': ledger['source_fingerprint'] == source_fingerprint(),
              'completed_episodes': 0, 'assigned_episodes': 360, 'jobs': []}
    for job in ledger['jobs']:
        run_name = job.get('run_directory')
        row = {k: job[k] for k in ('fold', 'scene', 'gpu', 'pid', 'status')}
        row['launcher_live_and_matching'] = process_matches(job['pid'], [str(ROOT.parents[1]/'run.py'), job['scene']])
        if not run_name:
            result['jobs'].append(row)
            continue
        run = Path(run_name)
        metrics = [json.loads(p.read_text()) for p in run.rglob('per_episode_metrics.json')]
        complete = {key for group in metrics for key in group}
        events = [r for p in run.rglob('blp_events.jsonl') for r in read_complete_lines(p)]
        finals = [r for r in events if r['event'] == 'verification_final']
        for event in finals:
            if event['v'] not in (0, 1) or event['y'] not in (0, 1):
                raise ValueError('Nonbinary verification/diagnostic label')
            if not math.isfinite(event['E']) or not 0 <= event['N_views'] <= 12:
                raise ValueError('Invalid semantic evidence or view count')
            if not math.isclose(event['S'], 1 - math.exp(-event['N_views']/3), abs_tol=1e-12):
                raise ValueError('Observation coverage violates the fixed definition')
        snapshots = [r for p in run.rglob('blp_snapshots.jsonl') for r in read_complete_lines(p)]
        decisions = [r for p in run.rglob('blp_decisions.jsonl') for r in read_complete_lines(p)]
        checked = 0
        for snapshot, decision in list(zip(snapshots, decisions))[-10:]:
            replay = solve_snapshot(snapshot['snapshot'], BeliefParameters(**snapshot['parameters']),
                                    variant='full', active_raw_rank=snapshot['active_raw_rank'])
            if replay['chosen_raw_rank'] != decision['chosen_raw_rank']:
                raise ValueError('Recorded online and offline choices disagree')
            checked += 1
        row.update(completed_episodes=len(complete),
                   accepted_actions=sum(p.read_bytes().count(b'\n') for p in run.rglob('trajectory.jsonl')),
                   snapshots=len(snapshots), latest_replays_checked=checked,
                   event_counts=dict(Counter(r['event'] for r in events)),
                   labels_v=dict(Counter(str(r['v']) for r in finals)),
                   completed_episode_verification_events=sum(r['scene_id']+'_'+str(r['episode_id']) in complete for r in finals),
                   shadow_error_bytes=sum(p.stat().st_size for p in run.rglob('blp_errors.jsonl')))
        result['completed_episodes'] += len(complete)
        result['jobs'].append(row)
    write(ROOT/'meta/current-live-audit.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
