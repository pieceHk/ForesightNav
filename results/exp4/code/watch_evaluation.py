"""Read-only supervision of an existing Exp4 stage; never starts workers."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

from experiments.exp4_collect import source_fingerprint, write
from pbsp.navigation.blp_parameters import CalibrationDocument

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / 'results/exp4'


def load(path):
    return json.loads(path.read_text())


def rows(path):
    data = path.read_bytes()
    # The producer may currently be appending the last line.
    return [json.loads(line) for line in data.split(b'\n')[:-1] if line]


def process(pid):
    root = Path('/proc') / str(pid)
    try:
        stat = (root / 'stat').read_text()
        fields = stat[stat.rfind(')') + 2:].split()
        return {'live': fields[0] != 'Z', 'state': fields[0],
                'start_ticks': fields[19],
                'command': (root / 'cmdline').read_bytes().replace(b'\0', b' ').decode()}
    except FileNotFoundError:
        return {'live': False}


def audit(stage):
    ledger = load(OUT / f'meta/evaluation-{stage}-runs.json')
    calibration = OUT / 'meta/calibration.json'
    parameter_sha = hashlib.sha256(calibration.read_bytes()).hexdigest()
    document = CalibrationDocument.from_mapping(load(calibration))
    cp2b = load(OUT / 'meta/CP2b_acceptance.json')
    control = process(ledger['controller_pid'])
    control['boot_matches'] = ledger['boot_id'] == Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    control['command_matches'] = f'--stage {stage}' in control.get('command', '') and 'evaluate_sequence.py' in control.get('command', '')
    start = OUT / 'meta/smoke_start.json'
    if stage == 'smoke' and start.exists():
        expected = load(start)
        control['start_matches'] = (expected['pid'] == ledger['controller_pid'] and
                                    expected['start_ticks'] == control.get('start_ticks'))
    result = {'checked_at_utc': datetime.now(timezone.utc).isoformat(),
              'stage': stage, 'status': ledger['status'], 'error': ledger.get('error'),
              'controller_pid': ledger['controller_pid'], 'controller': control,
              'source_unchanged': source_fingerprint() == cp2b['source_fingerprint'],
              'parameter_unchanged': parameter_sha == cp2b['parameter_sha256'], 'jobs': []}
    for job in ledger['jobs']:
        item = {k: job.get(k) for k in ('scene', 'gpu', 'pid', 'status', 'run_directory')}
        item['launcher'] = process(job['pid'])
        if not job.get('run_directory'):
            result['jobs'].append(item)
            continue
        run = Path(job['run_directory'])
        item['native_status'] = load(run / 'status.json') if (run / 'status.json').exists() else None
        item['blp_error_bytes'] = sum(p.stat().st_size for p in run.rglob('blp_errors.jsonl'))
        snapshots = [row for p in run.rglob('blp_snapshots.jsonl') for row in rows(p)]
        theta, binding = document.resolve(job['scene'])
        binding = dict(binding, calibration_sha256=parameter_sha)
        item['snapshot_count'] = len(snapshots)
        item['binding_error_count'] = sum(row.get('shadow') is not False or
            row.get('parameters') != asdict(theta) or row.get('parameter_binding') != binding
            for row in snapshots)
        trace = [row for p in run.rglob('trajectory.jsonl') for row in rows(p)]
        item['action_count'] = len(trace)
        item['missing_action_targets'] = sum('blp_action_target' not in row for row in trace)
        item['unresolved_query_targets'] = sum(
            (row.get('blp_action_target') or {}).get('source') == 'query_candidate' and
            (row.get('blp_action_target') or {}).get('query_identity_status') != 'resolved'
            for row in trace)
        item['current_stage'] = trace[-1]['subtask_index'] if trace else None
        dispatches = [row for p in run.rglob('scsp_decisions.jsonl') for row in rows(p)]
        applied = [row for row in dispatches if row.get('refresh_applied') and
                   (row.get('blp') or {}).get('nearest_raw_rank') is not None and
                   row.get('selected_raw_ranks', {}).get('dispatched') is not None]
        item['applied_nonempty_dispatches'] = len(applied)
        item['dispatch_deviations_from_nearest'] = sum(
            row['selected_raw_ranks']['dispatched'] != row['blp']['nearest_raw_rank']
            for row in applied)
        item['structural_dispatch_errors'] = sum(
            row.get('fallback_reason') == 'planner_error' or 'recording_error' in row or
            row.get('status') == 'telemetry_error' for row in dispatches)
        result['jobs'].append(item)
    result['completed_scenes'] = sum(j['status'] == 'completed' for j in result['jobs'])
    write(OUT / 'meta/evaluation-live-audit.json', result)
    return result


def compact(value):
    return {k: value[k] for k in ('checked_at_utc', 'stage', 'status', 'error',
            'controller_pid', 'controller', 'source_unchanged', 'parameter_unchanged',
            'completed_scenes')} | {'totals': {k: sum(j.get(k, 0) for j in value['jobs'])
                for k in ('snapshot_count', 'action_count', 'blp_error_bytes',
                          'binding_error_count', 'missing_action_targets',
                          'unresolved_query_targets', 'structural_dispatch_errors')},
                'active_jobs': [
        {k: j[k] for k in ('scene', 'gpu', 'status', 'snapshot_count', 'action_count',
                           'current_stage', 'blp_error_bytes', 'binding_error_count',
                           'missing_action_targets', 'unresolved_query_targets',
                           'applied_nonempty_dispatches', 'dispatch_deviations_from_nearest',
                           'structural_dispatch_errors') if k in j}
        for j in value['jobs'] if j['status'] != 'completed']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', default='smoke', choices=('smoke', 'full', 'nearest', 'myopic', 'suffix_only'))
    parser.add_argument('--watch-seconds', type=int, default=0)
    args = parser.parse_args()
    deadline = time.monotonic() + args.watch_seconds
    while True:
        value = audit(args.stage)
        print(json.dumps(compact(value)), flush=True)
        if (not value['controller']['live'] or value['status'] != 'running' or
                time.monotonic() >= deadline):
            break
        time.sleep(min(30., max(0., deadline - time.monotonic())))


if __name__ == '__main__':
    main()
