"""Detach one remaining 400-case variant after the existing checkpoint gate."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from evaluate_sequence import OUT, REPO, GateError, build_jobs, expected_identities, validate_gate
from watch_evaluation import process

STAGES = ('nearest', 'myopic', 'suffix_only')
PREVIOUS = dict(nearest='full', myopic='nearest', suffix_only='myopic')
ATTEMPT = 'a3'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def claim(path, value):
    with path.open('x') as handle:
        handle.write(json.dumps(value, indent=2) + '\n')


def paths(stage):
    return (OUT / f'meta/evaluation-{stage}-runs.json',
            OUT / f'meta/{stage}-start-{ATTEMPT}.json',
            OUT / f'logs/evaluation-{stage}-controller-{ATTEMPT}.log')


def check(stage):
    jobs = build_jobs(stage, expected_identities(stage))
    assert len(jobs) == 20 and sum(len(job['identities']) for job in jobs) == 400
    reasons, gate = [], None
    try:
        gate = validate_gate(stage)
    except GateError as error:
        reasons.append(str(error))
    occupied = [str(path.relative_to(OUT)) for path in paths(stage) if path.exists()]
    if occupied:
        reasons.append('Existing launch evidence: inspect original process; never duplicate it')
    for following in STAGES[STAGES.index(stage) + 1:]:
        if any(path.exists() for path in paths(following)):
            reasons.append(f'Later stage {following} already has launch evidence')
    previous = OUT / 'meta/last-evaluation-observer.json'
    if not previous.exists() or read(previous)['stage'] != PREVIOUS[stage]:
        reasons.append('Observer predecessor metadata does not match the prescribed order')
    gpu_rows = subprocess.check_output(['nvidia-smi',
        '--query-gpu=index,name,memory.total,memory.free', '--format=csv,noheader,nounits'], text=True).splitlines()
    gpu_rows = [row for row in gpu_rows if row.split(',')[0].strip() in ('2', '3')]
    if len(gpu_rows) != 2 or any(float(row.split(',')[-1]) < 38 * 1024 for row in gpu_rows):
        reasons.append('GPU2/3 have not each reached the existing 38GiB-free launch threshold')
    disk = os.statvfs(OUT)
    free_bytes = disk.f_bavail * disk.f_frsize
    if free_bytes < 30 * 1024**3:
        reasons.append('Less than 30GiB of output space remains')
    return {'stage': stage, 'attempt': ATTEMPT, 'ready': not reasons, 'reasons': reasons,
            'scenes': 20, 'episodes': 400, 'existing_target_artifacts': occupied,
            'gpu_observation': gpu_rows, 'output_free_bytes': free_bytes,
            'parameter_sha256': sha(OUT / 'meta/calibration.json'),
            'launcher_sha256': sha(Path(__file__)),
            'controller_sha256': sha(OUT / 'code/evaluate_sequence.py'),
            'launches_workers': False}, gate


def attach_observer(stage, record):
    ledger_path, _, _ = paths(stage)
    if not ledger_path.exists():
        return {'status': 'controller_initializing', 'controller_pid': record['pid'],
                'action': 'Inspect this same PID or attach its observer later; never relaunch it'}
    ledger = read(ledger_path)
    assert ledger['controller_pid'] == record['pid'] and ledger['boot_id'] == record['boot_id']
    if ledger['status'] != 'running':
        return {'status': ledger['status'], 'observer_started': False}
    identity = process(record['pid'])
    assert identity['live'] and identity['start_ticks'] == record['start_ticks']
    assert f'--stage {stage}' in identity['command']
    assert ledger['source_fingerprint'] == record['source_fingerprint']
    assert ledger['parameter_sha256'] == record['parameter_sha256'] == sha(OUT / 'meta/calibration.json')
    observer_path = OUT / f'meta/{stage}-observer-{ATTEMPT}.json'
    if observer_path.exists():
        existing = read(observer_path)
        assert existing.get('observer_pid'), 'Observer claim exists: inspect it, never duplicate it'
        observed = process(existing['observer_pid'])
        assert observed['live'] and observed['start_ticks'] == existing['start_ticks'], 'Inspect the existing observer record'
        assert existing['stage'] == stage and f'--stage {stage}' in observed['command']
        assert 'watch_evaluation.py' in observed['command']
        assert existing['controller_pid'] == record['pid'] and existing['boot_id'] == record['boot_id']
        assert existing['controller_start_ticks'] == record['start_ticks']
        return existing
    previous_path = OUT / 'meta/last-evaluation-observer.json'
    previous = read(previous_path)
    assert previous['stage'] == PREVIOUS[stage]
    now = datetime.now(timezone.utc)
    stamp = now.strftime('%Y%m%dT%H%M%S%fZ')
    log = OUT / f'logs/evaluation-observer-{stage}-{ATTEMPT}-{stamp}.jsonl'
    stderr = log.with_name(log.stem + '.stderr.log')
    observer = {'stage': stage, 'attempt': ATTEMPT, 'status': 'observer_launch_claimed',
                'boot_id': record['boot_id'], 'started_at_utc': now.isoformat(),
                'watch_seconds': 604800, 'log': str(log), 'stderr_log': str(stderr),
                'watcher_sha256': sha(OUT / 'code/watch_evaluation.py'),
                'controller_pid': record['pid'], 'controller_start_ticks': record['start_ticks'],
                'detached_session': True, 'scope': 'Read-only; no native worker or parameter changes'}
    claim(observer_path, observer)
    argv = [sys.executable, '-u', str(OUT / 'code/watch_evaluation.py'),
            '--stage', stage, '--watch-seconds', '604800']
    with log.open('xb') as out, stderr.open('xb') as err:
        child = subprocess.Popen(argv, cwd=REPO, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''),
            stdin=subprocess.DEVNULL, stdout=out, stderr=err, start_new_session=True)
    observed = process(child.pid)
    assert observed['live'] and f'--stage {stage}' in observed['command']
    observer.update(status='running', observer_pid=child.pid, watcher_pid=child.pid,
                    start_ticks=observed['start_ticks'])
    save(observer_path, observer)
    backup = OUT / f'meta/{PREVIOUS[stage]}-observer-complete-{ATTEMPT}.json'
    if not backup.exists():
        claim(backup, previous)
    save(previous_path, observer)
    return observer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=STAGES, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--attach-observer', action='store_true')
    args = parser.parse_args()
    if args.check:
        print(json.dumps(check(args.stage)[0]))
        return
    with (OUT / 'meta/evaluation-launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger_path, start_path, log = paths(args.stage)
        boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        if args.attach_observer:
            record = read(start_path)
            assert record['boot_id'] == boot
            gate = validate_gate(args.stage)
            assert gate['source_fingerprint'] == record['source_fingerprint']
            print(json.dumps({'start': record, 'observer': attach_observer(args.stage, record)}))
            return
        readiness, gate = check(args.stage)
        if not readiness['ready']:
            raise GateError('; '.join(readiness['reasons']))
        record = {**readiness, 'status': 'launch_claimed', 'boot_id': boot,
                  'started_at_utc': datetime.now(timezone.utc).isoformat(),
                  'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
                  'source_fingerprint': gate['source_fingerprint'],
                  'CP3_acceptance_sha256': sha(OUT / 'meta/CP3_acceptance.json'),
                  'predecessor_ledger_sha256': sha(OUT / f'meta/evaluation-{PREVIOUS[args.stage]}-runs.json'),
                  'CUDA_VISIBLE_DEVICES_in_native_parent': None, 'detached_session': True}
        if args.stage == 'nearest':
            record['CP4_reported_sha256'] = sha(OUT / 'meta/CP4_reported.json')
        claim(start_path, record)
        native_env = dict(os.environ)
        native_env.pop('CUDA_VISIBLE_DEVICES', None)
        argv = [sys.executable, '-u', str(OUT / 'code/evaluate_sequence.py'),
                '--stage', args.stage, '--attempt', ATTEMPT]
        with log.open('xb') as stdout:
            child = subprocess.Popen(argv, cwd=REPO, env=native_env, stdin=subprocess.DEVNULL,
                stdout=stdout, stderr=subprocess.STDOUT, start_new_session=True)
        identity = process(child.pid)
        assert identity['live'] and f'--stage {args.stage}' in identity['command']
        record.update(status='controller_started', pid=child.pid, start_ticks=identity['start_ticks'],
                      command=argv, log=str(log), launches_workers=True)
        save(start_path, record)
        for _ in range(80):
            if ledger_path.exists():
                break
            if child.poll() is not None:
                raise RuntimeError('Controller exited; inspect its original log, do not relaunch')
            time.sleep(.25)
        observer = attach_observer(args.stage, record)
        if observer.get('observer_pid'):
            record.update(status='running', observer_pid=observer['observer_pid'],
                          observer_start_ticks=observer['start_ticks'])
            save(start_path, record)
        print(json.dumps({'start': record, 'observer': observer}))


if __name__ == '__main__':
    main()
