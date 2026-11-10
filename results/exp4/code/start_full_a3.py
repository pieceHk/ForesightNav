from pathlib import Path
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time

from evaluate_sequence import validate_gate, build_jobs, expected_identities
from watch_evaluation import process

repo = Path('/nvme/khy/ForesightNav_exp4_depthfix')
root = repo / 'results/exp4'
def save(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)
lock = (root / 'meta/evaluation-launch.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
stage, attempt = 'full', 'a3'
gate = validate_gate(stage)
jobs = build_jobs(stage, expected_identities(stage))
assert len(jobs) == 20 and sum(len(job['identities']) for job in jobs) == 400
ledger_path = root / f'meta/evaluation-{stage}-runs.json'
start_path = root / f'meta/{stage}-start-{attempt}.json'
assert not ledger_path.exists() and not start_path.exists(), 'Inspect existing launch/ledger; never duplicate it'
for other in ('nearest', 'myopic', 'suffix_only'):
    assert not (root / f'meta/evaluation-{other}-runs.json').exists()
gpu_rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,memory.total,memory.free',
    '--format=csv,noheader,nounits'], text=True).splitlines()
gpu_rows = [row for row in gpu_rows if row.split(',')[0].strip() in ('2', '3')]
assert len(gpu_rows) == 2 and all(float(row.split(',')[-1]) >= 38 * 1024 for row in gpu_rows), gpu_rows
disk = os.statvfs(root)
assert disk.f_bavail * disk.f_frsize >= 30 * 1024**3
started = datetime.now(timezone.utc)
boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
record = {'stage': stage, 'attempt': attempt, 'status': 'launch_claimed', 'started_at_utc': started.isoformat(),
    'boot_id': boot, 'source_commit': source, 'parameter_sha256': gate['parameter_sha256'],
    'CP3_acceptance_sha256': hashlib.sha256((root / 'meta/CP3_acceptance.json').read_bytes()).hexdigest(),
    'gpu_observation': gpu_rows, 'n_scenes': 20, 'n_episodes': 400}
with start_path.open('x') as output:
    output.write(json.dumps(record, indent=2) + '\n')
native_env = dict(os.environ)
native_env.pop('CUDA_VISIBLE_DEVICES', None)
log = root / f'logs/evaluation-{stage}-controller-{attempt}.log'
argv = [sys.executable, '-u', str(root / 'code/evaluate_sequence.py'), '--stage', stage, '--attempt', attempt]
with log.open('xb') as stdout:
    controller = subprocess.Popen(argv, cwd=repo, env=native_env, stdin=subprocess.DEVNULL,
        stdout=stdout, stderr=subprocess.STDOUT, start_new_session=True)
identity = process(controller.pid)
assert identity['live'] and '--stage full' in identity['command']
record.update(status='controller_started', pid=controller.pid, start_ticks=identity['start_ticks'],
    command=argv, log=str(log), CUDA_VISIBLE_DEVICES_in_native_parent=None, detached_session=True)
save(start_path, record)
for _ in range(80):
    if ledger_path.exists():
        break
    if controller.poll() is not None:
        raise RuntimeError('Native controller exited; inspect original log, do not retry launch')
    time.sleep(.25)
assert ledger_path.exists(), 'Controller initializing: inspect recorded PID; never duplicate it'
ledger = json.loads(ledger_path.read_text())
assert ledger['controller_pid'] == controller.pid and ledger['boot_id'] == boot and ledger['status'] == 'running'
stamp = started.strftime('%Y%m%dT%H%M%SZ')
observer_log = root / f'logs/evaluation-observer-{stage}-{attempt}-{stamp}.jsonl'
observer_err = observer_log.with_name(observer_log.stem + '.stderr.log')
observer_argv = [sys.executable, '-u', str(root / 'code/watch_evaluation.py'), '--stage', stage, '--watch-seconds', '604800']
observer_env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
with observer_log.open('xb') as out, observer_err.open('xb') as err:
    observer = subprocess.Popen(observer_argv, cwd=repo, env=observer_env, stdin=subprocess.DEVNULL,
        stdout=out, stderr=err, start_new_session=True)
observation_identity = process(observer.pid)
assert observation_identity['live'] and 'watch_evaluation.py' in observation_identity['command']
observer_record = {'stage': stage, 'attempt': attempt, 'observer_pid': observer.pid, 'watcher_pid': observer.pid,
    'start_ticks': observation_identity['start_ticks'], 'boot_id': boot, 'started_at_utc': datetime.now(timezone.utc).isoformat(),
    'watch_seconds': 604800, 'log': str(observer_log), 'stderr_log': str(observer_err),
    'watcher_sha256': hashlib.sha256((root / 'code/watch_evaluation.py').read_bytes()).hexdigest(),
    'controller_pid': controller.pid, 'controller_start_ticks': identity['start_ticks'], 'detached_session': True,
    'scope': 'Read-only observation of existing Full400; no native or parameter changes; independent of SSH'}
save(root / f'meta/{stage}-observer-{attempt}.json', observer_record)
previous = root / 'meta/last-evaluation-observer.json'
if previous.exists():
    old = json.loads(previous.read_text())
    assert old['stage'] == 'smoke'
    save(root / 'meta/smoke-observer-complete-a3.json', old)
save(previous, observer_record)
record.update(status='running', observer_pid=observer.pid, observer_start_ticks=observation_identity['start_ticks'])
save(start_path, record)
print(json.dumps({'start': record, 'observer': observer_record, 'stage_user_authorized': True}), flush=True)
