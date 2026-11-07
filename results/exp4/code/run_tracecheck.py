"""Run one excluded 2-MON plumbing check while primary GPU3 is idle."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import re
import yaml

from experiments.exp4_collect import source_fingerprint, write, run_path

ROOT = Path(__file__).resolve().parents[3]
WORK = Path('/nvme/khy/ForesightNav_exp4_tracecheck')


def primary():
    ledger = json.loads((ROOT / 'results/exp4/meta/calibration_runs.json').read_text())
    command = (Path('/proc') / str(ledger['controller_pid']) / 'cmdline').read_bytes()
    if b'experiments.exp4_collect' not in command or ledger['status'] != 'running':
        raise RuntimeError('Primary collector is not confirmed running')
    if ledger['source_fingerprint'] != source_fingerprint():
        raise RuntimeError('Primary frozen runtime changed')
    if any(j['gpu'] == 3 and j['status'] == 'running' for j in ledger['jobs']):
        raise RuntimeError('Primary collector has reclaimed GPU3')
    return ledger['controller_pid']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', default='6s7QHgap2fW')
    parser.add_argument('--episode', default='0')
    parser.add_argument('--name', default='action-trace-plumbing')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.name):
        raise ValueError('Diagnostic name must be a simple distinct tag')
    identities = json.loads((ROOT / 'results/exp4/meta/2mon-identities.json').read_text())
    if not any(r['scene_id'] == args.scene and str(r['runtime_episode_id']) == args.episode for r in identities):
        raise ValueError('This guard only permits known fixed 2-MON identities')
    primary_pid = primary()
    occupancy = subprocess.run(['nvidia-smi', '-i', '3', '--query-compute-apps=pid',
        '--format=csv,noheader'], check=True, text=True, capture_output=True).stdout.strip()
    if occupancy:
        raise RuntimeError('GPU3 is occupied; do not launch a diagnostic')
    out = ROOT / 'results/exp4/diagnostics' / args.name
    out.mkdir(parents=True, exist_ok=True)
    state_path, log_path = out / 'controller.json', out / 'native.log'
    if state_path.exists():
        raise RuntimeError('Diagnostic ledger exists; inspect it rather than relaunching')
    runtime = yaml.safe_load((ROOT / 'runtime.exp4.local.yaml').read_text())
    runtime['paths']['outputs'] = str(out / 'runs')
    runtime['gpus'] = {'mapping': 3, 'navigation': 3}
    runtime_name = f'runtime.tracecheck.{args.name}.local.yaml'
    (WORK / runtime_name).write_text(yaml.safe_dump(runtime, sort_keys=False))
    argv = [sys.executable, str(WORK / 'run.py'), '--config', 'configs/experiments/exp4_calibration.yaml',
            '--runtime', runtime_name, '--gpus', '3,3',
            '--scene', args.scene, '--episode', args.episode, '--tag', args.name]
    state = {'scope': 'fixed 2-MON plumbing only; excluded from calibration and 3-MON',
             'status': 'starting', 'primary_controller_pid': primary_pid,
             'guard_pid': os.getpid(), 'command': argv, 'worktree': str(WORK),
             'log': str(log_path), 'fixed_timeout_s': 900}
    write(state_path, state)
    with log_path.open('x') as handle:
        process = subprocess.Popen(argv, cwd=WORK, stdout=handle, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, start_new_session=True)
        state.update(status='running', launcher_pid=process.pid)
        write(state_path, state)
        print(json.dumps(state), flush=True)
        deadline = time.monotonic() + 900
        try:
            while process.poll() is None:
                try:
                    primary()
                except (FileNotFoundError, json.JSONDecodeError) as error:
                    print('Transient primary observation failure: ' + str(error), flush=True)
                except RuntimeError as error:
                    state.update(status='stopped_for_primary_priority', reason=str(error))
                    process.terminate()
                    break
                if time.monotonic() >= deadline:
                    state.update(status='stopped_at_fixed_diagnostic_timeout')
                    process.terminate()
                    break
                time.sleep(2)
            if process.poll() is None:
                process.wait(timeout=30)
            if state['status'] == 'running':
                state['status'] = 'finished' if process.returncode == 0 else 'failed'
            state['returncode'] = process.returncode
        finally:
            if process.poll() is None:
                process.terminate()
            run = run_path(log_path)
            state['run_directory'] = str(run) if run else None
            write(state_path, state)
    print(json.dumps(state, indent=2), flush=True)
    return 0 if state['status'] == 'finished' else 1


if __name__ == '__main__':
    sys.exit(main())
