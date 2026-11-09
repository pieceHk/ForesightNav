"""Wait for this frozen a3 collection, then produce an unfrozen CP2a proposal.

CPU postprocessing only. No native workers, parameter acceptance, freeze, or
3-MON evaluation can be launched here. An unexpected state stops for review.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from evaluate_sequence import validate_calibration_coverage
from experiments.exp4_collect import source_fingerprint

REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / 'results/exp4'
STATE = ROOT / 'meta/calibration-postprocess.json'
STEPS = ('audit_completed_calibration.py', 'verify_action_path_pipeline.py',
         'archive_calibration_logs.py', 'fit_calibration_reviewed.py', 'report_calibration.py')
ANALYSIS = ('report_variants.py', 'native_candidate_gt.py', 'action_path_metrics.py',
            'evaluate_sequence.py', 'finish_calibration.py', 'candidate_identities.py')
CHILD = None


def read(path):
    for attempt in range(3):
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            if attempt == 2:
                raise
            time.sleep(.2)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(value):
    temporary = STATE.with_suffix('.tmp.' + str(os.getpid()))
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temporary, STATE)


def runtime_guard():
    start = read(ROOT / 'meta/calibration-start-a3.json')
    cp1 = read(ROOT / 'meta/CP1_acceptance.json')
    ledger = read(ROOT / 'meta/calibration_runs.json')
    if (start['attempt'] != 'a3' or cp1.get('accepted') is not True or
            cp1.get('collection_attempt') != 'a3' or
            source_fingerprint() != start['source_fingerprint'] or
            ledger['source_fingerprint'] != start['source_fingerprint'] or
            cp1['source_fingerprint'] != start['source_fingerprint']):
        raise ValueError('The accepted a3 runtime binding differs')
    if Path('/proc/sys/kernel/random/boot_id').read_text().strip() != start['boot_id']:
        raise ValueError('The server rebooted after collection started')
    if (ROOT / 'meta/calibration.json').exists():
        raise ValueError('Do not postprocess after parameters have been frozen')
    if ledger['status'] == 'running':
        proc = Path('/proc') / str(start['pid'])
        stat = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
        command = (proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode()
        if (stat[0] == 'Z' or stat[19] != str(start['start_ticks']) or
                'experiments.exp4_collect' not in command or proc.joinpath('cwd').resolve() != REPO):
            raise ValueError('The original collection controller is not live and matching')
    elif ledger['status'] == 'completed_three_folds_pending_CP2a':
        validate_calibration_coverage(ROOT, ledger)
    else:
        raise ValueError('Collection stopped outside the expected completed state: ' + ledger['status'])
    return start, ledger


def stop_cpu_child(signum, frame):
    if CHILD is not None and CHILD.poll() is None:
        CHILD.terminate()
        try:
            CHILD.wait(timeout=10)
        except subprocess.TimeoutExpired:
            CHILD.kill()
            CHILD.wait()
    raise SystemExit(128 + signum)


def main():
    global CHILD
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    start, ledger = runtime_guard()
    script_hashes = {name: digest(ROOT / 'code' / name) for name in (*STEPS, *ANALYSIS)}
    if args.check:
        print(json.dumps({'ready_to_schedule': not STATE.exists(),
                          'collection_status': ledger['status'],
                          'ready_to_process': ledger['status'] == 'completed_three_folds_pending_CP2a',
                          'fixed_input_count': len(start['source_fingerprint']),
                          'cpu_steps': STEPS, 'freezes_parameters': False, 'launches_native_workers': False}))
        return
    lock = (ROOT / 'meta/calibration-postprocess.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if STATE.exists() or (ROOT / 'meta/calibration-fit-binding.json').exists():
        raise ValueError('Inspect the existing postprocessing state before attempting another run')
    stat = Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()
    value = {'pid': os.getpid(), 'start_ticks': stat[19], 'boot_id': start['boot_id'],
             'started_at_utc': datetime.now(timezone.utc).isoformat(), 'collection_attempt': 'a3',
             'collection_controller_pid': start['pid'], 'source_fingerprint': start['source_fingerprint'],
             'script_sha256': script_hashes, 'status': 'waiting_for_complete_collection',
             'completed_steps': [], 'freezes_parameters': False, 'launches_native_workers': False}
    write(value)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop_cpu_child)

    def guard():
        result = runtime_guard()
        if any(digest(ROOT / 'code' / name) != wanted for name, wanted in script_hashes.items()):
            raise ValueError('A bound postprocessing script changed while queued')
        return result

    try:
        while True:
            _, ledger = guard()
            value['last_checked_at_utc'] = datetime.now(timezone.utc).isoformat()
            value['collection_status'] = ledger['status']
            write(value)
            if ledger['status'] == 'completed_three_folds_pending_CP2a':
                break
            time.sleep(30)
        for name in STEPS:
            guard()
            value.update(status='processing', current_step=name)
            write(value)
            print(json.dumps({'starting_cpu_step': name}), flush=True)
            CHILD = subprocess.Popen([sys.executable, str(ROOT / 'code' / name)], cwd=REPO)
            while CHILD.poll() is None:
                guard()
                time.sleep(10)
            if CHILD.returncode:
                raise RuntimeError(f'{name} returned {CHILD.returncode}')
            guard()
            value['completed_steps'].append(name)
            write(value)
        proposal = ROOT / 'meta/calibration_per_fold.json'
        value.update(status='completed_pending_CP2a_review',
                     finished_at_utc=datetime.now(timezone.utc).isoformat(),
                     fit_sha256=digest(proposal), report_sha256=digest(ROOT / 'reports/CP2a.md'))
        write(value)
        print(json.dumps({'status': value['status'], 'freezes_parameters': False}), flush=True)
    except BaseException as error:
        if CHILD is not None and CHILD.poll() is None:
            CHILD.terminate()
            try:
                CHILD.wait(timeout=10)
            except subprocess.TimeoutExpired:
                CHILD.kill()
                CHILD.wait()
        value.update(status='stopped_for_review', error=repr(error),
                     stopped_at_utc=datetime.now(timezone.utc).isoformat())
        write(value)
        raise


if __name__ == '__main__':
    main()
