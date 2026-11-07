"""Collect three fixed 2-MON folds on GPUs 2/3; never launch evaluation."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/exp4'


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def source_fingerprint():
    paths = [p for folder in ('pbsp', 'experiments') for p in sorted((ROOT / folder).rglob('*.py'))
             if not any(part in {'runs', 'results', 'analysis', '__pycache__'} for part in p.parts)]
    paths.extend(p for p in sorted((ROOT / 'configs').rglob('*'))
                 if p.is_file() and p.suffix in {'.yaml', '.json'})
    paths.append(ROOT / 'runtime.exp4.local.yaml')
    paths.extend(OUT / 'meta' / name for name in ('folds.json', '2mon-identities.json',
                 '3mon-identities.json', 'smoke-20-identities.json', 'protocol_decisions.json'))
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def audit_run(run, scene):
    status = json.loads((run / 'status.json').read_text())
    if status['status'] != 'completed':
        raise RuntimeError(f'Calibration scene did not complete: {status}')
    metrics = json.loads((run / 'mon_metrics.json').read_text())
    if metrics['counts']['completed_episodes'] != 30:
        raise RuntimeError('Calibration must contain exactly 30 native episodes per scene')
    errors = list(run.rglob('blp_errors.jsonl'))
    if any(p.stat().st_size for p in errors):
        raise RuntimeError('Shadow planning error; do not fit this run')
    events = [r for p in run.rglob('blp_events.jsonl') for r in read_jsonl(p)]
    snapshots = [r for p in run.rglob('blp_snapshots.jsonl') for r in read_jsonl(p)]
    expected = {str(r['runtime_episode_id']) for r in json.loads((OUT / 'meta/2mon-identities.json').read_text())
                if r['scene_id'] == scene}
    observed = {str(r['episode_id']) for r in snapshots}
    if expected != observed:
        raise RuntimeError('Shadow snapshots do not cover the frozen episode list')
    for row in events:
        if row['event'] == 'verification_final':
            if row.get('gt_diagnostic', {}).get('criterion') != 'pinned_GOAT_measures_at_candidate_position':
                raise RuntimeError('Wrong diagnostic label definition')
    return {'events': len(events), 'snapshots': len(snapshots),
            'verification_events': sum(r['event'] == 'verification_final' for r in events),
            'native_metrics': metrics['mon_metrics']}


def run_path(log):
    if not log.exists():
        return None
    values = [line[5:].strip() for line in log.read_text().splitlines() if line.startswith('Run: ')]
    return Path(values[-1]) if values else None


def main():
    acceptance = json.loads((OUT / 'meta/CP1_acceptance.json').read_text())
    fingerprint = source_fingerprint()
    if not acceptance.get('accepted') or acceptance['source_fingerprint'] != fingerprint:
        raise RuntimeError('CP1 implementation acceptance is missing or source-bound checks changed')
    path = OUT / 'meta/calibration_runs.json'
    if path.exists():
        raise RuntimeError('Collection ledger exists; inspect current jobs before explicit recovery')
    folds = json.loads((OUT / 'meta/folds.json').read_text())['folds']
    attempt = acceptance.get('collection_attempt', 'a1')
    ledger = {'status': 'running', 'controller_pid': os.getpid(),
              'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
              'source_fingerprint': fingerprint, 'jobs': []}
    active = {}
    def stop(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    write(path, ledger)
    try:
        for fold in range(3):
            pending = list(folds[fold])
            while pending or active:
                if source_fingerprint() != fingerprint:
                    raise RuntimeError('Source changed during calibration collection')
                for gpu in (2, 3):
                    if gpu in active or not pending:
                        continue
                    scene = pending.pop(0)
                    log = OUT / f'logs/calibration-{attempt}-fold{fold}-{scene}.controller.log'
                    handle = log.open('x')
                    command = [str(ROOT / 'run.py'), '--config', 'configs/experiments/exp4_calibration.yaml',
                               '--runtime', 'runtime.exp4.local.yaml', '--gpus', f'{gpu},{gpu}',
                               '--scene', scene, '--tag', f'calibration-{attempt}-fold{fold}-{scene}']
                    process = subprocess.Popen([os.sys.executable, *command], cwd=ROOT,
                                               stdout=handle, stderr=subprocess.STDOUT,
                                               stdin=subprocess.DEVNULL, start_new_session=True)
                    job = {'fold': fold, 'scene': scene, 'gpu': gpu, 'pid': process.pid,
                           'log': str(log), 'status': 'running', 'command': command}
                    ledger['jobs'].append(job)
                    active[gpu] = (process, handle, job)
                for gpu, (process, handle, job) in list(active.items()):
                    run = run_path(Path(job['log']))
                    if run:
                        job['run_directory'] = str(run)
                        if any(p.stat().st_size for p in run.rglob('blp_errors.jsonl')):
                            raise RuntimeError('Live shadow error; halt collection for review')
                    returncode = process.poll()
                    if returncode is not None:
                        handle.close()
                        if returncode != 0 or run is None:
                            raise RuntimeError(f'Calibration child failed: {job}, code={returncode}')
                        job['audit'] = audit_run(run, job['scene'])
                        job.update(status='completed', returncode=returncode)
                        del active[gpu]
                write(path, ledger)
                if active:
                    time.sleep(10)
            for kind in ('events', 'snapshots'):
                destination = OUT / f'logs/calibration_fold{fold}_{kind}.jsonl'
                with destination.open('x') as stream:
                    for job in ledger['jobs']:
                        if job['fold'] == fold:
                            for source in sorted(Path(job['run_directory']).rglob(f'blp_{kind}.jsonl')):
                                stream.write(source.read_text())
        ledger['status'] = 'completed_three_folds_pending_CP2a'
        write(path, ledger)
    except BaseException as exc:
        ledger.update(status='stopped_for_review', error=f'{type(exc).__name__}: {exc}')
        write(path, ledger)
        raise
    finally:
        for process, handle, job in active.values():
            if process.poll() is None:
                process.terminate()
        for process, handle, job in active.values():
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                # Native launchers own and close their separate worker groups;
                # retain identity evidence for recovery instead of broad pkill.
                job['termination_wait_exceeded'] = True
            handle.close()
        write(path, ledger)


if __name__ == '__main__':
    main()
