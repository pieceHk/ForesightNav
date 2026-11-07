"""Checkpoint-bound 3-MON evaluation; never substitutes partial episode sets.

Each invocation executes one stage. Check mode never launches workers. Smoke
artifacts have an isolated directory and cannot overwrite Full-400 results.
"""
from collections import defaultdict
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

from experiments.exp4_collect import source_fingerprint, run_path, write
from report_variants import normalize, format_summary, write_json, export_comparison
from pbsp.navigation.blp import BeliefParameters

REPO = Path(__file__).resolve().parents[3]
OUT = REPO/'results/exp4'
STAGES = ('smoke', 'full', 'nearest', 'myopic', 'suffix_only')


class GateError(RuntimeError):
    pass


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError as exc:
        raise GateError(f'Missing checkpoint artifact: {path}') from exc


def build_jobs(stage, identities):
    if stage not in STAGES:
        raise GateError('Unknown evaluation stage')
    expected_count = 20 if stage == 'smoke' else 400
    if len(identities) != expected_count:
        raise GateError(f'{stage} requires exactly {expected_count} frozen identities')
    grouped, seen = defaultdict(list), set()
    for identity in identities:
        scene, episode = str(identity['scene_id']), str(identity['runtime_episode_id'])
        if (scene, episode) in seen:
            raise GateError('Repeated frozen evaluation identity')
        if len(identity['ordered_categories']) != 3 or len(identity['source_episode_ids']) != 3:
            raise GateError('Every evaluation episode must retain three ordered targets')
        grouped[scene].append(identity)
        seen.add((scene, episode))
    per_scene = 1 if stage == 'smoke' else 20
    if len(grouped) != 20 or any(len(rows) != per_scene for rows in grouped.values()):
        raise GateError('Evaluation must cover all twenty scenes with the prescribed cardinality')
    return [{'scene': scene, 'identities': grouped[scene]} for scene in sorted(grouped)]


def validate_calibration_coverage(out, ledger):
    if ledger.get('status') not in {'completed_three_folds_pending_CP2a', 'completed_five_folds_pending_CP2a'}:
        raise GateError('Calibration collection has not completed')
    folds = load(out/'meta/folds.json')['folds']
    jobs = ledger.get('jobs', [])
    selected_folds = {j['fold'] for j in jobs}
    if selected_folds not in ({0, 1, 2}, {0, 1, 2, 3, 4}):
        raise GateError('Only complete three-fold or five-fold calibration is admissible')
    observed = [(j['fold'], j['scene']) for j in jobs]
    expected = [(f, scene) for f in selected_folds for scene in folds[f]]
    if len(observed) != len(set(observed)) or set(observed) != set(expected):
        raise GateError('Calibration scene/fold coverage is incomplete or duplicated')
    for job in jobs:
        run = Path(job['run_directory'])
        if job.get('status') != 'completed' or load(run/'status.json')['status'] != 'completed':
            raise GateError('A calibration job is not authoritatively complete')
        counts = load(run/'mon_metrics.json')['counts']
        if counts['completed_episodes'] != 30 or counts['targets_per_episode'] != 2:
            raise GateError('A calibration scene lacks thirty complete two-target episodes')


def validate_stage_proof(stage, out):
    """Inspect completed native jobs and the identity-bound normalized output."""
    out = Path(out)
    ledger = load(out/f'meta/evaluation-{stage}-runs.json')
    n = 20 if stage == 'smoke' else 400
    variant = 'full' if stage == 'smoke' else stage
    destination = out/'diagnostics/CP3' if stage == 'smoke' else out
    if ledger.get('status') not in {'completed', 'completed_requires_score_review'} or ledger.get('n_episodes') != n:
        raise GateError('The preceding evaluation stage is not complete')
    if len(ledger.get('jobs', [])) != 20:
        raise GateError('The preceding evaluation does not contain twenty scene assignments')
    for job in ledger['jobs']:
        if job.get('status') != 'completed':
            raise GateError('A preceding native scene job is not complete')
        audit_scene(Path(job['run_directory']), job)
    artifacts = ledger.get('artifact_sha256', {})
    for relative in (f'summary/{variant}.json', f'logs/{variant}_episodes.jsonl', f'logs/{variant}_decisions.jsonl'):
        if artifacts.get(relative) != digest(destination/relative):
            raise GateError('A preceding evaluation artifact changed after finalization')
    summary = load(destination/f'summary/{variant}.json')
    if summary['n_episodes'] != n or summary['variant'] != variant or summary.get('diagnostic_only'):
        raise GateError('A preceding summary has the wrong scope or variant')
    rows = [json.loads(line) for line in (destination/f'logs/{variant}_episodes.jsonl').read_text().splitlines()]
    wanted = {(r['scene_id'], str(r['runtime_episode_id'])): r for r in expected_identities(stage, out)}
    actual = {(r['scene_id'], str(r['episode_id'])) for r in rows}
    if len(rows) != n or actual != set(wanted):
        raise GateError('Preceding normalized episodes do not cover the fixed identities')
    for row in rows:
        identity = wanted[row['scene_id'], str(row['episode_id'])]
        if row['goals'] != identity['ordered_categories'] or row['source_episode_ids'] != identity['source_episode_ids'] or len(row['stages']) != 3:
            raise GateError('Preceding ordered task/source mapping differs from the frozen list')
    if ledger['status'] == 'completed_requires_score_review':
        review = load(out/f'meta/evaluation-{stage}-score-review.json')
        if review.get('accepted') is not True or review.get('summary_sha256') != digest(destination/f'summary/{variant}.json'):
            raise GateError('Anomalous score requires an explicit evidence-based review')
    return ledger


def validate_gate(stage, out=OUT, fingerprint_reader=source_fingerprint):
    out = Path(out)
    cal_path = out/'meta/calibration_runs.json'
    validate_calibration_coverage(out, load(cal_path))
    parameter_path = out/'meta/calibration.json'
    parameters = load(parameter_path)
    if parameters.get('frozen') is not True or parameters.get('label_source') != 'system_verification':
        raise GateError('Evaluation requires frozen self-verification parameters')
    if 'parameters_by_scene' in parameters or parameters.get('selection') in {'cross_fit', 'crossfit'}:
        raise GateError('Cross-fitting must first be enabled and tested in the runtime loader and analysis')
    try:
        BeliefParameters(**parameters['parameters'])
    except (KeyError, TypeError, ValueError) as error:
        raise GateError('Frozen pooled-parameter schema is invalid or needs cross-fit support') from error
    cp2 = load(out/'meta/CP2b_acceptance.json')
    fingerprint = fingerprint_reader()
    if cp2.get('accepted') is not True or cp2.get('source_fingerprint') != fingerprint:
        raise GateError('CP2b source-bound review is missing or obsolete')
    if cp2.get('parameter_sha256') != digest(parameter_path) or cp2.get('calibration_ledger_sha256') != digest(cal_path):
        raise GateError('Frozen parameters or calibration provenance changed after CP2b')
    if stage == 'full':
        smoke_path = out/'meta/evaluation-smoke-runs.json'
        smoke = validate_stage_proof('smoke', out)
        review = load(out/'meta/CP3_acceptance.json')
        if smoke.get('source_fingerprint') != fingerprint or smoke.get('parameter_sha256') != digest(parameter_path):
            raise GateError('Smoke used different runtime sources or parameters')
        if smoke.get('n_episodes') != 20:
            raise GateError('Full requires the complete twenty-episode smoke run')
        if (review.get('accepted') is not True or review.get('behavior_reviewed') is not True
                or review.get('source_fingerprint') != fingerprint
                or review.get('parameter_sha256') != digest(parameter_path)
                or review.get('smoke_ledger_sha256') != digest(smoke_path)):
            raise GateError('Full requires a current CP3 behavior review')
    predecessor = {'nearest': 'full', 'myopic': 'nearest', 'suffix_only': 'myopic'}.get(stage)
    if predecessor:
        prior = validate_stage_proof(predecessor, out)
        if prior.get('n_episodes') != 400:
            raise GateError(f'{stage} must follow a complete {predecessor} run')
        if prior['source_fingerprint'] != fingerprint or prior['parameter_sha256'] != digest(parameter_path):
            raise GateError('Variants cannot silently mix different runtime sources or parameters')
        if stage == 'nearest':
            reported = load(out/'meta/CP4_reported.json')
            if reported.get('reported') is not True or reported.get('summary_sha256') != digest(out/'summary/full.json'):
                raise GateError('Report CP4 before proceeding to the other three variants')
    return {'source_fingerprint': fingerprint, 'parameter_sha256': digest(parameter_path),
            'calibration_ledger_sha256': digest(cal_path), 'parameters': parameters['parameters']}


def expected_identities(stage, out=OUT):
    name = 'smoke-20-identities.json' if stage == 'smoke' else '3mon-identities.json'
    return load(Path(out)/'meta'/name)


def command(stage, scene, identities, gpu, attempt):
    variant = 'full' if stage == 'smoke' else stage
    argv = [sys.executable, str(REPO/'run.py'), '--config', f'configs/experiments/exp4_{variant}.yaml',
            '--runtime', 'runtime.exp4.local.yaml', '--gpus', f'{gpu},{gpu}',
            '--scene', scene, '--tag', f'exp4-{stage}-{attempt}-{scene}']
    if stage == 'smoke':
        argv.extend(['--episode', str(identities[0]['runtime_episode_id'])])
    return argv


def audit_scene(run, job):
    if load(run/'status.json')['status'] != 'completed':
        raise GateError('Native evaluation run did not complete')
    plan = load(run/'launch.json')
    selected = {(r['scene_id'], str(r['runtime_episode_id'])) for r in plan['identities']}
    expected = {(r['scene_id'], str(r['runtime_episode_id'])) for r in job['identities']}
    counts = load(run/'mon_metrics.json')['counts']
    if selected != expected or counts['completed_episodes'] != len(expected) or counts['targets_per_episode'] != 3:
        raise GateError('Native results differ from the fixed three-target scene assignment')
    if any(p.stat().st_size for p in run.rglob('blp_errors.jsonl')):
        raise GateError('Evaluation has unreviewed BLP errors')
    return {'completed_episodes': counts['completed_episodes'], 'identity_count': len(selected)}


def finalize(stage, ledger, out=OUT):
    out = Path(out)
    if len(ledger['jobs']) != 20 or any(j['status'] != 'completed' for j in ledger['jobs']):
        raise GateError('Finalization cannot substitute partial scene coverage')
    variant = 'full' if stage == 'smoke' else stage
    calibration = load(out/'meta/calibration.json')
    rows, decisions, summary = normalize(variant, [j['run_directory'] for j in ledger['jobs']],
        expected=expected_identities(stage, out), parameters=calibration['parameters'])
    destination = out/'diagnostics/CP3' if stage == 'smoke' else out
    write_json(destination/'summary'/f'{variant}.json', format_summary(summary))
    (destination/'logs').mkdir(parents=True, exist_ok=True)
    for kind, values in (('episodes', rows), ('decisions', decisions)):
        (destination/'logs'/f'{variant}_{kind}.jsonl').write_text(
            ''.join(json.dumps(r, allow_nan=False) + '\n' for r in values))
    ledger['artifact_sha256'] = {relative: digest(destination/relative) for relative in (
        f'summary/{variant}.json', f'logs/{variant}_episodes.jsonl', f'logs/{variant}_decisions.jsonl')}
    ledger.update(n_episodes=summary['n_episodes'], summary_path=str(destination/'summary'/f'{variant}.json'),
                  n_error=summary['n_error'])
    if summary['SR'] == 0. or summary['SR'] > 90.:
        ledger['status'] = 'completed_requires_score_review'
    else:
        ledger['status'] = 'completed'
    if stage == 'suffix_only' and ledger['status'] == 'completed':
        variants = {}
        for name in ('full', 'nearest', 'myopic', 'suffix_only'):
            episodes = [json.loads(s) for s in (out/f'logs/{name}_episodes.jsonl').read_text().splitlines()]
            variants[name] = (episodes, [], load(out/f'summary/{name}.json'))
        export_comparison(out, variants)
    return summary


def execute(stage, attempt, finalize_only=False):
    gate = validate_gate(stage)
    expected = expected_identities(stage)
    assignments = build_jobs(stage, expected)
    path = OUT/f'meta/evaluation-{stage}-runs.json'
    if finalize_only:
        ledger = load(path)
        if ledger['source_fingerprint'] != gate['source_fingerprint'] or ledger['parameter_sha256'] != gate['parameter_sha256']:
            raise GateError('Cannot finalize results under changed source/parameters')
        summary = finalize(stage, ledger)
        write(path, ledger)
        return summary
    if path.exists():
        raise GateError('Stage ledger already exists; inspect original process/results before recovery')
    ledger = {**{k: v for k, v in gate.items() if k != 'parameters'},
              'stage': stage, 'attempt': attempt, 'status': 'running', 'controller_pid': os.getpid(),
              'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(), 'jobs': []}
    script_hash = digest(__file__)
    active, pending = {}, list(assignments)
    write(path, ledger)
    def stop(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        while pending or active:
            if source_fingerprint() != gate['source_fingerprint'] or digest(OUT/'meta/calibration.json') != gate['parameter_sha256'] or digest(__file__) != script_hash:
                raise GateError('Runtime, frozen parameters or controller changed during evaluation')
            for gpu in (2, 3):
                if gpu in active or not pending:
                    continue
                assignment = pending.pop(0)
                log = OUT/f'logs/evaluation-{stage}-{attempt}-{assignment["scene"]}.controller.log'
                handle = log.open('x')
                argv = command(stage, assignment['scene'], assignment['identities'], gpu, attempt)
                process = subprocess.Popen(argv, cwd=REPO, stdout=handle, stderr=subprocess.STDOUT,
                                           stdin=subprocess.DEVNULL, start_new_session=True)
                job = dict(assignment, gpu=gpu, pid=process.pid, log=str(log), command=argv, status='running')
                ledger['jobs'].append(job)
                active[gpu] = (process, handle, job)
            for gpu, (process, handle, job) in list(active.items()):
                run = run_path(Path(job['log']))
                if run:
                    job['run_directory'] = str(run)
                    if any(p.stat().st_size for p in run.rglob('blp_errors.jsonl')):
                        raise GateError('Live BLP error; stop for review')
                code = process.poll()
                if code is not None:
                    handle.close()
                    if code != 0 or run is None:
                        raise GateError(f'Native evaluation failed: {job["scene"]}, returncode={code}')
                    job.update(status='completed', returncode=code, audit=audit_scene(run, job))
                    del active[gpu]
            write(path, ledger)
            if active:
                time.sleep(10)
        ledger['status'] = 'completed_pending_process_audit'
        write(path, ledger)
        summary = finalize(stage, ledger)
        write(path, ledger)
        return summary
    except BaseException as error:
        ledger.update(status='stopped_for_review', error=f'{type(error).__name__}: {error}')
        write(path, ledger)
        raise
    finally:
        for process, handle, job in active.values():
            if process.poll() is None:
                process.terminate()
        for process, handle, job in active.values():
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                job['termination_wait_exceeded'] = True
            handle.close()
        write(path, ledger)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=STAGES)
    parser.add_argument('--attempt', default='a1')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--finalize-only', action='store_true')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.attempt):
        raise GateError('Attempt must be a simple distinct tag')
    if args.check:
        jobs = build_jobs(args.stage, expected_identities(args.stage))
        try:
            validate_gate(args.stage)
            ready, reason = True, None
        except GateError as error:
            ready, reason = False, str(error)
        print(json.dumps({'stage': args.stage, 'ready': ready, 'reason': reason,
                          'scenes': len(jobs), 'episodes': sum(len(j['identities']) for j in jobs),
                          'launches_workers': False}, indent=2))
        return
    result = execute(args.stage, args.attempt, args.finalize_only)
    print(json.dumps({key: result[key] for key in ('variant', 'n_episodes', 'SR', 'SPL', 'PR', 'PPL')}, indent=2))


if __name__ == '__main__':
    main()
