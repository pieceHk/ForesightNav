"""CP2 analysis with a checked numerical retry for the same logistic MLE.

The frozen runtime source is untouched. Only an optimizer-status failure is
retried with SLSQP on mean negative log likelihood (identical minimizer). True
one-class, separation or missing-data failures retain the original error.
"""
from collections import Counter
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

import experiments.blp_calibration as calibration
import experiments.exp4_fit as driver
from pbsp.navigation.blp import BeliefParameters
from candidate_identities import canonicalize, DEFINITION as IDENTITY_DEFINITION, SOURCE_SHA256 as IDENTITY_SHA256
from experiments.exp4_collect import source_fingerprint

RAW_FIT = calibration.fit
RETRIES = Counter()


def reviewed_decision_flip_rate(snapshots, theta, reference):
    """Report all-snapshot flips plus the nonempty-root conditional rate."""
    if not snapshots:
        raise calibration.CalibrationDataError('No replay snapshots')
    flips = eligible = 0
    for row in snapshots:
        snapshot = row['snapshot']
        if not snapshot['layers'][snapshot['k']]:
            continue  # None equals None: no selection can flip on an empty root.
        eligible += 1
        kwargs = {'variant': 'full', 'active_raw_rank': row['active_raw_rank']}
        left = calibration.solve_snapshot(snapshot, theta, **kwargs)['chosen_raw_rank']
        right = calibration.solve_snapshot(snapshot, reference, **kwargs)['chosen_raw_rank']
        flips += left != right
    return {'flips': flips, 'all_snapshots': len(snapshots), 'eligible_snapshots': eligible,
            'rate': flips / len(snapshots), 'eligible_rate': flips / eligible if eligible else None,
            'denominator': 'all recorded snapshots; empty-root selections are both None'}


def robust_fit(events, label='v'):
    events, _, _ = canonicalize(events)
    try:
        return RAW_FIT(events, label=label)
    except calibration.CalibrationDataError as error:
        if not str(error).startswith('Logistic optimizer failed:'):
            raise
        RETRIES[label] += 1
    rows = [r for r in events if r['event'] == 'verification_final']
    e = np.asarray([r['E'] for r in rows], dtype=float)
    y = np.asarray([r[label] for r in rows], dtype=float)
    center, scale = float(e.mean()), max(float(e.std()), 1e-8)
    x = (e - center) / scale
    lower = 1e-8 * scale

    def objective(theta):
        z = theta[0] * x + theta[1]
        residual = expit(z) - y
        return float(np.mean(np.logaddexp(0., z) - y * z)), np.array([
            np.mean(residual * x), np.mean(residual)])

    rate = float(y.mean())
    result = minimize(objective, [10. * scale, np.log(rate / (1. - rate))], jac=True,
                      bounds=[(lower, None), (None, None)], method='SLSQP',
                      options={'ftol': 1e-12, 'maxiter': 2000})
    _, gradient = objective(result.x)
    if result.x[0] <= lower + 1e-7:
        gradient[0] = min(gradient[0], 0.)
    if not np.isfinite(result.x).all() or float(np.max(np.abs(gradient))) > 1e-6:
        raise calibration.CalibrationDataError('Retry failed constrained convex KKT check')
    a = float(result.x[0] / scale)
    b0 = float(result.x[1] - a * center)
    if a > 1e6 or abs(b0) > 1e6:
        raise calibration.CalibrationDataError('Retry is separated/unbounded')
    unique = {}
    for row in rows:
        key = (row.get('_resample_instance'), row['scene_id'], row['episode_id'],
               row['k'], row['candidate_id'])
        unique[key] = max(unique.get(key, 0), int(row[label]))
    exploration = [r['path_to_first_candidate_m'] for r in events if r['event'] == 'empty_activation']
    return BeliefParameters(a=a, b0=b0, pi=float(np.mean(list(unique.values()))),
                            c_exp=float(np.mean(exploration)))


def install_analysis_retry():
    calibration.fit = robust_fit
    driver.fit = robust_fit
    driver.decision_flip_rate = reviewed_decision_flip_rate


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def input_binding():
    root = driver.ROOT
    ledger = json.loads((root / 'meta/calibration_runs.json').read_text())
    start = json.loads((root / 'meta/calibration-start-a3.json').read_text())
    audit = json.loads((root / 'meta/completed-calibration-scene-audits.json').read_text())
    if ledger['status'] != 'completed_three_folds_pending_CP2a':
        raise ValueError('The corrected three-fold collection is incomplete')
    if start['attempt'] != 'a3' or start['source_fingerprint'] != ledger['source_fingerprint']:
        raise ValueError('The calibration ledger is not bound to the corrected collection')
    if source_fingerprint() != ledger['source_fingerprint']:
        raise ValueError('Runtime inputs changed after collection')
    if audit['complete_episodes'] != 360 or not audit['frozen_runtime_unchanged']:
        raise ValueError('The current 360-episode native audit is required')
    jobs = ledger['jobs']
    runs = {str(Path(j['run_directory']).resolve()) for j in jobs}
    audited = {str(Path(j['run_directory']).resolve()) for j in audit['completed_scenes']}
    if len(jobs) != 12 or runs != audited or len(audited) != 12:
        raise ValueError('Calibration audit and collection run sets differ')
    for j in jobs:
        Path(j['run_directory']).resolve().relative_to((root / 'runs').resolve())
        if j['status'] != 'completed' or j['returncode'] != 0:
            raise ValueError('Every native calibration job must complete')
    if (root / 'meta/calibration.json').exists():
        raise ValueError('Do not fit after parameters have been frozen')
    paths = ['meta/calibration_runs.json', 'meta/calibration-start-a3.json',
             'meta/completed-calibration-scene-audits.json', 'meta/folds.json',
             'meta/CP1_acceptance.json', 'code/fit_calibration_reviewed.py', 'code/candidate_identities.py']
    paths += [f'logs/calibration_fold{f}_{kind}.jsonl'
              for f in range(3) for kind in ('events', 'snapshots')]
    return {'collection_attempt': 'a3', 'collection_source_commit': start['source_commit'],
            'run_directories': sorted(runs),
            'input_sha256': {p: digest(root / p) for p in paths}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    try:
        before = input_binding()
    except (ValueError, FileNotFoundError, KeyError) as error:
        if args.check:
            print(json.dumps({'ready': False, 'reason': str(error), 'fits_parameters': False}))
            return
        raise
    if args.check:
        print(json.dumps({'ready': True, 'collection_attempt': 'a3', 'fits_parameters': False}))
        return
    if (driver.ROOT / 'meta/calibration-fit-binding.json').exists():
        raise ValueError('Inspect the existing fitted output before fitting again')
    install_analysis_retry()
    driver.main()
    output = driver.ROOT / 'meta/calibration_per_fold.json'
    value = json.loads(output.read_text())
    value['checked_candidate_identity'] = {'definition': IDENTITY_DEFINITION, 'source_sha256': IDENTITY_SHA256,
        'per_fold': {str(f): canonicalize(driver.read(driver.ROOT / f'logs/calibration_fold{f}_events.jsonl'))[2]
                     for f in range(3)}}
    value['numerical_optimizer_retry'] = {
        'trigger': 'original optimizer returned a failure status',
        'objective': 'same constrained positive-slope logistic maximum likelihood',
        'retry': 'SLSQP, analytic gradient, constrained KKT residual <=1e-6',
        'retry_counts_by_label': dict(RETRIES),
        'runtime_source_modified': False}
    output.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    if input_binding() != before:
        raise ValueError('Calibration evidence changed during fitting')
    outputs = ('meta/calibration_per_fold.json', 'meta/calibration.proposed.json',
               'figures/calibration_evidence.png', 'figures/calibration_reliability.png')
    binding = dict(before, fitted_at_utc=datetime.now(timezone.utc).isoformat(),
                   output_sha256={p: digest(driver.ROOT / p) for p in outputs},
                   optimizer_objective_unchanged=True, runtime_source_modified=False)
    (driver.ROOT / 'meta/calibration-fit-binding.json').write_text(
        json.dumps(binding, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
