"""Self-label calibration and GT-only diagnostics for the exp4 protocol."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from pbsp.navigation.blp import BeliefParameters, solve_snapshot


class CalibrationDataError(ValueError):
    pass


def fit(events, label='v'):
    checked = [r for r in events if r['event'] == 'verification_final']
    explorations = [r['path_to_first_candidate_m'] for r in events if r['event'] == 'empty_activation']
    if not checked or not explorations:
        raise CalibrationDataError('Need verification and uncensored empty-activation events')
    e = np.array([r['E'] for r in checked], dtype=float)
    y = np.array([r[label] for r in checked], dtype=float)
    if len(set(y)) != 2:
        raise CalibrationDataError(f'Cannot identify logistic slope with one label class ({label})')
    center, scale = float(e.mean()), max(float(e.std()), 1e-8)
    x = (e - center) / scale
    def objective(theta):
        z = theta[0] * x + theta[1]
        nll = np.logaddexp(0., z).sum() - np.dot(y, z)
        residual = expit(z) - y
        return nll, np.array([np.dot(residual, x), residual.sum()])
    event_rate = float(y.mean())
    candidate_labels = {}
    for row in checked:
        key = (row.get('_resample_instance'), row['scene_id'], row['episode_id'],
               row['k'], row['candidate_id'])
        candidate_labels[key] = max(candidate_labels.get(key, 0), int(row[label]))
    pi = float(np.mean(list(candidate_labels.values())))
    result = minimize(objective, [10 * scale, np.log(event_rate / (1 - event_rate))], jac=True,
                      method='L-BFGS-B', bounds=[(1e-8 * scale, None), (None, None)],
                      options={'ftol': 1e-13, 'gtol': 1e-8, 'maxiter': 2000})
    if not result.success:
        raise CalibrationDataError('Logistic optimizer failed: ' + str(result.message))
    a = float(result.x[0] / scale)
    b0 = float(result.x[1] - a * center)
    if a > 1e6 or abs(b0) > 1e6:
        raise CalibrationDataError('Logistic fit is separated/unbounded; collect more data')
    return BeliefParameters(a=a, b0=b0, pi=pi, c_exp=float(np.mean(explorations)))


def bootstrap(events, seed, count=200):
    groups = {}
    for event in events:
        groups.setdefault((event['scene_id'], event['episode_id']), []).append(event)
    episodes = sorted(groups)
    rng = np.random.RandomState(seed)
    estimates = []
    failures = []
    for iteration in range(count):
        sampled = [dict(r, _resample_instance=instance)
                   for instance, index in enumerate(rng.randint(len(episodes), size=len(episodes)))
                   for r in groups[episodes[index]]]
        try:
            estimates.append(list(asdict(fit(sampled)).values()))
        except CalibrationDataError as exc:
            failures.append({'iteration': iteration, 'error': str(exc)})
    if len(estimates) < 0.9 * count:
        raise CalibrationDataError(f'Only {len(estimates)}/{count} bootstrap fits identifiable')
    bounds = np.percentile(estimates, [2.5, 97.5], axis=0)
    return {'n_resamples': count, 'n_success': len(estimates), 'failed_resamples': failures,
            'ci95': {k: bounds[:, i].tolist() for i, k in enumerate(asdict(BeliefParameters()))}}


def diagnose(events, theta):
    rows = [r for r in events if r['event'] == 'verification_final']
    v, y = [np.array([int(r[k]) for r in rows]) for k in ('v', 'y')]
    tp, tn = int(((v == 1) & (y == 1)).sum()), int(((v == 0) & (y == 0)).sum())
    fp, fn = int(((v == 1) & (y == 0)).sum()), int(((v == 0) & (y == 1)).sum())
    q = expit(theta.a * np.array([r['E'] for r in rows]) + theta.b0)
    bins, ece = [], 0.
    for i in range(10):
        mask = (q >= i / 10) & (q < (i + 1) / 10 if i < 9 else q <= 1.)
        n = int(mask.sum())
        predicted, valid = (float(q[mask].mean()), float(y[mask].mean())) if n else (None, None)
        bins.append({'lower': i / 10, 'upper': (i + 1) / 10, 'n': n,
                     'predicted_q': predicted, 'gt_valid_fraction': valid})
        if n:
            ece += n / len(y) * abs(predicted - valid)
    return {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
            'agreement': (tp + tn) / len(y),
            'false_positive_rate': fp / (fp + tn) if fp + tn else None,
            'false_negative_rate': fn / (fn + tp) if fn + tp else None,
            'reliability_bins': bins, 'ece_vs_gt': ece}


def decision_flip_rate(snapshots, theta, reference):
    if not snapshots:
        raise CalibrationDataError('No replay snapshots')
    flips = total = 0
    for row in snapshots:
        snapshot = row['snapshot']
        if not snapshot['layers'][snapshot['k']]:
            continue
        args = dict(variant='full', active_raw_rank=row['active_raw_rank'])
        a = solve_snapshot(snapshot, theta, **args)['chosen_raw_rank']
        b = solve_snapshot(snapshot, reference, **args)['chosen_raw_rank']
        flips += a != b
        total += 1
    if not total:
        raise CalibrationDataError('No eligible root decisions')
    return {'flips': flips, 'eligible_snapshots': total, 'rate': flips / total}
