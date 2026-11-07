"""Generate CP2a statistics and a non-frozen parameter proposal."""
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from experiments.blp_calibration import bootstrap, diagnose, decision_flip_rate, fit

ROOT = Path(__file__).resolve().parents[1] / 'results/exp4'


def read(path):
    return [json.loads(row) for row in path.read_text().splitlines() if row]


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def figures(events, diagnostic):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    checked = [r for r in events if r['event'] == 'verification_final']
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), constrained_layout=True)
    for value, color in ((0, 'tab:red'), (1, 'tab:blue')):
        axes[0].hist([r['E'] for r in checked if r['v'] == value], bins=20,
                     alpha=.5, color=color, label=f'Confirmed={value}')
    axes[0].set(xlabel='Semantic score E', ylabel='Verification events')
    axes[0].legend()
    axes[1].hist([r['N_views'] for r in checked], bins=np.arange(14) - .5)
    axes[1].set(xlabel='Distinct visible bearing bins', ylabel='Verification events', xticks=range(13))
    fig.savefig(ROOT / 'figures/calibration_evidence.png', dpi=180)
    plt.close(fig)
    rows = [b for b in diagnostic['reliability_bins'] if b['n']]
    fig, ax = plt.subplots(figsize=(4.5, 4.2), constrained_layout=True)
    ax.plot([0, 1], [0, 1], '--', color='gray')
    ax.plot([b['predicted_q'] for b in rows], [b['gt_valid_fraction'] for b in rows], 'o-')
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel='Self-label q prediction',
           ylabel='Candidate validity under native GT criterion', title=f'ECE = {diagnostic["ece_vs_gt"]:.3f}')
    fig.savefig(ROOT / 'figures/calibration_reliability.png', dpi=180)
    plt.close(fig)


def main():
    ledger = json.loads((ROOT / 'meta/calibration_runs.json').read_text())
    if ledger['status'] != 'completed_three_folds_pending_CP2a':
        raise RuntimeError('The first three complete, audited folds are required')
    groups, snapshots = {}, []
    for fold in range(3):
        groups[fold] = read(ROOT / f'logs/calibration_fold{fold}_events.jsonl')
        snapshots.extend(read(ROOT / f'logs/calibration_fold{fold}_snapshots.jsonl'))
    all_events = [r for group in groups.values() for r in group]
    pool, pool_gt = fit(all_events), fit(all_events, label='y')
    per_fold = {}
    for fold, events in groups.items():
        theta, theta_gt = fit(events), fit(events, label='y')
        # Bootstrap resamples episodes, including all their verification and
        # exploration records together; GT never enters the self-label fit.
        intervals = bootstrap(events, seed=20261107 + fold)
        per_fold[str(fold)] = {'theta': asdict(theta), 'theta_gt': asdict(theta_gt),
                               'bootstrap': intervals, 'diagnostics': diagnose(events, theta),
                               'flip_vs_pool': decision_flip_rate(snapshots, theta, pool)}
    stats, overlap = {}, {}
    for key in asdict(pool):
        values = [g['theta'][key] for g in per_fold.values()]
        stats[key] = {'mean': float(np.mean(values)), 'std_population': float(np.std(values)),
                      'min': min(values), 'max': max(values), 'range': max(values) - min(values)}
        overlap[key] = max(g['bootstrap']['ci95'][key][0] for g in per_fold.values()) <= min(
                          g['bootstrap']['ci95'][key][1] for g in per_fold.values())
    stable = all(g['flip_vs_pool']['rate'] <= .1 for g in per_fold.values()) and all(overlap.values())
    value = {'label_source': 'system_verification', 'per_fold': per_fold,
             'theta_pool': asdict(pool), 'theta_pool_gt_diagnostic_only': asdict(pool_gt),
             'between_fold_statistics': stats, 'all_fold_ci_overlap': overlap,
             'pool_gt_decision_flip': decision_flip_rate(snapshots, pool_gt, pool),
             'pooled_diagnostics': diagnose(all_events, pool),
             'selection_bias': 'Only checked candidates enter the prior and verifier-labelled fit.',
             'stable_under_protocol_rule': stable,
             'recommended_next_step': ('review_CP2a_then_freeze_pool' if stable
                                       else 'review_CP2a_and_cross_fit_or_collect_remaining_folds')}
    if not stable:
        value['cross_fit_proposals'] = {
            str(fold): asdict(fit([r for other, events in groups.items() if other != fold for r in events]))
            for fold in groups}
    write(ROOT / 'meta/calibration_per_fold.json', value)
    write(ROOT / 'meta/calibration.proposed.json', {
        'frozen': False, 'label_source': 'system_verification',
        'parameters': asdict(pool), 'stable_under_protocol_rule': stable})
    figures(all_events, value['pooled_diagnostics'])
    print(json.dumps(value, indent=2))


if __name__ == '__main__':
    main()
