"""Exercise the CP2 statistical driver on temporary synthetic data only.

No production events/parameters are read or overwritten. Both the pooled and
cross-fitting recommendation branches must work before real folds finish.
"""
from dataclasses import asdict
import contextlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import experiments.exp4_fit as driver
from experiments.blp_calibration import fit


def write_jsonl(path, rows):
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))


def snapshot():
    points = [0., 1., 2., 3., 4., 5., 6.]
    return {'k': 0, 'layers': [[{'raw_rank': i, 'E': .2 + i * .05, 'N_views': i,
                                 'location': 1 + 3 * k + i} for i in range(3)] for k in range(2)],
            'distances': [[abs(a - b) for b in points] for a in points],
            'nearest_frontier': [None] * 7, 'frontier_available': False,
            'all_raw_ranks': [0, 1, 2]}


def exercise(c_exp_by_fold):
    original_root = driver.ROOT
    with TemporaryDirectory(prefix='exp4_synthetic_fit_') as folder:
        root = Path(folder)
        for name in ('meta', 'logs', 'figures'):
            (root/name).mkdir()
        (root/'meta/calibration_runs.json').write_text(json.dumps({'status': 'completed_three_folds_pending_CP2a'}))
        pooled = []
        for fold, c_exp in enumerate(c_exp_by_fold):
            events = []
            for episode in range(120):
                index = episode % 6
                e, v = [(.2, 0), (.25, 0), (.3, 1), (.32, 0), (.35, 1), (.4, 1)][index]
                y = 1 if index in (1, 4, 5) else 0
                identity = {'scene_id': f'synthetic_{fold}_{episode // 30}', 'episode_id': str(episode % 30)}
                events.extend([dict(identity, event='verification_final', E=e, v=v, y=y,
                                    k=0, candidate_id='0:0', N_views=index, position=[float(index), 0., 0.]),
                               dict(identity, event='empty_activation', path_to_first_candidate_m=c_exp)])
            write_jsonl(root/f'logs/calibration_fold{fold}_events.jsonl', events)
            write_jsonl(root/f'logs/calibration_fold{fold}_snapshots.jsonl',
                        [{'snapshot': snapshot(), 'active_raw_rank': None} for _ in range(3)] +
                        [{'snapshot': {'k': 0, 'layers': [[]], 'distances': [[0.]],
                                       'nearest_frontier': [None], 'frontier_available': False,
                                       'all_raw_ranks': []}, 'active_raw_rank': None}])
            pooled.extend(events)
        driver.ROOT = root
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                driver.main()
        finally:
            driver.ROOT = original_root
        output = json.loads((root/'meta/calibration_per_fold.json').read_text())
        proposed = json.loads((root/'meta/calibration.proposed.json').read_text())
        assert proposed['parameters'] == asdict(fit(pooled, label='v'))
        assert proposed['parameters'] != asdict(fit(pooled, label='y'))
        assert proposed['frozen'] is False
        assert not (root/'meta/calibration.json').exists()
        assert len(output['per_fold']) == 3
        for fold in output['per_fold'].values():
            assert fold['bootstrap']['n_resamples'] == 200
            assert fold['bootstrap']['n_success'] >= 180
            assert all(len(bound) == 2 for bound in fold['bootstrap']['ci95'].values())
            assert fold['flip_vs_pool']['eligible_snapshots'] == 9
        assert len(list((root/'figures').glob('*.png'))) == 2
        if len(set(c_exp_by_fold)) == 1:
            assert output['stable_under_protocol_rule'] is True
            assert 'cross_fit_proposals' not in output
        else:
            assert output['stable_under_protocol_rule'] is False
            assert len(output['cross_fit_proposals']) == 3
            for fold in range(3):
                expected = sum(value for i, value in enumerate(c_exp_by_fold) if i != fold) / 2
                assert output['cross_fit_proposals'][str(fold)]['c_exp'] == expected
        return {'stable': output['stable_under_protocol_rule'],
                'bootstrap_successes': [v['bootstrap']['n_success'] for v in output['per_fold'].values()]}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--retry-optimizer-status', action='store_true')
    args = parser.parse_args()
    if args.retry_optimizer_status:
        from fit_calibration_reviewed import install_analysis_retry, RETRIES
        install_analysis_retry()
    result = {'synthetic_only': True, 'pooled_case': exercise([2., 2., 2.]),
              'crossfit_case': exercise([1., 5., 9.]),
              'GT_never_used_for_proposed_parameters': True,
              'no_frozen_calibration_written': True}
    if args.retry_optimizer_status:
        result['optimizer_status_retries'] = dict(RETRIES)
    print(json.dumps(result, indent=2))
