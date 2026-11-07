"""Replay each pre-change planner call with the modified, disabled-BLP code."""
import gzip
import json
from pathlib import Path
import pickle

from dataclasses import asdict
from pbsp.navigation.planning import compute_candidate_plan


def stable(result):
    value = asdict(result)
    value.pop('timing_ms')
    value['suffix_plan']['compute_time_ms'] = 0.0
    return value


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1] / 'results/exp4'
    counts = {}
    for path in sorted((root / 'code/regression_inputs').glob('*.pkl.gz')):
        count = 0
        with gzip.open(path, 'rb') as stream:
            while True:
                try:
                    kwargs, distances, expected = pickle.load(stream)
                except EOFError:
                    break
                kwargs['geodesic'] = lambda a, b: distances[(tuple(map(float, a)), tuple(map(float, b)))]
                kwargs['clock'] = lambda: 0.0
                kwargs['started_at'] = 0.0
                actual = compute_candidate_plan(**kwargs)
                if stable(actual) != stable(expected):
                    raise AssertionError(f'Planner regression: episode={path.name} call={count}')
                count += 1
        counts[path.name] = count
    source = json.loads((root / 'meta/regression_source.json').read_text())
    run = Path(source['run_directory'])
    status = json.loads((run / 'status.json').read_text())
    plan = json.loads((run / 'launch.json').read_text())
    expected = {r['runtime_episode_id'] + '.pkl.gz' for r in plan['identities']}
    complete = status['status'] == 'completed' and set(counts) == expected and all(counts.values())
    if complete:
        summary = json.loads((run / 'mon_metrics.json').read_text())
        complete = summary['counts']['completed_episodes'] == len(expected)
    print(json.dumps({'episodes': len(counts), 'calls_per_episode': counts,
                      'all_decisions_identical': True, 'coverage_complete': bool(complete),
                      'baseline_run_directory': str(run)}, indent=2))
    if args.require_complete and not complete:
        raise RuntimeError('Baseline episodes are not all complete; only a prefix was compared')


if __name__ == '__main__':
    main()
