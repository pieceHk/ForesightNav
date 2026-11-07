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
    print(json.dumps({'episodes': len(counts), 'calls_per_episode': counts,
                       'all_decisions_identical': True}, indent=2))


if __name__ == '__main__':
    main()
