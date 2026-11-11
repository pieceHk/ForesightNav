"""Export six exact 400-episode pairs from complete CP5 normalized evidence."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics

from evaluate_sequence import GateError
from report_full_evaluation import verify, VARIANTS

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(variants, comparison, evidence):
    identities = json.loads((ROOT / 'meta/3mon-identities.json').read_text())
    order = [(str(row['scene_id']), str(row['runtime_episode_id'])) for row in identities]
    assert len(order) == len(set(order)) == 400
    by_name = {}
    for name in VARIANTS:
        episodes, _, summary = variants[name]
        records = {(str(row['scene_id']), str(row['episode_id'])): row for row in episodes}
        assert len(episodes) == len(records) == 400 and set(records) == set(order)
        for identity, key in zip(identities, order):
            row = records[key]
            assert row['variant'] == name and row['task'] == '3-MON' and len(row['stages']) == 3
            assert row['goals'] == identity['ordered_categories']
            assert row['source_episode_ids'] == identity['source_episode_ids']
            assert row['episode_success'] == bool(row['native_mon_metrics']['sr'])
            assert row['episode_success'] == all(stage['success'] for stage in row['stages'])
            assert math.isclose(row['PR'], row['native_mon_metrics']['psr'], rel_tol=0., abs_tol=1e-12)
        for metric, values in {
            'SR': [float(row['episode_success']) for row in episodes],
            'SPL': [row['native_mon_metrics']['spl'] for row in episodes],
            'PR': [row['PR'] for row in episodes], 'PPL': [row['PPL'] for row in episodes]
        }.items():
            assert all(math.isfinite(value) and 0 <= value <= 1 for value in values)
            assert math.isclose(100 * statistics.mean(values), summary['rates_pct_unrounded'][metric],
                rel_tol=0., abs_tol=1e-10), (name, metric)
        by_name[name] = records
    destination = ROOT / 'summary/paired'
    destination.mkdir(parents=True, exist_ok=True)
    fields = ['scene_id', 'episode_id', 'ordered_categories_json', 'source_episode_ids_json',
        'left_variant', 'right_variant', 'left_success', 'right_success',
        'left_SPL', 'right_SPL', 'left_PR', 'right_PR', 'left_PPL', 'right_PPL',
        'left_steps', 'right_steps', 'left_stage_success_json', 'right_stage_success_json']
    pairs = {}
    for left, right in itertools.combinations(VARIANTS, 2):
        rows, counts = [], Counter()
        for scene, episode in order:
            a, b = by_name[left][scene, episode], by_name[right][scene, episode]
            counts[bool(a['episode_success']), bool(b['episode_success'])] += 1
            row = {'scene_id': scene, 'episode_id': episode,
                'ordered_categories_json': json.dumps(a['goals'], separators=(',', ':')),
                'source_episode_ids_json': json.dumps(a['source_episode_ids'], separators=(',', ':')),
                'left_variant': left, 'right_variant': right}
            for side, record in (('left', a), ('right', b)):
                row.update({side + '_success': int(record['episode_success']),
                    side + '_SPL': record['native_mon_metrics']['spl'], side + '_PR': record['PR'],
                    side + '_PPL': record['PPL'], side + '_steps': record['total_steps'],
                    side + '_stage_success_json': json.dumps([int(stage['success']) for stage in record['stages']],
                        separators=(',', ':'))})
            rows.append(row)
        name = left + '_vs_' + right
        counted = {'both_success': counts[True, True], 'left_only_success': counts[True, False],
            'right_only_success': counts[False, True], 'both_failure': counts[False, False], 'n_paired': 400}
        assert counted == comparison['paired_episode_success'][name]
        path = destination / (name + '.csv')
        with path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        with path.open(newline='') as handle:
            restored = list(csv.DictReader(handle))
        assert len(restored) == 400
        for expected, actual in zip(rows, restored):
            assert actual == {key: str(value) for key, value in expected.items()}
            for side in ('left', 'right'):
                for metric in ('SPL', 'PR', 'PPL'):
                    key = side + '_' + metric
                    assert float(actual[key]) == expected[key], 'CSV roundtrip changed an unrounded value'
        pairs[name] = {'path': path.relative_to(ROOT).as_posix(), 'sha256': digest(path),
            'counts': counted, 'csv_rows': 400, 'unrounded_numeric_roundtrip_verified': True}
    assert len(pairs) == 6
    result = {'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'source': 'code/export_paired_episodes.py', 'source_sha256': digest(Path(__file__)),
        'input_sha256': evidence, 'identity_order': 'meta/3mon-identities.json',
        'identity_sha256': digest(ROOT / 'meta/3mon-identities.json'),
        'parameter_sha256': digest(ROOT / 'meta/calibration.json'),
        'n_pairs': 6, 'n_episodes_each_pair': 400, 'pairs': pairs,
        'units': 'SPL, PR and PPL are unrounded episode-level fractions in [0,1]; success is 0 or 1; steps are native action counts.',
        'pair_orientation': 'left/right as named; counts match the six canonical effects.json contingency tables.',
        'scope': 'Complete frozen 3-MON normalized inputs only; no refitting, resampling, exclusions or replacement episodes.',
        'native_states_modified': False, 'launches_workers': False}
    (destination / 'manifest.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    try:
        variants, comparison, evidence = verify('CP5')
    except (GateError, FileNotFoundError, ValueError, KeyError) as error:
        if not args.check:
            raise
        print(json.dumps({'ready': False, 'reason': str(error), 'writes_outputs': False,
            'native_states_modified': False, 'launches_workers': False}))
        return
    if args.check:
        print(json.dumps({'ready': True, 'writes_outputs': False, 'native_states_modified': False,
            'launches_workers': False}))
        return
    print(json.dumps(export(variants, comparison, evidence)))


if __name__ == '__main__':
    main()
