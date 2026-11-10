"""Inspect exact frozen dataset provenance and a height-only availability proxy.

This does not relabel outcomes, remove cases, modify data, or tune parameters.
"""
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ledger = json.loads((ROOT / 'meta/evaluation-smoke-runs.json').read_text())
    launch = json.loads((Path(ledger['jobs'][0]['run_directory']) / 'launch.json').read_text())
    config = launch['configuration']['dataset']
    dataset = Path(launch['resources']['data']) / config['root']
    manifest_path, trace_path = dataset / 'dataset_manifest.json', dataset / 'source_trace.json'
    assert digest(manifest_path) == config['manifest_sha256']
    assert digest(trace_path) == config['source_trace_sha256']
    manifest = json.loads(manifest_path.read_text())
    trace = json.loads(trace_path.read_text())
    sources, historical = set(), Counter()
    for episode in trace:
        for source in episode['sources']:
            key = source['content_file'], source['content_file_sha256'], source['in_file_sequence_index']
            assert key not in sources
            sources.add(key)
            historical[source['outcome']] += 1
    assert len(trace) == 400 and len(sources) == 1200
    scope = {'checked_at_utc': datetime.now(timezone.utc).isoformat(),
             'dataset_name': manifest['dataset_name'], 'manifest_sha256': digest(manifest_path),
             'source_trace_sha256': digest(trace_path), 'scientific_scope': manifest['scientific_scope'],
             'runtime_contract': manifest['runtime_contract'], 'source_count': len(sources),
             'historical_source_outcome_counts': dict(historical),
             'historical_labels_are_current_variant_scores': False,
             'current_3MON_outcomes_used_for_fitting_or_selection': False,
             'data_changed': False, 'parameters_changed': False,
             'source_sha256': digest(Path(__file__))}
    (ROOT / 'meta/dataset-scientific-scope.json').write_text(json.dumps(scope, indent=2) + '\n')
    expected = {(row['scene_id'], str(row['runtime_episode_id'])): row
                for row in json.loads((ROOT / 'meta/3mon-identities.json').read_text())}
    smoke = {(row['scene_id'], str(row['runtime_episode_id']))
             for row in json.loads((ROOT / 'meta/smoke-20-identities.json').read_text())}
    inverse_alias = {value: key for key, value in manifest['runtime_category_aliases'].items()}
    records, observed = [], set()
    for scene in manifest['scene_summary']:
        name = scene['scene_id']
        path = dataset / 'content' / (name + '.json.gz')
        assert digest(path) == scene['output_sha256']
        value = json.loads(gzip.decompress(path.read_bytes()))
        heights = {}
        for category_key, objects in value['goals'].items():
            heights[category_key] = sorted({float(point['agent_state']['position'][1])
                                            for goal in objects for point in goal.get('view_points', [])})
        for episode in value['episodes']:
            key = name, str(episode['episode_id'])
            identity = expected[key]
            categories = [task[0] for task in episode['tasks']]
            assert categories == identity['ordered_categories']
            observed.add(key)
            start_y = float(episode['start_position'][1])
            deltas = []
            for category in categories:
                raw = inverse_alias.get(category, category)
                keys = [item for item in heights if item.endswith('_' + raw)]
                assert len(keys) == 1, (name, category, keys)
                ys = heights[keys[0]]
                deltas.append(min(abs(y - start_y) for y in ys) if ys else None)
            records.append({'scene_id': name, 'episode_id': key[1], 'in_fixed_smoke': key in smoke,
                            'start_height_m': start_y, 'categories': categories,
                            'min_abs_viewpoint_height_delta_m': deltas})
    assert observed == set(expected) and len(records) == 400
    def summarize(selected):
        result = {'episodes': len(selected), 'targets': 3 * len(selected),
                  'targets_without_recorded_viewpoints': sum(delta is None for row in selected for delta in row['min_abs_viewpoint_height_delta_m'])}
        for tolerance in (.5, 1., 1.5):
            result['all_targets_have_height_compatible_viewpoint_' + str(tolerance) + 'm'] = sum(
                all(delta is not None and delta <= tolerance for delta in row['min_abs_viewpoint_height_delta_m'])
                for row in selected)
        return result
    height = {'checked_at_utc': datetime.now(timezone.utc).isoformat(),
              'definition': 'minimum vertical separation from initial agent height to any recorded category viewpoint',
              'limitation': 'height-only proxy; does not prove same-floor assignment, navigability, or cause of an outcome',
              'data_manifest_sha256': digest(manifest_path), 'source_sha256': digest(Path(__file__)),
              'full400': summarize(records), 'smoke20': summarize([row for row in records if row['in_fixed_smoke']]),
              'records': records, 'data_or_parameters_changed': False}
    destination = ROOT / 'diagnostics/CP3/dataset-height-availability.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(height, indent=2) + '\n')
    print(json.dumps({'scope': scope, 'height_full400': height['full400'], 'height_smoke20': height['smoke20']}))


if __name__ == '__main__':
    main()
