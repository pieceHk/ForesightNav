"""Read-only native navmesh availability diagnostic for the frozen dataset.

Distances are from the initial episode pose, not from later task transitions.
Height filtering is a proxy, not a floor, policy, success, or feasibility test.
"""
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path

from native_candidate_gt import NativeCandidateGT, SOURCE_SHA256

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ledger = json.loads((ROOT / 'meta/evaluation-smoke-runs.json').read_text())
    gt = NativeCandidateGT(ledger['jobs'][0]['run_directory'])
    identities = json.loads((ROOT / 'meta/3mon-identities.json').read_text())
    expected = {(row['scene_id'], str(row['runtime_episode_id'])) for row in identities}
    smoke = {(row['scene_id'], str(row['runtime_episode_id']))
             for row in json.loads((ROOT / 'meta/smoke-20-identities.json').read_text())}
    inverse_alias = {value: key for key, value in gt.manifest['runtime_category_aliases'].items()}
    records = []
    for scene in gt.manifest['scene_summary']:
        name = scene['scene_id']
        bundle = gt.scene(name)
        value = json.loads(gzip.decompress((gt.dataset_root / 'content' / (name + '.json.gz')).read_bytes()))
        points = {key: [point['agent_state']['position'] for goal in objects
                        for point in goal.get('view_points', [])]
                  for key, objects in value['goals'].items()}
        for raw_episode in value['episodes']:
            episode = bundle['episodes'][str(raw_episode['episode_id'])]
            assert episode.start_position == raw_episode['start_position']
            tasks = []
            for task in raw_episode['tasks']:
                category = task[0]
                raw_category = inverse_alias.get(category, category)
                keys = [key for key in points if key.endswith('_' + raw_category)]
                assert len(keys) == 1
                all_points = points[keys[0]]
                compatible = [point for point in all_points
                              if abs(point[1] - episode.start_position[1]) <= .5]
                distances = []
                for endpoints in (all_points, compatible):
                    distance = bundle['env'].sim.geodesic_distance(episode.start_position, endpoints) if endpoints else math.inf
                    distances.append(float(distance) if math.isfinite(distance) else None)
                tasks.append({'category': category, 'recorded_viewpoints': len(all_points),
                              'viewpoints_within_0.5m_initial_height': len(compatible),
                              'initial_to_any_viewpoint_geodesic_m': distances[0],
                              'initial_to_height_compatible_viewpoint_geodesic_m': distances[1]})
            key = name, str(episode.episode_id)
            records.append({'scene_id': name, 'episode_id': key[1], 'in_fixed_smoke': key in smoke,
                            'initial_position': episode.start_position, 'tasks': tasks})
        print(json.dumps({'audited_scene': name, 'episodes': len(bundle['episodes'])}), flush=True)
    assert len(records) == 400 and {(row['scene_id'], row['episode_id']) for row in records} == expected
    def summarize(selected):
        return {'episodes': len(selected), 'targets': 3 * len(selected),
                'targets_with_finite_initial_geodesic': sum(task['initial_to_any_viewpoint_geodesic_m'] is not None for row in selected for task in row['tasks']),
                'episodes_all_targets_have_finite_initial_geodesic': sum(all(task['initial_to_any_viewpoint_geodesic_m'] is not None for task in row['tasks']) for row in selected),
                'targets_with_finite_geodesic_to_height_compatible_viewpoint': sum(task['initial_to_height_compatible_viewpoint_geodesic_m'] is not None for row in selected for task in row['tasks']),
                'episodes_all_targets_have_finite_geodesic_to_height_compatible_viewpoint': sum(all(task['initial_to_height_compatible_viewpoint_geodesic_m'] is not None for task in row['tasks']) for row in selected)}
    result = {'checked_at_utc': datetime.now(timezone.utc).isoformat(),
              'method': 'pinned native HabitatSim geodesic wrapper and CPU PathFinder; no renderer, policy, or model',
              'limitation': 'Initial-pose category-viewpoint availability only. Height filtering does not establish floors, policy reachability, success, or an outcome cause; later stages start at actual transition poses.',
              'manifest_sha256': gt.plan['configuration']['dataset']['manifest_sha256'],
              'native_gt_source_sha256': SOURCE_SHA256, 'source_sha256': digest(Path(__file__)),
              'frozen_identities_sha256': digest(ROOT / 'meta/3mon-identities.json'),
              'native_inputs_or_data_or_parameters_changed': False, 'uses_GT_for_policy_or_fit': False,
              'full400': summarize(records), 'smoke20': summarize([row for row in records if row['in_fixed_smoke']]),
              'native_source_provenance': gt.provenance(), 'records': records}
    destination = ROOT / 'diagnostics/CP3/dataset-navmesh-availability.json'
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'full400': result['full400'], 'smoke20': result['smoke20'], 'output_sha256': digest(destination)}), flush=True)


if __name__ == '__main__':
    main()
