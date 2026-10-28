"""Recompute native ObjectNav results from identities, timings and trajectories."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from experiments.metrics import summarize_frame_times
from experiments.objectnav import EPISODE_SCHEMA, ObjectNavMetricProtocol, _number
from pbsp.adapters.objectnav import policy_category
from pbsp.runtime.plan import load_plan


def _read(path):
    def reject(value):
        raise ValueError('Non-finite JSON value: ' + value)
    return json.loads(Path(path).read_text(), parse_constant=reject)


def _pose(value):
    if (not isinstance(value, list) or len(value) != 7
            or any(type(item) not in (int, float) or not math.isfinite(item) for item in value)):
        raise ValueError('Invalid native trajectory pose')
    return value[4:]


def _trajectory(path, record, source):
    metadata = _read(path.parent/'identity.json')
    identity, actions = metadata['identity'], metadata['action_space']
    if (identity['runtime_episode_id'] != record['runtime_episode_id']
            or identity['source_episode_ids'] != [source['source_episode_id']]
            or identity['ordered_categories'] != [policy_category(source['category'])]):
        raise ValueError('Native trajectory identity differs from the source')
    count, travelled, previous, last = 0, 0., None, None
    for count, line in enumerate(path.read_text().splitlines(), 1):
        row = json.loads(line)
        if (row['schema_version'] != 'pbsp_trajectory_v1' or row['frame_index'] != count
                or row['task_step'] != count or row['subtask_index'] != 0
                or type(row['action_id']) is not int
                or row['episode_id'] != record['runtime_episode_id']
                or row['scene_id'] != Path(source['scene_id']).name.split('.')[0]
                or type(row['is_stop']) is not bool
                or row['action_name'] not in actions or row['action_id'] != actions[row['action_name']]
                or row['is_stop'] != (row['action_id'] == actions['STOP'])
                or (last is not None and last['is_stop'])):
            raise ValueError('Native trajectory action coverage differs')
        before, after = _pose(row['world_pose']), _pose(row['world_pose_after'])
        expected = source['start_position'] if previous is None else previous
        if any(abs(a-b) > 1e-4 for a, b in zip(before, expected)):
            raise ValueError('Native trajectory position is discontinuous or has the wrong start')
        travelled += math.sqrt(sum((a-b)**2 for a, b in zip(before, after)))
        previous, last = after, row
    if (count != record['steps'] or last is None
            or last['is_stop'] != record['native_stop_called']
            or (last['stop_reason'] if last['is_stop'] else 'episode_limit') != record['stop_reason']):
        raise ValueError('Native trajectory termination differs from the recorded outcome')
    if not math.isclose(travelled, record['actual_path_length_m'], rel_tol=1e-6, abs_tol=1e-4):
        raise ValueError('Native SPL path length differs from the complete trajectory')


def aggregate_run(directory, *, require_complete=True):
    directory = Path(directory)
    plan = load_plan(directory/'launch.json')
    config = plan.experiment.dataset
    if getattr(config, 'format', None) != 'objectnav':
        raise ValueError('Expected a native ObjectNav run')
    raw = (directory/'dataset_manifest.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != config.manifest_sha256:
        raise ValueError('Saved native ObjectNav dataset manifest differs')
    manifest = json.loads(raw)
    source = {(Path(row['scene_id']).name.split('.')[0], str(row['content_index'])): row
              for row in manifest['entries']}
    selected = {(row.scene_id, row.runtime_episode_id): row for row in plan.identities}
    grouped, records, seen = {}, {}, set()
    for path in sorted(directory.glob('results_*/*/timing_log.csv')):
        with path.open(newline='') as stream:
            for row in csv.DictReader(stream):
                key = row['scene_id'], row['episode_id']
                times = grouped.setdefault(key, [])
                if key not in selected or int(row['subtask_index']) != 0 or int(row['frame_index']) != len(times)+1:
                    raise ValueError('Native timing coverage differs from the selected identities')
                times.append(_number(row['dt_total_ms'], 'frame time'))
    for path in sorted(directory.glob('results_*/*/per_episode_metrics.json')):
        for name, episode in _read(path).items():
            if len(episode['metrics']) != 1:
                raise ValueError('Native ObjectNav episode contains multiple targets')
            record = episode['metrics'][0]
            scene = Path(record['source_scene_id']).name.split('.')[0]
            key = scene, record['runtime_episode_id']
            if key in seen or key not in selected or name != scene+'_'+key[1]:
                raise ValueError('Unknown or duplicate native ObjectNav result')
            seen.add(key)
            original = source[key]
            if (record['schema'] != EPISODE_SCHEMA
                    or record['source_episode_id'] != original['source_episode_id']
                    or record['raw_episode_id'] != original['raw_episode_id']
                    or record['source_scene_id'] != original['scene_id']
                    or record['category'] != original['category']
                    or tuple(selected[key].ordered_categories) != (policy_category(record['category']),)
                    or selected[key].source_episode_ids != (record['source_episode_id'],)):
                raise ValueError('Native result source identity differs')
            success, spl = _number(record['success'], 'success'), _number(record['spl'], 'spl')
            if success not in (0., 1.) or spl > 1. or (not success and spl != 0.):
                raise ValueError('Invalid native success or SPL')
            if (type(record['native_stop_called']) is not bool
                    or (success and not record['native_stop_called'])
                    or record['stop_reason'] not in {'verified_target', 'max_steps', 'episode_limit'}):
                raise ValueError('Invalid native STOP outcome')
            actual = _number(record['actual_path_length_m'], 'path length')
            shortest = record['shortest_path_length_m']
            if shortest is not None:
                shortest = _number(shortest, 'shortest path')
                if not max(actual, shortest) or not math.isclose(spl, success*shortest/max(actual, shortest),
                                                               rel_tol=1e-6, abs_tol=1e-8):
                    raise ValueError('Native SPL does not match its distance state')
            elif record['shortest_path_status'] != 'unreachable' or success:
                raise ValueError('Invalid unreachable native path')
            native_spl_status = record.get('native_spl_status')
            if native_spl_status == 'finite':
                if _number(record.get('native_spl_value'), 'native SPL') != spl:
                    raise ValueError('Recorded native SPL differs from the reported SPL')
            elif native_spl_status == 'nan_unreachable_failure':
                if (record.get('native_spl_value') is not None or success != 0. or spl != 0.
                        or shortest is not None or record['shortest_path_status'] != 'unreachable'):
                    raise ValueError('Invalid native SPL unreachable failure boundary')
            else:
                raise ValueError('Unknown native SPL status')
            timing = summarize_frame_times(grouped.get(key, []))
            if (timing != record['frame_timing'] or timing['frame_count'] != record['steps']
                    or episode['total_num_steps'] != record['steps']):
                raise ValueError('Native timing and result step counts differ')
            from experiments.trajectory import component
            trajectory = path.parent/'episodes'/component(scene)/component(key[1])/'trajectory.jsonl'
            _trajectory(trajectory, record, original)
            records[name] = episode
    if set(grouped) != seen or (require_complete and seen != set(selected)):
        raise ValueError('Native ObjectNav episode coverage is incomplete')
    times = [value for key in sorted(seen) for value in grouped[key]]
    result = ObjectNavMetricProtocol.summarize_records(records, frame_times_ms=times)
    result['coverage'] = dict(expected_episodes=len(selected), completed_episodes=len(seen),
                              complete=seen == set(selected), dataset_episodes=config.expected_episodes,
                              full_dataset_evaluation=len(selected) == config.expected_episodes,
                              dataset_manifest_sha256=config.manifest_sha256)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path)
    args = parser.parse_args(argv)
    results = {}
    for directory in args.runs:
        if _read(directory/'status.json')['status'] != 'completed':
            raise ValueError('Native ObjectNav run has not completed: ' + str(directory))
        result = aggregate_run(directory)
        if result != _read(directory/'objectnav_metrics.json'):
            raise ValueError('Saved native ObjectNav summary differs from raw records')
        results[str(directory)] = result
    print(json.dumps(results, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
