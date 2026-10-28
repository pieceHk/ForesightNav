"""Read native Habitat ObjectNav outcomes without changing its measurements."""

import math
from pathlib import Path
from statistics import median

from experiments.metrics import summarize_frame_times

EPISODE_SCHEMA = 'pbsp_native_objectnav_episode_v2'


def _number(value, name, *, allow_infinite=False):
    if isinstance(value, bool):
        raise ValueError(name + ' must be numeric')
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(name + ' must be numeric') from error
    if math.isnan(number) or number < 0 or (math.isinf(number) and not allow_infinite):
        raise ValueError(name + ' must be nonnegative and finite')
    return number


def _native_spl(value, *, success, shortest):
    """Retain Habitat's undefined 0 * (inf / inf) as an unreachable failure."""
    if success == 0. and math.isinf(shortest) and not isinstance(value, bool):
        try:
            if math.isnan(float(value)):
                return 0., 'nan_unreachable_failure'
        except (TypeError, ValueError):
            pass
    return _number(value, 'spl'), 'finite'


def validate_episode_identity(episode, identity):
    """Check the loaded native episode against an audited source record."""
    if (identity['source_episode_id'] != str(identity['content_index'])
            or str(episode.episode_id) != str(identity['content_index'])
            or Path(episode.scene_id).name != Path(identity['scene_id']).name
            or episode.object_category != identity['category']):
        raise ValueError('Loaded ObjectNav episode differs from the source identity')
    for name, length in (('start_position', 3), ('start_rotation', 4)):
        actual, expected = getattr(episode, name), identity[name]
        if (len(actual) != length or len(expected) != length
                or any(not math.isfinite(float(a)) or not math.isfinite(float(b))
                       or not math.isclose(float(a), float(b), rel_tol=0., abs_tol=1e-6)
                       for a, b in zip(actual, expected))):
            raise ValueError('Loaded ObjectNav ' + name + ' differs from the source')


class ObjectNavEvaluator:
    """Use Habitat's success, distance and SPL; never reset them mid-episode."""

    def __init__(self, habitat_env):
        self.habitat_env = habitat_env

    def complete_episode(self, identity, *, steps, stop_reason):
        env = self.habitat_env
        if not env.episode_over:
            raise ValueError('Native ObjectNav episode has not ended')
        if type(steps) is not int or steps < 1:
            raise ValueError('ObjectNav step count must be a positive integer')
        validate_episode_identity(env.current_episode, identity)
        raw = env.get_metrics()
        success = _number(raw['success'], 'success')
        if success not in (0., 1.):
            raise ValueError('Invalid native ObjectNav success')
        stopped = bool(getattr(env.task, 'is_stop_called', False))
        if success and not stopped:
            raise ValueError('Native ObjectNav success requires STOP')
        distance = _number(raw['distance_to_goal'], 'distance_to_goal',
                           allow_infinite=not success)
        measure = env.task.measurements.measures.get('spl')
        if measure is None:
            raise ValueError('Native SPL measurement is missing')
        actual = _number(getattr(measure, '_agent_episode_distance', None), 'actual_path_length')
        shortest = _number(getattr(measure, '_start_end_episode_distance', None),
                           'shortest_path_length', allow_infinite=not success)
        spl, native_spl_status = _native_spl(raw['spl'], success=success, shortest=shortest)
        if not 0. <= spl <= 1.:
            raise ValueError('Invalid native ObjectNav SPL')
        if not success and spl != 0.:
            raise ValueError('Native ObjectNav failure has nonzero SPL')
        if math.isfinite(shortest):
            denominator = max(actual, shortest)
            expected_spl = success * shortest / denominator if denominator > 0 else None
            if expected_spl is None or not math.isclose(spl, expected_spl, rel_tol=1e-6, abs_tol=1e-8):
                raise ValueError('Native SPL differs from its recorded distance state')
        return {
            'schema': EPISODE_SCHEMA,
            'source_scene_id': identity['scene_id'],
            'source_episode_id': identity['source_episode_id'],
            'raw_episode_id': identity['raw_episode_id'],
            'runtime_episode_id': str(env.current_episode.episode_id),
            'content_index': identity['content_index'],
            'category': identity['category'], 'success': int(success), 'spl': spl,
            'native_spl_value': spl if native_spl_status == 'finite' else None,
            'native_spl_status': native_spl_status,
            'distance_to_goal_m': distance if math.isfinite(distance) else None,
            'distance_to_goal_status': 'finite' if math.isfinite(distance) else 'unreachable',
            'actual_path_length_m': actual,
            'shortest_path_length_m': shortest if math.isfinite(shortest) else None,
            'shortest_path_status': 'finite' if math.isfinite(shortest) else 'unreachable',
            'native_stop_called': stopped, 'steps': steps, 'stop_reason': stop_reason,
        }


class ObjectNavMetricProtocol:
    """Capture native outcomes through the shared runner's measurement port."""

    metric_keys = ('distance_to_goal_m', 'success', 'spl')
    distance_key = 'distance_to_goal_m'
    capture_on_episode_end = True

    def __init__(self, source_identities):
        self.source_identities = source_identities

    def capture(self, evaluator, *, steps, stop_reason, frame_times_ms):
        episode = evaluator.habitat_env.current_episode
        key = (Path(episode.scene_id).name.split('.')[0], str(episode.episode_id))
        identity = self.source_identities.get(key)
        if identity is None:
            raise ValueError('Missing source identity for native ObjectNav evaluation')
        record = evaluator.complete_episode(identity, steps=steps, stop_reason=stop_reason)
        record['frame_timing'] = summarize_frame_times(frame_times_ms)
        if record['frame_timing']['frame_count'] != steps:
            raise ValueError('Native ObjectNav timing coverage differs from the executed steps')
        return record

    @staticmethod
    def metric_summary(records, name):
        values, unreachable = [], 0
        for record in records:
            value = record[name]
            if value is None and name == 'distance_to_goal_m' and record['distance_to_goal_status'] == 'unreachable':
                unreachable += 1
            else:
                values.append(_number(value, name))
        return dict(mean=round(math.fsum(values) / len(values), 4) if values else None,
                    median=round(median(values), 4) if values else None,
                    finite_count=len(values), unreachable_count=unreachable)

    @staticmethod
    def summarize_episode(records, *, total_steps, stop_reasons, frame_times_ms, categories):
        if (len(records) != 1 or total_steps != records[0]['steps']
                or stop_reasons != [records[0]['stop_reason']] or len(categories) != 1):
            raise ValueError('Native ObjectNav requires one complete single-target episode')
        return {'objectnav_metrics': {'sr': records[0]['success'], 'spl': records[0]['spl']},
                'frame_timing': summarize_frame_times(frame_times_ms)}

    @staticmethod
    def summarize_records(records, *, frame_times_ms=None):
        tasks = []
        for episode in records.values():
            if len(episode['metrics']) != 1:
                raise ValueError('Native ObjectNav result contains multiple targets')
            tasks.extend(episode['metrics'])
        if not tasks:
            raise ValueError('No complete native ObjectNav episodes')
        identities = {(row['source_scene_id'], row['source_episode_id']) for row in tasks}
        if len(identities) != len(tasks):
            raise ValueError('Duplicate native ObjectNav source episode')
        if frame_times_ms is None:
            count = sum(row['frame_timing']['frame_count'] for row in tasks)
            time_sum = math.fsum(row['frame_timing']['frame_time_sum_ms'] for row in tasks)
            timing = dict(frame_count=count, frame_time_sum_ms=time_sum,
                          mean_frame_time_ms=time_sum / count if count else None,
                          median_frame_time_ms=None, p95_frame_time_ms=None,
                          fps=count * 1000 / time_sum if time_sum else None)
        else:
            timing = summarize_frame_times(frame_times_ms)
            if timing['frame_count'] != sum(row['steps'] for row in tasks):
                raise ValueError('Native ObjectNav run timing coverage differs')
        return {'schema_version': 'pbsp_native_objectnav_metrics_v2',
                'counts': {'completed_episodes': len(tasks), 'completed_subtasks': len(tasks),
                           'successful_episodes': sum(row['success'] for row in tasks),
                           'unreachable_start_episodes': sum(row['shortest_path_status'] == 'unreachable' for row in tasks),
                           'undefined_native_spl_episodes': sum(row['native_spl_status'] == 'nan_unreachable_failure' for row in tasks)},
                'objectnav_metrics': {'sr': math.fsum(row['success'] for row in tasks) / len(tasks),
                                     'spl': math.fsum(row['spl'] for row in tasks) / len(tasks)},
                'frame_timing': timing}
