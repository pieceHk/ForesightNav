"""Keep task-specific measurements outside the shared navigation loop."""

from experiments.metrics import (
    DISTANCE_TO_GOAL_KEY, SPL_KEY, SUCCESS_KEY, build_subtask_telemetry,
    summarize_episode, summarize_records,
)
from experiments.recorder import finite_metric_summary


class GoatMetricProtocol:
    metric_keys = (DISTANCE_TO_GOAL_KEY, SUCCESS_KEY, SPL_KEY)
    distance_key = DISTANCE_TO_GOAL_KEY
    capture_on_episode_end = False

    def capture(self, evaluator, *, steps, stop_reason, frame_times_ms):
        metric = evaluator.get_episode_metrics()
        metric.pop('goat_top_down_map', None)
        lengths = evaluator.get_subtask_path_lengths()
        metric.update(build_subtask_telemetry(
            metric, actual_path_length=lengths['actual_path_length'],
            shortest_path_length=lengths['shortest_path_length'],
            steps=steps, frame_times_ms=frame_times_ms,
        ))
        evaluator.reset_subtask()
        return metric

    summarize_episode = staticmethod(summarize_episode)
    summarize_records = staticmethod(summarize_records)
    metric_summary = staticmethod(finite_metric_summary)
