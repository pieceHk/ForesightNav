"""Read GOAT evaluation state and reset measurements at task boundaries."""

# Portions adapted from LagMemo, Copyright (c) Meta Platforms, Inc. and affiliates.
# Upstream MIT terms and provenance are retained in THIRD_PARTY_NOTICES.md.

import numpy as np
from copy import copy
from types import SimpleNamespace


class GoatEvaluator:
    """Own access to success labels, shortest paths and evaluation maps."""

    def __init__(self, habitat_env):
        self.habitat_env = habitat_env

    def get_episode_metrics(self):
        return self.habitat_env.get_metrics()

    def label_candidate(self, position, task_idx):
        """Apply the pinned evaluator at a virtual candidate pose for diagnostics.

        Copies of both measures use a read-only simulator proxy. Neither the
        physical agent, real measure state nor episode goal order is modified.
        """
        original_sim = self.habitat_env.sim

        class VirtualPosition:
            def get_agent_state(self, *args, **kwargs):
                return SimpleNamespace(position=np.asarray(position, dtype=np.float32))

            def __getattr__(self, name):
                return getattr(original_sim, name)

        measures = self.habitat_env.task.measurements.measures
        distance = copy(measures['goat_distance_to_sub-goal'])
        distance._sim = VirtualPosition()
        distance._previous_position = None
        distance.force_metric_update = True
        episode = self.habitat_env.current_episode
        distance.update_goal_viewpoints(episode, current_goal_idx=int(task_idx))
        task = SimpleNamespace(update_goal=False, is_stop_called=False,
                               measurements=SimpleNamespace(measures={'goat_distance_to_sub-goal': distance}))
        distance.update_metric(episode=episode, task=task)
        success = copy(measures['goat_sub-task_success'])
        success.update_metric(episode=episode, task=task)
        value = float(distance.get_metric())
        return {'y': int(bool(success.get_metric())),
                'distance_to_goal_m': value if np.isfinite(value) else None,
                'criterion': 'pinned_GOAT_measures_at_candidate_position'}

    def top_down_map(self):
        return self.get_episode_metrics()["goat_top_down_map"]

    def get_subtask_path_lengths(self):
        """Read GOAT SPL path state before ``reset_subtask`` clears it."""
        measure = self.habitat_env.task.measurements.measures.get("goat_sub-task_spl")
        if measure is None:
            raise RuntimeError("goat_sub-task_spl measure is unavailable")
        actual = getattr(measure, "_agent_episode_distance", None)
        shortest = getattr(measure, "_start_end_episode_distance", None)
        values = {
            "actual_path_length": float(actual) if actual is not None else np.nan,
            "shortest_path_length": float(shortest) if shortest is not None else np.nan,
        }
        actual_value = values["actual_path_length"]
        if not np.isfinite(actual_value) or actual_value < 0.0:
            raise RuntimeError(f"invalid actual_path_length: {actual_value!r}")
        shortest_value = values["shortest_path_length"]
        if np.isnan(shortest_value) or np.isneginf(shortest_value) or shortest_value < 0.0:
            raise RuntimeError(f"invalid shortest_path_length: {shortest_value!r}")
        if np.isposinf(shortest_value):
            success_measure = self.habitat_env.task.measurements.measures.get(
                "goat_sub-task_success"
            )
            if success_measure is None:
                raise RuntimeError("goat_sub-task_success measure is unavailable")
            success = float(success_measure.get_metric())
            if not np.isfinite(success) or success != 0.0:
                raise RuntimeError(
                    "shortest_path_length=+inf is only valid for a failed subtask; "
                    f"success={success!r}"
                )
        return values

    def reset_subtask(self):
        for measure in self.habitat_env._task.measurements.measures.values():
            if measure.uuid == "goat_distance_to_sub-goal":
                measure.update_metric(
                    episode=self.habitat_env.current_episode, task=self.habitat_env.task
                )
        for measure in self.habitat_env._task.measurements.measures.values():
            if measure.uuid in {"goat_top_down_map", "goat_sub-task_spl"}:
                measure.reset_metric(
                    episode=self.habitat_env.current_episode, task=self.habitat_env.task
                )
