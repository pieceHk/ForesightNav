"""Read GOAT evaluation state and reset measurements at task boundaries."""

# Portions adapted from LagMemo, Copyright (c) Meta Platforms, Inc. and affiliates.
# Upstream MIT terms and provenance are retained in THIRD_PARTY_NOTICES.md.

import numpy as np


class GoatEvaluator:
    """Own access to success labels, shortest paths and evaluation maps."""

    def __init__(self, habitat_env):
        self.habitat_env = habitat_env

    def get_episode_metrics(self):
        return self.habitat_env.get_metrics()

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
