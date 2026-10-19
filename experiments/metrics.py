"""Publication-grade MON metrics and navigation-efficiency telemetry.

For ``M`` episodes with a fixed ``n`` ordered targets per episode:

``PSR = sum_ij(S_ij) / (M * n)``

``PeSPL = sum_ij(S_ij * D_ij / max(P_ij, D_ij)) / (M * n)``

Both metrics therefore lie in ``[0, 1]``.  Their raw numerators and the common
``M * n`` denominator are retained in every artifact for auditability.
"""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping, Sequence


SUCCESS_KEY = "goat_sub-task_success"
SPL_KEY = "goat_sub-task_spl"
DISTANCE_TO_GOAL_KEY = "goat_distance_to_sub-goal"
DISTANCE_TO_GOAL_STATUS_KEY = "goat_distance_to_sub-goal_status"
PATH_LENGTH_KEY = "goat_sub-task_path_length_m"
SHORTEST_PATH_LENGTH_KEY = "goat_sub-task_shortest_path_length_m"
SHORTEST_PATH_STATUS_KEY = "goat_sub-task_shortest_path_status"
STEPS_KEY = "goat_sub-task_num_steps"
FRAME_COUNT_KEY = "goat_sub-task_frame_count"
FRAME_TIME_SUM_KEY = "goat_sub-task_frame_time_sum_ms"

MON_METRICS_SCHEMA_VERSION = "mon_metrics_v3"
SHORTEST_PATH_FINITE = "finite"
SHORTEST_PATH_UNREACHABLE = "unreachable"


class MonMetricError(ValueError):
    """Raised when MON telemetry cannot produce a valid metric."""


def _reject_json_constant(value: str) -> Any:
    raise MonMetricError(f"non-standard JSON constant {value!r}")


def _finite_number(value: Any, name: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool):
        raise MonMetricError(f"{name} must be numeric, got bool")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise MonMetricError(f"{name} must be numeric, got {value!r}") from exc
    if not math.isfinite(number) or number < minimum:
        raise MonMetricError(
            f"{name} must be finite and >= {minimum}, got {number!r}"
        )
    return number


def _binary_success(value: Any, name: str = SUCCESS_KEY) -> int:
    number = _finite_number(value, name)
    if number not in (0.0, 1.0):
        raise MonMetricError(f"{name} must be 0 or 1, got {number!r}")
    return int(number)


def _nonnegative_int(value: Any, name: str) -> int:
    number = _finite_number(value, name)
    integer = int(number)
    if number != integer:
        raise MonMetricError(f"{name} must be an integer, got {number!r}")
    return integer


def _path_efficiency(actual: float, shortest: float) -> float:
    denominator = max(actual, shortest)
    return 1.0 if denominator == 0.0 else shortest / denominator


def _capture_shortest_path(
    value: Any,
    *,
    success: int,
) -> tuple[float | None, str]:
    """Classify Habitat's raw shortest path without serializing non-finite data."""

    if isinstance(value, bool):
        raise MonMetricError(
            f"{SHORTEST_PATH_LENGTH_KEY} must be numeric, got bool"
        )
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise MonMetricError(
            f"{SHORTEST_PATH_LENGTH_KEY} must be numeric, got {value!r}"
        ) from exc
    if math.isinf(number) and number > 0.0:
        if success != 0:
            raise MonMetricError(
                "a successful subtask cannot have an unreachable shortest path"
            )
        return None, SHORTEST_PATH_UNREACHABLE
    if not math.isfinite(number) or number < 0.0:
        raise MonMetricError(
            f"{SHORTEST_PATH_LENGTH_KEY} must be finite and >= 0, or +inf "
            f"for a failed unreachable subtask; got {number!r}"
        )
    return number, SHORTEST_PATH_FINITE


def _recorded_shortest_path(
    metric: Mapping[str, Any],
    *,
    success: int,
) -> tuple[float | None, str]:
    """Validate strict-JSON shortest-path telemetry, including legacy finite rows."""

    status = metric.get(SHORTEST_PATH_STATUS_KEY, SHORTEST_PATH_FINITE)
    if status == SHORTEST_PATH_FINITE:
        result = (
            _finite_number(
                metric.get(SHORTEST_PATH_LENGTH_KEY),
                SHORTEST_PATH_LENGTH_KEY,
            ),
            SHORTEST_PATH_FINITE,
        )
    elif status != SHORTEST_PATH_UNREACHABLE:
        raise MonMetricError(
            f"invalid {SHORTEST_PATH_STATUS_KEY}: {status!r}"
        )
    else:
        if success != 0:
            raise MonMetricError(
                "a successful subtask cannot be marked shortest-path unreachable"
            )
        if metric.get(SHORTEST_PATH_LENGTH_KEY) is not None:
            raise MonMetricError(
                "an unreachable shortest path must be serialized as null"
            )
        if _finite_number(metric.get(SPL_KEY), SPL_KEY) != 0.0:
            raise MonMetricError(
                "an unreachable failed subtask must contribute zero SPL"
            )
        for key in (
            "goat_sub-task_path_efficiency",
            "goat_sub-task_path_stretch",
            "goat_sub-task_excess_path_length_m",
        ):
            if metric.get(key) is not None:
                raise MonMetricError(
                    f"{key} must be null for an unreachable shortest path"
                )
        result = (None, SHORTEST_PATH_UNREACHABLE)

    if DISTANCE_TO_GOAL_KEY in metric:
        distance_status = metric.get(
            DISTANCE_TO_GOAL_STATUS_KEY, SHORTEST_PATH_FINITE
        )
        distance_value = metric.get(DISTANCE_TO_GOAL_KEY)
        if distance_status == SHORTEST_PATH_FINITE:
            _finite_number(distance_value, DISTANCE_TO_GOAL_KEY)
        elif distance_status == SHORTEST_PATH_UNREACHABLE:
            if result[1] != SHORTEST_PATH_UNREACHABLE or distance_value is not None:
                raise MonMetricError(
                    "a null unreachable goal distance requires an unreachable "
                    "shortest path"
                )
        else:
            raise MonMetricError(
                f"invalid {DISTANCE_TO_GOAL_STATUS_KEY}: {distance_status!r}"
            )
    return result


def _capture_distance_to_goal(
    value: Any,
    *,
    shortest_status: str,
) -> tuple[float | None, str]:
    """Sanitize Habitat's +inf goal distance only for a known unreachable target."""

    if isinstance(value, bool):
        raise MonMetricError(f"{DISTANCE_TO_GOAL_KEY} must be numeric, got bool")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise MonMetricError(
            f"{DISTANCE_TO_GOAL_KEY} must be numeric, got {value!r}"
        ) from exc
    if math.isinf(number) and number > 0.0:
        if shortest_status != SHORTEST_PATH_UNREACHABLE:
            raise MonMetricError(
                f"{DISTANCE_TO_GOAL_KEY}=+inf requires an unreachable shortest path"
            )
        return None, SHORTEST_PATH_UNREACHABLE
    if not math.isfinite(number) or number < 0.0:
        raise MonMetricError(
            f"{DISTANCE_TO_GOAL_KEY} must be finite and >= 0, or +inf for a "
            f"failed unreachable subtask; got {number!r}"
        )
    return number, SHORTEST_PATH_FINITE


def _percentile(samples: Sequence[float], quantile: float) -> float:
    if not samples:
        raise MonMetricError("cannot compute a percentile without samples")
    ordered = sorted(samples)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_frame_times(frame_times_ms: Iterable[Any]) -> dict[str, Any]:
    samples = [
        _finite_number(value, f"frame_times_ms[{index}]")
        for index, value in enumerate(frame_times_ms)
    ]
    if not samples:
        raise MonMetricError("each completed subtask must contain at least one frame")
    total_ms = sum(samples)
    mean_ms = total_ms / len(samples)
    return {
        "frame_count": len(samples),
        "frame_time_sum_ms": total_ms,
        "mean_frame_time_ms": mean_ms,
        "median_frame_time_ms": float(median(samples)),
        "p95_frame_time_ms": _percentile(samples, 0.95),
        "fps": None if mean_ms == 0.0 else 1000.0 / mean_ms,
    }


def build_subtask_telemetry(
    metric: Mapping[str, Any],
    *,
    actual_path_length: Any,
    shortest_path_length: Any,
    steps: Any,
    frame_times_ms: Iterable[Any],
) -> dict[str, Any]:
    """Capture one completed subtask before Habitat resets its SPL state."""

    success = _binary_success(metric.get(SUCCESS_KEY))
    actual = _finite_number(actual_path_length, PATH_LENGTH_KEY)
    shortest, shortest_status = _capture_shortest_path(
        shortest_path_length,
        success=success,
    )
    distance_to_goal: float | None = None
    distance_to_goal_status: str | None = None
    if DISTANCE_TO_GOAL_KEY in metric:
        distance_to_goal, distance_to_goal_status = _capture_distance_to_goal(
            metric[DISTANCE_TO_GOAL_KEY],
            shortest_status=shortest_status,
        )
    step_count = _nonnegative_int(steps, STEPS_KEY)
    timing = summarize_frame_times(frame_times_ms)
    if timing["frame_count"] != step_count:
        raise MonMetricError(
            "subtask frame/step mismatch: "
            f"frames={timing['frame_count']} steps={step_count}"
        )

    efficiency = (
        None if shortest is None else _path_efficiency(actual, shortest)
    )
    expected_spl = 0.0 if efficiency is None else success * efficiency
    raw_spl = metric.get(SPL_KEY)
    raw_spl_nonfinite_zero_length = False
    raw_spl_nonfinite_unreachable = False
    try:
        observed_spl = _finite_number(raw_spl, SPL_KEY)
    except MonMetricError as raw_spl_error:
        try:
            raw_spl_number = float(raw_spl)
        except (TypeError, ValueError):
            raise raw_spl_error
        if actual == 0.0 and shortest == 0.0 and not math.isfinite(raw_spl_number):
            observed_spl = expected_spl
            raw_spl_nonfinite_zero_length = True
        elif shortest is None and math.isnan(raw_spl_number):
            observed_spl = expected_spl
            raw_spl_nonfinite_unreachable = True
        else:
            raise raw_spl_error
    if not math.isclose(observed_spl, expected_spl, rel_tol=1e-6, abs_tol=1e-8):
        raise MonMetricError(
            "captured path lengths do not reproduce Habitat subtask SPL: "
            f"observed={observed_spl} expected={expected_spl}"
        )

    stretch = None
    if shortest is not None and shortest > 0.0:
        stretch = actual / shortest
    elif shortest == 0.0 and actual == 0.0:
        stretch = 1.0

    telemetry = {
        SPL_KEY: expected_spl,
        "goat_sub-task_spl_raw_nonfinite_zero_length": (
            raw_spl_nonfinite_zero_length
        ),
        "goat_sub-task_spl_raw_nonfinite_unreachable": (
            raw_spl_nonfinite_unreachable
        ),
        PATH_LENGTH_KEY: actual,
        SHORTEST_PATH_LENGTH_KEY: shortest,
        SHORTEST_PATH_STATUS_KEY: shortest_status,
        "goat_sub-task_path_efficiency": efficiency,
        "goat_sub-task_path_stretch": stretch,
        "goat_sub-task_excess_path_length_m": (
            None if shortest is None else max(0.0, actual - shortest)
        ),
        STEPS_KEY: step_count,
        FRAME_COUNT_KEY: timing["frame_count"],
        FRAME_TIME_SUM_KEY: timing["frame_time_sum_ms"],
        "goat_sub-task_mean_frame_time_ms": timing["mean_frame_time_ms"],
        "goat_sub-task_median_frame_time_ms": timing["median_frame_time_ms"],
        "goat_sub-task_p95_frame_time_ms": timing["p95_frame_time_ms"],
        "goat_sub-task_fps": timing["fps"],
    }
    if distance_to_goal_status is not None:
        telemetry[DISTANCE_TO_GOAL_KEY] = distance_to_goal
        telemetry[DISTANCE_TO_GOAL_STATUS_KEY] = distance_to_goal_status
    return telemetry


def summarize_episode(
    subtasks: Sequence[Mapping[str, Any]],
    *,
    total_steps: Any,
    stop_reasons: Sequence[str],
    frame_times_ms: Iterable[Any],
    categories: Sequence[str],
) -> dict[str, Any]:
    """Compute the four paper metrics and efficiency for one MON episode."""

    if not subtasks:
        raise MonMetricError("a completed MON episode must contain subtasks")
    if len(stop_reasons) != len(subtasks):
        raise MonMetricError(
            f"stop reason count {len(stop_reasons)} != subtask count {len(subtasks)}"
        )
    if len(categories) != len(subtasks):
        raise MonMetricError(
            f"category count {len(categories)} != subtask count {len(subtasks)}"
        )

    step_count = _nonnegative_int(total_steps, "total_num_steps")
    successes = [_binary_success(item.get(SUCCESS_KEY)) for item in subtasks]
    actuals = [
        _finite_number(item.get(PATH_LENGTH_KEY), PATH_LENGTH_KEY)
        for item in subtasks
    ]
    recorded_shortest = [
        _recorded_shortest_path(item, success=success)
        for item, success in zip(subtasks, successes)
    ]
    shortest = [value for value, _status in recorded_shortest]
    shortest_statuses = [status for _value, status in recorded_shortest]
    subtask_spls = [
        _finite_number(item.get(SPL_KEY), SPL_KEY) for item in subtasks
    ]
    subtask_steps = [
        _nonnegative_int(item.get(STEPS_KEY), STEPS_KEY) for item in subtasks
    ]
    if sum(subtask_steps) != step_count:
        raise MonMetricError(
            f"episode subtask steps {sum(subtask_steps)} != total steps {step_count}"
        )

    timing = summarize_frame_times(frame_times_ms)
    if timing["frame_count"] != step_count:
        raise MonMetricError(
            "episode frame/step mismatch: "
            f"frames={timing['frame_count']} steps={step_count}"
        )

    episode_success = int(all(successes))
    actual_total = sum(actuals)
    finite_indices = [
        index
        for index, status in enumerate(shortest_statuses)
        if status == SHORTEST_PATH_FINITE
    ]
    finite_actual_total = sum(actuals[index] for index in finite_indices)
    finite_shortest_total = sum(
        shortest[index] for index in finite_indices if shortest[index] is not None
    )
    finite_shortest_count = len(finite_indices)
    unreachable_count = len(subtasks) - finite_shortest_count
    shortest_complete = unreachable_count == 0
    finite_efficiency = (
        _path_efficiency(finite_actual_total, finite_shortest_total)
        if finite_shortest_count
        else None
    )
    episode_efficiency = finite_efficiency if shortest_complete else None
    if episode_success and episode_efficiency is None:
        raise MonMetricError(
            "a successful episode cannot contain an unreachable shortest path"
        )
    episode_spl = 0.0 if not episode_success else episode_efficiency
    successful_subtasks = sum(successes)
    subtask_count = len(subtasks)
    pespl_numerator = sum(subtask_spls)
    psr_numerator = successful_subtasks
    psr_by_category: dict[str, dict[str, Any]] = {}
    for category, success in zip(categories, successes):
        clean_category = str(category).strip().lower()
        if not clean_category:
            raise MonMetricError("subtask category must be non-empty")
        bucket = psr_by_category.setdefault(
            clean_category, {"targets": 0, "successes": 0}
        )
        bucket["targets"] += 1
        bucket["successes"] += success
    for bucket in psr_by_category.values():
        bucket["success_rate"] = bucket["successes"] / bucket["targets"]
        bucket["psr_contribution"] = bucket["successes"] / subtask_count

    return {
        "schema_version": MON_METRICS_SCHEMA_VERSION,
        "mon_metrics": {
            "sr": episode_success,
            "spl": episode_spl,
            "pespl": pespl_numerator / subtask_count,
            "psr": psr_numerator / subtask_count,
            "pespl_numerator": pespl_numerator,
            "pespl_denominator": subtask_count,
            "psr_numerator": psr_numerator,
            "psr_denominator": subtask_count,
            "subtask_count": subtask_count,
            "successful_subtask_count": successful_subtasks,
            "psr_by_category": psr_by_category,
        },
        "navigation_efficiency": {
            "actual_path_length_m": actual_total,
            "shortest_path_length_m": (
                finite_shortest_total if shortest_complete else None
            ),
            "shortest_path_status": (
                SHORTEST_PATH_FINITE
                if shortest_complete
                else "contains_unreachable"
            ),
            "path_efficiency": episode_efficiency,
            "path_stretch": (
                actual_total / finite_shortest_total
                if shortest_complete and finite_shortest_total > 0.0
                else (
                    1.0
                    if shortest_complete
                    and finite_shortest_total == 0.0
                    and actual_total == 0.0
                    else None
                )
            ),
            "excess_path_length_m": (
                max(0.0, actual_total - finite_shortest_total)
                if shortest_complete
                else None
            ),
            "finite_actual_path_length_m": finite_actual_total,
            "finite_shortest_path_length_m": finite_shortest_total,
            "finite_path_efficiency": finite_efficiency,
            "finite_shortest_path_subtask_count": finite_shortest_count,
            "unreachable_subtask_count": unreachable_count,
            "finite_shortest_path_coverage": finite_shortest_count / subtask_count,
            "total_steps": step_count,
            "mean_steps_per_subtask": step_count / subtask_count,
            "steps_per_successful_subtask": (
                None if successful_subtasks == 0 else step_count / successful_subtasks
            ),
            "successful_subtasks_per_100_steps": (
                None if step_count == 0 else 100.0 * successful_subtasks / step_count
            ),
            "max_steps_subtask_count": sum(
                reason == "max_steps" for reason in stop_reasons
            ),
        },
        "frame_timing": timing,
    }


def summarize_records(
    records: Mapping[str, Mapping[str, Any]],
    *,
    frame_times_ms: Iterable[Any] | None = None,
) -> dict[str, Any]:
    """Aggregate complete episode records with paper-exact denominators."""

    if not records:
        raise MonMetricError("cannot summarize zero completed MON episodes")

    episodes = list(records.values())
    mon = [episode["mon_metrics"] for episode in episodes]
    efficiency = [episode["navigation_efficiency"] for episode in episodes]
    all_subtasks = [item for episode in episodes for item in episode["metrics"]]
    episode_count = len(episodes)
    subtask_count = sum(
        _nonnegative_int(item["subtask_count"], "subtask_count") for item in mon
    )
    success_count = sum(
        _nonnegative_int(item["successful_subtask_count"], "successful_subtask_count")
        for item in mon
    )
    if subtask_count != len(all_subtasks):
        raise MonMetricError(
            f"declared subtask count {subtask_count} != observed {len(all_subtasks)}"
        )
    episode_subtask_counts = {
        _nonnegative_int(item["subtask_count"], "subtask_count") for item in mon
    }
    if len(episode_subtask_counts) != 1:
        raise MonMetricError(
            "PSR/PeSPL require a fixed number of targets per episode; "
            f"observed {sorted(episode_subtask_counts)}"
        )
    targets_per_episode = next(iter(episode_subtask_counts))
    if subtask_count != episode_count * targets_per_episode:
        raise MonMetricError(
            "PSR/PeSPL denominator mismatch: "
            f"subtasks={subtask_count} episodes={episode_count} "
            f"targets_per_episode={targets_per_episode}"
        )

    actual_total = sum(
        _finite_number(item["actual_path_length_m"], "actual_path_length_m")
        for item in efficiency
    )
    all_subtask_successes = [
        _binary_success(item.get(SUCCESS_KEY)) for item in all_subtasks
    ]
    recorded_shortest = [
        _recorded_shortest_path(item, success=success)
        for item, success in zip(all_subtasks, all_subtask_successes)
    ]
    finite_subtask_indices = [
        index
        for index, (_value, status) in enumerate(recorded_shortest)
        if status == SHORTEST_PATH_FINITE
    ]
    finite_shortest_count = len(finite_subtask_indices)
    unreachable_count = subtask_count - finite_shortest_count
    finite_shortest_total = sum(
        recorded_shortest[index][0]
        for index in finite_subtask_indices
        if recorded_shortest[index][0] is not None
    )
    finite_actual_total = sum(
        _finite_number(all_subtasks[index].get(PATH_LENGTH_KEY), PATH_LENGTH_KEY)
        for index in finite_subtask_indices
    )
    shortest_complete = unreachable_count == 0
    shortest_total = finite_shortest_total if shortest_complete else None
    total_steps = sum(
        _nonnegative_int(item["total_steps"], "total_steps") for item in efficiency
    )
    max_steps_count = sum(
        _nonnegative_int(item["max_steps_subtask_count"], "max_steps_subtask_count")
        for item in efficiency
    )
    psr_by_category: dict[str, dict[str, Any]] = {}
    for episode in mon:
        for category, source in episode["psr_by_category"].items():
            bucket = psr_by_category.setdefault(
                category, {"targets": 0, "successes": 0}
            )
            bucket["targets"] += _nonnegative_int(
                source["targets"], f"{category} targets"
            )
            bucket["successes"] += _nonnegative_int(
                source["successes"], f"{category} successes"
            )
    for bucket in psr_by_category.values():
        bucket["success_rate"] = bucket["successes"] / bucket["targets"]
        bucket["psr_contribution"] = bucket["successes"] / subtask_count

    timing_count = sum(
        _nonnegative_int(episode["frame_timing"]["frame_count"], "frame_count")
        for episode in episodes
    )
    timing_sum = sum(
        _finite_number(
            episode["frame_timing"]["frame_time_sum_ms"], "frame_time_sum_ms"
        )
        for episode in episodes
    )
    if timing_count != total_steps:
        raise MonMetricError(
            f"aggregate frame count {timing_count} != total steps {total_steps}"
        )
    if frame_times_ms is None:
        timing = {
            "frame_count": timing_count,
            "frame_time_sum_ms": timing_sum,
            "mean_frame_time_ms": timing_sum / timing_count,
            "median_frame_time_ms": None,
            "p95_frame_time_ms": None,
            "fps": None if timing_sum == 0.0 else 1000.0 * timing_count / timing_sum,
        }
    else:
        timing = summarize_frame_times(frame_times_ms)
        if timing["frame_count"] != timing_count:
            raise MonMetricError(
                f"raw timing rows {timing['frame_count']} != completed frames {timing_count}"
            )

    finite_subtask_efficiencies = [
        _finite_number(
            all_subtasks[index]["goat_sub-task_path_efficiency"],
            "subtask efficiency",
        )
        for index in finite_subtask_indices
    ]
    finite_episode_efficiencies: list[float] = []
    for episode, item in zip(episodes, efficiency):
        episode_has_unreachable = any(
            _recorded_shortest_path(
                subtask,
                success=_binary_success(subtask.get(SUCCESS_KEY)),
            )[1]
            == SHORTEST_PATH_UNREACHABLE
            for subtask in episode["metrics"]
        )
        value = item.get("path_efficiency")
        if episode_has_unreachable:
            if value is not None:
                raise MonMetricError(
                    "episode path_efficiency must be null when shortest-path "
                    "coverage is incomplete"
                )
        else:
            finite_episode_efficiencies.append(
                _finite_number(value, "episode efficiency")
            )

    pespl_numerator = sum(
        _finite_number(item.get(SPL_KEY), SPL_KEY) for item in all_subtasks
    )

    return {
        "schema_version": MON_METRICS_SCHEMA_VERSION,
        "counts": {
            "completed_episodes": episode_count,
            "completed_subtasks": subtask_count,
            "targets_per_episode": targets_per_episode,
            "successful_episodes": sum(
                _binary_success(item["sr"], "episode SR contribution") for item in mon
            ),
            "successful_subtasks": success_count,
            "finite_shortest_path_episodes": len(finite_episode_efficiencies),
            "episodes_with_unreachable": (
                episode_count - len(finite_episode_efficiencies)
            ),
            "finite_shortest_path_subtasks": finite_shortest_count,
            "unreachable_subtasks": unreachable_count,
        },
        "mon_metrics": {
            "sr": sum(item["sr"] for item in mon) / episode_count,
            "spl": sum(_finite_number(item["spl"], "episode SPL") for item in mon)
            / episode_count,
            "pespl": pespl_numerator / subtask_count,
            "psr": success_count / subtask_count,
            "pespl_numerator": pespl_numerator,
            "pespl_denominator": subtask_count,
            "psr_numerator": success_count,
            "psr_denominator": subtask_count,
            "psr_by_category": psr_by_category,
        },
        "navigation_efficiency": {
            "total_actual_path_length_m": actual_total,
            "total_shortest_path_length_m": shortest_total,
            "shortest_path_status": (
                SHORTEST_PATH_FINITE
                if shortest_complete
                else "contains_unreachable"
            ),
            "aggregate_path_efficiency": (
                _path_efficiency(actual_total, finite_shortest_total)
                if shortest_complete
                else None
            ),
            "mean_episode_path_efficiency": (
                sum(finite_episode_efficiencies) / episode_count
                if shortest_complete
                else None
            ),
            "mean_subtask_path_efficiency": (
                sum(finite_subtask_efficiencies) / subtask_count
                if shortest_complete
                else None
            ),
            "finite_actual_path_length_m": finite_actual_total,
            "finite_shortest_path_length_m": finite_shortest_total,
            "finite_aggregate_path_efficiency": (
                _path_efficiency(finite_actual_total, finite_shortest_total)
                if finite_shortest_count
                else None
            ),
            "mean_finite_episode_path_efficiency": (
                sum(finite_episode_efficiencies) / len(finite_episode_efficiencies)
                if finite_episode_efficiencies
                else None
            ),
            "mean_finite_subtask_path_efficiency": (
                sum(finite_subtask_efficiencies) / finite_shortest_count
                if finite_shortest_count
                else None
            ),
            "finite_shortest_path_subtask_count": finite_shortest_count,
            "unreachable_subtask_count": unreachable_count,
            "finite_shortest_path_episode_count": len(
                finite_episode_efficiencies
            ),
            "episodes_with_unreachable": (
                episode_count - len(finite_episode_efficiencies)
            ),
            "finite_shortest_path_coverage": finite_shortest_count / subtask_count,
            "mean_episode_steps": total_steps / episode_count,
            "mean_subtask_steps": total_steps / subtask_count,
            "successful_subtasks_per_100_steps": (
                None if total_steps == 0 else 100.0 * success_count / total_steps
            ),
            "max_steps_rate": max_steps_count / subtask_count,
        },
        "frame_timing": timing,
    }


def merge_episode_metric_files(paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for path in sorted(Path(item) for item in paths):
        with path.open(encoding="utf-8") as handle:
            try:
                payload = json.load(handle, parse_constant=_reject_json_constant)
            except json.JSONDecodeError as exc:
                raise MonMetricError(f"invalid JSON metric file: {path}") from exc
        if not isinstance(payload, dict):
            raise MonMetricError(f"metric file must contain an object: {path}")
        for episode_key, record in payload.items():
            if episode_key in merged:
                raise MonMetricError(f"duplicate completed episode {episode_key!r}")
            merged[episode_key] = record
    return merged


def load_timing_samples(paths: Iterable[Path]) -> list[float]:
    samples: list[float] = []
    identities: set[tuple[str, str, str, str]] = set()
    for path in sorted(Path(item) for item in paths):
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                identity = (
                    row["scene_id"],
                    row["episode_id"],
                    row["subtask_index"],
                    row["frame_index"],
                )
                if identity in identities:
                    raise MonMetricError(f"duplicate timing row {identity!r}")
                identities.add(identity)
                samples.append(_finite_number(row["dt_total_ms"], "dt_total_ms"))
    return samples


def aggregate_run_artifacts(run: Path) -> dict[str, Any]:
    run = Path(run)
    metric_paths = list(run.glob("results_*/**/per_episode_metrics.json"))
    timing_paths = list(run.glob("results_*/**/timing_log.csv"))
    records = merge_episode_metric_files(metric_paths)
    timing_samples = load_timing_samples(timing_paths)
    return summarize_records(records, frame_times_ms=timing_samples)


def write_json_strict(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
