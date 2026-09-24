"""Exact MON metric aggregation and navigation-efficiency telemetry."""

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
PATH_LENGTH_KEY = "goat_sub-task_path_length"
SHORTEST_PATH_LENGTH_KEY = "goat_sub-task_shortest_path_length"
STEPS_KEY = "goat_sub-task_num_steps"
FRAME_COUNT_KEY = "goat_sub-task_frame_count"
FRAME_TIME_SUM_KEY = "goat_sub-task_frame_time_sum_ms"


class MonMetricError(ValueError):
    """Raised when MON telemetry cannot produce a mathematically valid metric."""


def _finite_number(value: Any, name: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool):
        raise MonMetricError(f"{name} must be numeric, got bool")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise MonMetricError(f"{name} must be numeric, got {value!r}") from exc
    if not math.isfinite(number) or number < minimum:
        raise MonMetricError(f"{name} must be finite and >= {minimum}, got {number!r}")
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
    """Build telemetry from the already-updated Habitat subtask measures."""

    success = _binary_success(metric.get(SUCCESS_KEY))
    observed_spl = _finite_number(metric.get(SPL_KEY), SPL_KEY)
    actual = _finite_number(actual_path_length, PATH_LENGTH_KEY)
    shortest = _finite_number(shortest_path_length, SHORTEST_PATH_LENGTH_KEY)
    step_count = _nonnegative_int(steps, STEPS_KEY)
    timing = summarize_frame_times(frame_times_ms)
    if timing["frame_count"] != step_count:
        raise MonMetricError(
            "subtask frame/step mismatch: "
            f"frames={timing['frame_count']} steps={step_count}"
        )

    efficiency = _path_efficiency(actual, shortest)
    expected_spl = success * efficiency
    if not math.isclose(observed_spl, expected_spl, rel_tol=1e-6, abs_tol=1e-8):
        raise MonMetricError(
            "captured path lengths do not reproduce Habitat subtask SPL: "
            f"observed={observed_spl} expected={expected_spl}"
        )

    stretch = None
    if shortest > 0.0:
        stretch = actual / shortest
    elif actual == 0.0:
        stretch = 1.0

    return {
        PATH_LENGTH_KEY: actual,
        SHORTEST_PATH_LENGTH_KEY: shortest,
        "goat_sub-task_path_efficiency": efficiency,
        "goat_sub-task_path_stretch": stretch,
        "goat_sub-task_excess_path_length": max(0.0, actual - shortest),
        STEPS_KEY: step_count,
        FRAME_COUNT_KEY: timing["frame_count"],
        FRAME_TIME_SUM_KEY: timing["frame_time_sum_ms"],
        "goat_sub-task_mean_frame_time_ms": timing["mean_frame_time_ms"],
        "goat_sub-task_median_frame_time_ms": timing["median_frame_time_ms"],
        "goat_sub-task_p95_frame_time_ms": timing["p95_frame_time_ms"],
        "goat_sub-task_fps": timing["fps"],
    }


def summarize_episode(
    subtasks: Sequence[Mapping[str, Any]],
    *,
    total_steps: Any,
    stop_reasons: Sequence[str],
    frame_times_ms: Iterable[Any],
) -> dict[str, Any]:
    if not subtasks:
        raise MonMetricError("a completed MON episode must contain at least one subtask")
    if len(stop_reasons) != len(subtasks):
        raise MonMetricError(
            f"stop reason count {len(stop_reasons)} != subtask count {len(subtasks)}"
        )

    step_count = _nonnegative_int(total_steps, "total_num_steps")
    successes = [_binary_success(item.get(SUCCESS_KEY)) for item in subtasks]
    actuals = [
        _finite_number(item.get(PATH_LENGTH_KEY), PATH_LENGTH_KEY) for item in subtasks
    ]
    shortest = [
        _finite_number(item.get(SHORTEST_PATH_LENGTH_KEY), SHORTEST_PATH_LENGTH_KEY)
        for item in subtasks
    ]
    subtask_spls = [_finite_number(item.get(SPL_KEY), SPL_KEY) for item in subtasks]
    subtask_steps = [_nonnegative_int(item.get(STEPS_KEY), STEPS_KEY) for item in subtasks]
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
    shortest_total = sum(shortest)
    episode_efficiency = _path_efficiency(actual_total, shortest_total)
    successful_subtasks = sum(successes)
    subtask_count = len(subtasks)

    return {
        "mon_episode_metrics": {
            "episode_success": episode_success,
            "spl": episode_success * episode_efficiency,
            "pespl_contribution": sum(subtask_spls),
            "psr_contribution": successful_subtasks,
            "normalized_pespl": sum(subtask_spls) / subtask_count,
            "normalized_psr": successful_subtasks / subtask_count,
            "subtask_count": subtask_count,
            "successful_subtask_count": successful_subtasks,
        },
        "navigation_efficiency": {
            "actual_path_length": actual_total,
            "shortest_path_length": shortest_total,
            "path_efficiency": episode_efficiency,
            "path_stretch": (
                actual_total / shortest_total
                if shortest_total > 0.0
                else (1.0 if actual_total == 0.0 else None)
            ),
            "excess_path_length": max(0.0, actual_total - shortest_total),
            "total_steps": step_count,
            "steps_per_subtask": step_count / subtask_count,
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
    if not records:
        raise MonMetricError("cannot summarize zero completed MON episodes")

    episodes = list(records.values())
    mon = [episode["mon_episode_metrics"] for episode in episodes]
    efficiency = [episode["navigation_efficiency"] for episode in episodes]
    all_subtasks = [item for episode in episodes for item in episode["metrics"]]
    episode_count = len(episodes)
    subtask_count = sum(_nonnegative_int(item["subtask_count"], "subtask_count") for item in mon)
    success_count = sum(
        _nonnegative_int(item["successful_subtask_count"], "successful_subtask_count")
        for item in mon
    )
    if subtask_count != len(all_subtasks):
        raise MonMetricError(
            f"declared subtask count {subtask_count} != observed {len(all_subtasks)}"
        )

    actual_total = sum(
        _finite_number(item["actual_path_length"], "actual_path_length")
        for item in efficiency
    )
    shortest_total = sum(
        _finite_number(item["shortest_path_length"], "shortest_path_length")
        for item in efficiency
    )
    total_steps = sum(_nonnegative_int(item["total_steps"], "total_steps") for item in efficiency)
    max_steps_count = sum(
        _nonnegative_int(item["max_steps_subtask_count"], "max_steps_subtask_count")
        for item in efficiency
    )

    timing_count = sum(
        _nonnegative_int(episode["frame_timing"]["frame_count"], "frame_count")
        for episode in episodes
    )
    timing_sum = sum(
        _finite_number(episode["frame_timing"]["frame_time_sum_ms"], "frame_time_sum_ms")
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

    subtask_efficiencies = [
        _finite_number(item["goat_sub-task_path_efficiency"], "subtask path efficiency")
        for item in all_subtasks
    ]
    episode_efficiencies = [
        _finite_number(item["path_efficiency"], "episode path efficiency")
        for item in efficiency
    ]

    return {
        "schema_version": "mon_metrics_v1",
        "counts": {
            "completed_episodes": episode_count,
            "completed_subtasks": subtask_count,
            "successful_episodes": sum(
                _binary_success(item["episode_success"], "episode_success") for item in mon
            ),
            "successful_subtasks": success_count,
        },
        "mon_metrics": {
            "sr": sum(item["episode_success"] for item in mon) / episode_count,
            "spl": sum(_finite_number(item["spl"], "episode SPL") for item in mon)
            / episode_count,
            "pespl": sum(
                _finite_number(item["pespl_contribution"], "PeSPL contribution")
                for item in mon
            )
            / episode_count,
            "psr": sum(
                _finite_number(item["psr_contribution"], "PSR contribution")
                for item in mon
            )
            / episode_count,
            "normalized_pespl": sum(
                _finite_number(item.get(SPL_KEY), SPL_KEY) for item in all_subtasks
            )
            / subtask_count,
            "normalized_psr": success_count / subtask_count,
        },
        "navigation_efficiency": {
            "total_actual_path_length": actual_total,
            "total_shortest_path_length": shortest_total,
            "aggregate_path_efficiency": _path_efficiency(actual_total, shortest_total),
            "mean_episode_path_efficiency": sum(episode_efficiencies) / episode_count,
            "mean_subtask_path_efficiency": sum(subtask_efficiencies) / subtask_count,
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
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise MonMetricError(f"episode metric file must contain an object: {path}")
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
