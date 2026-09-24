import csv
import json
import math
from pathlib import Path

import pytest

from tdonav_navigation.mon_metrics import (
    DISTANCE_TO_GOAL_KEY,
    DISTANCE_TO_GOAL_STATUS_KEY,
    MON_METRICS_SCHEMA_VERSION,
    MonMetricError,
    SHORTEST_PATH_STATUS_KEY,
    SHORTEST_PATH_UNREACHABLE,
    aggregate_run_artifacts,
    build_subtask_telemetry,
    merge_episode_metric_files,
    summarize_episode,
    summarize_records,
    write_json_strict,
)


def _subtask(success, actual, shortest, frame_times, raw_spl=None):
    efficiency = 1.0 if max(actual, shortest) == 0 else shortest / max(actual, shortest)
    metric = {
        "goat_distance_to_sub-goal": 0.0 if success else 1.0,
        "goat_sub-task_success": float(success),
        "goat_sub-task_spl": success * efficiency if raw_spl is None else raw_spl,
    }
    metric.update(
        build_subtask_telemetry(
            metric,
            actual_path_length=actual,
            shortest_path_length=shortest,
            steps=len(frame_times),
            frame_times_ms=frame_times,
        )
    )
    return metric


def _episode(subtasks, frame_times, categories=None, stop_reasons=None):
    categories = categories or ["chair"] * len(subtasks)
    stop_reasons = stop_reasons or ["verified_target"] * len(subtasks)
    total_steps = sum(item["goat_sub-task_num_steps"] for item in subtasks)
    record = {
        "metrics": subtasks,
        "total_num_steps": total_steps,
        "stop_reasons": stop_reasons,
    }
    record.update(
        summarize_episode(
            subtasks,
            total_steps=total_steps,
            stop_reasons=stop_reasons,
            frame_times_ms=frame_times,
            categories=categories,
        )
    )
    return record


def test_exact_normalized_paper_metrics_and_audit_denominators():
    first = [
        _subtask(1, 2.0, 1.0, [10.0, 20.0]),
        _subtask(1, 4.0, 2.0, [30.0]),
    ]
    second = [
        _subtask(0, 4.0, 2.0, [40.0]),
        _subtask(1, 4.0, 1.0, [50.0]),
    ]
    records = {
        "scene_0": _episode(first, [10.0, 20.0, 30.0], ["chair", "bed"]),
        "scene_1": _episode(
            second,
            [40.0, 50.0],
            ["chair", "toilet"],
            ["max_steps", "verified_target"],
        ),
    }

    summary = summarize_records(
        records, frame_times_ms=[10.0, 20.0, 30.0, 40.0, 50.0]
    )

    assert summary["counts"]["completed_episodes"] == 2
    assert summary["counts"]["completed_subtasks"] == 4
    assert summary["counts"]["targets_per_episode"] == 2
    assert summary["counts"]["successful_episodes"] == 1
    assert summary["counts"]["successful_subtasks"] == 3
    assert summary["counts"]["finite_shortest_path_episodes"] == 2
    assert summary["counts"]["episodes_with_unreachable"] == 0
    assert summary["counts"]["finite_shortest_path_subtasks"] == 4
    assert summary["counts"]["unreachable_subtasks"] == 0
    assert summary["mon_metrics"]["sr"] == pytest.approx(0.5)
    assert summary["mon_metrics"]["spl"] == pytest.approx(0.25)
    assert summary["mon_metrics"]["pespl"] == pytest.approx(1.25 / 4)
    assert summary["mon_metrics"]["psr"] == pytest.approx(3 / 4)
    assert summary["mon_metrics"]["pespl_numerator"] == pytest.approx(1.25)
    assert summary["mon_metrics"]["pespl_denominator"] == 4
    assert summary["mon_metrics"]["psr_numerator"] == 3
    assert summary["mon_metrics"]["psr_denominator"] == 4
    chair = summary["mon_metrics"]["psr_by_category"]["chair"]
    assert chair["successes"] == 1
    assert chair["targets"] == 2
    assert chair["success_rate"] == pytest.approx(0.5)
    assert summary["navigation_efficiency"]["max_steps_rate"] == pytest.approx(1 / 4)
    assert summary["frame_timing"]["mean_frame_time_ms"] == pytest.approx(30.0)
    assert summary["frame_timing"]["median_frame_time_ms"] == pytest.approx(30.0)
    assert summary["frame_timing"]["p95_frame_time_ms"] == pytest.approx(48.0)


def test_each_episode_contains_all_four_standard_metrics():
    record = _episode(
        [_subtask(1, 2.0, 1.0, [5.0]), _subtask(0, 3.0, 1.0, [7.0])],
        [5.0, 7.0],
        ["bed", "tv"],
    )
    assert set(("sr", "spl", "pespl", "psr")) <= set(record["mon_metrics"])
    assert record["schema_version"] == MON_METRICS_SCHEMA_VERSION
    assert record["mon_metrics"]["sr"] == 0
    assert record["mon_metrics"]["spl"] == 0.0
    assert record["mon_metrics"]["pespl"] == pytest.approx(0.25)
    assert record["mon_metrics"]["psr"] == pytest.approx(0.5)
    assert record["mon_metrics"]["pespl_numerator"] == pytest.approx(0.5)
    assert record["mon_metrics"]["pespl_denominator"] == 2
    assert record["mon_metrics"]["psr_numerator"] == 1
    assert record["mon_metrics"]["psr_denominator"] == 2


def test_aggregate_rejects_variable_targets_per_episode():
    records = {
        "scene_0": _episode([_subtask(1, 1.0, 1.0, [1.0])], [1.0]),
        "scene_1": _episode(
            [_subtask(1, 1.0, 1.0, [1.0]), _subtask(0, 1.0, 1.0, [1.0])],
            [1.0, 1.0],
        ),
    }
    with pytest.raises(MonMetricError, match="fixed number of targets"):
        summarize_records(records)


def test_failed_subtask_retains_lengths_but_has_zero_spl():
    metric = _subtask(0, 9.0, 3.0, [5.0])
    assert metric["goat_sub-task_path_length_m"] == 9.0
    assert metric["goat_sub-task_shortest_path_length_m"] == 3.0
    assert metric["goat_sub-task_spl"] == 0.0
    assert metric["goat_sub-task_path_efficiency"] == pytest.approx(1 / 3)


def test_failed_unreachable_subtask_is_zero_contribution_and_strict_json(tmp_path):
    source = {
        DISTANCE_TO_GOAL_KEY: math.inf,
        "goat_sub-task_success": 0.0,
        "goat_sub-task_spl": math.nan,
    }
    source.update(
        build_subtask_telemetry(
            source,
            actual_path_length=5.0,
            shortest_path_length=math.inf,
            steps=1,
            frame_times_ms=[3.0],
        )
    )

    assert source["goat_sub-task_spl"] == 0.0
    assert source["goat_sub-task_shortest_path_length_m"] is None
    assert source[SHORTEST_PATH_STATUS_KEY] == SHORTEST_PATH_UNREACHABLE
    assert source["goat_sub-task_path_efficiency"] is None
    assert source["goat_sub-task_path_stretch"] is None
    assert source["goat_sub-task_excess_path_length_m"] is None
    assert source[DISTANCE_TO_GOAL_KEY] is None
    assert source[DISTANCE_TO_GOAL_STATUS_KEY] == SHORTEST_PATH_UNREACHABLE
    assert source["goat_sub-task_spl_raw_nonfinite_unreachable"] is True

    finite = _subtask(1, 2.0, 1.0, [2.0])
    record = _episode(
        [finite, source],
        [2.0, 3.0],
        ["chair", "plant"],
        ["verified_target", "max_steps"],
    )
    summary = summarize_records({"scene_0": record}, frame_times_ms=[2.0, 3.0])

    assert record["mon_metrics"]["sr"] == 0
    assert record["mon_metrics"]["spl"] == 0.0
    assert record["mon_metrics"]["psr"] == pytest.approx(0.5)
    assert record["mon_metrics"]["psr_denominator"] == 2
    assert record["mon_metrics"]["pespl"] == pytest.approx(0.25)
    assert record["mon_metrics"]["pespl_denominator"] == 2
    assert record["navigation_efficiency"]["shortest_path_length_m"] is None
    assert record["navigation_efficiency"]["path_efficiency"] is None
    assert summary["navigation_efficiency"]["unreachable_subtask_count"] == 1
    assert summary["navigation_efficiency"]["finite_shortest_path_subtask_count"] == 1
    assert summary["navigation_efficiency"]["finite_shortest_path_coverage"] == 0.5
    assert summary["navigation_efficiency"]["total_shortest_path_length_m"] is None
    assert summary["navigation_efficiency"]["aggregate_path_efficiency"] is None
    assert summary["counts"]["finite_shortest_path_episodes"] == 0
    assert summary["counts"]["episodes_with_unreachable"] == 1

    output = tmp_path / "unreachable.json"
    write_json_strict(output, {"scene_0": record, "summary": summary})
    raw = output.read_text(encoding="utf-8")
    assert "NaN" not in raw and "Infinity" not in raw
    assert json.loads(raw)["scene_0"]["metrics"][1][SHORTEST_PATH_STATUS_KEY] == (
        SHORTEST_PATH_UNREACHABLE
    )


def test_zero_length_success_sanitizes_only_raw_nonfinite_spl():
    metric = _subtask(1, 0.0, 0.0, [1.0], raw_spl=math.nan)
    record = _episode([metric], [1.0], ["toilet"])
    assert metric["goat_sub-task_spl"] == 1.0
    assert metric["goat_sub-task_spl_raw_nonfinite_zero_length"] is True
    assert record["mon_metrics"]["spl"] == 1.0


@pytest.mark.parametrize(
    "actual,shortest",
    [
        (math.nan, 1.0),
        (math.inf, 1.0),
        (1.0, math.nan),
        (1.0, -math.inf),
        (-1.0, 1.0),
    ],
)
def test_invalid_lengths_fail_loudly(actual, shortest):
    metric = {
        "goat_sub-task_success": 0.0,
        "goat_sub-task_spl": 0.0,
    }
    with pytest.raises(MonMetricError):
        build_subtask_telemetry(
            metric,
            actual_path_length=actual,
            shortest_path_length=shortest,
            steps=1,
            frame_times_ms=[1.0],
        )


def test_success_with_unreachable_shortest_path_fails_loudly():
    metric = {
        DISTANCE_TO_GOAL_KEY: math.inf,
        "goat_sub-task_success": 1.0,
        "goat_sub-task_spl": math.nan,
    }
    with pytest.raises(MonMetricError, match="successful subtask"):
        build_subtask_telemetry(
            metric,
            actual_path_length=1.0,
            shortest_path_length=math.inf,
            steps=1,
            frame_times_ms=[1.0],
        )


@pytest.mark.parametrize("raw_spl", [math.inf, -math.inf])
def test_unreachable_non_nan_raw_spl_fails_loudly(raw_spl):
    metric = {
        DISTANCE_TO_GOAL_KEY: math.inf,
        "goat_sub-task_success": 0.0,
        "goat_sub-task_spl": raw_spl,
    }
    with pytest.raises(MonMetricError):
        build_subtask_telemetry(
            metric,
            actual_path_length=1.0,
            shortest_path_length=math.inf,
            steps=1,
            frame_times_ms=[1.0],
        )


def test_frame_step_mismatch_fails_loudly():
    metric = {"goat_sub-task_success": 1.0, "goat_sub-task_spl": 1.0}
    with pytest.raises(MonMetricError, match="frame/step mismatch"):
        build_subtask_telemetry(
            metric,
            actual_path_length=1.0,
            shortest_path_length=1.0,
            steps=2,
            frame_times_ms=[1.0],
        )


def test_duplicate_completed_episode_is_rejected(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(json.dumps({"scene_0": {}}), encoding="utf-8")
    second.write_text(json.dumps({"scene_0": {}}), encoding="utf-8")
    with pytest.raises(MonMetricError, match="duplicate completed episode"):
        merge_episode_metric_files([first, second])


def test_metric_reader_rejects_non_standard_json_constants(tmp_path):
    invalid = tmp_path / "invalid.json"
    invalid.write_text('{"scene_0": {"extra": NaN}}', encoding="utf-8")
    with pytest.raises(MonMetricError, match="non-standard JSON constant"):
        merge_episode_metric_files([invalid])


def test_run_artifact_aggregation_and_strict_json(tmp_path):
    subtasks = [_subtask(1, 2.0, 1.0, [10.0, 20.0])]
    record = _episode(subtasks, [10.0, 20.0], ["plant"])
    results = tmp_path / "results_scene" / "experiment"
    results.mkdir(parents=True)
    write_json_strict(results / "per_episode_metrics.json", {"scene_0": record})
    with (results / "timing_log.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "scene_id",
                "episode_id",
                "subtask_index",
                "frame_index",
                "dt_total_ms",
            ),
        )
        writer.writeheader()
        for index, value in enumerate((10.0, 20.0), 1):
            writer.writerow(
                {
                    "scene_id": "scene",
                    "episode_id": "0",
                    "subtask_index": "0",
                    "frame_index": str(index),
                    "dt_total_ms": str(value),
                }
            )

    summary = aggregate_run_artifacts(tmp_path)
    assert summary["counts"]["completed_episodes"] == 1
    assert summary["mon_metrics"]["sr"] == 1.0
    assert summary["schema_version"] == MON_METRICS_SCHEMA_VERSION
    assert summary["frame_timing"]["mean_frame_time_ms"] == 15.0

    with pytest.raises(ValueError):
        write_json_strict(tmp_path / "nonfinite.json", {"value": math.nan})


def test_capture_precedes_subtask_reset_and_is_not_policy_input():
    root = Path(__file__).resolve().parents[2]
    entrypoint = (
        root / "Navigation/runtime/evaluators/goat/entrypoint.py"
    ).read_text(encoding="utf-8")
    capture = entrypoint.index("path_lengths = env.get_subtask_path_lengths()")
    build = entrypoint.index("ep_metrics.update(build_subtask_telemetry(")
    reset = entrypoint.index("env.reset_subtask()", capture)
    policy = entrypoint.index("action, info = agent.act(obs)")
    assert policy < capture < build < reset
    assert entrypoint.count("build_subtask_telemetry(") == 1
    assert "_finite_metric_summary" in entrypoint
    assert "in-place resume is unsupported" in entrypoint

    env_source = (
        root
        / "Navigation/runtime/lagmemo/src/lagmemo/lagmemo/env/habitat_lagmemo_env.py"
    ).read_text(encoding="utf-8")
    assert "np.isposinf(shortest_value)" in env_source
    assert "success != 0.0" in env_source

    run_source = (root / "Exp/scripts/run_goat.py").read_text(encoding="utf-8")
    assert '"mon_metrics_schema": "mon_metrics_v3"' in run_source
    assert '"unreachable_shortest_path_convention"' in run_source
