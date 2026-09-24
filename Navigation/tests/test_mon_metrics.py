import csv
import json
import math
from pathlib import Path

import pytest

from tdonav_navigation.mon_metrics import (
    DISTANCE_TO_GOAL_STATUS_KEY,
    MON_METRICS_SCHEMA_VERSION,
    MonMetricError,
    aggregate_run_artifacts,
    build_subtask_telemetry,
    merge_episode_metric_files,
    summarize_episode,
    summarize_records,
    write_json_strict,
)


def _subtask(success, actual, shortest, frame_times):
    efficiency = 1.0 if max(actual, shortest) == 0 else shortest / max(actual, shortest)
    metric = {
        "goat_distance_to_sub-goal": 0.0 if success else 1.0,
        "goat_sub-task_success": float(success),
        "goat_sub-task_spl": success * efficiency,
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


def _episode(subtasks, frame_times, stop_reasons=None):
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
        )
    )
    return record


def test_exact_paper_metrics_and_normalized_variants():
    first_subtasks = [
        _subtask(1, 2.0, 1.0, [10.0, 20.0]),
        _subtask(1, 4.0, 2.0, [30.0]),
    ]
    second_subtasks = [_subtask(0, 4.0, 2.0, [40.0])]
    records = {
        "scene_0": _episode(first_subtasks, [10.0, 20.0, 30.0]),
        "scene_1": _episode(second_subtasks, [40.0], ["max_steps"]),
    }

    summary = summarize_records(
        records, frame_times_ms=[10.0, 20.0, 30.0, 40.0]
    )

    assert summary["counts"] == {
        "completed_episodes": 2,
        "completed_subtasks": 3,
        "successful_episodes": 1,
        "successful_subtasks": 2,
    }
    assert summary["mon_metrics"]["sr"] == pytest.approx(0.5)
    assert summary["mon_metrics"]["spl"] == pytest.approx(0.25)
    assert summary["mon_metrics"]["pespl"] == pytest.approx(0.5)
    assert summary["mon_metrics"]["psr"] == pytest.approx(1.0)
    assert summary["mon_metrics"]["normalized_pespl"] == pytest.approx(1 / 3)
    assert summary["mon_metrics"]["normalized_psr"] == pytest.approx(2 / 3)
    assert summary["navigation_efficiency"]["max_steps_rate"] == pytest.approx(1 / 3)
    assert summary["frame_timing"]["mean_frame_time_ms"] == pytest.approx(25.0)
    assert summary["frame_timing"]["median_frame_time_ms"] == pytest.approx(25.0)
    assert summary["frame_timing"]["p95_frame_time_ms"] == pytest.approx(38.5)


def test_failed_subtask_retains_lengths_but_has_zero_spl():
    metric = _subtask(0, 9.0, 3.0, [5.0])
    assert metric["goat_sub-task_path_length"] == 9.0
    assert metric["goat_sub-task_shortest_path_length"] == 3.0
    assert metric["goat_sub-task_spl"] == 0.0
    assert metric["goat_sub-task_path_efficiency"] == pytest.approx(1 / 3)


def test_zero_length_success_has_unit_efficiency():
    metric = _subtask(1, 0.0, 0.0, [1.0])
    record = _episode([metric], [1.0])
    assert metric["goat_sub-task_path_efficiency"] == 1.0
    assert record["mon_episode_metrics"]["spl"] == 1.0


def test_unreachable_failed_subtask_preserves_exact_success_weighted_metrics(tmp_path):
    unreachable = {
        "goat_distance_to_sub-goal": math.inf,
        "goat_sub-task_success": 0.0,
        "goat_sub-task_spl": math.nan,
    }
    unreachable.update(
        build_subtask_telemetry(
            unreachable,
            actual_path_length=9.0,
            shortest_path_length=math.inf,
            steps=1,
            frame_times_ms=[20.0],
        )
    )
    assert unreachable["goat_distance_to_sub-goal"] is None
    assert unreachable[DISTANCE_TO_GOAL_STATUS_KEY] == "unreachable"
    assert unreachable["goat_sub-task_shortest_path_length"] is None
    assert unreachable["goat_sub-task_shortest_path_status"] == "unreachable"
    assert unreachable["goat_sub-task_spl"] == 0.0
    assert unreachable["goat_sub-task_path_efficiency"] is None
    assert unreachable["goat_sub-task_path_stretch"] is None
    assert unreachable["goat_sub-task_excess_path_length"] is None

    subtasks = [
        _subtask(1, 2.0, 1.0, [10.0]),
        unreachable,
        _subtask(1, 4.0, 2.0, [30.0]),
    ]
    record = _episode(
        subtasks,
        [10.0, 20.0, 30.0],
        ["verified_target", "max_steps", "verified_target"],
    )
    assert record["mon_episode_metrics"]["episode_success"] == 0
    assert record["mon_episode_metrics"]["spl"] == 0.0
    assert record["mon_episode_metrics"]["normalized_psr"] == pytest.approx(2 / 3)
    assert record["mon_episode_metrics"]["normalized_pespl"] == pytest.approx(1 / 3)
    assert record["navigation_efficiency"]["shortest_path_length"] is None
    assert record["navigation_efficiency"]["path_efficiency"] is None
    assert record["navigation_efficiency"]["shortest_path_coverage"] == pytest.approx(2 / 3)
    assert record["navigation_efficiency"]["unreachable_subtask_count"] == 1
    assert record["navigation_efficiency"]["finite_only_actual_path_length"] == 6.0
    assert record["navigation_efficiency"]["finite_only_shortest_path_length"] == 3.0
    assert record["navigation_efficiency"]["finite_only_path_efficiency"] == 0.5

    summary = summarize_records(
        {"scene_0": record}, frame_times_ms=[10.0, 20.0, 30.0]
    )
    assert summary["schema_version"] == MON_METRICS_SCHEMA_VERSION
    assert summary["mon_metrics"]["sr"] == 0.0
    assert summary["mon_metrics"]["spl"] == 0.0
    assert summary["mon_metrics"]["normalized_psr"] == pytest.approx(2 / 3)
    assert summary["mon_metrics"]["normalized_pespl"] == pytest.approx(1 / 3)
    efficiency = summary["navigation_efficiency"]
    assert efficiency["total_shortest_path_length"] is None
    assert efficiency["aggregate_path_efficiency"] is None
    assert efficiency["mean_episode_path_efficiency"] is None
    assert efficiency["mean_subtask_path_efficiency"] is None
    assert efficiency["shortest_path_coverage"] == pytest.approx(2 / 3)
    assert efficiency["unreachable_subtask_count"] == 1
    assert efficiency["finite_only_aggregate_path_efficiency"] == 0.5
    assert efficiency["finite_only_mean_episode_path_efficiency"] == 0.5

    output = tmp_path / "unreachable.json"
    write_json_strict(output, {"scene_0": record, "summary": summary})
    encoded = output.read_text(encoding="utf-8")
    assert "NaN" not in encoded
    assert "Infinity" not in encoded
    assert json.loads(encoded)["scene_0"]["metrics"][1][
        "goat_sub-task_shortest_path_length"
    ] is None

    results = tmp_path / "run" / "results_scene" / "experiment"
    results.mkdir(parents=True)
    write_json_strict(results / "per_episode_metrics.json", {"scene_0": record})
    with (results / "timing_log.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
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
        for index, value in enumerate((10.0, 20.0, 30.0)):
            writer.writerow(
                {
                    "scene_id": "scene",
                    "episode_id": "0",
                    "subtask_index": str(index),
                    "frame_index": "1",
                    "dt_total_ms": str(value),
                }
            )
    cumulative = aggregate_run_artifacts(tmp_path / "run")
    write_json_strict(results / "cumulative_metrics.json", cumulative)
    assert cumulative["mon_metrics"]["normalized_pespl"] == pytest.approx(1 / 3)

    record["metrics"][1]["goat_sub-task_spl"] = 0.25
    with pytest.raises(MonMetricError, match="zero persisted SPL"):
        summarize_records({"scene_0": record})


def test_successful_unreachable_subtask_fails_loudly():
    metric = {
        "goat_distance_to_sub-goal": math.inf,
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


def test_unreachable_subtask_rejects_nonzero_spl_and_invalid_distance():
    base = {
        "goat_distance_to_sub-goal": math.inf,
        "goat_sub-task_success": 0.0,
        "goat_sub-task_spl": 0.5,
    }
    with pytest.raises(MonMetricError, match="zero or NaN"):
        build_subtask_telemetry(
            base,
            actual_path_length=1.0,
            shortest_path_length=math.inf,
            steps=1,
            frame_times_ms=[1.0],
        )

    base["goat_sub-task_spl"] = math.nan
    base["goat_distance_to_sub-goal"] = math.nan
    with pytest.raises(MonMetricError, match=r"finite or \+inf"):
        build_subtask_telemetry(
            base,
            actual_path_length=1.0,
            shortest_path_length=math.inf,
            steps=1,
            frame_times_ms=[1.0],
        )


def test_finite_only_episode_mean_includes_partial_coverage_episode():
    unreachable = {
        "goat_distance_to_sub-goal": math.inf,
        "goat_sub-task_success": 0.0,
        "goat_sub-task_spl": math.nan,
    }
    unreachable.update(
        build_subtask_telemetry(
            unreachable,
            actual_path_length=8.0,
            shortest_path_length=math.inf,
            steps=1,
            frame_times_ms=[1.0],
        )
    )
    partial = _episode(
        [_subtask(1, 4.0, 1.0, [1.0]), unreachable],
        [1.0, 1.0],
        ["verified_target", "max_steps"],
    )
    complete = _episode([_subtask(1, 1.0, 1.0, [1.0])], [1.0])
    summary = summarize_records(
        {"scene_0": partial, "scene_1": complete},
        frame_times_ms=[1.0, 1.0, 1.0],
    )
    assert summary["navigation_efficiency"][
        "finite_only_mean_episode_path_efficiency"
    ] == pytest.approx(0.625)


@pytest.mark.parametrize(
    "actual,shortest",
    [
        (math.nan, 1.0),
        (math.inf, 1.0),
        (-1.0, 1.0),
        (1.0, math.nan),
        (1.0, -math.inf),
        (1.0, -1.0),
    ],
)
def test_invalid_lengths_fail_loudly(actual, shortest):
    metric = {
        "goat_distance_to_sub-goal": 1.0,
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


def test_frame_step_mismatch_fails_loudly():
    metric = {
        "goat_sub-task_success": 1.0,
        "goat_sub-task_spl": 1.0,
    }
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


def test_nonstandard_json_constants_are_rejected_on_load(tmp_path):
    path = tmp_path / "metrics.json"
    path.write_text('{"scene_0": {"value": NaN}}', encoding="utf-8")
    with pytest.raises(MonMetricError, match="non-standard JSON constant"):
        merge_episode_metric_files([path])


def test_run_artifact_aggregation_and_strict_json(tmp_path):
    subtasks = [_subtask(1, 2.0, 1.0, [10.0, 20.0])]
    record = _episode(subtasks, [10.0, 20.0])
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
        writer.writerow(
            {
                "scene_id": "scene",
                "episode_id": "0",
                "subtask_index": "0",
                "frame_index": "1",
                "dt_total_ms": "10.0",
            }
        )
        writer.writerow(
            {
                "scene_id": "scene",
                "episode_id": "0",
                "subtask_index": "0",
                "frame_index": "2",
                "dt_total_ms": "20.0",
            }
        )

    summary = aggregate_run_artifacts(tmp_path)
    assert summary["counts"]["completed_episodes"] == 1
    assert summary["mon_metrics"]["sr"] == 1.0
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
    assert entrypoint.count("metric_summary = _finite_metric_summary(") == 2


def test_environment_allows_only_positive_infinite_shortest_path():
    root = Path(__file__).resolve().parents[2]
    environment = (
        root
        / "Navigation/runtime/lagmemo/src/lagmemo/lagmemo/env/habitat_lagmemo_env.py"
    ).read_text(encoding="utf-8")
    assert "np.isposinf(shortest_value)" in environment
    assert "np.isneginf(shortest_value)" in environment
    assert "shortest_path_status=unreachable" in environment
    assert "success != 0.0" in environment

    run_goat = (root / "Exp/scripts/run_goat.py").read_text(encoding="utf-8")
    assert "MON_METRICS_SCHEMA_VERSION" in run_goat
    assert '"unreachable_shortest_path_convention"' in run_goat
