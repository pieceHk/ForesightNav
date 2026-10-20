"""Bind raw MON measurements to completed-run identity and provenance records."""

import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import pytest

from experiments.metrics import build_subtask_telemetry, summarize_episode, summarize_records
from pbsp.types import EpisodeIdentity
from runtime.plan import digest
from scripts.summarize import main, markdown, summarize_runs

ROOT = Path(__file__).resolve().parents[2]


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def fixture(tmp_path, episode=0, *, name=None):
    run = tmp_path / (name or f"run{episode}")
    result = run / "results_sceneA/native"
    result.mkdir(parents=True)
    manifest = {"scene_summary": [{"scene_id": "sceneA", "mon_episode_count": 2}]}
    write(run / "dataset_manifest.json", manifest)
    manifest_hash = hashlib.sha256((run / "dataset_manifest.json").read_bytes()).hexdigest()
    identity = EpisodeIdentity(dataset_manifest_sha256=manifest_hash, scene_id="sceneA", runtime_episode_id=str(episode),
                               source_episode_ids=(f"source-{episode}-0", f"source-{episode}-1"),
                               ordered_categories=("chair", "toilet"))
    dataset = {"name": "toy_2mon", "manifest_sha256": manifest_hash, "source_trace_sha256": "c" * 64,
               "expected_tasks_per_episode": 2, "expected_episodes": 2, "expected_scenes": 1}
    config = {"method": {"name": "full_pbsp"}, "dataset": dataset,
              "evaluation": {"max_steps_per_task": 500}, "seed": 1, "runtime": {"mapping_gpu": episode * 2}}
    plan = {"schema_version": "pbsp_launch_v1", "configuration": config, "model_specification": {"input": "fixed"},
            "source_commit": "a" * 40, "identities": [asdict(identity)],
            "scenes": [{"scene_id": "sceneA", "episode_ids": [str(episode)]}]}
    plan["configuration_sha256"] = digest({"experiment": config, "mapping_model": plan["model_specification"]})
    plan["identity_sha256"] = digest(plan["identities"])
    write(run / "launch.json", plan)
    model_manifest = (ROOT / "assets/models.json").read_bytes()
    (run / "model_manifest.json").write_bytes(model_manifest)
    provenance = {"commit": plan["source_commit"], "status": [], "selected_episodes": 1, "selected_tasks": 2,
                  "configuration_sha256": plan["configuration_sha256"], "identity_sha256": plan["identity_sha256"],
                  "resource_checks": {"models": {"status": "passed", "files": 22,
                                                 "manifest_sha256": hashlib.sha256(model_manifest).hexdigest()},
                                      "habitat_revision": "d" * 40}}
    write(run / "provenance.json", provenance)
    (run / "source.patch").write_text("")
    write(run / "status.json", {"status": "completed", "completed_episodes": 1})
    for name in ("mapping_process.json", "monitor_process.json", "navigation_sceneA_process.json"):
        write(run / name, {"returncode": 0})
    times = [10.0 + episode * 20, 20.0 + episode * 20]
    subtasks = []
    for index, elapsed in enumerate(times):
        success = int(episode == 1 or index == 0)
        metric = {"goat_distance_to_sub-goal": 0.0 if success else 1.0,
                  "goat_sub-task_success": success, "goat_sub-task_spl": success * 0.5}
        metric.update(build_subtask_telemetry(metric, actual_path_length=2.0, shortest_path_length=1.0,
                                             steps=1, frame_times_ms=[elapsed]))
        subtasks.append(metric)
    record = {"metrics": subtasks, "tasks": [{"category": category} for category in identity.ordered_categories],
              "total_num_steps": 2, "stop_reasons": ["verified_target", "verified_target" if episode == 1 else "max_steps"]}
    record.update(summarize_episode(subtasks, total_steps=2, stop_reasons=record["stop_reasons"],
                                    frame_times_ms=times, categories=identity.ordered_categories))
    records = {f"sceneA_{episode}": record}
    write(result / "per_episode_metrics.json", records)
    with (result / "timing_log.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["scene_id", "episode_id", "subtask_index", "frame_index", "dt_total_ms"])
        writer.writeheader()
        for index, elapsed in enumerate(times):
            writer.writerow({"scene_id": "sceneA", "episode_id": episode, "subtask_index": index,
                             "frame_index": index + 1, "dt_total_ms": elapsed})
    write(run / "mon_metrics.json", summarize_records(records, frame_times_ms=times))
    return run


def mutate_json(path, change):
    value = json.loads(path.read_text())
    change(value)
    write(path, value)


def change_protocol(run, field, value):
    plan = json.loads((run / "launch.json").read_text())
    if field == "source_commit":
        plan[field] = value
    else:
        plan["configuration"][field] = value
    plan["configuration_sha256"] = digest({"experiment": plan["configuration"], "mapping_model": plan["model_specification"]})
    write(run / "launch.json", plan)
    mutate_json(run / "provenance.json", lambda data: data.update(commit=plan["source_commit"], configuration_sha256=plan["configuration_sha256"]))


def test_merge_recomputes_exact_denominators_and_full_split_coverage(tmp_path):
    first, second = fixture(tmp_path), fixture(tmp_path, 1)
    compared = summarize_runs([first, second])
    assert [row["coverage"]["kind"] for row in compared["rows"]] == ["selected_episodes"] * 2
    report = summarize_runs([second, first], merge=True)
    row = report["rows"][0]
    assert row["coverage"]["kind"] == "full_dataset"
    assert row["metrics"]["counts"]["completed_subtasks"] == 4
    assert row["metrics"]["mon_metrics"]["sr"] == 0.5
    assert row["metrics"]["mon_metrics"]["psr"] == 0.75
    assert row["metrics"]["mon_metrics"]["pespl_denominator"] == 4
    assert report == summarize_runs([first, second], merge=True)
    assert "| merged | toy_2mon | 2 | full |" in markdown(report)


@pytest.mark.parametrize("field,value", [("seed", 2), ("method", {"name": "another"}), ("source_commit", "b" * 40)])
def test_valid_runs_with_different_protocols_are_compared_but_not_merged(tmp_path, field, value):
    first, second = fixture(tmp_path), fixture(tmp_path, 1)
    change_protocol(second, field, value)
    assert len(summarize_runs([first, second])["rows"]) == 2
    with pytest.raises(ValueError, match="different code"):
        summarize_runs([first, second], merge=True)


def test_overlapping_episodes_and_duplicate_input_paths_are_rejected(tmp_path):
    first, duplicate = fixture(tmp_path), fixture(tmp_path, name="duplicate")
    with pytest.raises(ValueError, match="overlapping episode"):
        summarize_runs([first, duplicate], merge=True)
    with pytest.raises(ValueError, match="distinct"):
        summarize_runs([first, first])


def test_different_checked_model_manifests_cannot_be_merged(tmp_path):
    first, second = fixture(tmp_path), fixture(tmp_path, 1)
    path = second / "model_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["scope"] = "Another checked model declaration"
    write(path, manifest)
    manifest_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    mutate_json(second / "provenance.json", lambda data: data["resource_checks"]["models"].update(manifest_sha256=manifest_hash))
    assert len(summarize_runs([first, second])["rows"]) == 2
    with pytest.raises(ValueError, match="different code"):
        summarize_runs([first, second], merge=True)


@pytest.mark.parametrize("case", ["running", "worker", "task_order", "episode_metric", "aggregate", "identity", "model_manifest"])
def test_incomplete_or_inconsistent_artifacts_cannot_produce_a_summary(tmp_path, case):
    run = fixture(tmp_path)
    result = run / "results_sceneA/native/per_episode_metrics.json"
    if case == "running":
        mutate_json(run / "status.json", lambda data: data.update(status="running"))
    elif case == "worker":
        mutate_json(run / "navigation_sceneA_process.json", lambda data: data.update(returncode=1))
    elif case == "task_order":
        mutate_json(result, lambda data: data["sceneA_0"]["tasks"].reverse())
    elif case == "episode_metric":
        mutate_json(result, lambda data: data["sceneA_0"]["mon_metrics"].update(sr=1))
    elif case == "aggregate":
        mutate_json(run / "mon_metrics.json", lambda data: data["mon_metrics"].update(psr=0.0))
    elif case == "identity":
        mutate_json(run / "launch.json", lambda data: data["identities"][0].update(runtime_episode_id="other"))
    else:
        (run / "model_manifest.json").write_text("different")
    destination = tmp_path / "summary"
    with pytest.raises(ValueError):
        main([str(run), "--output", str(destination)])
    assert not destination.exists()


@pytest.mark.parametrize("case", ["duplicate", "gap", "task", "time"])
def test_timing_rows_must_cover_the_selected_tasks_and_frames(tmp_path, case):
    run = fixture(tmp_path)
    path = run / "results_sceneA/native/timing_log.csv"
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames, list(reader)
    if case == "duplicate":
        rows.append(rows[0])
    elif case == "gap":
        rows.pop()
    elif case == "task":
        rows[0]["subtask_index"] = "1"
    else:
        rows[0]["dt_total_ms"] = "12"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError):
        summarize_runs([run])


def test_dirty_source_is_identified_in_comparison_and_cannot_be_merged(tmp_path):
    first, second = fixture(tmp_path), fixture(tmp_path, 1)
    mutate_json(second / "provenance.json", lambda data: data.update(status=[" M pbsp/agent.py"]))
    (second / "source.patch").write_text("recorded change\n")
    report = summarize_runs([first, second])
    assert report["rows"][1]["source_dirty"] is True
    assert "An asterisk" in markdown(report)
    with pytest.raises(ValueError, match="clean source"):
        summarize_runs([first, second], merge=True)


def test_cli_writes_a_new_summary_without_modifying_the_run(tmp_path):
    run = fixture(tmp_path)
    before = (run / "mon_metrics.json").read_bytes()
    destination = tmp_path / "summary"
    main([str(run), "--output", str(destination)])
    assert json.loads((destination / "summary.json").read_text())["mode"] == "compare"
    assert "selection of 2" in (destination / "summary.md").read_text()
    assert (run / "mon_metrics.json").read_bytes() == before
    with pytest.raises(FileExistsError):
        main([str(run), "--output", str(destination)])
