"""Reject incomplete traces and environment claims using independent run records."""

import hashlib
import importlib.metadata
import json
import shutil
import sys

import pytest

from scripts.summarize import summarize_runs
from tests.experiments.test_result_summary import ROOT, fixture, change_protocol, mutate_json, write


def with_recordings(tmp_path, episode=0):
    run = fixture(tmp_path, episode)
    runtime = {"mapping_python": "/map/bin/python", "navigation_python": "/nav/bin/python",
               "mapping_gpu": episode * 2, "navigation_gpu": episode * 2 + 1}
    change_protocol(run, "runtime", runtime)
    plan = json.loads((run / "launch.json").read_text())
    plan["scenes"][0]["scene_index"] = 0
    write(run / "launch.json", plan)
    provenance = json.loads((run / "provenance.json").read_text())
    provenance["artifact_schemas"] = {"environment": "pbsp_environment_v1", "trajectory": "pbsp_trajectory_v1"}
    provenance["environment_locks"] = {}
    (run / "environment_locks").mkdir()
    for role in ("mapping", "navigation"):
        data = (ROOT / "envs" / role / "linux-64.lock.json").read_bytes()
        (run / "environment_locks" / (role + ".json")).write_bytes(data)
        digest = hashlib.sha256(data).hexdigest()
        provenance["environment_locks"][role] = digest
        lock = json.loads(data)
        report = {"schema_version": "pbsp_environment_v1", "role": role,
                  "scene_id": "sceneA" if role == "navigation" else None,
                  "launch_source_commit": plan["source_commit"],
                  "configuration_sha256": plan["configuration_sha256"], "identity_sha256": plan["identity_sha256"],
                  "python_executable": runtime[role + "_python"], "cuda_visible_devices": str(runtime[role + "_gpu"]),
                  "user_site_enabled": False, "lock_sha256": digest,
                  "python_version": ".".join(map(str, lock["python"])) + ".0",
                  "packages": [{"name": item.get("installed_distribution", item["name"]),
                                "version": item["version"], "location": f"/{role}/site-packages"}
                               for item in lock["packages"]]}
        write(run / ("mapping_environment.json" if role == "mapping" else "navigation_environment_0.json"), report)
    write(run / "provenance.json", provenance)
    directory = run / "results_sceneA/native/episodes/sceneA" / str(episode)
    directory.mkdir(parents=True)
    write(directory / "identity.json", {"schema_version": "pbsp_trajectory_v1", "identity": plan["identities"][0],
          "coordinate_frame": "habitat_world_xyz_y_up", "pose_format": "qw_qx_qy_qz_x_y_z",
          "pose_observed": "before_action", "action_space": {"STOP": 0, "MOVE_FORWARD": 1}})
    raw = json.loads((run / "results_sceneA/native/per_episode_metrics.json").read_text())[f"sceneA_{episode}"]
    rows = [{"schema_version": "pbsp_trajectory_v1", "scene_id": "sceneA", "episode_id": str(episode),
             "frame_index": index + 1, "subtask_index": index, "task_step": 1,
             "action_id": 0, "action_name": "STOP", "is_stop": True,
             "stop_reason": raw["stop_reasons"][index], "world_pose": [1., 0., 0., 0., index * .2, 0., 0.]}
            for index in range(2)]
    (directory / "trajectory.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    return run, directory


def test_current_recordings_are_checked_and_merge_ignores_machine_paths_and_gpu_ids(tmp_path):
    first, _ = with_recordings(tmp_path)
    second, _ = with_recordings(tmp_path, 1)
    row = summarize_runs([first])["rows"][0]
    assert row["environment_identity_checked"] is True
    assert row["trajectory_checked"] == {"episodes": 1, "frames": 2}
    merged = summarize_runs([first, second], merge=True)["rows"][0]
    assert merged["coverage"]["kind"] == "full_dataset"
    assert merged["environment_identity_checked"] is True
    assert merged["trajectory_checked"] == {"episodes": 2, "frames": 4}


@pytest.mark.parametrize("case", ["missing_environment", "interpreter", "package", "lock", "source", "user_site"])
def test_bad_worker_environment_cannot_produce_a_summary(tmp_path, case):
    run, _ = with_recordings(tmp_path)
    path = run / "mapping_environment.json"
    if case == "missing_environment":
        path.unlink()
    elif case == "lock":
        (run / "environment_locks/mapping.json").write_text("{}")
    else:
        data = json.loads(path.read_text())
        if case == "interpreter": data["python_executable"] = "/foreign/bin/python"
        elif case == "package": data["packages"][0]["version"] = "0"
        elif case == "source": data["launch_source_commit"] = "b" * 40
        else: data["user_site_enabled"] = True
        write(path, data)
    with pytest.raises(ValueError):
        summarize_runs([run])


@pytest.mark.parametrize("case", ["missing", "identity", "gap", "task", "action", "pose", "stop", "unfinished"])
def test_bad_trajectory_cannot_produce_a_summary(tmp_path, case):
    run, directory = with_recordings(tmp_path)
    path = directory / "trajectory.jsonl"
    if case == "missing":
        path.unlink()
    elif case == "identity":
        mutate_json(directory / "identity.json", lambda data: data["identity"].update(runtime_episode_id="other"))
    else:
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if case == "gap": rows[0]["frame_index"] = 2
        elif case == "task": rows[1]["subtask_index"] = 0
        elif case == "action": rows[0]["action_id"] = 1
        elif case == "pose": rows[0]["world_pose"][4] = float("nan")
        elif case == "stop": rows[0]["stop_reason"] = "max_steps"
        else: rows.pop()
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError):
        summarize_runs([run])


def test_an_extra_installed_version_changes_the_merge_protocol(tmp_path):
    first, _ = with_recordings(tmp_path)
    second, _ = with_recordings(tmp_path, 1)
    mutate_json(second / "mapping_environment.json", lambda data: data["packages"].append(
        {"name": "extra-backend", "version": "1.0", "location": "/another/environment"}))
    assert len(summarize_runs([first, second])["rows"]) == 2
    with pytest.raises(ValueError, match="different code"):
        summarize_runs([first, second], merge=True)


@pytest.mark.parametrize("vendor_first", [False, True])
def test_duplicate_metadata_matches_real_python_lookup_and_remains_portable(tmp_path, monkeypatch, vendor_first):
    run, _ = with_recordings(tmp_path)
    path = run / "navigation_environment_0.json"
    report = json.loads(path.read_text())
    expected = next(row["version"] for row in report["packages"] if row["name"] == "tomli")
    site = tmp_path / "metadata"
    vendor = site / "setuptools/_vendor"
    for directory, version in ((site, expected), (vendor, "2.0.1")):
        metadata = directory / f"tomli-{version}.dist-info"
        metadata.mkdir(parents=True)
        (metadata / "METADATA").write_text(f"Metadata-Version: 2.1\nName: tomli\nVersion: {version}\n")
    paths = [str(vendor), str(site)] if vendor_first else [str(site), str(vendor)]
    with monkeypatch.context() as patch:
        patch.setattr(sys, "path", paths)
        assert importlib.metadata.version("tomli") == ("2.0.1" if vendor_first else expected)
    for row in report["packages"]:
        row["location"] = str(site)
    report["source_search_paths"] = paths
    report["packages"].append({"name": "TOMLI", "version": "2.0.1", "location": str(vendor)})
    write(path, report)
    shutil.rmtree(site)
    if vendor_first:
        with pytest.raises(ValueError, match="differs from the lock: tomli"):
            summarize_runs([run])
    else:
        result = summarize_runs([run])
        assert result["rows"][0]["environment_identity_checked"] is True
        assert json.loads(path.read_text()) == report


@pytest.mark.parametrize("case", ["missing_paths", "unknown_location", "same_location"])
def test_duplicate_versions_require_unambiguous_recorded_precedence(tmp_path, case):
    run, _ = with_recordings(tmp_path)
    path = run / "navigation_environment_0.json"
    report = json.loads(path.read_text())
    location = "/navigation/site-packages"
    vendor = location + "/setuptools/_vendor"
    if case != "missing_paths":
        report["source_search_paths"] = [location, vendor]
    duplicate = location if case == "same_location" else "/unrecorded" if case == "unknown_location" else vendor
    report["packages"].append({"name": "tomli", "version": "2.0.1", "location": duplicate})
    write(path, report)
    with pytest.raises(ValueError, match="package (precedence|metadata)"):
        summarize_runs([run])


def test_active_version_precedence_changes_merge_protocol_for_unlocked_packages(tmp_path):
    first, _ = with_recordings(tmp_path)
    second, _ = with_recordings(tmp_path, 1)
    for run, reverse in ((first, False), (second, True)):
        path = run / "navigation_environment_0.json"
        report = json.loads(path.read_text())
        paths = ["/navigation/site-packages", "/navigation/site-packages/vendor"]
        report["source_search_paths"] = paths[::-1] if reverse else paths
        report["packages"].extend({"name": "extra-backend", "version": version, "location": location}
                                  for version, location in zip(("1.0", "2.0"), paths))
        write(path, report)
    assert len(summarize_runs([first, second])["rows"]) == 2
    with pytest.raises(ValueError, match="different code"):
        summarize_runs([first, second], merge=True)
