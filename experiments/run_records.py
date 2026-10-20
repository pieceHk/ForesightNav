"""Cross-check environment and trajectory artifacts against a native run plan."""

from dataclasses import asdict
import hashlib
import json
import math
import re


def environment_identity(directory, plan, provenance, read_json):
    signatures, expected_files = {}, {"mapping_environment.json"}
    selected = {item["scene_id"]: item["scene_index"] for item in plan["scenes"]}
    if (len(selected) != len(plan["scenes"]) or set(selected) != {item["scene_id"] for item in plan["identities"]}
            or any(type(index) is not int or index < 0 for index in selected.values())
            or len(set(selected.values())) != len(selected)):
        raise ValueError("Environment worker selection differs from the selected episode scenes")
    expected_files.update(f"navigation_environment_{index}.json" for index in selected.values())
    if {path.name for path in directory.glob("*_environment*.json")} != expected_files:
        raise ValueError("Environment inventory files differ from the selected workers")
    for name in sorted(expected_files):
        report = read_json(directory / name)
        role = "mapping" if name == "mapping_environment.json" else "navigation"
        if report["schema_version"] != "pbsp_environment_v1" or report["role"] != role:
            raise ValueError("Environment inventory schema or role differs")
        if role == "mapping" and report["scene_id"] is not None:
            raise ValueError("Mapping environment inventory must cover the whole run")
        if role == "navigation" and name != f"navigation_environment_{selected.get(report['scene_id'])}.json":
            raise ValueError("Navigation environment scene differs from the run selection")
        runtime = plan["configuration"]["runtime"]
        if (report["launch_source_commit"] != plan["source_commit"]
                or report["configuration_sha256"] != plan["configuration_sha256"]
                or report["identity_sha256"] != plan["identity_sha256"]
                or report["python_executable"] != runtime[role + "_python"]
                or report["cuda_visible_devices"] != str(runtime[role + "_gpu"])
                or report["user_site_enabled"] is not False):
            raise ValueError("Environment inventory differs from the launch identity or interpreter")
        lock_path = directory / "environment_locks" / (role + ".json")
        lock_hash = hashlib.sha256(lock_path.read_bytes()).hexdigest()
        if lock_hash != report["lock_sha256"] or lock_hash != provenance["environment_locks"][role]:
            raise ValueError("Recorded environment lock identity differs")
        lock = read_json(lock_path)
        if list(map(int, report["python_version"].split(".")[:2])) != lock["python"]:
            raise ValueError("Recorded Python differs from its environment lock")
        packages = set()
        for package in report["packages"]:
            if not all(isinstance(package[key], str) and package[key] for key in ("name", "version", "location")):
                raise ValueError("Invalid environment package record")
            packages.add((re.sub(r"[-_.]+", "-", package["name"]).lower(), package["version"]))
        for item in lock["packages"]:
            name = re.sub(r"[-_.]+", "-", item.get("installed_distribution", item["name"])).lower()
            if {version for package, version in packages if package == name} != {item["version"]}:
                raise ValueError(f"Recorded installed package differs from the lock: {name}")
        signature = {"python_version": report["python_version"], "lock_sha256": lock_hash,
                     "packages": sorted(packages)}
        if role in signatures and signatures[role] != signature:
            raise ValueError("Installed navigation environment changed between scenes")
        signatures[role] = signature
    return signatures


def validate_trajectories(directory, identities, raw, times, read_json):
    seen, frames_checked = set(), 0
    for path in sorted(directory.glob("results_*/**/trajectory.jsonl")):
        metadata = read_json(path.parent / "identity.json")
        identity = metadata["identity"]
        key = (identity["scene_id"], identity["runtime_episode_id"])
        if key not in identities or key in seen:
            raise ValueError("Trajectory episode is repeated or outside the selected identities")
        expected = json.loads(json.dumps(asdict(identities[key])))
        if (metadata["schema_version"] != "pbsp_trajectory_v1" or identity != expected
                or metadata["coordinate_frame"] != "habitat_world_xyz_y_up"
                or metadata["pose_format"] != "qw_qx_qy_qz_x_y_z"
                or metadata["pose_observed"] != "before_action"):
            raise ValueError("Trajectory identity or pose convention differs")
        space = metadata["action_space"]
        if "STOP" not in space or not all(isinstance(name, str) and type(code) is int for name, code in space.items()):
            raise ValueError("Invalid trajectory action space")
        record = raw[f"{key[0]}_{key[1]}"]
        timing = sorted(times[key])
        task_steps, stopped, count = {}, set(), 0
        with path.open(encoding="utf-8") as stream:
            for count, line in enumerate(stream, 1):
                def reject(value):
                    raise ValueError(f"Non-finite trajectory value: {value}")
                row = json.loads(line, parse_constant=reject)
                task = row["subtask_index"]
                pose = row["world_pose"]
                if (row["schema_version"] != "pbsp_trajectory_v1" or (row["scene_id"], row["episode_id"]) != key
                        or type(row["frame_index"]) is not int or row["frame_index"] != count
                        or type(task) is not int or count > len(timing) or task != timing[count - 1][1]
                        or task in stopped or type(row["task_step"]) is not int
                        or row["task_step"] != task_steps.get(task, 0) + 1):
                    raise ValueError("Trajectory frame/task coverage differs from raw timing")
                if (not isinstance(pose, list) or len(pose) != 7
                        or not all(type(value) in (float, int) and math.isfinite(value) for value in pose)):
                    raise ValueError("Invalid trajectory world pose")
                if (type(row["action_id"]) is not int or row["action_name"] not in space
                        or row["action_id"] != space[row["action_name"]]
                        or type(row["is_stop"]) is not bool
                        or row["is_stop"] != (row["action_id"] == space["STOP"])):
                    raise ValueError("Trajectory action differs from its recorded action space")
                if row["is_stop"]:
                    if row["stop_reason"] != record["stop_reasons"][task]:
                        raise ValueError("Trajectory STOP reason differs from raw task metrics")
                    stopped.add(task)
                elif row["stop_reason"] is not None:
                    raise ValueError("Non-STOP trajectory action carries a stop reason")
                task_steps[task] = row["task_step"]
        if count != record["total_num_steps"] or stopped != set(range(len(identities[key].ordered_categories))):
            raise ValueError("Trajectory is incomplete for the selected ordered episode")
        frames_checked += count
        seen.add(key)
    if seen != set(identities):
        raise ValueError("Missing selected episode trajectory")
    return {"episodes": len(seen), "frames": frames_checked}
