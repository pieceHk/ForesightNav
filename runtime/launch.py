"""Own native worker processes, immutable run inputs and result aggregation."""

from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time
import uuid

from runtime.bootstrap import worker_environment
from runtime.protocol import request


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def status(plan, state, **fields):
    write_json(
        plan.run_directory / "status.json",
        {
            "status": state,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **fields,
        },
    )


def source_info(root):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True)

    try:
        return {
            "commit": git("rev-parse", "HEAD").strip(),
            "status": git("status", "--porcelain").splitlines(),
            "patch": git("diff", "--binary", "HEAD"),
        }
    except subprocess.CalledProcessError:
        return {
            "commit": "unknown",
            "status": ["Git metadata unavailable"],
            "patch": "",
        }


def worker_command(plan, role, *, scene=None):
    runtime = plan.experiment.runtime
    if role == "monitor":
        return [
            str(runtime.navigation_python),
            "-m",
            "runtime.monitor",
            "--output",
            str(plan.run_directory / "resource_timeline.jsonl"),
            "--root-pid",
            str(os.getpid()),
            "--gpus",
            f"{runtime.mapping_gpu},{runtime.navigation_gpu}",
        ]
    python = runtime.mapping_python if role == "mapping" else runtime.navigation_python
    command = [
        str(python),
        "-m",
        f"runtime.{role}_worker",
        "--run-plan",
        str(plan.run_directory / "launch.json"),
    ]
    if role == "navigation":
        command += ["--scene", scene]
    return command


def terminate_group(process, timeout=30):
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=10)


def wait_for_mapping(process, socket_path, timeout=300):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"Mapping worker exited before readiness: {process.returncode}"
            )
        try:
            response = request(socket_path, {"cmd": "status"}, timeout_s=1)
            if response.get("ok"):
                return
        except (OSError, ValueError):
            pass
        time.sleep(0.2)
    raise TimeoutError(f"Mapping worker readiness timed out: {socket_path}")


@contextmanager
def managed_process(plan, role, *, scene=None):
    label = role if scene is None else f"{role}_{scene}"
    command = worker_command(plan, role, scene=scene)
    env = worker_environment(plan, "navigation" if role == "monitor" else role)
    completed = False
    with (plan.run_directory / "logs" / f"{label}.log").open("w") as log:
        process = subprocess.Popen(
            command,
            cwd=plan.resources.root,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            yield process
            completed = True
        finally:
            if role == "mapping" and process.poll() is None:
                try:
                    response = request(
                        plan.socket_path, {"cmd": "shutdown"}, timeout_s=120
                    )
                    if response.get("ok"):
                        process.wait(timeout=30)
                except (OSError, ValueError, subprocess.TimeoutExpired):
                    pass
            terminate_group(process)
            write_json(
                plan.run_directory / f"{label}_process.json",
                {
                    "command": command,
                    "pid": process.pid,
                    "returncode": process.returncode,
                    "scene": scene,
                    "role": role,
                },
            )
            if completed and process.returncode != 0:
                raise RuntimeError(f"{label} worker exited with {process.returncode}")


@contextmanager
def termination_scope():
    def stop(signum, _frame):
        raise SystemExit(128 + signum)

    previous = {
        name: signal.signal(name, stop) for name in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        yield
    finally:
        for name, handler in previous.items():
            signal.signal(name, handler)


def allocate_run(plan, *, tag="", resource_checks=None):
    if not re.fullmatch(r"[A-Za-z0-9_-]*", tag):
        raise ValueError("Run tag may contain letters, digits, underscores and hyphens")
    if resource_checks is not None:
        import hashlib

        model_manifest = (plan.resources.root / "assets/models.json").read_bytes()
        if hashlib.sha256(model_manifest).hexdigest() != resource_checks["models"]["manifest_sha256"]:
            raise ValueError("Model manifest changed after the resource check")
    if plan.run_directory is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        name = f"{stamp}_{plan.experiment.name}" + (f"_{tag}" if tag else "")
        plan = replace(plan, run_directory=plan.experiment.runtime.outputs / name)
    plan.run_directory.mkdir(parents=True, exist_ok=False)
    (plan.run_directory / "logs").mkdir()
    (plan.run_directory / "artifacts").mkdir()
    write_json(plan.run_directory / "launch.json", plan.snapshot())
    import yaml

    (plan.run_directory / "resolved_config.yaml").write_text(
        yaml.safe_dump(
            {
                "experiment": plan.experiment.snapshot(),
                "mapping_model": plan.model_specification,
            },
            sort_keys=False,
        )
    )
    source = source_info(plan.resources.root)
    (plan.run_directory / "source.patch").write_text(source.pop("patch"))
    write_json(
        plan.run_directory / "provenance.json",
        {
            **source,
            "configuration_sha256": plan.configuration_sha256,
            "identity_sha256": plan.identity_sha256,
            "selected_episodes": len(plan.identities),
            "selected_tasks": len(plan.identities)
            * plan.experiment.dataset.expected_tasks_per_episode,
            "mapping_environment": worker_environment(plan, "mapping", {}),
            "navigation_environment": worker_environment(plan, "navigation", {}),
            "in_place_resume_supported": False,
            **({"resource_checks": resource_checks} if resource_checks is not None else {}),
        },
    )
    if resource_checks is not None:
        (plan.run_directory / "model_manifest.json").write_bytes(model_manifest)
    shutil.copyfile(
        plan.resources.data
        / plan.experiment.dataset.root
        / plan.experiment.dataset.manifest,
        plan.run_directory / "dataset_manifest.json",
    )
    return plan


def launch(plan, *, tag="", resource_checks=None):
    from experiments.metrics import aggregate_run_artifacts

    plan = allocate_run(plan, tag=tag, resource_checks=resource_checks)
    status(plan, "starting")
    print(f"Run: {plan.run_directory}", flush=True)
    try:
        with termination_scope(), managed_process(plan, "monitor"):
            with managed_process(plan, "mapping") as mapping:
                wait_for_mapping(mapping, plan.socket_path)
                for selection in plan.scenes:
                    status(
                        plan,
                        "running",
                        scene=selection.scene_id,
                        episode_ids=selection.episode_ids,
                    )
                    with managed_process(
                        plan, "navigation", scene=selection.scene_id
                    ) as navigation:
                        navigation.wait()
                    result = aggregate_run_artifacts(plan.run_directory)
                    write_json(plan.run_directory / "mon_metrics.json", result)
                    response = request(
                        plan.socket_path, {"cmd": "flush"}, timeout_s=120
                    )
                    if not response.get("ok"):
                        raise RuntimeError(f"Mapping flush failed: {response}")
                expected = len(plan.identities)
                if (
                    result["counts"]["completed_episodes"] != expected
                    or result["counts"]["completed_subtasks"]
                    != expected * plan.experiment.dataset.expected_tasks_per_episode
                ):
                    raise RuntimeError(
                        "Completed metric counts differ from the selected complete episodes"
                    )
            status(plan, "finishing")
        status(plan, "completed", completed_episodes=expected)
    except BaseException as exc:
        status(plan, "failed", error=repr(exc))
        raise
    finally:
        print(plan.run_directory, flush=True)
    return plan.run_directory
