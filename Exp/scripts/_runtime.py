"""Shared process and artifact lifecycle for 3DONav experiment runners."""

import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("TDONAV_ROOT", str(ROOT))
MAPPING_ENGINE = ROOT / "Mapping" / "runtime" / "embodied_splat"
NAVIGATION_ENGINE = ROOT / "Navigation" / "runtime" / "lagmemo"


def load_toml(path: Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def expand(value: str) -> str:
    return os.path.expandvars(os.path.expanduser(value))


def create_run(profile: dict, suffix: str = "") -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = profile["name"] + (f"_{suffix}" if suffix else "")
    run = ROOT / "Exp" / "runs" / f"{timestamp}_{name}"
    run.mkdir(parents=True)
    (run / "logs").mkdir()
    (run / "artifacts").mkdir()
    return run


def socket_for_run(run: Path) -> Path:
    """Return a short, deterministic AF_UNIX socket path for ``run``."""
    digest = hashlib.sha256(str(run).encode("utf-8")).hexdigest()[:16]
    return Path("/tmp") / f"3donav_{digest}.sock"


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_status(run: Path, status: str, **fields) -> None:
    payload = {"status": status, "timestamp": datetime.now().astimezone().isoformat(), **fields}
    write_json(run / "status.json", payload)


def git_head() -> str:
    result = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def send_socket(socket_path: Path, payload: dict, timeout_s: float = 120.0) -> dict:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(timeout_s)
        client.connect(str(socket_path))
        client.sendall(json.dumps(payload).encode() + b"\n")
        data = b""
        while b"\n" not in data:
            block = client.recv(65536)
            if not block:
                break
            data += block
    return json.loads(data.split(b"\n", 1)[0].decode())


def wait_for_server(process: subprocess.Popen, socket_path: Path, timeout_s: int = 300) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Mapping process exited with {process.returncode}")
        if socket_path.exists():
            try:
                response = send_socket(socket_path, {"cmd": "status"}, timeout_s=5)
                if response.get("ok"):
                    return
            except (ConnectionError, OSError, ValueError):
                pass
        time.sleep(2)
    raise TimeoutError(f"Mapping server did not become ready: {socket_path}")


def terminate_group(process: subprocess.Popen, timeout_s: int = 30) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


@contextmanager
def resource_monitor(run: Path, gpus: tuple):
    command = [
        sys.executable,
        str(ROOT / "Exp/scripts/monitor.py"),
        "--output",
        str(run / "resource_timeline.jsonl"),
        "--root-pid",
        str(os.getpid()),
        "--gpus",
        ",".join(str(gpu) for gpu in gpus),
    ]
    process = subprocess.Popen(command, start_new_session=True)
    try:
        yield process
    finally:
        terminate_group(process, timeout_s=5)


@contextmanager
def mapping_server(runtime: dict, profile: dict, run: Path, socket_path: Path, es_gpu: int):
    python = expand(runtime["python"]["mapping"])
    if not MAPPING_ENGINE.is_dir():
        raise FileNotFoundError(MAPPING_ENGINE)
    env = os.environ.copy()
    mapping = profile["mapping"]
    # A caller's shell must not re-enable per-episode Gaussian checkpoints in
    # this live-only runtime. PNG/BEV visualization follows a separate path.
    env.pop("ES_EXPORT_FINAL_MAP_DIR", None)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(es_gpu),
            "TDONAV_ROOT": str(ROOT),
            "ES_ROOT": str(MAPPING_ENGINE),
            "ES_SNAP_LIVE": str(run / "artifacts" / "mapping_tmp"),
            "ES_FINAL_BEV_DIR": str(run / "artifacts" / "mapping_final"),
            "ES_MAX_GAUSSIANS": "0",
            "ES_MAX_KEYFRAMES": "0",
            "ES_FRAME_CHANGE_RATIO": str(mapping["keyframe_threshold"]),
            "ES_FRAME_PIXEL_DELTA": str(mapping["pixel_delta"]),
            "ES_WARMUP_KEYFRAMES": str(mapping["warmup_keyframes"]),
            "ES_REBUILD_INTERVAL": str(mapping["rebuild_interval"]),
            "ES_RAW_FRAME_BUFFER": str(mapping.get("raw_frame_buffer", 2)),
            "ES_OFFLOAD_FASTSAM_AFTER_REBUILD": "0",
            "ES_QUERY_MIN_SIMILARITY": str(mapping["query_similarity"]),
            "ES_QUERY_SUPPORT_RADIUS": str(mapping["query_support_radius_m"]),
            "ES_QUERY_MIN_SUPPORT": str(mapping["query_minimum_support"]),
            "ES_QUERY_NMS_RADIUS": str(mapping["query_nms_radius_m"]),
            "ES_QUERY_POOL_SIZE": str(mapping.get("query_pool_size", 2000)),
            "ES_QUERY_SUPPORT_RERANK_WEIGHT": str(
                mapping.get("query_support_rerank_weight", 0.0)
            ),
            # The online mapper consumes in-process Gaussian state. Periodic P3
            # snapshots are offline-only, unused here, and can exhaust /nvme.
            "ES_DISABLE_P3_SNAPSHOT": "1",
            "ES_ENABLE_BEV": "0",
            "MALLOC_ARENA_MAX": "2",
            "MALLOC_TRIM_THRESHOLD_": "131072",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "PYTHONUNBUFFERED": "1",
        }
    )
    if "hf_home" in runtime.get("shared", {}):
        env["HF_HOME"] = expand(runtime["shared"]["hf_home"])
        env["HF_HUB_CACHE"] = str(Path(env["HF_HOME"]) / "hub")
        env["HF_HUB_OFFLINE"] = "1"
    pythonpath = [ROOT / "Mapping" / "src", MAPPING_ENGINE, MAPPING_ENGINE / "src", MAPPING_ENGINE / "src" / "third_party"]
    env["PYTHONPATH"] = os.pathsep.join(map(str, pythonpath)) + os.pathsep + env.get("PYTHONPATH", "")
    log = (run / "logs" / "mapping.log").open("w", encoding="utf-8")
    command = [
        python,
        "-m",
        "tdonav_mapping.adapters.embodied_splat_server",
        "--socket_path",
        str(socket_path),
        "--device",
        "cuda",
        "--top_k",
        str(profile["mapping"]["top_k"]),
        "--live_mode",
        "--live_only",
    ]
    process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        wait_for_server(process, socket_path)
        yield process
    finally:
        try:
            if process.poll() is None and socket_path.exists():
                send_socket(socket_path, {"cmd": "shutdown"}, timeout_s=120)
        except (ConnectionError, OSError, ValueError, TimeoutError):
            pass
        terminate_group(process)
        log.close()
        socket_path.unlink(missing_ok=True)


def navigation_environment(runtime: dict, profile: dict, nav_gpu: int) -> dict:
    if not NAVIGATION_ENGINE.is_dir():
        raise FileNotFoundError(NAVIGATION_ENGINE)
    env = os.environ.copy()
    navigation = profile["navigation"]

    def navigation_value(name: str, default):
        """Keep profiles cross-dataset compatible while preserving explicit baselines."""
        return navigation.get(name, default)

    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(nav_gpu),
            "TDONAV_ROOT": str(ROOT),
            "TDONAV_LAGMEMO_ROOT": str(NAVIGATION_ENGINE),
            "LAGMEMO_SEMANTIC_SOURCE": profile["semantic_source"],
            "ES_NAV_PROJECTION_MAX_DISTANCE": str(
                navigation_value("projection_tolerance_m", 1.5)
            ),
            "ES_CANDIDATE_TRACK_RADIUS": str(
                navigation_value("candidate_track_radius_m", 1.0)
            ),
            "ES_CANDIDATE_BLACKLIST_RADIUS": str(
                navigation_value("candidate_blacklist_radius_m", 1.0)
            ),
            "ES_TRAPPED_WARMUP_STEPS": str(navigation_value("trapped_warmup_steps", 30)),
            "ES_TRAPPED_WINDOW_STEPS": str(navigation_value("trapped_window_steps", 25)),
            "ES_TRAPPED_RADIUS": str(navigation_value("trapped_radius_m", 0.05)),
            "HM3D_EXP_TWO_STAGE_VERIFY": "1"
            if navigation_value("two_stage_verification", False)
            else "0",
            "HM3D_EXP_VERIFY_WINDOW_STEPS": str(navigation_value("verification_window_steps", 6)),
            "HM3D_EXP_VERIFY_MIN_CONFIRMATIONS": str(
                navigation_value("verification_min_confirmations", 2)
            ),
            "HM3D_EXP_VERIFY_POSE_RADIUS": str(navigation_value("verification_pose_radius_m", 1.5)),
            "HM3D_EXP_VERIFY_MIN_PEAK_SEEM": str(navigation_value("verification_min_peak_seem", 1.15)),
            "HM3D_EXP_APPROACH_RING": "1" if navigation_value("approach_ring", False) else "0",
            "HM3D_EXP_CANDIDATE_EXHAUSTION_EXPLORE": "1"
            if navigation_value("candidate_exhaustion_exploration", False)
            else "0",
            "HM3D_EXP_VERIFY_ES_AGREEMENT": "1"
            if navigation_value("verify_es_agreement", False)
            else "0",
            "MALLOC_ARENA_MAX": "2",
            "MALLOC_TRIM_THRESHOLD_": "131072",
            "PYTHONUNBUFFERED": "1",
        }
    )
    if "hf_home" in runtime.get("shared", {}):
        env["HF_HOME"] = expand(runtime["shared"]["hf_home"])
        env["HF_HUB_CACHE"] = str(Path(env["HF_HOME"]) / "hub")
        env["HF_HUB_OFFLINE"] = "1"
    third_party = NAVIGATION_ENGINE / "src" / "third_party"
    pythonpath = [
        ROOT / "Navigation" / "src",
        NAVIGATION_ENGINE / "src" / "lagmemo",
        third_party / "seem" / "src",
        third_party / "detectron2",
        third_party / "ml-mobileclip",
        third_party / "frontier_exploration",
    ]
    env["PYTHONPATH"] = os.pathsep.join(map(str, pythonpath)) + os.pathsep + env.get("PYTHONPATH", "")
    return env
