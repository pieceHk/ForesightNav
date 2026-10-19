"""Shared process and artifact lifecycle for 3DONav experiment runners."""

import hashlib
import json
import math
import os
import signal
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("TDONAV_ROOT", str(ROOT))
MAPPING_ENGINE = ROOT / "Mapping" / "runtime" / "embodied_splat"
NAVIGATION_ENGINE = ROOT / "Navigation" / "runtime" / "lagmemo"
_LOADED_TOML_SOURCES = {}


def load_toml(path: Path) -> dict:
    source = Path(path).resolve()
    raw = source.read_bytes()
    value = tomllib.loads(raw.decode("utf-8"))
    _LOADED_TOML_SOURCES[id(value)] = (
        value,
        source,
        hashlib.sha256(raw).hexdigest(),
    )
    return value


def resolve_scsp_profile_text(common_text: str, *, arm: str, mode: str) -> str:
    """Resolve one SCSP arm while preserving every non-arm/mode byte.

    The profile loader intentionally remains plain TOML.  This small pure helper
    makes the no-include fallback reproducible and lets contract tests prove that
    the checked-in resolved profiles differ only at the two allowlisted fields.
    """

    safe = frozenset("abcdefghijklmnopqrstuvwxyz0123456789_-.")
    for name, value in (("arm", arm), ("mode", mode)):
        if (
            not isinstance(value, str)
            or not value
            or any(char not in safe for char in value)
        ):
            raise ValueError(f"SCSP {name} must be a lowercase identifier")

    section = ""
    replacements = {"arm": 0, "mode": 0}
    resolved = []
    for line in common_text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped
        key = None
        if section == "[scsp]":
            for candidate in replacements:
                if stripped.startswith(candidate + " ="):
                    key = candidate
                    break
        if key is None:
            resolved.append(line)
            continue
        ending = "\r\n" if line.endswith("\r\n") else "\n" if line.endswith("\n") else ""
        value = arm if key == "arm" else mode
        resolved.append(f'{key} = "{value}"{ending}')
        replacements[key] += 1

    if replacements != {"arm": 1, "mode": 1}:
        raise ValueError("common SCSP profile must contain one arm and one mode field")
    return "".join(resolved)


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


def _scsp_runtime_provenance(profile: dict) -> dict:
    """Resolve frozen SCSP provenance without trusting caller-supplied hashes."""

    frozen_path = ROOT / "Exp/experiments/scsp_idea2_frozen_pairs_v1.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    arm = str(profile["scsp"]["arm"])
    arms = {row["arm"]: row for row in frozen["arms"]}
    if arm not in arms:
        raise ValueError(f"SCSP arm is absent from frozen protocol: {arm!r}")
    arm_contract = arms[arm]
    profile_path = (ROOT / arm_contract["profile_path"]).resolve()
    source = _LOADED_TOML_SOURCES.get(id(profile))
    if source is None or source[0] is not profile:
        raise ValueError("SCSP profile provenance is unavailable")
    _, observed_profile_path, profile_sha256 = source

    portable_dataset = profile["scsp"].get("portable_dataset", False)
    if portable_dataset is not False and portable_dataset is not True:
        raise ValueError("SCSP portable_dataset must be boolean")
    if portable_dataset:
        if arm != "ordered_suffix_mvp":
            raise ValueError(
                "portable SCSP datasets are supported only by ordered_suffix_mvp"
            )
        profiles_dir = (ROOT / "Exp" / "profiles").resolve()
        if observed_profile_path.parent != profiles_dir:
            raise ValueError("portable SCSP profile must be checked in under Exp/profiles")

        frozen_profile = tomllib.loads(profile_path.read_text(encoding="utf-8"))
        normalized = deepcopy(profile)
        normalized["name"] = frozen_profile["name"]
        normalized["dataset"] = frozen_profile["dataset"]
        portable_scsp = normalized["scsp"]
        dataset_sha256 = portable_scsp.pop("dataset_manifest_sha256", None)
        portable_scsp.pop("portable_dataset", None)
        portable_scsp["ordered_subtasks"] = frozen_profile["scsp"][
            "ordered_subtasks"
        ]
        if normalized != frozen_profile:
            raise ValueError(
                "portable SCSP profile differs outside dataset identity fields"
            )
        if (
            not isinstance(dataset_sha256, str)
            or len(dataset_sha256) != 64
            or any(char not in "0123456789abcdef" for char in dataset_sha256)
        ):
            raise ValueError(
                "portable SCSP dataset_manifest_sha256 must be lowercase SHA-256"
            )
        return {
            "profile_sha256": profile_sha256,
            "dataset_manifest_sha256": dataset_sha256,
            "code_sha": git_head(),
        }

    if observed_profile_path != profile_path:
        raise ValueError("SCSP profile path differs from the frozen protocol")
    frozen_profile = tomllib.loads(profile_path.read_text(encoding="utf-8"))
    if profile != frozen_profile:
        raise ValueError("SCSP profile was mutated after frozen loading")
    if profile_sha256 != arm_contract["profile_sha256"]:
        raise ValueError("SCSP profile hash differs from the frozen protocol")
    code_sha = git_head()
    expected_code_sha = arm_contract.get("code_sha")
    if arm == "exact_474a752" and code_sha != expected_code_sha:
        raise ValueError(
            "exact_474a752 must run from detached exact commit "
            f"{expected_code_sha}, got {code_sha}"
        )
    dataset_sha256 = str(frozen["dataset"]["dataset_manifest_sha256"])
    return {
        "profile_sha256": profile_sha256,
        "dataset_manifest_sha256": dataset_sha256,
        "code_sha": code_sha,
    }


def resolve_scsp_runtime_mode(profile_mode: str, override: str = "") -> str:
    """Resolve an active SCSP mode while allowing only safe rollbacks."""

    defaults = {
        "exact_baseline": "off",
        "idea1_only": "idea1_only",
        "ordered_suffix_mvp": "ordered_suffix_mvp",
    }
    allowed = {
        "exact_baseline": {"off"},
        "idea1_only": {"shadow", "idea1_only"},
        "ordered_suffix_mvp": {
            "shadow",
            "idea1_only",
            "ordered_suffix_mvp",
        },
    }
    if profile_mode not in defaults:
        raise ValueError(f"unsupported SCSP profile mode: {profile_mode!r}")
    requested = str(override).strip() or defaults[profile_mode]
    if requested not in allowed[profile_mode]:
        raise ValueError(
            "SCSP runtime mode is not a safe realization/rollback of "
            f"{profile_mode!r}: {requested!r}"
        )
    return requested


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
    pythonpath = [ROOT, ROOT / "Mapping" / "src", MAPPING_ENGINE, MAPPING_ENGINE / "src", MAPPING_ENGINE / "src" / "third_party"]
    env["PYTHONPATH"] = os.pathsep.join(map(str, pythonpath)) + os.pathsep + env.get("PYTHONPATH", "")
    log = (run / "logs" / "mapping.log").open("w", encoding="utf-8")
    command = [
        python,
        "-m",
        "runtime.mapping_worker",
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
    completed_body = False
    try:
        wait_for_server(process, socket_path)
        yield process
        completed_body = True
    finally:
        try:
            if process.poll() is None and socket_path.exists():
                send_socket(socket_path, {"cmd": "shutdown"}, timeout_s=120)
        except (ConnectionError, OSError, ValueError, TimeoutError):
            pass
        terminate_group(process)
        log.close()
        socket_path.unlink(missing_ok=True)
        write_json(run / "mapping_process.json", {
            "pid": process.pid,
            "command": command,
            "returncode": process.returncode,
        })
        if completed_body and process.returncode != 0:
            raise RuntimeError(f"Mapping worker exited with {process.returncode}")


def navigation_environment(runtime: dict, profile: dict, nav_gpu: int) -> dict:
    if not NAVIGATION_ENGINE.is_dir():
        raise FileNotFoundError(NAVIGATION_ENGINE)
    env = os.environ.copy()
    navigation = profile["navigation"]
    scsp = profile.get("scsp")
    if scsp is None:
        scsp_arm = "legacy"
        scsp_profile_mode = "exact_baseline"
        scsp_runtime_mode = "off"
        scsp_candidate_cap = 5
        scsp_suffix_weight = 0.35
        scsp_provenance = {
            "profile_sha256": "",
            "dataset_manifest_sha256": "",
            "code_sha": "",
        }
    else:
        scsp_arm = str(scsp.get("arm", "")).strip()
        scsp_profile_mode = str(scsp.get("mode", "")).strip()
        if profile.get("ground_truth_semantics") is not False:
            raise ValueError("SCSP requires ground_truth_semantics=false")
        if profile.get("semantic_source") != "detic":
            raise ValueError("SCSP requires semantic_source=detic")
        if not scsp_arm:
            raise ValueError("SCSP arm must be non-empty")
        if scsp.get("strict_geodesic") is not True:
            raise ValueError("SCSP requires strict_geodesic=true")
        if scsp.get("euclidean_fallback") is not False:
            raise ValueError("SCSP requires euclidean_fallback=false")
        scsp_candidate_cap = scsp.get("candidate_cap")
        if (
            isinstance(scsp_candidate_cap, bool)
            or not isinstance(scsp_candidate_cap, int)
            or not 1 <= scsp_candidate_cap <= 5
        ):
            raise ValueError("SCSP candidate_cap must be an integer in [1, 5]")
        scsp_suffix_weight = scsp.get("suffix_weight")
        if (
            isinstance(scsp_suffix_weight, bool)
            or not isinstance(scsp_suffix_weight, (int, float))
            or not math.isfinite(float(scsp_suffix_weight))
            or float(scsp_suffix_weight) < 0.0
        ):
            raise ValueError("SCSP suffix_weight must be finite and non-negative")
        scsp_runtime_mode = resolve_scsp_runtime_mode(
            scsp_profile_mode,
            os.environ.get("ES_SCSP_RUNTIME_MODE_OVERRIDE", ""),
        )
        scsp_ordered_subtasks = scsp.get("ordered_subtasks")
        if (
            isinstance(scsp_ordered_subtasks, bool)
            or not isinstance(scsp_ordered_subtasks, int)
            or scsp_ordered_subtasks < 1
        ):
            raise ValueError("SCSP ordered_subtasks must be a positive integer")
        scsp_provenance = _scsp_runtime_provenance(profile)

    def navigation_value(name: str, default):
        """Keep profiles cross-dataset compatible while preserving explicit controls."""
        return navigation.get(name, default)

    category_cooldowns = profile.get("query_switch_cooldown_by_category", {})
    if not isinstance(category_cooldowns, dict):
        raise ValueError("query_switch_cooldown_by_category must be a TOML table")
    for category, steps in category_cooldowns.items():
        if not str(category).strip():
            raise ValueError("category cooldown names must be non-empty")
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
            raise ValueError(
                f"cooldown for {category!r} must be a non-negative integer"
            )
    if category_cooldowns and not navigation_value(
        "query_action_rerank_enabled", False
    ):
        raise ValueError(
            "category cooldown requires query_action_rerank_enabled=true"
        )

    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(nav_gpu),
            "TDONAV_ROOT": str(ROOT),
            "TDONAV_LAGMEMO_ROOT": str(NAVIGATION_ENGINE),
            "LAGMEMO_SEMANTIC_SOURCE": profile["semantic_source"],
            "ES_SCSP_ARM": scsp_arm,
            "ES_SCSP_PROFILE_MODE": scsp_profile_mode,
            "ES_SCSP_RUNTIME_MODE": scsp_runtime_mode,
            "ES_SCSP_CANDIDATE_CAP": str(scsp_candidate_cap),
            "ES_SCSP_SUFFIX_WEIGHT": str(scsp_suffix_weight),
            "ES_SCSP_ORDERED_SUBTASKS": str(scsp_ordered_subtasks)
            if scsp is not None
            else "0",
            "ES_SCSP_STRICT_GEODESIC": "1",
            "ES_SCSP_EUCLIDEAN_FALLBACK": "0",
            "ES_SCSP_PROFILE_SHA256": scsp_provenance["profile_sha256"],
            "ES_SCSP_EXPECTED_DATASET_MANIFEST_SHA256": scsp_provenance[
                "dataset_manifest_sha256"
            ],
            "ES_SCSP_CODE_SHA": scsp_provenance["code_sha"],
            "ES_NAV_PROJECTION_MAX_DISTANCE": str(
                navigation_value("projection_tolerance_m", 1.5)
            ),
            "ES_CANDIDATE_TRACK_RADIUS": str(
                navigation_value("candidate_track_radius_m", 1.0)
            ),
            "ES_CANDIDATE_BLACKLIST_RADIUS": str(
                navigation_value("candidate_blacklist_radius_m", 1.0)
            ),
            "ES_CANDIDATE_EVIDENCE_RADIUS": str(
                navigation_value("candidate_evidence_radius_m", 0.75)
            ),
            "ES_CANDIDATE_PROMOTION_MARGIN": str(
                navigation_value("candidate_promotion_margin", 2)
            ),
            "ES_CANDIDATE_EVIDENCE_RANKING": "1"
            if navigation.get("candidate_evidence_ranking", True)
            else "0",
            "ES_QUERY_ACTION_RERANK_ENABLED": "1"
            if navigation_value("query_action_rerank_enabled", False)
            else "0",
            "ES_QUERY_W_SEMANTIC": str(navigation_value("query_w_semantic", 1.0)),
            "ES_QUERY_W_SUPPORT": str(navigation_value("query_w_support", 0.35)),
            "ES_QUERY_W_RECURRENCE": str(navigation_value("query_w_recurrence", 0.30)),
            "ES_QUERY_W_TRAVEL": str(navigation_value("query_w_travel", 0.35)),
            "ES_QUERY_W_NOVELTY": str(navigation_value("query_w_novelty", 0.15)),
            "ES_QUERY_W_REVISIT": str(navigation_value("query_w_revisit", 0.80)),
            "ES_QUERY_W_FAILURE": str(navigation_value("query_w_failure", 1.0)),
            "ES_QUERY_SWITCH_MARGIN": str(navigation_value("query_switch_margin", 0.15)),
            "ES_QUERY_SWITCH_COOLDOWN_BY_CATEGORY": json.dumps(
                category_cooldowns, sort_keys=True, separators=(",", ":")
            ),
            "ES_FRONTIER_RERANK_ENABLED": "1"
            if navigation_value("frontier_rerank_enabled", False)
            else "0",
            "ES_FRONTIER_W_GAIN": str(navigation_value("frontier_w_gain", 1.0)),
            "ES_FRONTIER_W_CLEARANCE": str(navigation_value("frontier_w_clearance", 0.25)),
            "ES_FRONTIER_W_TRAVEL": str(navigation_value("frontier_w_travel", 0.35)),
            "ES_FRONTIER_W_NOVELTY": str(navigation_value("frontier_w_novelty", 0.20)),
            "ES_FRONTIER_W_REVISIT": str(navigation_value("frontier_w_revisit", 0.75)),
            "ES_FRONTIER_W_FAILURE": str(navigation_value("frontier_w_failure", 1.0)),
            "ES_FRONTIER_SWITCH_MARGIN": str(navigation_value("frontier_switch_margin", 0.15)),
            "ES_TRAPPED_WARMUP_STEPS": str(
                navigation_value("trapped_warmup_steps", 30)
            ),
            "ES_TRAPPED_WINDOW_STEPS": str(
                navigation_value("trapped_window_steps", 25)
            ),
            "ES_TRAPPED_RADIUS": str(
                navigation_value("trapped_radius_m", 0.05)
            ),
            "ES_RECOVERY_HISTORY_SIZE": str(
                navigation_value("recovery_history_size", 8)
            ),
            "ES_RECOVERY_NOVELTY_WEIGHT": str(
                navigation_value("recovery_novelty_weight", 0.65)
            ),
            "HM3D_EXP_TWO_STAGE_VERIFY": "1"
            if navigation_value("two_stage_verification", False)
            else "0",
            "HM3D_EXP_VERIFY_WINDOW_STEPS": str(
                navigation_value("verification_window_steps", 6)
            ),
            "HM3D_EXP_VERIFY_MIN_CONFIRMATIONS": str(
                navigation_value("verification_min_confirmations", 2)
            ),
            "HM3D_EXP_VERIFY_POSE_RADIUS": str(
                navigation_value("verification_pose_radius_m", 1.5)
            ),
            "HM3D_EXP_VERIFY_MIN_PEAK_SEEM": str(
                navigation_value("verification_min_peak_seem", 1.15)
            ),
            "HM3D_EXP_VERIFY_DUAL_SCORE_GATE": "1"
            if navigation.get("verification_dual_score_gate", False)
            else "0",
            "HM3D_EXP_VERIFY_STRONG_PEAK_SEEM": str(
                navigation.get("verification_strong_peak_seem", 1.35)
            ),
            "HM3D_EXP_VERIFY_MIN_PEAK_CLIP": str(
                navigation.get("verification_min_peak_clip", 0.12)
            ),
            "HM3D_EXP_VERIFY_MIN_OBSERVER_TRANSLATION": str(
                navigation_value("verification_min_observer_translation_m", 0.0)
            ),
            "HM3D_EXP_VERIFY_MIN_OBSERVER_YAW_DEG": str(
                navigation_value("verification_min_observer_yaw_deg", 0.0)
            ),
            "HM3D_EXP_APPROACH_RING": "1"
            if navigation_value("approach_ring", False)
            else "0",
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
    if "ground_truth_semantics" in profile:
        env["ES_GROUND_TRUTH_SEMANTICS"] = (
            "1" if profile["ground_truth_semantics"] else "0"
        )
    if "hf_home" in runtime.get("shared", {}):
        env["HF_HOME"] = expand(runtime["shared"]["hf_home"])
        env["HF_HUB_CACHE"] = str(Path(env["HF_HOME"]) / "hub")
        env["HF_HUB_OFFLINE"] = "1"
    third_party = NAVIGATION_ENGINE / "src" / "third_party"
    pythonpath = [
        ROOT,
        ROOT / "Navigation" / "src",
        NAVIGATION_ENGINE / "src" / "lagmemo",
        third_party / "seem" / "src",
        third_party / "ml-mobileclip",
        third_party / "frontier_exploration",
    ]
    env["PYTHONPATH"] = os.pathsep.join(map(str, pythonpath))
    return env
