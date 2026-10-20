"""Record installed package metadata without importing model implementations."""

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import site
import sys

SCHEMA = "pbsp_environment_v1"


def record_environment(plan, role, *, scene=None):
    if role not in {"mapping", "navigation"}:
        raise ValueError(f"Unknown environment role: {role}")
    if role == "navigation":
        selected = next((item for item in plan.scenes if item.scene_id == scene), None)
        if selected is None:
            raise ValueError("Environment inventory requires a selected scene")
        name = f"navigation_environment_{selected.scene_index}.json"
    else:
        if scene is not None:
            raise ValueError("Mapping environment inventory has no scene selection")
        name = "mapping_environment.json"
    lock = plan.resources.root / "configs/environments" / role / "linux-64.lock.json"
    packages = []
    for distribution in importlib.metadata.distributions():
        packages.append({"name": distribution.metadata["Name"], "version": distribution.version,
                         "location": str(Path(distribution.locate_file("")).resolve())})
    packages.sort(key=lambda item: (item["name"].lower(), item["version"], item["location"]))
    report = {
        "schema_version": SCHEMA, "role": role, "scene_id": scene,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "launch_source_commit": plan.source_commit,
        "configuration_sha256": plan.configuration_sha256,
        "identity_sha256": plan.identity_sha256,
        "python_executable": sys.executable, "python_version": platform.python_version(),
        "prefix": sys.prefix, "base_prefix": sys.base_prefix,
        "platform": platform.platform(), "machine": platform.machine(),
        "user_site_enabled": site.ENABLE_USER_SITE,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "source_search_paths": list(sys.path),
        "lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
        "packages": packages,
    }
    path = plan.run_directory / name
    with path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return path
