"""Check the separately obtained SuperGlue inference dependency."""

import hashlib
import json
from pathlib import Path, PurePosixPath
import re

ROOT = Path(__file__).resolve().parents[2]
SUPERGLUE_FILES = {"LICENSE", "models/__init__.py", "models/matching.py",
                   "models/superpoint.py", "models/superglue.py", "models/utils.py"}


def load_source_manifest(path, expected_sha256=None):
    data = Path(path).read_bytes()
    if expected_sha256 is not None and hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError("External source manifest checksum differs from the navigation lock")
    manifest = json.loads(data)
    if manifest["schema"] != "pbsp_superglue_source_v1" or not re.fullmatch(r"[0-9a-f]{40}", manifest["revision"]):
        raise ValueError("Invalid SuperGlue source manifest")
    files = manifest["files"]
    if len(files) != len(SUPERGLUE_FILES) or {row["path"] for row in files} != SUPERGLUE_FILES:
        raise ValueError("Unexpected SuperGlue source inventory")
    for row in files:
        path = PurePosixPath(row["cache_path"])
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Unsafe external source cache path")
        if not all(re.fullmatch(r"[0-9a-f]{64}", row[name]) for name in ("sha256", "installed_sha256")):
            raise ValueError("Invalid external source file hash")
    return manifest


def verify_external_source(directory, manifest):
    directory = Path(directory)
    expected = {row["path"] for row in manifest["files"]}
    actual = {str(path.relative_to(directory)) for path in directory.rglob("*")
              if (path.is_symlink() or path.is_file()) and "__pycache__" not in path.relative_to(directory).parts}
    if actual != expected:
        raise ValueError(f"SuperGlue source inventory differs; prepare dependencies: {directory}")
    for row in manifest["files"]:
        path = directory / row["path"]
        path.resolve().relative_to(directory.resolve())
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != row["installed_sha256"]:
            raise ValueError(f"SuperGlue source file differs: {path}")
    content = [(row["path"], row["installed_sha256"]) for row in manifest["files"]]
    return {"repository": manifest["repository"], "revision": manifest["revision"],
            "files": len(expected), "kind": "external-inference-files",
            "content_sha256": hashlib.sha256(json.dumps(sorted(content), separators=(",", ":")).encode()).hexdigest()}


def pinned_superglue_manifest(root=ROOT):
    root = Path(root)
    lock = json.loads((root / "configs/environments/navigation/linux-64.lock.json").read_text())
    entries = [row for row in lock["runtime_source_manifests"] if row["path"] == "configs/resources/superglue.json"]
    if len(entries) != 1:
        raise ValueError("Expected one pinned SuperGlue source manifest")
    return load_source_manifest(root / entries[0]["path"], entries[0]["sha256"])


def superglue_directory(root=ROOT):
    root = Path(root)
    directory = root / "third_party/superglue"
    manifest = pinned_superglue_manifest(root)
    verify_external_source(directory, manifest)
    return directory
