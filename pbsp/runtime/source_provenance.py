"""Verify pinned source snapshots used by the default Habitat adapter."""

import hashlib
import json
from pathlib import Path
import subprocess

MARKER = ".pbsp-source.json"


def habitat_source(lock):
    modules = lock["source_groups"]["habitat_lab"]
    if len(modules) != 1 or modules[0]["path"] != ".":
        raise ValueError("Expected one Habitat-Lab source root")
    return next(item for item in lock["sources"] if item["id"] == modules[0]["source"])


def verify_source(directory, source):
    """Accept a clean pinned Git checkout or a complete checked archive export."""
    directory = Path(directory)
    marker = directory / MARKER
    if not marker.exists():
        revision = subprocess.check_output(["git", "-C", str(directory), "rev-parse", "HEAD"], text=True).strip()
        if revision != source["revision"]:
            raise ValueError(f"Habitat revision differs: {revision}")
        subprocess.run(["git", "-C", str(directory), "diff", "--quiet", "HEAD"], check=True)
        if subprocess.check_output(["git", "-C", str(directory), "status", "--porcelain", "--untracked-files=all"], text=True).strip():
            raise ValueError("Habitat source checkout has uncommitted files")
        return {"revision": revision, "kind": "git"}
    payload = json.loads(marker.read_text())
    entries = payload["entries"]
    digest = hashlib.sha256(json.dumps(sorted(entries), separators=(",", ":")).encode()).hexdigest()
    if payload["revision"] != source["revision"] or digest != source["content_sha256"] or len(entries) != source["files"]:
        raise ValueError("Habitat source manifest differs from the pinned archive")
    expected = {name for name, mode, value in entries}
    actual = {str(path.relative_to(directory)) for path in directory.rglob("*")
              if (path.is_symlink() or path.is_file()) and path.name != MARKER
              and "__pycache__" not in path.relative_to(directory).parts}
    if actual != expected:
        raise ValueError("Habitat source file inventory differs")
    for name, mode, value in entries:
        path = directory / name
        path.resolve().relative_to(directory.resolve())
        if mode == 0o120000:
            import os
            valid = path.is_symlink() and os.readlink(path) == value
        else:
            valid = (not path.is_symlink() and path.is_file()
                     and (0o755 if path.stat().st_mode & 0o111 else 0o644) == mode
                     and hashlib.sha256(path.read_bytes()).hexdigest() == value)
        if not valid:
            raise ValueError(f"Habitat source file differs: {name}")
    return {"revision": source["revision"], "kind": "archive", "files": len(entries), "content_sha256": digest}
