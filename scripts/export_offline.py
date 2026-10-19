#!/usr/bin/env python3
"""Export a committed PBSP checkout and verified Linux caches for offline transfer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from runtime.model_assets import load_manifest, relative_path, verify
from scripts._environment import file_hash, source_files, verify_artifact
from scripts.setup_navigation import checked_artifacts

GROUPS = ("environments", "models", "episodes")
SCHEMA = "pbsp_offline_export_v1"


def cache_inputs(root: Path, cache: Path, groups) -> list[dict]:
    """Check every selected input before creating an export directory."""
    if not groups or len(groups) != len(set(groups)) or not set(groups) <= set(GROUPS):
        raise ValueError("Select distinct offline resource groups")
    selected = {}

    def add(name, item, group, checker):
        relative_path(name)
        path = cache / name
        path.resolve().relative_to(cache.resolve())
        if path.is_symlink():
            raise ValueError(f"Offline cache input must be a regular file: {path}")
        checker(path, item)
        record = {"path": "cache/" + name, "sha256": file_hash(path),
                  "bytes": path.stat().st_size, "groups": [group], "mode": 0o644}
        existing = selected.get(name)
        if existing:
            if (existing["sha256"], existing["bytes"]) != (record["sha256"], record["bytes"]):
                raise ValueError(f"Conflicting offline cache identities: {name}")
            if group not in existing["groups"]:
                existing["groups"].append(group)
        else:
            selected[name] = record

    if "environments" in groups:
        for role in ("mapping", "navigation"):
            lock = json.loads((root / f"envs/{role}/linux-64.lock.json").read_text())
            if role == "mapping":
                artifacts = lock["conda"] + lock["packages"] + [row for row in lock["sources"] if "cache_path" in row]
                for row in lock["sources"]:
                    if "repository_path" in row:
                        source_files(root, row)
            else:
                artifacts = checked_artifacts(lock, root)
            for item in artifacts:
                add(item["cache_path"], item, "environments", verify_artifact)
    if "models" in groups:
        manifest = load_manifest(root / "assets/models.json")
        artifacts = [row for row in manifest["files"] if "url" in row] + manifest["archives"]
        for item in artifacts:
            add("models/" + item["sha256"], item, "models", verify)
    if "episodes" in groups:
        source = json.loads((root / "assets/datasets.json").read_text())["objectnav_source"]
        for item in source["files"]:
            add("datasets/" + source["content_prefix"] + item["scene_id"] + ".json.gz", item, "episodes", verify)
    return sorted(selected.values(), key=lambda row: row["path"])


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def bundle_head(path):
    rows = subprocess.check_output(["git", "bundle", "list-heads", str(path)], text=True).splitlines()
    refs = {}
    for row in rows:
        commit, ref = row.split(" ", 1)
        refs[ref] = commit
    branches = [ref for ref in refs if ref.startswith("refs/heads/")]
    if len(rows) != 2 or len(branches) != 1 or set(refs) != {"HEAD", branches[0]}:
        raise ValueError("Offline code bundle must contain one branch")
    ref = branches[0]
    commit = refs[ref]
    if refs["HEAD"] != commit:
        raise ValueError("Offline code bundle HEAD differs from its branch")
    return {"commit": commit, "branch": ref.removeprefix("refs/heads/")}


def verify_export(directory: Path) -> dict:
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "manifest.json").read_text())
    groups = manifest["groups"]
    if manifest["schema"] != SCHEMA or not groups or len(groups) != len(set(groups)) or not set(groups) <= set(GROUPS):
        raise ValueError("Invalid offline export manifest")
    expected = {"manifest.json"}
    for row in manifest["files"]:
        name = str(relative_path(row["path"]))
        if name in expected:
            raise ValueError(f"Duplicate offline export member: {name}")
        expected.add(name)
        path = directory / name
        path.resolve().relative_to(directory)
        if path.is_symlink():
            raise ValueError(f"Offline export member is a link: {name}")
        verify(path, row)
        if (0o755 if path.stat().st_mode & 0o111 else 0o644) != row["mode"]:
            raise ValueError(f"Offline export file mode differs: {name}")
    actual = {str(path.relative_to(directory)) for path in directory.rglob("*") if path.is_file() or path.is_symlink()}
    if actual != expected or "code.bundle" not in expected:
        raise ValueError("Offline export file inventory differs")
    if bundle_head(directory / "code.bundle") != manifest["code"]:
        raise ValueError("Offline code bundle revision differs")
    return {"status": "passed", "code": manifest["code"], "groups": manifest["groups"],
            "files": len(manifest["files"]), "bytes": sum(row["bytes"] for row in manifest["files"])}


def export_offline(*, root: Path, cache: Path, destination: Path, groups=GROUPS, micromamba=None) -> dict:
    root, cache, destination = Path(root).resolve(), Path(cache).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"Offline export destination already exists: {destination}")
    if git(root, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("Commit the checkout before exporting offline code")
    branch = git(root, "symbolic-ref", "--short", "HEAD")
    commit = git(root, "rev-parse", "HEAD")
    records = cache_inputs(root, cache, groups)
    if micromamba is not None:
        micromamba = Path(micromamba).resolve(strict=True)
        with micromamba.open("rb") as stream:
            header = stream.read(20)
        if header[:6] != b"\x7fELF\x02\x01" or header[18:20] != b">\x00":
            raise ValueError("Select a Linux x86_64 micromamba executable")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".pbsp-offline-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "export"
        staging.mkdir()
        bundle = staging / "code.bundle"
        subprocess.run(["git", "-C", str(root), "bundle", "create", str(bundle), "refs/heads/" + branch, "HEAD"], check=True)
        code = {"commit": commit, "branch": branch}
        if bundle_head(bundle) != code:
            raise ValueError("Code branch changed while creating the offline bundle")
        for row in records:
            target = staging / row["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(cache / row["path"].removeprefix("cache/"), target)
            target.chmod(row["mode"])
        files = [*records, {"path": "code.bundle", "sha256": file_hash(bundle),
                            "bytes": bundle.stat().st_size, "mode": 0o644}]
        if micromamba is not None:
            target = staging / "tools/micromamba"
            target.parent.mkdir()
            shutil.copyfile(micromamba, target)
            target.chmod(0o755)
            files.append({"path": "tools/micromamba", "sha256": file_hash(micromamba),
                          "bytes": micromamba.stat().st_size, "mode": 0o755})
        payload = {"schema": SCHEMA, "code": code, "groups": sorted(groups),
                   "scene_assets": "Obtain HM3D scenes separately through their provider",
                   "files": sorted(files, key=lambda row: row["path"])}
        (staging / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
        report = verify_export(staging)
        if git(root, "rev-parse", "HEAD") != commit or git(root, "status", "--porcelain", "--untracked-files=no"):
            raise ValueError("Code changed during offline export")
        staging.replace(destination)
    return {**report, "directory": str(destination)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--output", type=Path)
    action.add_argument("--verify", type=Path)
    parser.add_argument("--cache", type=Path, default=ROOT / ".cache/pbsp-offline")
    parser.add_argument("--group", action="append", choices=GROUPS, help="Export selected groups; default exports all three")
    parser.add_argument("--micromamba", type=Path, help="Include an existing Linux x86_64 micromamba executable")
    args = parser.parse_args(argv)
    if args.verify is not None:
        report = verify_export(args.verify)
    else:
        report = export_offline(root=ROOT, cache=args.cache, destination=args.output,
                                groups=args.group or GROUPS, micromamba=args.micromamba)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
