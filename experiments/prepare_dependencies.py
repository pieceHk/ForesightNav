#!/usr/bin/env python3
"""Prepare the pinned Habitat and SuperGlue inference dependencies."""

import argparse
import json
import hashlib
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pbsp.runtime.source_provenance import MARKER, habitat_source, verify_source
from experiments._environment import fetch_artifact
from experiments._sources import materialize_source_group, source_inventory
from pbsp.runtime.external_sources import pinned_superglue_manifest, verify_external_source


def prepare_habitat(*, root, cache, destination=None, offline=False):
    root, cache = Path(root).resolve(), Path(cache).resolve()
    destination = Path(destination).resolve() if destination else root / "third_party/habitat-lab"
    lock = json.loads((root / "configs/environments/navigation/linux-64.lock.json").read_text())
    source = habitat_source(lock)
    if destination.exists():
        return verify_source(destination, source)
    archive = fetch_artifact(cache, source, offline)
    entries = source_inventory(archive)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".pbsp-habitat-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "source"
        materialize_source_group(cache, [source], lock["source_groups"]["habitat_lab"], staging)
        (staging / MARKER).write_text(json.dumps({"revision": source["revision"], "entries": entries}, indent=2) + "\n")
        report = verify_source(staging, source)
        staging.replace(destination)
    return report


def prepare_superglue(*, root, cache, destination=None, offline=False, download_only=False):
    root, cache = Path(root).resolve(), Path(cache).resolve()
    destination = Path(destination).resolve() if destination else root / "third_party/superglue"
    manifest = pinned_superglue_manifest(root)
    if destination.exists() and not download_only:
        return verify_external_source(destination, manifest)
    files = {}
    for row in manifest["files"]:
        data = fetch_artifact(cache, row, offline).read_bytes()
        if len(data) != row["bytes"]:
            raise ValueError(f"External source length differs: {row['path']}")
        for patch in manifest["patches"].get(row["path"], []):
            before, after = patch["before"].encode(), patch["after"].encode()
            if data.count(before) != 1:
                raise ValueError(f"External source patch does not match exactly once: {row['path']}")
            data = data.replace(before, after)
        if hashlib.sha256(data).hexdigest() != row["installed_sha256"]:
            raise ValueError(f"Prepared external source differs: {row['path']}")
        files[row["path"]] = data
    if download_only:
        return {"status": "cached", "revision": manifest["revision"], "files": len(files)}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".pbsp-superglue-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "source"
        for name, data in files.items():
            path = staging / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        report = verify_external_source(staging, manifest)
        staging.replace(destination)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path.home() / ".cache/pbsp")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--download-only", action="store_true", help="Cache the small SuperGlue inference source files")
    args = parser.parse_args(argv)
    if args.download_only:
        result = prepare_superglue(root=ROOT, cache=args.cache, offline=args.offline, download_only=True)
    else:
        result = {"habitat": prepare_habitat(root=ROOT, cache=args.cache, destination=args.output, offline=args.offline),
                  "superglue": prepare_superglue(root=ROOT, cache=args.cache, offline=args.offline)}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
