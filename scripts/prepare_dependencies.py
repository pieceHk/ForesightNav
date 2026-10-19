#!/usr/bin/env python3
"""Prepare the pinned Habitat source snapshot from the environment cache."""

import argparse
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from runtime.source_provenance import MARKER, habitat_source, verify_source
from scripts._environment import fetch_artifact
from scripts._sources import materialize_source_group, source_inventory


def prepare_habitat(*, root, cache, destination=None, offline=False):
    root, cache = Path(root).resolve(), Path(cache).resolve()
    destination = Path(destination).resolve() if destination else root / "Exp/assets/dependencies/habitat-lab"
    lock = json.loads((root / "envs/navigation/linux-64.lock.json").read_text())
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=ROOT / ".cache/pbsp-offline")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(prepare_habitat(root=ROOT, cache=args.cache, destination=args.output, offline=args.offline), indent=2))


if __name__ == "__main__":
    main()
