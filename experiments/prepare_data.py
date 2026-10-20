#!/usr/bin/env python3
"""Connect released scene/episode assets and prepare complete task identities."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.dataset import build_identity_index


def connect(source: Path, destination: Path) -> None:
    source = source.resolve(strict=True)
    if destination.is_symlink():
        if destination.resolve() != source:
            raise ValueError(f"Resource link points to a different source: {destination}")
        return
    if destination.exists():
        raise ValueError(f"Resource destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(os.path.relpath(source, destination.parent), target_is_directory=source.is_dir())


def prepare_data(source: Path, destination: Path, catalog: dict, *, scenes: Path | None = None) -> list[dict]:
    source = Path(source).expanduser().resolve(strict=True)
    destination = Path(destination).expanduser().resolve()
    scene_source = Path(scenes).expanduser().resolve(strict=True) if scenes is not None else source / catalog["scenes"]
    if not scene_source.is_dir():
        raise FileNotFoundError(scene_source)
    prepared = []
    # Validate every selected dataset before creating links or generated files.
    for entry in catalog["datasets"]:
        dataset = source / entry["source"]
        if "val_seen_sha256" in entry and hashlib.sha256((dataset / "val_seen.json.gz").read_bytes()).hexdigest() != entry["val_seen_sha256"]:
            raise ValueError(f"Dataset split header checksum mismatch: {entry['name']}")
        payload = build_identity_index(dataset, entry["tasks_per_episode"], entry["manifest_sha256"], entry["source_trace_sha256"])
        records = payload["episodes"]
        if len(records) != entry["episodes"] or len({row["scene_id"] for row in records}) != entry["scenes"]:
            raise ValueError(f"Dataset cardinality mismatch: {entry['name']}")
        prepared.append((entry, dataset, payload))

    common = catalog.get("common")
    common_source = source / common["source"] if common else None
    if common_source is not None and common_source.exists():
        for record in common["files"]:
            if hashlib.sha256((common_source / record["path"]).read_bytes()).hexdigest() != record["sha256"]:
                raise ValueError(f"Common episode provenance checksum mismatch: {record['path']}")

    connect(scene_source, destination / "scenes/hm3dv1")
    if common_source is not None and common_source.exists():
        connect(common_source, destination / common["target"])
    summaries = []
    for entry, dataset, payload in prepared:
        target = destination / entry["target"]
        if target.is_symlink():
            raise ValueError(f"Prepared dataset must have its own identity index directory: {target}")
        target.mkdir(parents=True, exist_ok=True)
        for asset in sorted(dataset.iterdir()):
            if asset.name != "identities.json":
                connect(asset, target / asset.name)
        path = target / "identities.json"
        text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if path.exists() and path.read_text(encoding="utf-8") != text:
            raise ValueError(f"Existing identity index differs from the selected dataset: {path}")
        if not path.exists():
            path.write_text(text, encoding="utf-8")
        summaries.append({"dataset": entry["name"], "episodes": len(payload["episodes"]), "identities": str(path)})
    return summaries


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Connect an existing catalog instead of rebuilding the bundled fixed episodes")
    parser.add_argument("--scenes", type=Path, help="Scene asset directory containing hm3d/")
    parser.add_argument("--cache", type=Path, default=ROOT / ".cache/pbsp-offline/datasets")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--download-only", action="store_true", help="Rebuild and verify episode cache without scene assets")
    parser.add_argument("--destination", type=Path, default=ROOT / "data")
    parser.add_argument("--catalog", type=Path, default=ROOT / "configs/resources/datasets.json")
    args = parser.parse_args(argv)
    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    if args.source is not None and args.download_only:
        parser.error("--download-only cannot be combined with --source")
    if args.source is None:
        if args.scenes is None and not args.download_only:
            parser.error("--scenes is required when preparing the bundled episodes")
        from experiments._episode_sources import materialize_episodes

        args.source = materialize_episodes(ROOT, catalog, args.cache.expanduser().resolve(), offline=args.offline)
    if args.download_only:
        print(json.dumps({"episode_cache": str(args.source), "status": "verified"}, indent=2))
        return
    print(json.dumps(prepare_data(args.source, args.destination, catalog, scenes=args.scenes), indent=2))


if __name__ == "__main__":
    main()
