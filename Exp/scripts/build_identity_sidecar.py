#!/usr/bin/env python3
"""Bind ordered MON runtime episodes to their recorded source identities."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

from _runtime import ROOT, load_toml


FROZEN = ROOT / "Exp/experiments/scsp_idea2_frozen_pairs_v1.json"


def build_sidecar(dataset, ordered_subtasks, expected_hash, frozen):
    raw_manifest = (dataset / "dataset_manifest.json").read_bytes()
    observed_hash = hashlib.sha256(raw_manifest).hexdigest()
    if observed_hash != expected_hash:
        raise ValueError("Dataset manifest does not match the selected profile")
    manifest = json.loads(raw_manifest)
    trace_rows = json.loads((dataset / "source_trace.json").read_text())
    traces = {}
    for row in trace_rows:
        key = (str(row["scene_id"]), str(row["mon_episode_id"]))
        if key in traces:
            raise ValueError(f"Duplicate source identity: {key}")
        traces[key] = row["sources"]

    frozen_keys = set()
    if frozen["dataset"]["dataset_manifest_sha256"] == observed_hash:
        for row in frozen["episodes"]:
            identity = row["identity"]
            frozen_keys.add((str(identity["scene_id"]), str(identity["runtime_episode_id"])))

    records = []
    observed_keys = set()
    frozen_matches = set()
    for scene in manifest["scene_summary"]:
        scene_id = str(scene["scene_id"])
        content = dataset / "content" / f"{scene_id}.json.gz"
        if hashlib.sha256(content.read_bytes()).hexdigest() != scene["output_sha256"]:
            raise ValueError(f"Runtime episode content hash mismatch: {scene_id}")
        with gzip.open(content, "rt") as stream:
            episodes = json.load(stream)["episodes"]
        if len(episodes) != scene["mon_episode_count"]:
            raise ValueError(f"Episode count mismatch: {scene_id}")
        for episode in episodes:
            key = (scene_id, str(episode["episode_id"]))
            runtime_scene = Path(episode["scene_id"]).name.split(".", 1)[0]
            if runtime_scene != scene_id or key in observed_keys:
                raise ValueError(f"Ambiguous runtime episode identity: {key}")
            observed_keys.add(key)
            sources = traces.get(key)
            tasks = episode["tasks"]
            if sources is None or len(sources) != ordered_subtasks or len(tasks) != ordered_subtasks:
                raise ValueError(f"Incomplete ordered identity: {key}")
            source_ids, categories = [], []
            for index, (source, task) in enumerate(zip(sources, tasks)):
                category = str(source["runtime_category"])
                source_id = str(source["source_episode_id"])
                if (source["position"] != index or str(source["scene_id"]) != scene_id
                        or task[0] != category or task[1] != "object" or not source_id.strip()):
                    raise ValueError(f"Source/runtime task alignment mismatch: {key}, task {index}")
                source_ids.append(source_id)
                categories.append(category)
            record = {
                "scene_id": scene_id,
                "runtime_episode_id": key[1],
                "source_episode_ids": source_ids,
                "ordered_categories": categories,
            }
            if key in frozen_keys:
                frozen_identity = next(row["identity"] for row in frozen["episodes"]
                                       if (str(row["identity"]["scene_id"]),
                                           str(row["identity"]["runtime_episode_id"])) == key)
                if any(record[name] != frozen_identity[name] for name in record):
                    raise ValueError(f"Frozen identity disagrees with runtime/source trace: {key}")
                frozen_matches.add(key)
            else:
                records.append(record)
    if observed_keys != set(traces) or frozen_matches != frozen_keys:
        raise ValueError("Source, runtime and frozen identity sets do not align")
    if not records:
        raise ValueError("No additional episode identities are needed")
    return {
        "schema_version": "scsp_identity_sidecar_v1",
        "dataset_manifest_sha256": observed_hash,
        "episodes": records,
    }


def expected_manifest_hash(mon):
    name = "hm3dv1_mon_scsp_suffix.toml" if mon == 3 else f"hm3dv1_{mon}mon_scsp_suffix.toml"
    profile = load_toml(ROOT / "Exp/profiles" / name)
    return profile["scsp"].get("dataset_manifest_sha256") or json.loads(FROZEN.read_text())["dataset"]["dataset_manifest_sha256"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--mon", type=int, choices=(2, 3, 4), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_sidecar(args.dataset, args.mon, expected_manifest_hash(args.mon),
                            json.loads(FROZEN.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output.exists() and args.output.read_text() != text:
        raise RuntimeError(f"Refusing to replace a different identity sidecar: {args.output}")
    if not args.output.exists():
        args.output.write_text(text)
    print(f"Wrote {len(payload['episodes'])} identity-only records to {args.output}")
    print(f"Set ES_SCSP_IDENTITY_SIDECAR={args.output.absolute()}")


if __name__ == "__main__":
    main()
