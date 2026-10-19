"""Validate released episodes and bind every ordered task to its source identity."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Iterator, Mapping, Optional

from pbsp.types import EpisodeIdentity


IDENTITY_SCHEMA = "pbsp_episode_identities_v1"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_identity_index(dataset: Path, expected_tasks: int, expected_manifest_sha256: str,
                         expected_source_sha256: Optional[str] = None) -> dict:
    """Build a complete index using identity fields; preserve manifest episode order."""
    dataset = Path(dataset)
    if type(expected_tasks) is not int or expected_tasks < 1:
        raise ValueError("Expected task count must be a positive integer")
    manifest_path = dataset / "dataset_manifest.json"
    observed_hash = file_sha256(manifest_path)
    if observed_hash != expected_manifest_sha256:
        raise ValueError("Dataset manifest does not match the selected dataset version")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    trace_path = dataset / "source_trace.json"
    if expected_source_sha256 is not None and file_sha256(trace_path) != expected_source_sha256:
        raise ValueError("Source trace does not match the selected dataset version")
    traces = {}
    for row in json.loads(trace_path.read_text(encoding="utf-8")):
        key = (str(row["scene_id"]), str(row["mon_episode_id"]))
        if key in traces:
            raise ValueError(f"Duplicate source identity: {key}")
        traces[key] = row["sources"]

    observed_keys = set()
    scenes = set()
    records = []
    for scene in manifest["scene_summary"]:
        scene_id = str(scene["scene_id"])
        if Path(scene_id).name != scene_id or scene_id in scenes:
            raise ValueError(f"Ambiguous scene identity: {scene_id}")
        scenes.add(scene_id)
        content = dataset / "content" / f"{scene_id}.json.gz"
        if file_sha256(content) != scene["output_sha256"]:
            raise ValueError(f"Runtime episode content hash mismatch: {scene_id}")
        with gzip.open(content, "rt", encoding="utf-8") as stream:
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
            if sources is None or len(sources) != expected_tasks or len(tasks) != expected_tasks:
                raise ValueError(f"Incomplete ordered identity: {key}")
            source_ids, categories = [], []
            for index, (source, task) in enumerate(zip(sources, tasks)):
                category = str(source["runtime_category"])
                source_id = str(source["source_episode_id"])
                if (type(source["position"]) is not int or source["position"] != index
                        or str(source["scene_id"]) != scene_id or not source_id.strip()
                        or len(task) < 2 or task[0] != category or task[1] != "object"):
                    raise ValueError(f"Source/runtime task alignment mismatch: {key}, task {index}")
                source_ids.append(source_id)
                categories.append(category)
            identity = EpisodeIdentity(
                dataset_manifest_sha256=observed_hash, scene_id=scene_id,
                runtime_episode_id=key[1], source_episode_ids=tuple(source_ids),
                ordered_categories=tuple(categories),
            )
            records.append({
                "scene_id": identity.scene_id,
                "runtime_episode_id": identity.runtime_episode_id,
                "source_episode_ids": list(identity.source_episode_ids),
                "ordered_categories": list(identity.ordered_categories),
            })
    if observed_keys != set(traces):
        raise ValueError("Source and runtime identity sets do not align")
    if not records:
        raise ValueError("Dataset contains no episode identities")
    return {
        "schema_version": IDENTITY_SCHEMA,
        "dataset_manifest_sha256": observed_hash,
        "episodes": records,
    }


def load_identity_index(path: Path, expected_manifest_sha256: str) -> Mapping[tuple[str, str], EpisodeIdentity]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(payload) != {"schema_version", "dataset_manifest_sha256", "episodes"}:
        raise ValueError("Unexpected identity index fields")
    if payload["schema_version"] != IDENTITY_SCHEMA or payload["dataset_manifest_sha256"] != expected_manifest_sha256:
        raise ValueError("Identity index schema or dataset version mismatch")
    identities = {}
    for record in payload["episodes"]:
        if set(record) != {"scene_id", "runtime_episode_id", "source_episode_ids", "ordered_categories"}:
            raise ValueError("Identity records may contain only task identity fields")
        identity = EpisodeIdentity(dataset_manifest_sha256=expected_manifest_sha256, **record)
        key = (identity.scene_id, identity.runtime_episode_id)
        if key in identities:
            raise ValueError(f"Duplicate runtime identity: {key}")
        identities[key] = identity
    if not identities:
        raise ValueError("Identity index is empty")
    return identities


def iter_episodes(dataset: Path, identities: Mapping[tuple[str, str], EpisodeIdentity]) -> Iterator[tuple[dict, EpisodeIdentity]]:
    """Yield evaluator-owned raw episodes with their validated identities."""
    dataset = Path(dataset)
    manifest_path = dataset / "dataset_manifest.json"
    digest = file_sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    seen = set()
    for scene in manifest["scene_summary"]:
        scene_id = str(scene["scene_id"])
        content = dataset / "content" / f"{scene_id}.json.gz"
        if file_sha256(content) != scene["output_sha256"]:
            raise ValueError(f"Runtime episode content hash mismatch: {scene_id}")
        with gzip.open(content, "rt", encoding="utf-8") as stream:
            episodes = json.load(stream)["episodes"]
        if len(episodes) != scene["mon_episode_count"]:
            raise ValueError(f"Episode count mismatch: {scene_id}")
        for episode in episodes:
            key = (scene_id, str(episode["episode_id"]))
            identity = identities.get(key)
            if identity is None or key in seen or identity.dataset_manifest_sha256 != digest:
                raise ValueError(f"Missing, duplicate or outdated episode identity: {key}")
            categories = tuple(task[0] for task in episode["tasks"])
            runtime_scene = Path(episode["scene_id"]).name.split(".", 1)[0]
            if identity.ordered_categories != categories or runtime_scene != scene_id:
                raise ValueError(f"Episode task identity mismatch: {key}")
            seen.add(key)
            yield episode, identity
    if seen != set(identities):
        raise ValueError("Unused records in the episode identity index")


def task_category(task):
    for name in ('category', 'object_category'):
        value = task.get(name)
        if value is not None and str(value).strip():
            return str(value)
    raise ValueError(f'MON task is missing its target category: {task}')
