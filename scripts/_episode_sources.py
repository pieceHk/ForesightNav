"""Rebuild pinned MON episodes from fixed task orders and official ObjectNav inputs."""

from __future__ import annotations

import copy
import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import struct
import tarfile
import tempfile
import urllib.request
import zipfile
import zlib


def checked_bytes(path: Path, digest: str, size: int | None = None) -> bytes:
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != digest or (size is not None and len(data) != size):
        raise ValueError(f"Dataset resource checksum mismatch: {path}")
    return data


def relative_path(value: str) -> Path:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in ("..", ".") for part in path.parts):
        raise ValueError(f"Invalid dataset resource path: {value}")
    return Path(*path.parts)


def read_recipe(root: Path, recipe: dict, expected: dict[str, str]) -> dict[str, bytes]:
    data = checked_bytes(root / relative_path(recipe["path"]), recipe["sha256"], recipe["bytes"])
    files = {}
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        for member in archive:
            relative_path(member.name)
            if not member.isfile() or member.name not in expected or member.name in files:
                raise ValueError(f"Unexpected episode recipe member: {member.name}")
            body = archive.extractfile(member).read()
            if hashlib.sha256(body).hexdigest() != expected[member.name]:
                raise ValueError(f"Episode recipe member checksum mismatch: {member.name}")
            files[member.name] = body
    if set(files) != set(expected):
        raise ValueError("Incomplete episode recipe")
    return files


def _range_read(url: str, value: str, source: dict, opener) -> tuple[bytes, int]:
    request = urllib.request.Request(url, headers={"Range": "bytes=" + value})
    with opener(request, timeout=120) as response:
        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
        if response.status != 206 or match is None:
            raise ValueError("ObjectNav provider did not return a byte range")
        first, last, total = map(int, match.groups())
        expected = (max(0, total - int(value[1:])), total - 1) if value.startswith("-") else tuple(map(int, value.split("-")))
        if (first, last) != expected or not 0 <= first <= last < total:
            raise ValueError("ObjectNav provider returned a different byte range")
        if response.headers.get("x-amz-version-id") != source["version_id"]:
            raise ValueError("ObjectNav archive version differs from the dataset catalog")
        if response.headers.get("ETag") != source["etag"]:
            raise ValueError("ObjectNav archive ETag differs from the dataset catalog")
        body = response.read(last - first + 2)
        if len(body) != last - first + 1:
            raise ValueError("Truncated or oversized ObjectNav byte range")
        return body, first


def prepare_objectnav(source: dict, cache: Path, *, offline: bool, opener=None) -> dict[str, Path]:
    """Fetch only the selected validation members, checking each pinned SHA-256."""
    opener = opener or urllib.request.urlopen
    paths, missing = {}, []
    for entry in source["files"]:
        scene = entry["scene_id"]
        relative_path(scene)
        if len(PurePosixPath(scene).parts) != 1:
            raise ValueError(f"Invalid ObjectNav scene: {scene}")
        path = cache / relative_path(source["content_prefix"]) / f"{scene}.json.gz"
        if scene in paths:
            raise ValueError(f"Duplicate ObjectNav scene: {scene}")
        paths[scene] = path
        if path.exists():
            checked_bytes(path, entry["sha256"], entry["bytes"])
        else:
            missing.append(entry)
    if not missing:
        return paths
    if offline:
        raise FileNotFoundError(f"Offline ObjectNav cache is incomplete: {paths[missing[0]['scene_id']]}")

    tail, tail_start = _range_read(source["url"], "-65536", source, opener)
    with zipfile.ZipFile(io.BytesIO(tail)) as archive:
        # ZipFile offsets are relative to this tail buffer, including negative offsets.
        infos = []
        for entry in missing:
            info = archive.getinfo(source["content_prefix"] + entry["scene_id"] + ".json.gz")
            if info.file_size != entry["bytes"] or info.compress_type not in (0, 8):
                raise ValueError(f"Unexpected ObjectNav ZIP member: {info.filename}")
            infos.append((entry, info))
        first = min(info.header_offset + tail_start for _, info in infos)
        last = max(info.header_offset + tail_start + 30 + len(info.filename.encode()) + len(info.extra) + info.compress_size for _, info in infos) + 512
        data, _ = _range_read(source["url"], f"{first}-{last}", source, opener)
        for entry, info in infos:
            offset = info.header_offset + tail_start - first
            header = struct.unpack("<4s5H3I2H", data[offset:offset + 30])
            if header[0] != b"PK\x03\x04" or header[2] & 1 or header[3] != info.compress_type:
                raise ValueError(f"Invalid ObjectNav local header: {info.filename}")
            position = offset + 30 + header[-2] + header[-1]
            compressed = data[position:position + info.compress_size]
            body = zlib.decompress(compressed, -15) if info.compress_type == 8 else compressed
            if (len(body) != entry["bytes"] or zlib.crc32(body) != info.CRC
                    or hashlib.sha256(body).hexdigest() != entry["sha256"]):
                raise ValueError(f"ObjectNav member checksum mismatch: {info.filename}")
            path = paths[entry["scene_id"]]
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(body)
            temporary.replace(path)
    return paths


def compose_scene(source: bytes, rows: list[dict], manifest: dict, tasks: int) -> bytes:
    source_sha = hashlib.sha256(source).hexdigest()
    original = json.loads(gzip.decompress(source))
    aliases = manifest["runtime_category_aliases"]
    episodes = []
    for row in rows:
        selected = row["sources"]
        if len(selected) != tasks or [record["position"] for record in selected] != list(range(tasks)):
            raise ValueError("Episode recipe has incomplete task positions")
        for record in selected:
            index = record["in_file_sequence_index"]
            if type(index) is not int or not 0 <= index < len(original["episodes"]):
                raise ValueError("ObjectNav source sequence index is out of bounds")
            raw = original["episodes"][index]
            if (record["content_file_sha256"] != source_sha
                    or str(record["source_episode_id"]) != str(index)
                    or record["scene_id"] != row["scene_id"]
                    or raw["object_category"] != record["object_category"]
                    or record["runtime_category"] != aliases.get(raw["object_category"], raw["object_category"])):
                raise ValueError("ObjectNav source/task identity mismatch")
        first = original["episodes"][selected[0]["in_file_sequence_index"]]
        episodes.append({
            **{key: first[key] for key in ("scene_dataset_config", "scene_id", "start_position", "start_rotation")},
            "episode_id": row["mon_episode_id"],
            "tasks": [[record["runtime_category"], "object", None] for record in selected],
        })
    goals = copy.deepcopy(original["goals_by_category"])
    for objects in goals.values():
        for goal in objects:
            goal["object_category"] = aliases.get(goal["object_category"], goal["object_category"])
    raw = (json.dumps({"episodes": episodes, "goals": goals}, sort_keys=True, separators=(",", ":")) + "\n").encode()
    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", fileobj=buffer, mode="wb", mtime=0) as stream:
        stream.write(raw)
    return buffer.getvalue()


def _publish_directory(target: Path, files: dict[str, bytes]) -> None:
    if target.exists():
        for name, body in files.items():
            if (target / name).read_bytes() != body:
                raise ValueError(f"Existing dataset cache differs: {target / name}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=target.name + ".", dir=target.parent))
    try:
        for name, body in files.items():
            path = staging / relative_path(name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(body)
        staging.rename(target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def materialize_episodes(root: Path, catalog: dict, cache: Path, *, offline: bool) -> Path:
    """Produce the catalog's original episode bytes without sampling or reordering."""
    recipes = []
    for entry in catalog["datasets"]:
        expected = {"dataset_manifest.json": entry["manifest_sha256"],
                    "source_trace.json": entry["source_trace_sha256"],
                    "val_seen.json.gz": entry["val_seen_sha256"]}
        recipes.append((entry, read_recipe(root, entry["recipe"], expected)))
    common = catalog["common"]
    common_files = read_recipe(root, common["recipe"], {row["path"]: row["sha256"] for row in common["files"]})
    sources = prepare_objectnav(catalog["objectnav_source"], cache, offline=offline)
    destination = cache / "pbsp-hm3dv1-balanced1200-v1"
    notice = (root / "assets/episodes/NOTICE.md").read_bytes()
    for entry, metadata in recipes:
        manifest = json.loads(metadata["dataset_manifest.json"])
        trace = json.loads(metadata["source_trace.json"])
        files = {**metadata, "NOTICE.md": notice}
        for scene in manifest["scene_summary"]:
            scene_id = scene["scene_id"]
            rows = [row for row in trace if row["scene_id"] == scene_id]
            body = compose_scene(sources[scene_id].read_bytes(), rows, manifest, entry["tasks_per_episode"])
            if hashlib.sha256(body).hexdigest() != scene["output_sha256"]:
                raise ValueError(f"Rebuilt episode bytes differ from the released dataset: {scene_id}")
            files[f"content/{scene_id}.json.gz"] = body
        _publish_directory(destination / relative_path(entry["source"]), files)
    _publish_directory(destination / relative_path(common["source"]), {**common_files, "NOTICE.md": notice})
    return destination
