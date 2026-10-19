"""Pinned model inventories and read-only content verification."""

import hashlib
import json
from pathlib import Path, PurePosixPath
import re


def relative_path(value):
    path = PurePosixPath(value)
    if not path.parts or path.is_absolute() or ".." in path.parts or str(path) != value:
        raise ValueError(f"Expected a normalized relative asset path: {value}")
    return path


def target_path(root, value):
    path = root / str(relative_path(value))
    path.resolve().relative_to(root.resolve())
    return path


def load_manifest(path):
    manifest = json.loads(Path(path).read_text())
    if manifest["schema"] != "pbsp_models_v1":
        raise ValueError("Unsupported model manifest")
    names = set()
    archives = {item["id"]: item for item in manifest["archives"]}
    if len(archives) != len(manifest["archives"]):
        raise ValueError("Duplicate model archive ID")
    for item in [*manifest["files"], *manifest["refs"]]:
        name = str(relative_path(item["path"]))
        if name in names or any(name.startswith(p + "/") or p.startswith(name + "/") for p in names):
            raise ValueError(f"Overlapping asset paths: {name}")
        names.add(name)
    for item in [*manifest["files"], *archives.values()]:
        if not re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) or type(item["bytes"]) is not int or item["bytes"] < 1:
            raise ValueError("Invalid asset digest or size")
        if "archive" in item:
            if item["archive"] not in archives or "url" in item:
                raise ValueError("Invalid asset archive reference")
            relative_path(item["member"])
        elif not item["url"].startswith("https://"):
            raise ValueError("Expected an HTTPS model source")
    for item in manifest["refs"]:
        if not re.fullmatch(r"[0-9a-f]{40}", item["value"]):
            raise ValueError("Expected a pinned Hugging Face revision")
    return manifest


def verify(path, item):
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != item["bytes"]:
        raise ValueError(f"Asset size differs: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(data)
    if digest.hexdigest() != item["sha256"]:
        raise ValueError(f"Asset checksum differs: {path}")


def verify_model_set(directory, manifest_path):
    directory = Path(directory).resolve()
    manifest_path = Path(manifest_path)
    manifest = load_manifest(manifest_path)
    for item in manifest['files']:
        verify(target_path(directory, item['path']), item)
    for item in manifest['refs']:
        if target_path(directory, item['path']).read_text() != item['value']:
            raise ValueError(f"Model cache revision differs: {item['path']}")
    return {
        'status': 'passed',
        'files': len(manifest['files']),
        'bytes': sum(item['bytes'] for item in manifest['files']),
        'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    }
