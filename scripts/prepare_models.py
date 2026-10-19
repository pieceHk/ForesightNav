#!/usr/bin/env python3
"""Prepare the pinned default model assets without importing model packages."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


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


def fetch(cache, item, offline):
    path = cache / item["sha256"]
    if path.exists():
        verify(path, item)
        return path
    if offline:
        raise FileNotFoundError(f"Missing cached model: {path}")
    cache.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial")
    offset = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(item["url"], headers={"Range": f"bytes={offset}-"} if offset else {})
    with urllib.request.urlopen(request, timeout=120) as response:
        resumed = response.status == 206
        if resumed and response.headers.get("Content-Range", "").split("/")[0] != f"bytes {offset}-{item['bytes'] - 1}":
            raise ValueError("Model server returned an unexpected byte range")
        with partial.open("ab" if resumed else "wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
    verify(partial, item)
    partial.replace(path)
    print(f"Verified {item.get('path', item.get('id'))}", flush=True)
    return path


def extract_selected(archive, items, destination, *, zstd="zstd"):
    """Stream only the listed regular members into a private staging directory."""
    wanted = {item["member"]: item for item in items}
    if len(wanted) != len(items):
        raise ValueError("Duplicate selected archive member")
    found = set()
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen([str(zstd), "-dc", str(archive)], stdout=subprocess.PIPE, stderr=errors)
        try:
            with process.stdout, tarfile.open(fileobj=process.stdout, mode="r|") as stream:
                for member in stream:
                    name = str(relative_path(member.name.rstrip("/")))
                    if name not in wanted:
                        continue
                    if name in found or not member.isfile():
                        raise ValueError(f"Invalid selected model archive member: {name}")
                    item = wanted[name]
                    if member.size != item["bytes"]:
                        raise ValueError(f"Model archive member size differs: {name}")
                    target = target_path(destination, item["path"])
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with stream.extractfile(member) as source, target.open("wb") as output:
                        shutil.copyfileobj(source, output, length=1024 * 1024)
                    verify(target, item)
                    found.add(name)
            code = process.wait()
            if code:
                errors.seek(0)
                raise RuntimeError(f"zstd exited {code}: {errors.read().decode(errors='replace')}")
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait()
    if found != wanted.keys():
        raise FileNotFoundError(f"Model archive members missing: {sorted(wanted.keys() - found)}")


def prepare(manifest, *, destination, cache, source=None, offline=False, download_only=False, verify_only=False, jobs=4, zstd="zstd"):
    destination, cache = Path(destination).resolve(), Path(cache).resolve()
    if jobs < 1 or sum(map(bool, (source, download_only, verify_only))) > 1:
        raise ValueError("Choose one model preparation mode and a positive job count")
    if verify_only:
        for item in manifest["files"]:
            verify(target_path(destination, item["path"]), item)
        for item in manifest["refs"]:
            if target_path(destination, item["path"]).read_text() != item["value"]:
                raise ValueError(f"Model cache revision differs: {item['path']}")
        return len(manifest["files"])
    # Validate existing targets before creating or changing any asset files.
    for item in manifest["files"]:
        path = target_path(destination, item["path"])
        if path.exists():
            verify(path, item)
    for item in manifest["refs"]:
        path = target_path(destination, item["path"])
        if path.exists() and path.read_text() != item["value"]:
            raise ValueError(f"Existing model cache revision differs: {path}")
    if source:
        inputs = {item["path"]: Path(source).resolve() / item["path"] for item in manifest["files"]}
        for item in manifest["files"]:
            verify(inputs[item["path"]], item)
    else:
        artifacts = [item for item in manifest["files"] if "url" in item] + manifest["archives"]
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            cached = list(pool.map(lambda item: fetch(cache, item, offline), artifacts))
        downloaded = {item["sha256"]: path for item, path in zip(artifacts, cached)}
        if download_only:
            return len(artifacts)
        inputs = {item["path"]: downloaded[item["sha256"]] for item in manifest["files"] if "url" in item}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".pbsp-models-", dir=destination.parent) as temporary:
        staging = Path(temporary)
        if not source:
            for archive in manifest["archives"]:
                selected = [item for item in manifest["files"] if item.get("archive") == archive["id"]]
                extract_selected(downloaded[archive["sha256"]], selected, staging, zstd=zstd)
                inputs.update({item["path"]: staging / item["path"] for item in selected})
        for item in manifest["files"]:
            target = target_path(destination, item["path"])
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary_file = staging / (item["sha256"] + ".copy")
            shutil.copyfile(inputs[item["path"]], temporary_file)
            verify(temporary_file, item)
            temporary_file.replace(target)
        for item in manifest["refs"]:
            target = target_path(destination, item["path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(item["value"])
    return prepare(manifest, destination=destination, cache=cache, verify_only=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "assets/models.json")
    parser.add_argument("--output", type=Path, default=ROOT / "Exp/assets/weights")
    parser.add_argument("--cache", type=Path, default=ROOT / ".cache/pbsp-offline/models")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--source", type=Path, help="Import an existing prepared weights directory")
    modes.add_argument("--download-only", action="store_true")
    modes.add_argument("--verify", dest="verify_only", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--zstd", default="zstd", help="zstd executable for the upstream pretrained archive")
    args = parser.parse_args(argv)
    options = vars(args)
    manifest = load_manifest(options.pop("manifest"))
    options["destination"] = options.pop("output")
    print(f"Verified {prepare(manifest, **options)} model assets")


if __name__ == "__main__":
    main()
