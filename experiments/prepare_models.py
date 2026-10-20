#!/usr/bin/env python3
"""Prepare the pinned default model assets without importing model packages."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import http.client
import json
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pbsp.runtime.model_assets import load_manifest, relative_path, target_path, verify

CHUNK_BYTES = 32 * 1024 * 1024
DOWNLOAD_ATTEMPTS = 3


class ShortDownload(OSError):
    """A valid response ended before its declared byte range was complete."""


def _fetch_range(partial, item):
    offset = partial.stat().st_size if partial.exists() else 0
    end = min(offset + CHUNK_BYTES, item["bytes"]) - 1
    request = urllib.request.Request(item["url"], headers={
        "Range": f"bytes={offset}-{end}", "Accept-Encoding": "identity",
    })
    with urllib.request.urlopen(request, timeout=120) as response:
        if response.headers.get("Content-Encoding", "identity") != "identity":
            raise ValueError("Model server returned encoded bytes")
        if response.status == 206:
            expected = f"bytes {offset}-{end}/{item['bytes']}"
            if response.headers.get("Content-Range") != expected:
                raise ValueError("Model server returned an unexpected byte range")
            remaining, mode = end - offset + 1, "ab"
        elif response.status == 200:
            # A server may ignore Range. A full response starts at byte zero.
            remaining, mode = item["bytes"], "wb"
        else:
            raise ValueError(f"Unexpected model response status: {response.status}")
        length = response.headers.get("Content-Length")
        if length is not None and (not re.fullmatch(r"[0-9]+", length) or int(length) != remaining):
            raise ValueError("Model server returned an unexpected content length")
        with partial.open(mode) as output:
            while remaining:
                block = response.read(min(1024 * 1024, remaining))
                if not block:
                    raise ShortDownload("Model response ended early; the partial file is retained")
                output.write(block)
                remaining -= len(block)
            if response.read(1):
                raise ValueError("Model response exceeded its declared byte range")


def _transient(error):
    if isinstance(error, urllib.error.HTTPError):
        return error.code in {408, 429, 500, 502, 503, 504}
    return isinstance(error, (ShortDownload, http.client.IncompleteRead,
                              urllib.error.URLError, ConnectionError, socket.timeout))


def fetch(cache, item, offline):
    path = cache / item["sha256"]
    if path.exists():
        verify(path, item)
        return path
    if offline:
        raise FileNotFoundError(f"Missing cached model: {path}")
    cache.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial")
    if partial.exists() and partial.stat().st_size > item["bytes"]:
        raise ValueError(f"Cached partial model is larger than the pinned asset: {partial}")
    while not partial.exists() or partial.stat().st_size < item["bytes"]:
        for attempt in range(DOWNLOAD_ATTEMPTS):
            try:
                _fetch_range(partial, item)
                break
            except Exception as error:
                if not _transient(error) or attempt + 1 == DOWNLOAD_ATTEMPTS:
                    raise
                print(f"Retry {attempt + 1}/{DOWNLOAD_ATTEMPTS - 1}: "
                      f"{item.get('path', item.get('id'))} ({type(error).__name__})", flush=True)
                time.sleep(2 ** attempt)
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
        downloaded, failures = {}, []
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            pending = {pool.submit(fetch, cache, item, offline): item for item in artifacts}
            for future in as_completed(pending):
                item = pending[future]
                try:
                    downloaded[item["sha256"]] = future.result()
                except Exception as error:
                    failures.append(error)
                    print(f"Failed {item.get('path', item.get('id'))}: {error}", file=sys.stderr, flush=True)
        if failures:
            raise failures[0]
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
    parser.add_argument("--manifest", type=Path, default=ROOT / "configs/resources/models.json")
    parser.add_argument("--output", type=Path, default=ROOT / "third_party/weights")
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
