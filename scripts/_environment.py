"""Shared artifact and environment checks for the Linux installers."""

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def content_hash(files):
    return hashlib.sha256(json.dumps(sorted(files), separators=(",", ":")).encode()).hexdigest()


def archive_files(path):
    """Read regular files under one archive root before writing anything to disk."""
    entries = []
    with tarfile.open(path) as archive:
        roots = set()
        names = set()
        for member in archive:
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts:
                raise ValueError(f"Unsafe archive path: {member.name}")
            if not name.parts:
                continue
            roots.add(name.parts[0])
            if member.isdir():
                continue
            if not member.isfile() or len(name.parts) < 2:
                raise ValueError(f"Unsupported archive entry: {member.name}")
            relative = str(PurePosixPath(*name.parts[1:]))
            if relative in names:
                raise ValueError(f"Duplicate archive entry: {relative}")
            names.add(relative)
            mode = 0o755 if member.mode & 0o111 else 0o644
            entries.append((relative, mode, archive.extractfile(member).read()))
        if len(roots) != 1:
            raise ValueError("Source archive must contain exactly one root directory")
    return entries


def archive_hash(path):
    return content_hash([(name, mode, hashlib.sha256(data).hexdigest())
                         for name, mode, data in archive_files(path)])


def verify_artifact(path, item):
    if not path.is_file():
        raise FileNotFoundError(f"Missing cached artifact: {path}")
    if item.get("source_archive"):
        from scripts._sources import source_archive_hash
        actual = source_archive_hash(path)
    else:
        actual = archive_hash(path) if "content_sha256" in item else file_hash(path)
    expected = item.get("content_sha256", item.get("sha256"))
    if actual != expected:
        raise ValueError(f"Checksum mismatch: {path}")


def fetch_artifact(cache, item, offline):
    path = cache / item["cache_path"]
    if path.exists():
        verify_artifact(path, item)
        return path
    if offline:
        raise FileNotFoundError(f"Missing cached artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
        try:
            with urllib.request.urlopen(item["url"], timeout=120) as response:
                shutil.copyfileobj(response, temporary, length=1024 * 1024)
            temporary.flush()
            verify_artifact(temporary_path, item)
            temporary_path.replace(path)
        finally:
            temporary_path.unlink(missing_ok=True)
    return path


def source_files(root, source):
    relative = source["repository_path"]
    names = subprocess.check_output(
        ["git", "-C", str(root), "ls-files", "-z", "--", relative]
    ).decode().split("\0")
    entries = []
    for name in filter(None, names):
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Source must be a regular tracked file: {path}")
        path.resolve().relative_to(root.resolve())
        mode = 0o755 if path.stat().st_mode & 0o111 else 0o644
        entries.append((str(path.relative_to(root / relative)), mode, file_hash(path)))
    if not entries or content_hash(entries) != source["content_sha256"]:
        raise ValueError(f"Tracked source differs from the mapping lock: {relative}")
    return entries


def materialize_source(root, cache, source, destination):
    if "repository_path" in source:
        entries = source_files(root, source)
        for name, mode, _ in entries:
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / source["repository_path"] / name, target)
            target.chmod(mode)
    else:
        path = cache / source["cache_path"]
        verify_artifact(path, source)
        for name, mode, data in archive_files(path):
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            target.chmod(mode)


def check_prefix(prefix, lock):
    python = prefix / "bin/python"
    if not python.is_file():
        raise ValueError(f"Existing prefix is not a Python environment: {prefix}")
    observed = json.loads(subprocess.check_output([
        str(python), "-I", "-c",
        "import importlib.util,json,sys;print(json.dumps({'prefix':sys.prefix,"
        "'python':list(sys.version_info[:2]),'pip':importlib.util.find_spec('pip') is not None}))",
    ], text=True))
    if Path(observed["prefix"]).resolve() != prefix.resolve() or observed["python"] != lock["python"]:
        raise ValueError(f"Prefix does not match the locked Python version: {prefix}")
    for package in lock["conda"]:
        records = list((prefix / "conda-meta").glob(
            f"{package['name']}-{package['version']}-{package['build']}.json"
        ))
        if len(records) != 1:
            raise ValueError(f"Prefix is missing the locked Conda base package: {package['name']}")
    if not observed["pip"]:
        raise ValueError(
            "Conda did not link pip into the target Python environment. "
            "Recreate this prefix with a current Conda or micromamba."
        )
    return python
