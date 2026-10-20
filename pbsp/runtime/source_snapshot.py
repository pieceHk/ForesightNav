"""Preserve untracked working-tree objects alongside the tracked Git patch."""

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess

SCHEMA = "pbsp_source_snapshot_v1"
MAX_UNTRACKED_BYTES = 64 * 1024 * 1024


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def relative(value):
    path = PurePosixPath(value)
    if not value or not path.parts or path.is_absolute() or ".." in path.parts or str(path) != value:
        raise ValueError("Invalid source snapshot path")
    return path


def untracked(root, destination):
    raw = subprocess.check_output(["git", "-C", str(root), "ls-files", "--others", "--exclude-standard", "-z"])
    values = [os.fsdecode(value) for value in raw.split(b"\0") if value]
    try:
        excluded = destination.relative_to(root)
    except ValueError:
        excluded = None
    return sorted(value for value in values if excluded is None or not (
        Path(value) == excluded or excluded in Path(value).parents))


def stamp(value):
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_mode


def capture_source_snapshot(root, directory, source, *, state_reader, max_bytes=MAX_UNTRACKED_BYTES):
    if type(max_bytes) is not int or max_bytes < 0:
        raise ValueError("Source snapshot byte budget must be a non-negative integer")
    root, directory = Path(root).resolve(), Path(directory).resolve()
    destination = directory / "source_untracked"
    destination.mkdir(exist_ok=False)
    manifest = {"schema_version": SCHEMA, "commit": source["commit"], "status": source["status"],
                "tracked_patch_sha256": hashlib.sha256(source["patch"].encode()).hexdigest(),
                "max_untracked_bytes": max_bytes, "untracked": [], "complete": True,
                "inventory_stable": True, "tracked_state_stable": True,
                "scope": "Git working-tree objects; symlink targets are recorded without following them. The current run directory is excluded."}
    observed, used = {}, 0
    try:
        paths = untracked(root, directory)
    except subprocess.CalledProcessError:
        paths = []
        manifest.update(complete=False, inventory_stable=False, inventory_error="Git inventory unavailable")
    for value in paths:
        name = relative(value)
        original = root / str(name)
        row = {"path": value, "captured": False}
        copied = None
        manifest["untracked"].append(row)
        try:
            before = original.lstat()
            observed[value] = stamp(before)
            row["mode"] = stat.S_IMODE(before.st_mode)
            if stat.S_ISLNK(before.st_mode):
                target = os.readlink(original)
                row.update(kind="symlink", captured=True, target=target,
                           sha256=hashlib.sha256(os.fsencode(target)).hexdigest())
            elif stat.S_ISREG(before.st_mode):
                row.update(kind="file", bytes=before.st_size)
                original.parent.resolve().relative_to(root)
                if before.st_size > max_bytes - used:
                    row["reason"] = "untracked_byte_budget"
                    continue
                descriptor = os.open(original, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                                     | getattr(os, "O_NONBLOCK", 0))
                with os.fdopen(descriptor, "rb") as stream:
                    if stamp(os.fstat(stream.fileno())) != stamp(before):
                        row["reason"] = "changed_during_capture"
                        continue
                    data = stream.read(max_bytes - used + 1)
                    if len(data) != before.st_size or stamp(os.fstat(stream.fileno())) != stamp(before):
                        row["reason"] = "changed_during_capture"
                        continue
                copied = destination / str(name)
                copied.parent.mkdir(parents=True, exist_ok=True)
                with copied.open("xb") as stream:
                    stream.write(data)
                copied.chmod(row["mode"])
                row.update(captured=True, sha256=hashlib.sha256(data).hexdigest())
                used += len(data)
            else:
                row.update(kind="special", reason="non_regular_working_tree_object")
        except (OSError, ValueError) as error:
            if copied is not None:
                raise
            row["reason"] = type(error).__name__
    for value, expected in observed.items():
        try:
            if stamp((root / value).lstat()) != expected:
                manifest["inventory_stable"] = False
        except OSError:
            manifest["inventory_stable"] = False
    try:
        manifest["inventory_stable"] &= paths == untracked(root, directory)
    except subprocess.CalledProcessError:
        manifest["inventory_stable"] = False
    manifest["tracked_state_stable"] = state_reader(root) == source
    manifest["complete"] &= (manifest["inventory_stable"] and manifest["tracked_state_stable"]
                             and all(row["captured"] for row in manifest["untracked"]))
    path = directory / "source_snapshot.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return {"schema_version": SCHEMA, "manifest_sha256": digest(path),
            "complete": manifest["complete"], "untracked_objects": len(paths), "captured_file_bytes": used}


def verify_source_snapshot(directory, proof, source, read_json):
    path = directory / "source_snapshot.json"
    if proof["schema_version"] != SCHEMA or digest(path) != proof["manifest_sha256"]:
        raise ValueError("Source snapshot manifest identity differs")
    manifest = read_json(path)
    if (manifest["schema_version"] != SCHEMA or manifest["commit"] != source["commit"]
            or manifest["status"] != source["status"]
            or manifest["tracked_patch_sha256"] != digest(directory / "source.patch")
            or any(type(manifest[key]) is not bool for key in ("complete", "inventory_stable", "tracked_state_stable"))
            or type(proof["complete"]) is not bool):
        raise ValueError("Source snapshot differs from recorded Git state")
    folder, expected, seen, used = directory / "source_untracked", set(), set(), 0
    if folder.is_symlink() or not folder.is_dir():
        raise ValueError("Source snapshot directory must be a regular directory")
    for row in manifest["untracked"]:
        name = str(relative(row["path"]))
        if name in seen or type(row["captured"]) is not bool:
            raise ValueError("Repeated or invalid source snapshot object")
        seen.add(name)
        if not row["captured"]:
            if "reason" not in row:
                raise ValueError("Uncaptured source object has no reason")
            continue
        if row["kind"] == "file":
            if type(row["bytes"]) is not int or row["bytes"] < 0 or type(row["mode"]) is not int:
                raise ValueError("Invalid source snapshot file metadata")
            expected.add(name)
            copied = folder / name
            copied.parent.resolve().relative_to(folder.resolve())
            if (copied.is_symlink() or not copied.is_file() or copied.stat().st_size != row["bytes"]
                    or digest(copied) != row["sha256"]
                    or (stat.S_IMODE(copied.stat().st_mode) & 0o111) != (row["mode"] & 0o111)):
                raise ValueError(f"Source snapshot file differs: {name}")
            used += row["bytes"]
        elif row["kind"] == "symlink":
            if (not isinstance(row["target"], str) or not row["target"] or "\0" in row["target"]
                    or hashlib.sha256(os.fsencode(row["target"])).hexdigest() != row["sha256"]):
                raise ValueError("Source snapshot symlink record differs")
        else:
            raise ValueError("Unknown captured source object type")
    actual = set()
    for member in folder.rglob("*"):
        if member.is_symlink() or not (member.is_file() or member.is_dir()):
            raise ValueError("Unexpected non-regular source snapshot member")
        if member.is_file():
            actual.add(str(member.relative_to(folder)))
    complete = (manifest["inventory_stable"] and manifest["tracked_state_stable"]
                and all(row["captured"] for row in manifest["untracked"]))
    if (actual != expected or complete != manifest["complete"] or proof["complete"] != manifest["complete"]
            or proof["untracked_objects"] != len(seen) or proof["captured_file_bytes"] != used):
        raise ValueError("Source snapshot coverage differs from its inventory")
    return {"complete": manifest["complete"], "untracked_objects": len(seen), "captured_file_bytes": used}
