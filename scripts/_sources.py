"""Check and assemble pinned source archives, including recursive submodules."""

import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile


SYMLINK = 0o120000


def source_archive_entries(path):
    """Yield root-relative names, normalized modes and file bytes/link targets."""
    with tarfile.open(path) as archive:
        roots, names = set(), set()
        for member in archive:
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts or not name.parts:
                raise ValueError(f"Unsafe source archive path: {member.name}")
            roots.add(name.parts[0])
            if member.isdir():
                continue
            relative = str(PurePosixPath(*name.parts[1:]))
            if len(name.parts) < 2 or relative in names:
                raise ValueError(f"Invalid or duplicate source entry: {member.name}")
            names.add(relative)
            if member.issym():
                if PurePosixPath(member.linkname).is_absolute():
                    raise ValueError(f"Absolute source link: {relative}")
                yield relative, SYMLINK, member.linkname
            elif member.isfile():
                yield relative, 0o755 if member.mode & 0o111 else 0o644, archive.extractfile(member).read()
            else:
                raise ValueError(f"Unsupported source entry: {relative}")
        if len(roots) != 1:
            raise ValueError("Source archive must contain one root")


def source_inventory(path):
    return [(name, mode, body if mode == SYMLINK else hashlib.sha256(body).hexdigest())
            for name, mode, body in source_archive_entries(path)]


def source_archive_hash(path):
    return hashlib.sha256(json.dumps(sorted(source_inventory(path)), separators=(",", ":")).encode()).hexdigest()


def materialize_source_group(cache, artifacts, modules, destination):
    """Write regular files before links and keep every link within the group."""
    destination = Path(destination)
    if destination.exists():
        raise ValueError(f"Source destination already exists: {destination}")
    by_id = {item["id"]: item for item in artifacts}
    if len(by_id) != len(artifacts):
        raise ValueError("Duplicate source IDs")
    paths, links, archives = set(), {}, []
    for module in modules:
        relative = PurePosixPath(module["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe module path: {relative}")
        source = by_id[module["source"]]
        archive = Path(cache) / source["cache_path"]
        inventory = source_inventory(archive)
        digest = hashlib.sha256(json.dumps(sorted(inventory), separators=(",", ":")).encode()).hexdigest()
        if digest != source["content_sha256"]:
            raise ValueError(f"Source checksum mismatch: {archive}")
        archives.append((relative, archive))
        for name, mode, body in inventory:
            combined = str(relative / name)
            if combined in paths:
                raise ValueError(f"Source groups overlap: {combined}")
            paths.add(combined)
            if mode == SYMLINK:
                target = destination / relative / PurePosixPath(name).parent / body
                try:
                    target.resolve(strict=False).relative_to(destination.resolve())
                except ValueError as error:
                    raise ValueError(f"Source link escapes group: {combined}") from error
                links[combined] = body
    for name in paths:
        if any(str(parent) in links for parent in PurePosixPath(name).parents):
            raise ValueError(f"Source entry passes through a link: {name}")
    for relative, archive in archives:
        for name, mode, body in source_archive_entries(archive):
            if mode == SYMLINK:
                continue
            target = destination / relative / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(body)
            target.chmod(mode)
    for name, body in links.items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(body)
    for name in links:
        try:
            (destination / name).resolve(strict=False).relative_to(destination.resolve())
        except (ValueError, RuntimeError) as error:
            raise ValueError(f"Invalid source link chain: {name}") from error
    return len(paths)
