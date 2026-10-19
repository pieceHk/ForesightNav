"""Exercise pinned recursive-source assembly and offline installation checks."""

import hashlib
import io
from pathlib import Path
import tarfile
from types import SimpleNamespace

import pytest

from scripts._sources import materialize_source_group, source_archive_hash
from scripts.setup_navigation import checked_artifacts, install


def make_archive(path, entries):
    with tarfile.open(path, "w:gz") as archive:
        for name, body in entries:
            member = tarfile.TarInfo(name)
            if isinstance(body, tuple):
                member.type, member.linkname = body
                archive.addfile(member)
            else:
                member.size = len(body)
                member.mode = 0o755 if name.endswith(".sh") else 0o644
                archive.addfile(member, io.BytesIO(body))
    return path


def artifact(path, key):
    return {"id": key, "cache_path": path.name, "content_sha256": source_archive_hash(path), "source_archive": True}


def test_recursive_sources_keep_bounded_links_and_shared_submodule_archives(tmp_path):
    main = make_archive(tmp_path / "main.tar.gz", [("project/README", b"main"), ("project/docs/readme", (tarfile.SYMTYPE, "../README"))])
    child = make_archive(tmp_path / "child.tar.gz", [("child/build.sh", b"exit 0\n")])
    modules = [{"path": ".", "source": "main"}, {"path": "a/toolchains", "source": "child"}, {"path": "b/toolchains", "source": "child"}]
    destination = tmp_path / "source"
    assert materialize_source_group(tmp_path, [artifact(main, "main"), artifact(child, "child")], modules, destination) == 4
    assert (destination / "docs/readme").is_symlink()
    assert (destination / "docs/readme").read_bytes() == b"main"
    assert (destination / "a/toolchains/build.sh").read_bytes() == b"exit 0\n"
    assert (destination / "b/toolchains/build.sh").stat().st_mode & 0o111


@pytest.mark.parametrize("entries", [
    [("../outside", b"bad")],
    [("/outside", b"bad")],
    [("project/link", (tarfile.SYMTYPE, "/outside"))],
    [("project/link", (tarfile.LNKTYPE, "target"))],
    [("project/code.py", b"one"), ("project/code.py", b"two")],
    [("one/code.py", b"one"), ("two/code.py", b"two")],
])
def test_invalid_source_archive_rejected_before_writing(tmp_path, entries):
    archive = make_archive(tmp_path / "source.tar.gz", entries)
    destination = tmp_path / "source"
    with pytest.raises(ValueError):
        materialize_source_group(tmp_path, [{"id": "source", "cache_path": archive.name, "content_sha256": "0" * 64}], [{"path": ".", "source": "source"}], destination)
    assert not destination.exists()


def test_parent_link_cannot_redirect_a_source_file_write(tmp_path):
    archive = make_archive(tmp_path / "source.tar.gz", [("root/alias", (tarfile.SYMTYPE, "target")), ("root/alias/code.py", b"bad")])
    destination = tmp_path / "source"
    with pytest.raises(ValueError, match="passes through a link"):
        materialize_source_group(tmp_path, [artifact(archive, "source")], [{"path": ".", "source": "source"}], destination)
    assert not destination.exists()


def test_cross_module_relative_link_is_bounded_by_the_complete_group(tmp_path):
    main = make_archive(tmp_path / "main.tar.gz", [("root/README", b"main")])
    child = make_archive(tmp_path / "child.tar.gz", [("child/readme", (tarfile.SYMTYPE, "../README"))])
    modules = [{"path": ".", "source": "main"}, {"path": "child", "source": "child"}]
    materialize_source_group(tmp_path, [artifact(main, "main"), artifact(child, "child")], modules, tmp_path / "source")
    assert (tmp_path / "source/child/readme").read_bytes() == b"main"


def test_link_outside_complete_group_fails_before_materialization(tmp_path):
    archive = make_archive(tmp_path / "source.tar.gz", [("root/readme", (tarfile.SYMTYPE, "../outside"))])
    destination = tmp_path / "source"
    with pytest.raises(ValueError, match="escapes group"):
        materialize_source_group(tmp_path, [artifact(archive, "source")], [{"path": ".", "source": "source"}], destination)
    assert not destination.exists()


def test_module_overlap_or_changed_source_never_overwrites_destination(tmp_path):
    archive = make_archive(tmp_path / "source.tar.gz", [("root/code.py", b"source")])
    source = artifact(archive, "source")
    module = {"path": ".", "source": "source"}
    with pytest.raises(ValueError, match="overlap"):
        materialize_source_group(tmp_path, [source], [module, module], tmp_path / "overlap")
    source["content_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="checksum mismatch"):
        materialize_source_group(tmp_path, [source], [module], tmp_path / "changed")
    assert not (tmp_path / "overlap").exists() and not (tmp_path / "changed").exists()


def minimal_lock(item):
    return {"conda": [], "packages": [item], "sources": [{"id": "source", "cache_path": "source.tar.gz", "sha256": hashlib.sha256(b"source").hexdigest()}], "source_groups": {name: [{"path": ".", "source": "source"}] for name in ("habitat_sim", "habitat_lab", "imath", "detectron2")}, "tracked_sources": []}


def test_download_only_checks_cache_without_creating_a_prefix(tmp_path):
    (tmp_path / "wheel.whl").write_bytes(b"wheel")
    (tmp_path / "source.tar.gz").write_bytes(b"source")
    lock = minimal_lock({"cache_path": "wheel.whl", "sha256": hashlib.sha256(b"wheel").hexdigest()})
    args = SimpleNamespace(cache=tmp_path, offline=True, download_only=True, prefix=tmp_path / "env")
    install(args, lock, root=tmp_path)
    assert not args.prefix.exists() and not (tmp_path / "Exp").exists()
    (tmp_path / "wheel.whl").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        install(args, lock, root=tmp_path)
    assert not args.prefix.exists()


@pytest.mark.parametrize("mutation", ("duplicate", "traversal", "unknown_source"))
def test_invalid_locks_fail_before_download_or_environment_mutation(tmp_path, mutation):
    lock = minimal_lock({"cache_path": "wheel.whl", "sha256": "0" * 64})
    if mutation == "duplicate":
        lock["packages"].append(lock["packages"][0])
    elif mutation == "traversal":
        lock["packages"][0]["cache_path"] = "../outside"
    else:
        lock["source_groups"]["habitat_sim"][0]["source"] = "missing"
    with pytest.raises(ValueError):
        checked_artifacts(lock, tmp_path)
    assert not list(tmp_path.iterdir())
