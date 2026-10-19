"""Verify cached sources and environment identity before installation."""

import hashlib
import io
from pathlib import Path
import subprocess
import sys
import tarfile
from types import SimpleNamespace
import venv

import pytest

from scripts.setup_mapping import (
    archive_hash, check_prefix, content_hash, fetch_artifact,
    install, materialize_source, source_files,
)


def archive(path, entries):
    with tarfile.open(path, "w:gz") as output:
        for name, data, mode, kind in entries:
            member = tarfile.TarInfo(name)
            member.mode, member.type = mode, kind
            member.size = len(data) if kind == tarfile.REGTYPE else 0
            if kind == tarfile.SYMTYPE:
                member.linkname = "outside"
            output.addfile(member, io.BytesIO(data) if member.isfile() else None)
    return path


def test_source_archive_hash_ignores_export_root_and_group_write_bit(tmp_path):
    first = archive(tmp_path / "main.tar.gz", [
        ("repo-main/code.py", b"value = 1\n", 0o664, tarfile.REGTYPE),
        ("repo-main/build.sh", b"exit 0\n", 0o775, tarfile.REGTYPE),
    ])
    second = archive(tmp_path / "commit.tar.gz", [
        ("repo-abcdef/build.sh", b"exit 0\n", 0o755, tarfile.REGTYPE),
        ("repo-abcdef/code.py", b"value = 1\n", 0o644, tarfile.REGTYPE),
    ])
    assert archive_hash(first) == archive_hash(second)
    source = {"cache_path": second.name, "content_sha256": archive_hash(first)}
    materialize_source(tmp_path, tmp_path, source, tmp_path / "extracted")
    assert (tmp_path / "extracted/code.py").read_bytes() == b"value = 1\n"
    assert (tmp_path / "extracted/build.sh").stat().st_mode & 0o111


@pytest.mark.parametrize("entries", [
    [("../outside", b"bad", 0o644, tarfile.REGTYPE)],
    [("/outside", b"bad", 0o644, tarfile.REGTYPE)],
    [("repo/code.py", b"", 0o644, tarfile.SYMTYPE)],
    [("one/code.py", b"a", 0o644, tarfile.REGTYPE),
     ("two/code.py", b"a", 0o644, tarfile.REGTYPE)],
    [("one/code.py", b"a", 0o644, tarfile.REGTYPE),
     ("one/code.py", b"b", 0o644, tarfile.REGTYPE)],
])
def test_invalid_source_archives_are_rejected_before_extraction(tmp_path, entries):
    path = archive(tmp_path / "source.tar.gz", entries)
    source = {"cache_path": path.name, "content_sha256": "0" * 64}
    with pytest.raises(ValueError):
        materialize_source(tmp_path, tmp_path, source, tmp_path / "extracted")
    assert not (tmp_path / "extracted").exists()


def test_cached_source_changes_fail_before_writing_a_build_tree(tmp_path):
    path = archive(tmp_path / "source.tar.gz", [("root/code.py", b"changed", 0o644, tarfile.REGTYPE)])
    source = {"cache_path": path.name, "content_sha256": "0" * 64}
    with pytest.raises(ValueError, match="Checksum mismatch"):
        materialize_source(tmp_path, tmp_path, source, tmp_path / "extracted")
    assert not (tmp_path / "extracted").exists()


def test_tracked_sources_exclude_existing_build_products_and_detect_edits(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    directory = tmp_path / "extension"
    directory.mkdir()
    (directory / "setup.py").write_bytes(b"source\n")
    (directory / "old.so").write_bytes(b"old compiled library")
    subprocess.run(["git", "-C", str(tmp_path), "add", "extension/setup.py"], check=True)
    source = {"repository_path": "extension", "content_sha256": content_hash([
        ("setup.py", 0o644, hashlib.sha256(b"source\n").hexdigest()),
    ])}
    materialize_source(tmp_path, tmp_path, source, tmp_path / "build")
    assert sorted(path.name for path in (tmp_path / "build").iterdir()) == ["setup.py"]
    (directory / "setup.py").write_bytes(b"changed\n")
    with pytest.raises(ValueError, match="differs from the mapping lock"):
        source_files(tmp_path, source)


def test_offline_missing_or_corrupt_cache_does_not_create_download_files(tmp_path):
    item = {"cache_path": "wheel.whl", "sha256": hashlib.sha256(b"good").hexdigest()}
    with pytest.raises(FileNotFoundError, match="Missing cached artifact"):
        fetch_artifact(tmp_path, item, offline=True)
    assert not list(tmp_path.iterdir())
    (tmp_path / "wheel.whl").write_bytes(b"bad")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        fetch_artifact(tmp_path, item, offline=True)
    assert list(tmp_path.iterdir()) == [tmp_path / "wheel.whl"]


def test_download_only_does_not_create_an_environment(tmp_path):
    (tmp_path / "wheel.whl").write_bytes(b"wheel")
    lock = {"conda": [], "sources": [], "packages": [
        {"cache_path": "wheel.whl", "sha256": hashlib.sha256(b"wheel").hexdigest()},
    ]}
    args = SimpleNamespace(cache=tmp_path, offline=True, download_only=True, prefix=tmp_path / "env")
    install(args, lock, root=tmp_path)
    assert not args.prefix.exists()
    assert not (tmp_path / "Exp").exists()


def test_existing_prefix_cannot_borrow_another_environments_python(tmp_path):
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin/python").symlink_to(sys.executable)
    with pytest.raises(ValueError, match="locked Python version"):
        check_prefix(tmp_path, {"python": list(sys.version_info[:2]), "conda": []})


def test_existing_prefix_must_have_the_locked_conda_base():
    prefix = Path(sys.prefix)
    with pytest.raises(ValueError, match="locked Conda base package"):
        check_prefix(prefix, {"python": list(sys.version_info[:2]), "conda": [
            {"name": "pbsp_missing_base", "version": "1", "build": "0"},
        ]})


def test_missing_pip_is_detected_even_when_python_and_metadata_match(tmp_path):
    prefix = tmp_path / "bare"
    venv.EnvBuilder(with_pip=False).create(prefix)
    with pytest.raises(ValueError, match="did not link pip"):
        check_prefix(prefix, {"python": list(sys.version_info[:2]), "conda": []})
