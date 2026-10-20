"""Preserve new source files without following external links or hiding gaps."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from runtime.launch import source_info
from runtime.source_snapshot import capture_source_snapshot, digest, verify_source_snapshot
from scripts.summarize import read_json, summarize_runs
from tests.experiments.test_result_summary import fixture as result_fixture, write


@pytest.fixture
def repository(tmp_path):
    root = tmp_path / "source"
    root.mkdir()

    def git(*args):
        return subprocess.run(["git", "-C", str(root), "-c", "core.hooksPath=/dev/null",
                               "-c", "commit.gpgsign=false", "-c", "user.name=Snapshot test",
                               "-c", "user.email=snapshot@example.invalid", *args],
                              check=True, capture_output=True, text=True)

    git("init", "-q")
    (root / ".gitignore").write_text("outputs/\nignored.bin\n")
    (root / "tracked.py").write_text("value = 1\n")
    git("add", ".")
    git("commit", "-qm", "Test source")
    return root


def capture(root, *, max_bytes=64 * 1024 * 1024, state_reader=source_info):
    output = root / "outputs/run"
    output.mkdir(parents=True)
    source = source_info(root)
    (output / "source.patch").write_text(source["patch"])
    proof = capture_source_snapshot(root, output, source, state_reader=state_reader, max_bytes=max_bytes)
    return output, source, proof


def test_dirty_snapshot_is_portable_after_original_source_is_removed(repository, tmp_path):
    (repository / "tracked.py").write_text("value = 2\n")
    (repository / "adapter 新.py").write_bytes(b"def plan(): return 1\n")
    (repository / "adapter 新.py").chmod(0o755)
    (repository / "binary\ninput.bin").write_bytes(b"\x00\xff\x01")
    (repository / "ignored.bin").write_bytes(b"not part of the source snapshot")
    outside = tmp_path / "external.bin"
    outside.write_bytes(b"external contents must not be copied")
    (repository / "external_link").symlink_to(outside)
    output, source, proof = capture(repository)
    assert proof["complete"] is True
    assert proof["untracked_objects"] == 3
    assert not (output / "source_untracked/external_link").exists()
    relocated = tmp_path / "relocated snapshot"
    shutil.copytree(output, relocated)
    shutil.rmtree(repository)
    outside.unlink()
    checked = verify_source_snapshot(relocated, proof, source, read_json)
    assert checked["complete"] is True
    assert (relocated / "source_untracked/adapter 新.py").read_bytes() == b"def plan(): return 1\n"
    assert b"value = 2" in (relocated / "source.patch").read_bytes()


def test_byte_budget_records_an_explicit_partial_snapshot(repository):
    (repository / "large.bin").write_bytes(b"0123456789")
    output, source, proof = capture(repository, max_bytes=4)
    assert proof["complete"] is False
    assert proof["captured_file_bytes"] == 0
    manifest = read_json(output / "source_snapshot.json")
    assert manifest["untracked"][0]["reason"] == "untracked_byte_budget"
    assert verify_source_snapshot(output, proof, source, read_json)["complete"] is False


def test_source_changes_during_capture_are_not_claimed_complete(repository):
    def changed(root):
        result = source_info(root)
        result["patch"] += "source changed during capture"
        return result

    output, source, proof = capture(repository, state_reader=changed)
    assert proof["complete"] is False
    assert verify_source_snapshot(output, proof, source, read_json)["complete"] is False


@pytest.mark.parametrize("case", ["bytes", "mode", "extra", "symlink", "manifest", "patch", "coverage", "traversal"])
def test_modified_or_inconsistent_snapshots_are_rejected(repository, case):
    (repository / "adapter.py").write_bytes(b"version 1\n")
    output, source, proof = capture(repository)
    copied = output / "source_untracked/adapter.py"
    if case == "bytes": copied.write_bytes(b"version 2\n")
    elif case == "mode": copied.chmod(0o755)
    elif case == "extra": (output / "source_untracked/extra.py").write_text("extra")
    elif case == "symlink":
        copied.unlink()
        copied.symlink_to(repository / "adapter.py")
    elif case == "patch": (output / "source.patch").write_text("changed patch")
    elif case == "coverage": proof["complete"] = False
    else:
        path = output / "source_snapshot.json"
        data = read_json(path)
        if case == "manifest": data["commit"] = "other"
        else: data["untracked"][0]["path"] = "../escape.py"
        write(path, data)
        if case == "traversal": proof["manifest_sha256"] = digest(path)
    with pytest.raises(ValueError):
        verify_source_snapshot(output, proof, source, read_json)


def test_non_git_source_gets_a_partial_record(tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    source = source_info(tmp_path)
    (output / "source.patch").write_text(source["patch"])
    proof = capture_source_snapshot(tmp_path, output, source, state_reader=source_info)
    assert proof["complete"] is False
    assert verify_source_snapshot(output, proof, source, read_json)["complete"] is False


def test_partial_snapshot_cannot_be_merged_even_with_clean_source_status(tmp_path):
    first, second = result_fixture(tmp_path), result_fixture(tmp_path, 1)
    provenance = read_json(first / "provenance.json")
    manifest = {"schema_version": "pbsp_source_snapshot_v1", "commit": provenance["commit"],
                "status": [], "tracked_patch_sha256": hashlib.sha256(b"").hexdigest(),
                "untracked": [], "complete": False, "inventory_stable": False, "tracked_state_stable": True}
    (first / "source_untracked").mkdir()
    write(first / "source_snapshot.json", manifest)
    provenance["source_snapshot"] = {"schema_version": "pbsp_source_snapshot_v1",
        "manifest_sha256": digest(first / "source_snapshot.json"), "complete": False,
        "untracked_objects": 0, "captured_file_bytes": 0}
    write(first / "provenance.json", provenance)
    assert summarize_runs([first])["rows"][0]["source_snapshot_complete"] is False
    with pytest.raises(ValueError, match="complete source snapshots"):
        summarize_runs([first, second], merge=True)
