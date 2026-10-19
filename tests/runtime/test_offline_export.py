"""Offline exports restore the committed code and reject incomplete transfers."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from scripts.export_offline import export_offline, verify_export


def fixture(tmp_path):
    root, cache = tmp_path / "repo", tmp_path / "cache"
    root.mkdir()
    cache.mkdir()
    files = {"packages/shared.whl": b"environment", "sources/source.tar": b"source",
             "datasets/objectnav/val/content/scene.json.gz": b"episodes"}
    model = b"model"
    model_sha = hashlib.sha256(model).hexdigest()
    files["models/" + model_sha] = model
    for name, body in files.items():
        path = cache / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
    def artifact(name):
        return {"cache_path": name, "sha256": hashlib.sha256(files[name]).hexdigest()}
    mapping = {"conda": [], "packages": [artifact("packages/shared.whl")], "sources": []}
    navigation = {**mapping, "sources": [{"id": "source", **artifact("sources/source.tar")}],
                  "tracked_sources": [], "source_groups": {
                      key: [{"path": ".", "source": "source"}]
                      for key in ("habitat_sim", "habitat_lab", "imath", "detectron2")}}
    for role, lock in (("mapping", mapping), ("navigation", navigation)):
        path = root / f"envs/{role}/linux-64.lock.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(lock))
    (root / "assets").mkdir()
    (root / "assets/models.json").write_text(json.dumps({
        "schema": "pbsp_models_v1", "archives": [], "refs": [], "files": [{
            "path": "model.bin", "url": "https://example.test/model", "bytes": len(model), "sha256": model_sha}]}))
    body = files["datasets/objectnav/val/content/scene.json.gz"]
    (root / "assets/datasets.json").write_text(json.dumps({"objectnav_source": {
        "content_prefix": "objectnav/val/content/", "files": [{
            "scene_id": "scene", "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}]}}))
    (root / "README.md").write_text("Committed PBSP fixture\n")
    (root / "readme-link").symlink_to("README.md")
    (root / ".gitignore").write_text("runtime.local.yaml\n")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "symbolic-ref", "HEAD", "refs/heads/test-offline"], check=True)
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(["git", "-C", str(root), "-c", "core.hooksPath=/dev/null", "-c", "commit.gpgsign=false",
                    "-c", "user.name=PBSP tests", "-c", "user.email=tests@invalid", "commit", "-qm", "fixture"], check=True)
    (root / "runtime.local.yaml").write_text("machine-local: excluded\n")
    (cache / "unfinished.partial").write_bytes(b"not an input")
    return root, cache, tmp_path / "export", files


def test_complete_export_can_be_transferred_and_cloned_with_the_same_commit(tmp_path):
    root, cache, target, files = fixture(tmp_path)
    result = export_offline(root=root, cache=cache, destination=target)
    assert result["groups"] == ["environments", "episodes", "models"]
    assert result["files"] == len(files) + 1
    moved = tmp_path / "transferred"
    shutil.copytree(target, moved)
    assert verify_export(moved)["code"] == result["code"]
    restored = tmp_path / "restored"
    subprocess.run(["git", "clone", "-q", str(moved / "code.bundle"), str(restored)], check=True)
    assert subprocess.check_output(["git", "-C", str(restored), "rev-parse", "HEAD"], text=True).strip() == result["code"]["commit"]
    assert (restored / "readme-link").is_symlink()
    assert (restored / "readme-link").read_bytes() == (root / "README.md").read_bytes()
    assert not (restored / "runtime.local.yaml").exists()
    assert not (moved / "cache/unfinished.partial").exists()
    for name, body in files.items():
        assert (moved / "cache" / name).read_bytes() == body


@pytest.mark.parametrize("change", ["missing", "corrupt", "link"])
def test_incomplete_cache_never_creates_an_export_directory(tmp_path, change):
    root, cache, target, files = fixture(tmp_path)
    model = next(cache / name for name in files if name.startswith("models/"))
    if change == "corrupt":
        model.write_bytes(b"changed")
    else:
        body = model.read_bytes()
        model.unlink()
        if change == "link":
            outside = tmp_path / "outside-model"
            outside.write_bytes(body)
            model.symlink_to(outside)
    with pytest.raises((ValueError, FileNotFoundError)):
        export_offline(root=root, cache=cache, destination=target)
    assert not target.exists() and not list(tmp_path.glob(".pbsp-offline-*"))


@pytest.mark.parametrize("change", ["content", "extra", "mode", "path"])
def test_transfer_verification_detects_corruption_and_inventory_changes(tmp_path, change):
    root, cache, target, files = fixture(tmp_path)
    export_offline(root=root, cache=cache, destination=target)
    if change == "content":
        (target / "cache/packages/shared.whl").write_bytes(b"broken")
    elif change == "extra":
        (target / "extra.bin").write_bytes(b"unlisted")
    elif change == "mode":
        (target / "cache/packages/shared.whl").chmod(0o755)
    else:
        path = target / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["files"][0]["path"] = "../escape"
        path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        verify_export(target)


def test_selected_groups_are_explicit_and_cannot_claim_a_complete_export(tmp_path):
    root, cache, target, files = fixture(tmp_path)
    for name in files:
        if name.startswith("models/"):
            (cache / name).unlink()
    report = export_offline(root=root, cache=cache, destination=target, groups=["environments"])
    assert report["groups"] == ["environments"] and report["files"] == 3
    assert not (target / "cache/models").exists()


def test_copy_failure_removes_staging_and_preserves_source_cache(tmp_path, monkeypatch):
    root, cache, target, files = fixture(tmp_path)
    def fail(*args):
        raise OSError("transfer failed")
    monkeypatch.setattr("scripts.export_offline.shutil.copyfile", fail)
    with pytest.raises(OSError, match="transfer failed"):
        export_offline(root=root, cache=cache, destination=target)
    assert not target.exists() and not list(tmp_path.glob(".pbsp-offline-*"))
    assert (cache / "packages/shared.whl").read_bytes() == files["packages/shared.whl"]


def test_dirty_code_and_existing_output_are_not_exported_or_overwritten(tmp_path):
    root, cache, target, files = fixture(tmp_path)
    (root / "README.md").write_text("uncommitted")
    with pytest.raises(ValueError, match="Commit the checkout"):
        export_offline(root=root, cache=cache, destination=target)
    target.mkdir()
    with pytest.raises(FileExistsError):
        export_offline(root=root, cache=cache, destination=target)


def test_linux_micromamba_is_optional_and_preserves_executable_mode(tmp_path):
    root, cache, target, files = fixture(tmp_path)
    tool = tmp_path / "micromamba"
    header = bytearray(64)
    header[:6], header[18:20] = b"\x7fELF\x02\x01", b">\x00"
    tool.write_bytes(header)
    result = export_offline(root=root, cache=cache, destination=target, micromamba=tool)
    assert (target / "tools/micromamba").stat().st_mode & 0o111
    assert (target / "tools/micromamba").read_bytes() == header
    assert result["files"] == len(files) + 2
