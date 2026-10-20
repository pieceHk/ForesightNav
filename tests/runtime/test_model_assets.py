"""Check relocatable offline assets, changed inputs and selected archive reads."""

import hashlib
import io
import json
from pathlib import Path
import shutil
import tarfile

import pytest

from runtime.resources import ResourcePaths
from scripts.prepare_models import extract_selected, load_manifest, prepare


def manifest(tmp_path):
    item = {"path": "model/weights.bin", "bytes": 7, "sha256": hashlib.sha256(b"weights").hexdigest(), "url": "https://example.org/model"}
    data = {"schema": "pbsp_models_v1", "files": [item], "archives": [], "refs": [{"path": "hf/refs/main", "value": "a" * 40}]}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(data))
    return load_manifest(path)


def test_imported_assets_are_independent_and_offline_reusable(tmp_path, monkeypatch):
    data = manifest(tmp_path)
    source, target = tmp_path / "source", tmp_path / "relocated weights"
    (source / "model").mkdir(parents=True)
    (source / "model/weights.bin").write_bytes(b"weights")
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: pytest.fail("offline import accessed network"))
    options = dict(destination=target, source=source, cache=tmp_path / "cache", offline=True)
    assert prepare(data, **options) == 1
    assert not (target / "model/weights.bin").is_symlink()
    shutil.rmtree(source)
    assert prepare(data, destination=target, cache=tmp_path / "cache", verify_only=True) == 1
    (target / "hf/refs/main").write_text("b" * 40)
    with pytest.raises(ValueError, match="revision differs"):
        prepare(data, destination=target, cache=tmp_path / "cache", verify_only=True)


def test_corrupt_source_or_target_never_replaces_existing_assets(tmp_path):
    data = manifest(tmp_path)
    source, target = tmp_path / "source", tmp_path / "target"
    (source / "model").mkdir(parents=True)
    (source / "model/weights.bin").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        prepare(data, destination=target, cache=tmp_path / "cache", source=source)
    assert not target.exists()
    (source / "model/weights.bin").write_bytes(b"weights")
    (target / "model").mkdir(parents=True)
    (target / "model/weights.bin").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        prepare(data, destination=target, cache=tmp_path / "cache", source=source)
    assert (target / "model/weights.bin").read_bytes() == b"corrupt"


def test_offline_cache_does_not_fall_back_to_network(tmp_path, monkeypatch):
    data = manifest(tmp_path)
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: pytest.fail("offline preparation accessed network"))
    with pytest.raises(FileNotFoundError, match="Missing cached model"):
        prepare(data, destination=tmp_path / "target", cache=tmp_path / "cache", offline=True)
    assert not (tmp_path / "target").exists()


def test_failed_download_keeps_processing_other_cache_artifacts(tmp_path, monkeypatch):
    data = manifest(tmp_path)
    data["files"] = [{**data["files"][0], "path": f"model/{index}.bin", "sha256": str(index) * 64} for index in range(3)]
    calls = []

    def download(cache, item, offline):
        calls.append(item["path"])
        if item["path"] == "model/0.bin":
            raise OSError("first source failed")
        return cache / item["sha256"]

    monkeypatch.setattr("scripts.prepare_models.fetch", download)
    with pytest.raises(OSError, match="first source failed"):
        prepare(data, destination=tmp_path / "target", cache=tmp_path / "cache", download_only=True, jobs=1)
    assert calls == ["model/0.bin", "model/1.bin", "model/2.bin"]
    assert not (tmp_path / "target").exists()


@pytest.mark.parametrize("path", ["../escape", "/absolute", "a/../escape", "model", "model/weights.bin/child"])
def test_manifest_rejects_traversal_and_file_directory_collisions(tmp_path, path):
    data = manifest(tmp_path)
    data["refs"].append({"path": path, "value": "b" * 40})
    output = tmp_path / "invalid.json"
    output.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_manifest(output)


@pytest.mark.skipif(shutil.which("zstd") is None, reason="zstd is required for compressed model archives")
@pytest.mark.parametrize("member_name", ("FastSAM-x.pt", "pretrained/checkpoint.bin"))
def test_model_archive_only_materializes_selected_regular_members(tmp_path, member_name):
    import subprocess

    data = manifest(tmp_path)
    item = {**data["files"][0], "member": member_name}
    archive = tmp_path / "models.tar"
    with tarfile.open(archive, "w") as output:
        for name, payload in [("unrelated/large.bin", b"unused"), (item["member"], b"weights")]:
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            output.addfile(member, io.BytesIO(payload))
    subprocess.run(["zstd", "-q", str(archive)], check=True)
    extract_selected(archive.with_suffix(".tar.zst"), [item], tmp_path / "staging")
    assert (tmp_path / "staging/model/weights.bin").read_bytes() == b"weights"
    assert not (tmp_path / "staging/unrelated").exists()


def test_resources_follow_selected_weights_without_source_bridges(tmp_path):
    paths = ResourcePaths(tmp_path / "code", tmp_path / "data", tmp_path / "weights elsewhere")
    resources = [*vars(paths.mapping_resources()).values(), *vars(paths.navigation_resources().matching).values(), paths.navigation_resources().seem_tokenizer, paths.navigation_resources().detic_category_frequency]
    for path in resources:
        path.relative_to(paths.weights)
        assert paths.root not in path.parents
    assert paths.weights / "clip/ViT-B-32.pt" in paths.required_files()


def test_public_model_manifest_loads_and_includes_no_optional_backends():
    data = load_manifest(Path(__file__).resolve().parents[2] / "assets/models.json")
    assert len(data["files"]) == 22
    assert sum(item["bytes"] for item in data["files"]) == 5631339793
    assert not any(item["path"].endswith((".h5", ".msgpack", "superglue_outdoor.pth")) for item in data["files"])
