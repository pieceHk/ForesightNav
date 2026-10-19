"""External matching sources retain their pinned behavior and offline provenance."""

import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from runtime.external_sources import SUPERGLUE_FILES, pinned_superglue_manifest, verify_external_source
from scripts.prepare_dependencies import prepare_superglue
from scripts.setup_navigation import checked_artifacts

ROOT = Path(__file__).resolve().parents[2]
PREFIX = "Navigation/runtime/lagmemo/src/lagmemo/lagmemo/agent/imagenav_agent/SuperGluePretrainedNetwork"


def fixture(tmp_path):
    root, cache = tmp_path / "project", tmp_path / "cache"
    files = []
    for name in sorted(SUPERGLUE_FILES):
        raw = b"VALUE = 'original'\n" if name.endswith(".py") else b"terms\n"
        prepared = raw.replace(b"original", b"selected") if name == "models/matching.py" else raw
        row = {"path": name, "cache_path": f"sources/superglue/{name}", "url": "https://example.test/" + name,
               "sha256": hashlib.sha256(raw).hexdigest(), "installed_sha256": hashlib.sha256(prepared).hexdigest(), "bytes": len(raw)}
        files.append(row)
        path = cache / row["cache_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    manifest = {"schema": "pbsp_superglue_source_v1", "revision": "a" * 40,
                "repository": "https://example.test/superglue", "files": files,
                "patches": {"models/matching.py": [{"before": "original", "after": "selected"}]}}
    path = root / "assets/superglue.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(manifest))
    lock = {"runtime_source_manifests": [{"path": "assets/superglue.json", "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}]}
    lock_path = root / "envs/navigation/linux-64.lock.json"
    lock_path.parent.mkdir(parents=True)
    lock_path.write_text(json.dumps(lock))
    return root, cache, manifest


def test_external_source_preparation_is_offline_exact_and_repeatable(tmp_path, monkeypatch):
    root, cache, manifest = fixture(tmp_path)
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: pytest.fail("offline source accessed network"))
    options = dict(root=root, cache=cache, offline=True)
    assert prepare_superglue(**options, download_only=True)["files"] == 6
    assert not (root / "Exp").exists()
    first = prepare_superglue(**options)
    assert prepare_superglue(**options) == first
    target = root / "Exp/assets/dependencies/superglue"
    assert (target / "models/matching.py").read_text() == "VALUE = 'selected'\n"
    assert verify_external_source(target, manifest) == first


@pytest.mark.parametrize("change", ["content", "extra", "symlink"])
def test_changed_external_source_is_rejected_without_replacement(tmp_path, change):
    root, cache, manifest = fixture(tmp_path)
    prepare_superglue(root=root, cache=cache, offline=True)
    target = root / "Exp/assets/dependencies/superglue"
    code = target / "models/matching.py"
    if change == "content":
        code.write_text("modified")
    elif change == "extra":
        (target / "extra.py").touch()
    else:
        outside = tmp_path / "outside.py"
        outside.write_bytes(code.read_bytes())
        code.unlink()
        code.symlink_to(outside)
    before = code.read_bytes()
    with pytest.raises(ValueError):
        prepare_superglue(root=root, cache=cache, offline=True)
    assert code.read_bytes() == before


def test_manifest_and_offline_cache_are_both_pinned_before_preparation(tmp_path):
    root, cache, manifest = fixture(tmp_path)
    source = cache / manifest["files"][0]["cache_path"]
    source.write_bytes(b"different")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        prepare_superglue(root=root, cache=cache, offline=True)
    assert not (root / "Exp").exists()
    (root / "assets/superglue.json").write_text(json.dumps({**manifest, "revision": "b" * 40}))
    with pytest.raises(ValueError, match="manifest checksum"):
        pinned_superglue_manifest(root)


def test_compatibility_package_redirects_imports_to_verified_external_source(tmp_path, monkeypatch):
    root, cache, manifest = fixture(tmp_path)
    prepare_superglue(root=root, cache=cache, offline=True)
    target = root / "Exp/assets/dependencies/superglue"
    monkeypatch.setattr("runtime.external_sources.superglue_directory", lambda: target)
    spec = importlib.util.spec_from_file_location("pbsp_test_external_matching", ROOT / PREFIX / "__init__.py", submodule_search_locations=[])
    package = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, package)
    spec.loader.exec_module(package)
    try:
        module = importlib.import_module(spec.name + ".models.matching")
        assert module.VALUE == "selected" and Path(module.__file__).is_relative_to(target)
    finally:
        for name in list(sys.modules):
            if name.startswith(spec.name + "."):
                sys.modules.pop(name)


def test_prepared_file_hashes_preserve_the_frozen_matching_implementation():
    manifest = pinned_superglue_manifest(ROOT)
    for row in manifest["files"]:
        frozen = subprocess.check_output(["git", "show", f"42ca185:{PREFIX}/{row['path']}"], cwd=ROOT)
        assert hashlib.sha256(frozen).hexdigest() == row["installed_sha256"]


def test_navigation_cache_includes_all_separately_obtained_matching_files():
    lock = json.loads((ROOT / "envs/navigation/linux-64.lock.json").read_text())
    artifacts = checked_artifacts(lock, ROOT)
    selected = {row["cache_path"] for row in pinned_superglue_manifest(ROOT)["files"]}
    assert selected <= {row["cache_path"] for row in artifacts}


def test_existing_superglue_license_is_retained_byte_for_byte():
    frozen = subprocess.check_output(["git", "show", f"42ca185:{PREFIX}/LICENSE"], cwd=ROOT)
    assert (ROOT / "third_party/licenses/superglue/LICENSE").read_bytes() == frozen
