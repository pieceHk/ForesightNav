"""Verify archive exports without requiring an embedded Git checkout."""

import hashlib
import io
import json
from pathlib import Path
import tarfile

import pytest

from runtime.source_provenance import MARKER, verify_source
from scripts._sources import source_archive_hash
from scripts.prepare_dependencies import prepare_habitat


def fixture(tmp_path):
    root, cache = tmp_path / "project", tmp_path / "cache"
    archive = cache / "habitat.tar.gz"
    cache.mkdir()
    with tarfile.open(archive, "w:gz") as output:
        member = tarfile.TarInfo("export/habitat-lab/habitat/__init__.py")
        payload = b"VERSION = 'pinned'\n"
        member.size, member.mode = len(payload), 0o644
        output.addfile(member, io.BytesIO(payload))
    source = {"id": "habitat-pinned", "revision": "a" * 40, "url": "https://example.org/habitat.tar.gz", "cache_path": archive.name, "content_sha256": source_archive_hash(archive), "files": 1, "source_archive": True}
    lock = {"sources": [source], "source_groups": {"habitat_lab": [{"path": ".", "source": source["id"]}]}}
    (root / "envs/navigation").mkdir(parents=True)
    (root / "envs/navigation/linux-64.lock.json").write_text(json.dumps(lock))
    return root, cache, source


def test_archive_source_preparation_is_offline_and_idempotent(tmp_path, monkeypatch):
    root, cache, source = fixture(tmp_path)
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: pytest.fail("offline source preparation accessed network"))
    options = dict(root=root, cache=cache, offline=True)
    first = prepare_habitat(**options)
    assert first["kind"] == "archive" and first["content_sha256"] == source["content_sha256"]
    assert prepare_habitat(**options) == first
    path = root / "Exp/assets/dependencies/habitat-lab"
    assert not (path / ".git").exists()
    (path / "habitat-lab/habitat/__pycache__").mkdir()
    (path / "habitat-lab/habitat/__pycache__/runtime.pyc").write_bytes(b"cache")
    assert verify_source(path, source) == first


@pytest.mark.parametrize("change", ["content", "extra", "manifest"])
def test_changed_source_is_rejected_without_replacement(tmp_path, change):
    root, cache, source = fixture(tmp_path)
    prepare_habitat(root=root, cache=cache, offline=True)
    path = root / "Exp/assets/dependencies/habitat-lab"
    if change == "content":
        (path / "habitat-lab/habitat/__init__.py").write_text("changed")
    elif change == "extra":
        (path / "extra.py").touch()
    else:
        payload = json.loads((path / MARKER).read_text())
        payload["entries"][0][0] = "../../escape"
        (path / MARKER).write_text(json.dumps(payload))
    before = (path / MARKER).read_bytes()
    with pytest.raises(ValueError, match="Habitat source"):
        prepare_habitat(root=root, cache=cache, offline=True)
    assert (path / MARKER).read_bytes() == before
