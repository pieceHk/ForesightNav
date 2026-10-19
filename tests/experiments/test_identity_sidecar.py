"""Check episode alignment and keep recorded outcomes out of planner input."""

import gzip
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Exp" / "scripts"))
from build_identity_sidecar import build_sidecar
from experiments.dataset import build_identity_index, iter_episodes, load_identity_index
from scripts.prepare_data import prepare_data


@pytest.fixture
def dataset(tmp_path):
    (tmp_path / "content").mkdir()
    episode = {
        "episode_id": 0,
        "scene_id": "hm3d/val/00000-sceneA/sceneA.basis.glb",
        "tasks": [["chair", "object", None], ["toilet", "object", None]],
    }
    content = gzip.compress(json.dumps({"episodes": [episode]}).encode())
    (tmp_path / "content/sceneA.json.gz").write_bytes(content)
    manifest = {"scene_summary": [{
        "scene_id": "sceneA", "mon_episode_count": 1,
        "output_sha256": hashlib.sha256(content).hexdigest(),
    }]}
    raw = json.dumps(manifest).encode()
    (tmp_path / "dataset_manifest.json").write_bytes(raw)
    sources = [{
        "scene_id": "sceneA", "position": i, "source_episode_id": str(10 + i),
        "runtime_category": category, "baseline_success": 0,
        "candidate_success": 1, "outcome": "gain",
    } for i, category in enumerate(("chair", "toilet"))]
    (tmp_path / "source_trace.json").write_text(json.dumps([
        {"scene_id": "sceneA", "mon_episode_id": 0, "sources": sources}
    ]))
    return tmp_path, hashlib.sha256(raw).hexdigest(), {
        "dataset": {"dataset_manifest_sha256": "0" * 64}, "episodes": []
    }


def test_outcome_changes_do_not_change_identity_sidecar(dataset):
    root, digest, frozen = dataset
    expected = build_sidecar(root, 2, digest, frozen)
    path = root / "source_trace.json"
    trace = json.loads(path.read_text())
    for row in trace[0]["sources"]:
        row.update(baseline_success=1, candidate_success=0, outcome="regression")
    path.write_text(json.dumps(trace))
    assert build_sidecar(root, 2, digest, frozen) == expected
    assert set(expected["episodes"][0]) == {
        "scene_id", "runtime_episode_id", "source_episode_ids", "ordered_categories"
    }
    assert expected["episodes"][0]["source_episode_ids"] == ["10", "11"]


def test_reordered_source_tasks_are_rejected(dataset):
    root, digest, frozen = dataset
    path = root / "source_trace.json"
    trace = json.loads(path.read_text())
    trace[0]["sources"].reverse()
    path.write_text(json.dumps(trace))
    with pytest.raises(ValueError, match="task alignment mismatch"):
        build_sidecar(root, 2, digest, frozen)


def test_modified_episode_content_is_rejected(dataset):
    root, digest, frozen = dataset
    path = root / "content/sceneA.json.gz"
    episode = json.loads(gzip.decompress(path.read_bytes()))
    episode["episodes"][0]["episode_id"] = 100
    path.write_bytes(gzip.compress(json.dumps(episode).encode()))
    with pytest.raises(ValueError, match="content hash mismatch"):
        build_sidecar(root, 2, digest, frozen)


def test_frozen_and_source_identity_disagreement_is_rejected(dataset):
    root, digest, frozen = dataset
    frozen["dataset"]["dataset_manifest_sha256"] = digest
    frozen["episodes"] = [{"identity": {
        "scene_id": "sceneA", "runtime_episode_id": "0",
        "source_episode_ids": ["wrong", "11"],
        "ordered_categories": ["chair", "toilet"],
    }}]
    with pytest.raises(ValueError, match="Frozen identity disagrees"):
        build_sidecar(root, 2, digest, frozen)


def test_complete_index_preserves_identities_without_a_frozen_subset(dataset):
    root, digest, _ = dataset
    payload = build_identity_index(root, 2, digest)
    path = root / "identities.json"
    path.write_text(json.dumps(payload))
    identities = load_identity_index(path, digest)
    assert set(identities) == {("sceneA", "0")}
    assert identities[("sceneA", "0")].source_episode_ids == ("10", "11")
    episodes = list(iter_episodes(root, identities))
    assert len(episodes) == 1
    assert episodes[0][1] is identities[("sceneA", "0")]
    assert set(payload["episodes"][0]) == {
        "scene_id", "runtime_episode_id", "source_episode_ids", "ordered_categories"
    }


def test_new_identity_index_is_independent_of_recorded_outcomes(dataset):
    root, digest, _ = dataset
    expected = build_identity_index(root, 2, digest)
    path = root / "source_trace.json"
    trace = json.loads(path.read_text())
    for source in trace[0]["sources"]:
        source.update(baseline_success=1, candidate_success=0, outcome="different")
    path.write_text(json.dumps(trace))
    assert build_identity_index(root, 2, digest) == expected


def test_published_source_trace_checksum_is_verified(dataset):
    root, digest, _ = dataset
    with pytest.raises(ValueError, match="Source trace"):
        build_identity_index(root, 2, digest, "0" * 64)


def test_identity_loader_rejects_outcome_fields(dataset):
    root, digest, _ = dataset
    payload = build_identity_index(root, 2, digest)
    payload["episodes"][0]["success"] = True
    path = root / "identities.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="only task identity"):
        load_identity_index(path, digest)


def test_episode_iteration_checks_task_alignment(dataset):
    root, digest, _ = dataset
    payload = build_identity_index(root, 2, digest)
    payload["episodes"][0]["ordered_categories"].reverse()
    path = root / "identities.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="task identity mismatch"):
        list(iter_episodes(root, load_identity_index(path, digest)))


def test_data_preparation_is_repeatable_and_keeps_source_files_unchanged(dataset):
    root, digest, _ = dataset
    (root / "scenes").mkdir()
    trace_digest = hashlib.sha256((root / "source_trace.json").read_bytes()).hexdigest()
    catalog = {"scenes": f"{root.name}/scenes", "datasets": [{
        "name": "test", "source": root.name, "target": "episodes/test",
        "tasks_per_episode": 2, "episodes": 1, "scenes": 1,
        "manifest_sha256": digest, "source_trace_sha256": trace_digest,
    }]}
    destination = root.parent / (root.name + "-prepared")
    first = prepare_data(root.parent, destination, catalog)
    second = prepare_data(root.parent, destination, catalog)
    assert first == second
    assert (destination / "episodes/test/content").resolve() == root / "content"
    assert not (root / "identities.json").exists()
    assert hashlib.sha256((root / "dataset_manifest.json").read_bytes()).hexdigest() == digest
