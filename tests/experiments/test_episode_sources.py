import copy
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile
import zipfile

import pytest

from scripts._episode_sources import compose_scene, materialize_episodes, prepare_objectnav, read_recipe


@pytest.fixture
def source_episode():
    episodes = [{
        "episode_id": "0", "object_category": category,
        "scene_dataset_config": "default", "scene_id": "hm3d/scene/scene.basis.glb",
        "start_position": [index, 0, 0], "start_rotation": [0, 0, 0, 1],
    } for index, category in enumerate(("sofa", "tv_monitor"))]
    source = gzip.compress(json.dumps({"episodes": episodes, "goals_by_category": {
        "scene.basis.glb_sofa": [{"object_category": "sofa", "position": [1, 0, 0]}],
        "scene.basis.glb_tv_monitor": [{"object_category": "tv_monitor", "position": [2, 0, 0]}],
    }}).encode(), mtime=0)
    digest = hashlib.sha256(source).hexdigest()
    records = [{"position": position, "scene_id": "scene", "source_episode_id": str(index),
                "in_file_sequence_index": index, "object_category": category,
                "runtime_category": runtime, "content_file_sha256": digest}
               for position, (index, category, runtime) in enumerate(((1, "tv_monitor", "tv"), (0, "sofa", "couch")))]
    return source, [{"scene_id": "scene", "mon_episode_id": 7, "sources": records}], {
        "runtime_category_aliases": {"sofa": "couch", "tv_monitor": "tv"}}


def test_fixed_sequence_uses_source_index_and_first_start_despite_duplicate_official_ids(source_episode):
    body = compose_scene(*source_episode, 2)
    result = json.loads(gzip.decompress(body))
    episode = result["episodes"][0]
    assert episode["episode_id"] == 7
    assert episode["start_position"] == [1, 0, 0]
    assert episode["tasks"] == [["tv", "object", None], ["couch", "object", None]]
    assert result["goals"]["scene.basis.glb_sofa"][0]["object_category"] == "couch"
    assert "scene.basis.glb_couch" not in result["goals"]
    assert compose_scene(*source_episode, 2) == body


@pytest.mark.parametrize("field,value,error", [
    ("position", 1, "task positions"),
    ("in_file_sequence_index", -1, "out of bounds"),
    ("in_file_sequence_index", 0, "identity mismatch"),
    ("source_episode_id", "0", "identity mismatch"),
    ("runtime_category", "sofa", "identity mismatch"),
    ("content_file_sha256", "0" * 64, "identity mismatch"),
])
def test_composition_rejects_source_task_misalignment(source_episode, field, value, error):
    source, rows, manifest = copy.deepcopy(source_episode)
    rows[0]["sources"][0][field] = value
    with pytest.raises(ValueError, match=error):
        compose_scene(source, rows, manifest, 2)


def zip_provider(source):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("train/unrelated", "unused" * 100)
        archive.writestr("objectnav_hm3d_v1/val/content/scene.json.gz", source)
        archive.writestr("padding", bytes(range(256)) * 300, compress_type=zipfile.ZIP_STORED)
    data = buffer.getvalue()
    calls = []

    def opener(request, timeout):
        calls.append(request.get_header("Range"))
        value = calls[-1].removeprefix("bytes=")
        first, last = (max(0, len(data) - int(value[1:])), len(data) - 1) if value.startswith("-") else map(int, value.split("-"))
        response = io.BytesIO(data[first:last + 1])
        response.status = 206
        response.headers = {"Content-Range": f"bytes {first}-{last}/{len(data)}", "ETag": "pinned", "x-amz-version-id": "v1"}
        return response

    return opener, calls


def official_catalog(source):
    return {"url": "https://example.test/objectnav.zip", "version_id": "v1", "etag": "pinned",
            "content_prefix": "objectnav_hm3d_v1/val/content/", "files": [
                {"scene_id": "scene", "bytes": len(source), "sha256": hashlib.sha256(source).hexdigest()}]}


def test_partial_zip_fetch_and_verified_offline_reuse(tmp_path, source_episode):
    source = source_episode[0]
    catalog = official_catalog(source)
    opener, calls = zip_provider(source)
    paths = prepare_objectnav(catalog, tmp_path, offline=False, opener=opener)
    assert paths["scene"].read_bytes() == source
    assert len(calls) == 2 and calls[0] == "bytes=-65536"

    def forbidden(*args, **kwargs):
        pytest.fail("Offline cache attempted a network request")

    assert prepare_objectnav(catalog, tmp_path, offline=True, opener=forbidden) == paths
    paths["scene"].write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checksum mismatch"):
        prepare_objectnav(catalog, tmp_path, offline=False, opener=forbidden)


def test_incomplete_offline_cache_does_not_contact_provider(tmp_path, source_episode):
    def forbidden(*args, **kwargs):
        pytest.fail("Offline cache attempted a network request")

    with pytest.raises(FileNotFoundError, match="Offline ObjectNav cache"):
        prepare_objectnav(official_catalog(source_episode[0]), tmp_path, offline=True, opener=forbidden)


@pytest.mark.parametrize("change,error", [
    ({"status": 200}, "byte range"),
    ({"x-amz-version-id": "changed"}, "version differs"),
    ({"Content-Range": "bytes 0-1/100"}, "different byte range"),
])
def test_provider_must_return_requested_archive_and_range(tmp_path, source_episode, change, error):
    opener, _ = zip_provider(source_episode[0])

    def altered(request, timeout):
        response = opener(request, timeout)
        for field, value in change.items():
            if field == "status":
                response.status = value
            else:
                response.headers[field] = value
        return response

    with pytest.raises(ValueError, match=error):
        prepare_objectnav(official_catalog(source_episode[0]), tmp_path, offline=False, opener=altered)
    assert not list(tmp_path.rglob("*.json.gz"))


def write_recipe(root, name, files):
    target = root / name
    target.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(target, "w:gz") as archive:
        for path, body in files.items():
            member = tarfile.TarInfo(path)
            member.size = len(body)
            archive.addfile(member, io.BytesIO(body))
    body = target.read_bytes()
    return {"path": name, "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}


@pytest.mark.parametrize("name", ["../escaped", "/absolute", "unlisted"])
def test_recipe_rejects_unlisted_or_unsafe_members(tmp_path, name):
    recipe = write_recipe(tmp_path, "recipe.tar.gz", {name: b"test"})
    with pytest.raises(ValueError):
        read_recipe(tmp_path, recipe, {"allowed": hashlib.sha256(b"test").hexdigest()})
    assert not (tmp_path.parent / "escaped").exists()


def test_materializer_checks_expected_output_before_publishing(tmp_path, source_episode):
    source, rows, manifest = source_episode
    root = tmp_path / "repo"
    cache = tmp_path / "cache"
    (root / "assets/episodes").mkdir(parents=True)
    (root / "assets/episodes/NOTICE.md").write_text("dataset terms")
    original = official_catalog(source)
    source_path = cache / original["content_prefix"] / "scene.json.gz"
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(source)
    manifest["scene_summary"] = [{"scene_id": "scene", "output_sha256": "0" * 64}]
    metadata = {"dataset_manifest.json": json.dumps(manifest).encode(),
                "source_trace.json": json.dumps(rows).encode(), "val_seen.json.gz": b"stub"}
    entry = {"source": "pool/2MON", "tasks_per_episode": 2,
             "manifest_sha256": hashlib.sha256(metadata["dataset_manifest.json"]).hexdigest(),
             "source_trace_sha256": hashlib.sha256(metadata["source_trace.json"]).hexdigest(),
             "val_seen_sha256": hashlib.sha256(b"stub").hexdigest(),
             "recipe": write_recipe(root, "meta.tar.gz", metadata)}
    common = {"source": "pool/common", "files": [{"path": "plan.json", "sha256": hashlib.sha256(b"plan").hexdigest()}],
              "recipe": write_recipe(root, "common.tar.gz", {"plan.json": b"plan"})}
    with pytest.raises(ValueError, match="Rebuilt episode bytes differ"):
        materialize_episodes(root, {"datasets": [entry], "common": common, "objectnav_source": original}, cache, offline=True)
    assert not (cache / "pbsp-hm3dv1-balanced1200-v1/pool/2MON").exists()


def test_bundled_fixed_plans_match_every_released_task_order():
    root = Path(__file__).resolve().parents[2]
    catalog = json.loads((root / "assets/datasets.json").read_text())
    common = catalog["common"]
    plans = read_recipe(root, common["recipe"], {row["path"]: row["sha256"] for row in common["files"]})
    total = 0
    for entry in catalog["datasets"]:
        files = read_recipe(root, entry["recipe"], {
            "dataset_manifest.json": entry["manifest_sha256"],
            "source_trace.json": entry["source_trace_sha256"], "val_seen.json.gz": entry["val_seen_sha256"],
        })
        manifest = json.loads(files["dataset_manifest.json"])
        trace = json.loads(files["source_trace.json"])
        plan = json.loads(plans[f"{entry['tasks_per_episode']}mon_plan.json"])
        expected = {(row["scene_id"], row["mon_episode_id"]): row["sources"]
                    for rows in plan["episodes_by_scene"].values() for row in rows}
        observed = {(row["scene_id"], row["mon_episode_id"]): row["sources"] for row in trace}
        assert expected == observed and len(trace) == len(expected) == entry["episodes"]
        assert manifest["scientific_scope"]["selection_type"] == "historical paired-outcome-calibrated"
        total += len(trace)
    assert total == 1300
