"""Exercise native run selection, frozen inputs and actual process cleanup."""

from dataclasses import replace
from importlib.machinery import PathFinder
import gzip
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from experiments.dataset import build_identity_index
from runtime.bootstrap import worker_environment
from runtime.config import (
    DatasetConfig,
    RuntimeConfig,
    experiment_from_snapshot,
    load_experiment,
)
from runtime.factories import navigation_request
from runtime.launch import allocate_run, launch
from runtime.navigation_worker import select_habitat_episodes
from runtime.plan import load_plan, prepare_plan

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def plan(tmp_path):
    config = load_experiment(ROOT / "configs/experiments/hm3d_2mon.yaml")
    dataset = tmp_path / "data/episodes/toy"
    (dataset / "content").mkdir(parents=True)
    (dataset / "val_seen.json.gz").write_bytes(gzip.compress(b'{"episodes":[]}'))
    summary = []
    trace = []
    for scene in ("sceneB", "sceneA"):
        episodes = []
        for index in range(2):
            episodes.append(
                {
                    "episode_id": index,
                    "scene_id": f"hm3d/val/00000-{scene}/{scene}.basis.glb",
                    "tasks": [["chair", "object", None], ["toilet", "object", None]],
                }
            )
            trace.append(
                {
                    "scene_id": scene,
                    "mon_episode_id": index,
                    "sources": [
                        {
                            "scene_id": scene,
                            "position": task,
                            "source_episode_id": f"{scene}-{index}-{task}",
                            "runtime_category": category,
                            "candidate_success": 1,
                            "baseline_success": 0,
                        }
                        for task, category in enumerate(("chair", "toilet"))
                    ],
                }
            )
        raw = gzip.compress(json.dumps({"episodes": episodes}).encode())
        (dataset / "content" / f"{scene}.json.gz").write_bytes(raw)
        summary.append(
            {
                "scene_id": scene,
                "mon_episode_count": 2,
                "output_sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    (dataset / "source_trace.json").write_text(json.dumps(trace))
    (dataset / "dataset_manifest.json").write_text(
        json.dumps({"scene_summary": summary})
    )
    manifest_sha = hashlib.sha256(
        (dataset / "dataset_manifest.json").read_bytes()
    ).hexdigest()
    source_sha = hashlib.sha256(
        (dataset / "source_trace.json").read_bytes()
    ).hexdigest()
    (dataset / "identities.json").write_text(
        json.dumps(build_identity_index(dataset, 2, manifest_sha, source_sha))
    )
    data_config = DatasetConfig(
        "toy",
        "val",
        "episodes/toy",
        "dataset_manifest.json",
        "identities.json",
        manifest_sha,
        source_sha,
        2,
        4,
        2,
    )
    runtime = RuntimeConfig(
        tmp_path / "data",
        tmp_path / "weights",
        tmp_path / "outputs",
        Path(sys.executable),
        Path(sys.executable),
        6,
        7,
    )
    config = replace(config, dataset=data_config, runtime=runtime)
    return prepare_plan(config, root=ROOT, source_commit="test-source")


def test_complete_selection_uses_manifest_order_and_preserves_source_identity(plan):
    assert [item.scene_id for item in plan.scenes] == ["sceneB", "sceneA"]
    assert [item.episode_ids for item in plan.scenes] == [("0", "1"), ("0", "1")]
    assert len(plan.identities) == 4
    assert plan.identities[0].source_episode_ids == ("sceneB-0-0", "sceneB-0-1")
    assert "success" not in json.dumps(plan.snapshot()["identities"])


def test_limit_applies_to_complete_episodes_across_scenes(plan):
    limited = prepare_plan(plan.experiment, root=ROOT, source_commit="test", limit=3)
    assert [(item.scene_id, item.episode_ids) for item in limited.scenes] == [
        ("sceneB", ("0", "1")),
        ("sceneA", ("0",)),
    ]
    assert len(limited.identities) == 3


def test_scene_and_real_episode_id_select_exact_records(plan):
    selected = prepare_plan(
        plan.experiment,
        root=ROOT,
        source_commit="test",
        scenes=["sceneA"],
        episodes=["1"],
    )
    assert len(selected.identities) == 1
    assert selected.identities[0].source_episode_ids == ("sceneA-1-0", "sceneA-1-1")
    assert selected.scenes[0].scene_index == 1


@pytest.mark.parametrize(
    "options",
    [
        {"scenes": ["missing"]},
        {"scenes": ["sceneA", "sceneA"]},
        {"scenes": []},
        {"episodes": ["0"]},
        {"scenes": ["sceneA"], "episodes": ["99"]},
        {"scenes": ["sceneA"], "episodes": ["0", "0"]},
        {"limit": 0},
        {"limit": True},
    ],
)
def test_bad_selection_is_rejected_without_output_directory(plan, options):
    with pytest.raises(ValueError):
        prepare_plan(plan.experiment, root=ROOT, source_commit="test", **options)
    assert not plan.experiment.runtime.outputs.exists()


def test_prepared_identity_changes_are_rejected_even_when_manifest_is_intact(plan):
    path = plan.resources.data / plan.experiment.dataset.root / "identities.json"
    payload = json.loads(path.read_text())
    payload["episodes"][0]["source_episode_ids"][0] = "other-source"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="differs"):
        prepare_plan(plan.experiment, root=ROOT, source_commit="test")


def test_plan_roundtrip_freezes_configuration_model_and_identity_data(plan, tmp_path):
    prepared = allocate_run(plan, tag="freeze")
    path = prepared.run_directory / "launch.json"
    frozen = load_plan(path)
    assert frozen == prepared
    assert experiment_from_snapshot(plan.experiment.snapshot()) == plan.experiment
    request = navigation_request(frozen, "sceneA")
    assert request.episode_ids == ("0", "1") and request.seed == plan.experiment.seed
    assert request.settings == plan.experiment.method.navigation
    assert request.resources.scene_directory == plan.resources.data / "scenes/hm3dv1"
    assert len(request.planning_settings.episode_identities) == 2
    assert len(str(frozen.socket_path).encode()) < 100


@pytest.mark.parametrize(
    "field", ["configuration", "model_specification", "identities", "scenes"]
)
def test_changed_saved_inputs_or_selection_fail_before_model_loading(plan, field):
    prepared = allocate_run(plan)
    path = prepared.run_directory / "launch.json"
    value = json.loads(path.read_text())
    if field == "configuration":
        value[field]["method"]["suffix"]["suffix_weight"] = 0.9
    elif field == "model_specification":
        value[field]["camera"]["hfov_degrees"] = 90
    elif field == "identities":
        value[field][0]["source_episode_ids"][0] = "modified"
    else:
        value[field][0]["episode_ids"] = ["not-selected"]
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        load_plan(path)


def test_worker_environment_replaces_ambient_algorithm_source_and_device_values(plan):
    inherited = {
        "ES_SCSP_SUFFIX_WEIGHT": "999",
        "ES_ROOT": "/wrong",
        "HM3D_EXP_TWO_STAGE": "0",
        "LAGMEMO_SEMANTIC_SOURCE": "ground_truth",
        "TDONAV_ROOT": "/wrong",
        "PYTHONPATH": "/old/source",
        "CUDA_VISIBLE_DEVICES": "0",
        "HF_HOME": "/old/cache",
        "KEEP": "value",
    }
    mapping = worker_environment(plan, "mapping", inherited)
    nav = worker_environment(plan, "navigation", inherited)
    assert inherited["ES_SCSP_SUFFIX_WEIGHT"] == "999"
    assert mapping["CUDA_VISIBLE_DEVICES"] == "6" and nav["CUDA_VISIBLE_DEVICES"] == "7"
    for environment in (mapping, nav):
        assert environment["KEEP"] == "value" and environment["HF_HUB_OFFLINE"] == "1"
        assert environment["TORCH_HOME"] == str(plan.resources.weights / "torch")
        assert environment["PYTHONHASHSEED"] == str(plan.experiment.seed)
        assert all(
            key not in environment
            for key in (
                "ES_SCSP_SUFFIX_WEIGHT",
                "ES_ROOT",
                "HM3D_EXP_TWO_STAGE",
                "LAGMEMO_SEMANTIC_SOURCE",
            )
        )
        assert "/old/source" not in environment["PYTHONPATH"]
    assert str(plan.resources.habitat / "habitat-lab") in nav["PYTHONPATH"]


def test_navigation_source_paths_preserve_installed_detectron2_package(plan, tmp_path):
    installed = tmp_path / "site-packages"
    package = installed / "detectron2"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("__version__ = 'installed'\n")
    # The native extension belongs to this package. A source-only package
    # earlier on sys.path hides it, even when the extension is installed.
    environment = worker_environment(plan, "navigation", {})
    paths = environment["PYTHONPATH"].split(os.pathsep) + [str(installed)]
    spec = PathFinder.find_spec("detectron2", paths)
    assert spec is not None
    assert Path(spec.origin) == package / "__init__.py"
    assert list(spec.submodule_search_locations) == [str(package)]


def test_native_habitat_selection_preserves_requested_ids(plan):
    prepared = replace(plan, run_directory=plan.experiment.runtime.outputs / "unused")
    request = navigation_request(prepared, "sceneA")
    episodes = [
        SimpleNamespace(episode_id="extra"),
        SimpleNamespace(episode_id="1"),
        SimpleNamespace(episode_id="0"),
    ]
    selected, end = select_habitat_episodes(episodes, request)
    assert [item.episode_id for item in selected] == ["0", "1"] and end == 2
    with pytest.raises(ValueError, match="missing or ambiguous"):
        select_habitat_episodes(episodes[:-1], request)


def test_allocated_run_directory_cannot_be_reused(plan):
    prepared = allocate_run(plan)
    with pytest.raises(FileExistsError):
        allocate_run(prepared)


def test_root_cli_help_is_available_outside_checkout_without_model_imports(tmp_path):
    result = subprocess.run(
        [sys.executable, str(ROOT / "run.py"), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--limit-episodes" in result.stdout and "--check" in result.stdout
    script = f"import sys;sys.path.insert(0,{str(ROOT)!r});import run;assert not any(n in sys.modules for n in ('torch','numpy','habitat','lagmemo'))"
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=tmp_path, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_default_worker_commands_call_native_modules_and_frozen_plan(plan):
    from runtime.launch import worker_command

    prepared = replace(plan, run_directory=plan.experiment.runtime.outputs / "command")
    for role in ("mapping", "navigation"):
        command = worker_command(prepared, role, scene="sceneA")
        assert command[1:4] == ["-m", f"runtime.{role}_worker", "--run-plan"]
        assert command[4] == str(prepared.run_directory / "launch.json")
        assert not any("Exp/scripts" in item or "legacy" in item for item in command)


@pytest.mark.parametrize(
    "scenario",
    [
        "success",
        "navigation_failure",
        "mapping_startup_failure",
        "mapping_close_failure",
        "parent_sigterm",
    ],
)
def test_real_worker_entrypoints_and_parent_process_cleanup(
    plan, tmp_path, monkeypatch, scenario
):
    # Keep the actual native entrypoints, socket protocol and process owner.
    # Substitute model/simulator construction and metric aggregation only.
    script = tmp_path / "controlled_worker.py"
    script.write_text(
        """
import sys,json,os,time,signal
from pathlib import Path
from contextlib import ExitStack
sys.path.insert(0,ROOT)
role,plan_path,scenario,parent_pid,scene=sys.argv[1:]
from runtime.plan import load_plan
plan=load_plan(plan_path)
from runtime import bootstrap
bootstrap.configure_run_process=lambda *args:None
out=plan.run_directory

def marker(name):
    with (out/'lifecycle.jsonl').open('a') as stream:stream.write(json.dumps({'event':name,'pid':os.getpid()})+'\\n')

if role=='monitor':
    def stop(*args):marker('monitor_closed');raise SystemExit(0)
    signal.signal(signal.SIGTERM,stop)
    marker('monitor_ready')
    while True:time.sleep(.1)
elif role=='mapping':
    from runtime import factories
    class Mapper:
        def get_status(self):return {'ready':True}
        def flush_episode(self):return self.get_status()
        def close(self):
            marker('mapping_closed')
            if scenario=='mapping_close_failure':raise RuntimeError('controlled mapper close error')
    def build(*args,**kwargs):
        assert 'runtime.legacy_mapping' not in sys.modules
        if scenario=='mapping_startup_failure':raise RuntimeError('controlled mapping startup error')
        return Mapper()
    factories.create_mapper=build
    from runtime.mapping_worker import main
    main(['--run-plan',plan_path])
else:
    from runtime import navigation_worker
    def run(request):
        assert 'runtime.legacy_navigation_run' not in sys.modules
        with ExitStack() as stack:
            stack.callback(marker,'navigation_closed_'+scene)
            if scenario=='navigation_failure':raise RuntimeError('controlled navigation error')
            if scenario=='parent_sigterm':
                os.kill(int(parent_pid),signal.SIGTERM)
                while True:time.sleep(.1)
            (out/('finished_'+scene+'.json')).write_text(json.dumps({'scene':scene,'episodes':request.episode_ids,'categories':[item['ordered_categories'] for item in request.planning_settings.episode_identities.values()]}))
    navigation_worker.run_navigation=run
    navigation_worker.main(['--run-plan',plan_path,'--scene',scene])
""".replace(
            "ROOT", repr(str(ROOT))
        )
    )
    import runtime.launch as launcher

    original_wait = launcher.wait_for_mapping
    monkeypatch.setattr(
        launcher,
        "wait_for_mapping",
        lambda process, socket: original_wait(process, socket, timeout=10),
    )
    monkeypatch.setattr(
        launcher,
        "worker_command",
        lambda prepared, role, scene=None: [
            sys.executable,
            str(script),
            role,
            str(prepared.run_directory / "launch.json"),
            scenario,
            str(os.getpid()),
            scene or "",
        ],
    )

    def aggregate(run):
        done = [json.loads(path.read_text()) for path in run.glob("finished_*.json")]
        count = sum(len(item["episodes"]) for item in done)
        return {
            "counts": {"completed_episodes": count, "completed_subtasks": count * 2}
        }

    monkeypatch.setattr("experiments.metrics.aggregate_run_artifacts", aggregate)
    previous = signal.getsignal(signal.SIGTERM)
    if scenario == "success":
        run = launch(plan, tag="processes")
    else:
        expected = SystemExit if scenario == "parent_sigterm" else RuntimeError
        with pytest.raises(expected):
            launch(plan, tag="processes")
        run = next(plan.experiment.runtime.outputs.iterdir())
    assert signal.getsignal(signal.SIGTERM) == previous
    state = json.loads((run / "status.json").read_text())
    assert state["status"] == ("completed" if scenario == "success" else "failed")
    records = [json.loads(path.read_text()) for path in run.glob("*_process.json")]
    assert {row["role"] for row in records} >= {"mapping", "monitor"}
    for row in records:
        with pytest.raises(ProcessLookupError):
            os.kill(row["pid"], 0)
    events = [
        json.loads(line)["event"]
        for line in (run / "lifecycle.jsonl").read_text().splitlines()
    ]
    assert "monitor_closed" in events
    if scenario != "mapping_startup_failure":
        assert "mapping_closed" in events
    if scenario == "success":
        assert (
            "navigation_closed_sceneA" in events
            and "navigation_closed_sceneB" in events
        )
        assert all(row["returncode"] == 0 for row in records)
    if scenario == "mapping_close_failure":
        assert (
            next(row["returncode"] for row in records if row["role"] == "mapping") != 0
        )
    prepared = load_plan(run / "launch.json")
    assert not prepared.socket_path.exists()


@pytest.mark.parametrize("query_failure", [False, True])
def test_native_mapper_factory_uses_resolved_settings_and_releases_failed_model(
    plan, monkeypatch, query_failure
):
    import numpy as np
    from runtime.factories import create_mapper, benchmark_intrinsic

    prepared = replace(plan, run_directory=plan.experiment.runtime.outputs / "factory")
    method = replace(
        plan.experiment.method,
        mapping=replace(plan.experiment.method.mapping, keyframe_threshold=0.8),
        query=replace(plan.experiment.method.query, query_similarity=0.7, top_k=3),
    )
    prepared = replace(prepared, experiment=replace(prepared.experiment, method=method))
    calls = {}
    closed = []
    model = SimpleNamespace(
        encoder=SimpleNamespace(), close=lambda: closed.append(True)
    )

    def load_model(**kwargs):
        calls["model"] = kwargs
        return model

    def queries(encoder, *, settings, device):
        assert encoder is model.encoder
        calls["query"] = (settings, device)
        if query_failure:
            raise RuntimeError("query construction failed")
        return SimpleNamespace(top_k=settings.top_k)

    monkeypatch.setattr("adapters.embodied_splat.load_model", load_model)
    monkeypatch.setattr("adapters.embodied_splat_query.create_query_backend", queries)
    monkeypatch.setattr(
        "runtime.config.read_yaml",
        lambda *a: pytest.fail("factory must use frozen model inputs"),
    )
    monkeypatch.setenv("ES_FRAME_CHANGE_RATIO", "999")
    monkeypatch.setenv("ES_QUERY_MIN_SIMILARITY", "999")
    if query_failure:
        with pytest.raises(RuntimeError, match="query construction"):
            create_mapper(prepared, device="cpu")
        assert closed == [True]
    else:
        mapper = create_mapper(prepared, device="cpu")
        assert mapper.settings == method.mapping and mapper.top_k == 3
        assert calls["query"] == (method.query, "cpu")
        assert (
            calls["model"]["checkpoint"]
            == plan.resources.weights / "freesplatpp_scannet_gtdepth.ckpt"
        )
        assert (
            calls["model"]["work_directory"]
            == prepared.run_directory / "artifacts/mapping_tmp"
        )
        assert np.array_equal(
            mapper.camera_intrinsic,
            benchmark_intrinsic(plan.model_specification["camera"]),
        )
        assert not closed


def test_recording_switch_changes_artifacts_without_changing_the_complete_loop(
    tmp_path, monkeypatch
):
    from tests.experiments import test_runner_replay as replay
    from experiments.runner import RunSettings

    enabled = replay.execute(tmp_path / "enabled", monkeypatch, native=True)
    monkeypatch.setattr(
        replay,
        "RunSettings",
        lambda *args, **kwargs: RunSettings(*args, **kwargs, record_decisions=False),
    )
    disabled = replay.execute(tmp_path / "disabled", monkeypatch, native=True)
    assert disabled["error"] is None
    for name in ("events", "timing", "status", "candidates"):
        assert disabled[name] == enabled[name]
    from copy import deepcopy

    expected = deepcopy(enabled["metrics"])
    for key, record in expected.items():
        assert record["scsp_provider_event_summary"]["jsonl_recorded_count"] > 0
        record["scsp_provider_event_summary"]["jsonl_recorded_count"] = 0
    assert disabled["metrics"] == expected
    assert set(disabled["files"]) == set(enabled["files"]) - {
        "scsp_provider_events.jsonl",
        "scsp_decisions.jsonl",
    }
    for name, content in disabled["files"].items():
        if name == "per_episode_metrics.json":
            assert json.loads(content) == expected
        else:
            assert content == enabled["files"][name]


def test_runtime_preparation_copies_required_cache_and_is_idempotent(tmp_path):
    from scripts.prepare_runtime import prepare_runtime, TORCH_FILES
    from runtime.config import load_runtime

    data = tmp_path / "data"
    data.mkdir()
    weights = tmp_path / "weights"
    weights.mkdir()
    source = tmp_path / "source/hub/checkpoints"
    source.mkdir(parents=True)
    for name in TORCH_FILES:
        (source / name).write_bytes(name.encode())
    arguments = dict(
        mapping_python=sys.executable,
        navigation_python=sys.executable,
        data=data,
        weights=weights,
        outputs=tmp_path / "outputs",
        mapping_gpu=6,
        navigation_gpu=7,
        destination=tmp_path / "runtime.yaml",
        torch_home_source=tmp_path / "source",
    )
    first = prepare_runtime(**arguments)
    assert prepare_runtime(**arguments) == first
    loaded = load_runtime(arguments["destination"])
    assert loaded.mapping_gpu == 6 and loaded.navigation_gpu == 7
    for name in TORCH_FILES:
        assert (weights / "torch/hub/checkpoints" / name).read_bytes() == name.encode()
    assert not (tmp_path / "outputs").exists()


def test_runtime_cache_conflict_leaves_other_files_and_configuration_untouched(
    tmp_path,
):
    from scripts.prepare_runtime import prepare_runtime, TORCH_FILES

    data = tmp_path / "data"
    data.mkdir()
    weights = tmp_path / "weights"
    weights.mkdir()
    source = tmp_path / "source/hub/checkpoints"
    source.mkdir(parents=True)
    for name in TORCH_FILES:
        (source / name).write_bytes(b"selected")
    target = weights / "torch/hub/checkpoints"
    target.mkdir(parents=True)
    (target / TORCH_FILES[1]).write_bytes(b"existing")
    with pytest.raises(ValueError, match="cache differs"):
        prepare_runtime(
            mapping_python=sys.executable,
            navigation_python=sys.executable,
            data=data,
            weights=weights,
            outputs=tmp_path / "outputs",
            mapping_gpu=0,
            navigation_gpu=1,
            destination=tmp_path / "runtime.yaml",
            torch_home_source=tmp_path / "source",
        )
    assert not (target / TORCH_FILES[0]).exists()
    assert not (tmp_path / "runtime.yaml").exists()
    assert (target / TORCH_FILES[1]).read_bytes() == b"existing"


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("input", "image_height", 383),
        ("camera", "hfov_degrees", 180),
        ("camera", "width", True),
    ],
)
def test_invalid_model_inputs_fail_without_loading_models(plan, section, key, value):
    from copy import deepcopy

    specification = deepcopy(plan.model_specification)
    specification[section][key] = value
    with pytest.raises(ValueError):
        replace(plan, model_specification=specification)
