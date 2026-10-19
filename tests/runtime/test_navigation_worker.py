"""Compare native startup with the frozen startup, then fault its resource owners."""

import argparse
import ast
from copy import deepcopy
from dataclasses import asdict, replace
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import ModuleType, SimpleNamespace

import pytest

from experiments.runner import PlanningRunSettings
from experiments.telemetry import to_jsonable
from pbsp.navigation.cooldown import CategorySwitchCooldown, parse_category_cooldowns
from pbsp.navigation.ranking import CandidateActionReranker, CandidateRerankConfig
from runtime.legacy_navigation_run import parse_args, resolve_request
from runtime.navigation_worker import (
    NavigationRequest,
    NavigationResources,
    build_services,
    run_navigation,
)


ROOT = Path(__file__).resolve().parents[2]
REFERENCE = "d7b7b25b573a5d4dd8dda1ccf72c1ec1a91fcabb"
SOURCE = "Navigation/runtime/evaluators/goat/entrypoint.py"


def frozen_source(path=SOURCE):
    return subprocess.check_output(
        ["git", "-C", str(ROOT), "show", f"{REFERENCE}:{path}"], text=True
    )


def config():
    class Config(dict):
        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError as exc:
                raise AttributeError(name) from exc

        def __setattr__(self, name, value):
            self[name] = value

    def wrap(value):
        return (
            Config({key: wrap(item) for key, item in value.items()})
            if isinstance(value, dict)
            else value
        )

    return wrap(
        dict(
            GROUND_TRUTH_SEMANTICS=0,
            EXP_FIRST=True,
            DUMP_LOCATION="old",
            NUM_ENVIRONMENTS=35,
            PRINT_IMAGES=1,
            EXP_NAME="frozen",
            AGENT={"max_steps": 200},
            ENVIRONMENT={"min_depth": 0.5, "max_depth": 5.0},
            habitat={
                "environment": {"iterator_options": {"shuffle": True}},
                "dataset": {"data_path": "old", "content_scenes": []},
            },
        )
    )


def fixture_request(tmp_path, mode):
    dataset = tmp_path / "dataset"
    (dataset / "content").mkdir(parents=True)
    (dataset / "val_seen.json.gz").touch()
    (dataset / "content/scene.json.gz").touch()
    (dataset / "dataset_manifest.json").write_text("{}")
    digest = hashlib.sha256(b"{}").hexdigest()
    frozen = tmp_path / "Exp/experiments/scsp_idea2_frozen_pairs_v1.json"
    frozen.parent.mkdir(parents=True)
    frozen.write_text('{"dataset":{"dataset_manifest_sha256":"other"}}')
    sidecar = tmp_path / "identities.json"
    sidecar.write_text(
        json.dumps(
            dict(
                schema_version="scsp_identity_sidecar_v1",
                dataset_manifest_sha256=digest,
                episodes=[
                    dict(
                        scene_id="scene",
                        runtime_episode_id="0",
                        source_episode_ids=["a", "b", "c", "d"],
                        ordered_categories=["bed", "tv", "bed", "chair"],
                    )
                ],
            )
        )
    )
    environment = {
        "ES_SCSP_RUNTIME_MODE": mode,
        "ES_SCSP_PROFILE_MODE": "exact_baseline" if mode == "off" else mode,
        "ES_SCSP_ARM": "legacy" if mode == "off" else mode,
        "LAGMEMO_SEMANTIC_SOURCE": "detic",
        "ES_GROUND_TRUTH_SEMANTICS": "0",
        "ES_SCSP_ORDERED_SUBTASKS": "4",
        "ES_SCSP_EXPECTED_DATASET_MANIFEST_SHA256": digest,
        "ES_SCSP_IDENTITY_SIDECAR": str(sidecar),
        "ES_QUERY_ACTION_RERANK_ENABLED": "1",
        "ES_QUERY_SWITCH_COOLDOWN_BY_CATEGORY": '{"bed":30}',
        "ES_QUERY_W_TRAVEL": "0.6",
        "ES_CANDIDATE_EVIDENCE_RADIUS": "0.8",
        "ES_NAV_PROJECTION_MAX_DISTANCE": "1.25",
        "ES_NAVMESH_SAMPLE_COUNT": "7",
        "ES_NAVMESH_RANDOM_MIN_DISTANCE": "1.8",
    }
    argv = [
        "--dataset_root",
        str(dataset),
        "--output_path",
        str(tmp_path / "output"),
        "--scenes",
        "scene",
        "--episode_start",
        "1",
        "--episode_end",
        "2",
        "--max_steps_per_subtask",
        "123",
        "--scene_idx",
        "5",
        "--live_mode",
        "--enable_timing",
        "--no-print-images",
    ]
    args = parse_args(argv, tmp_path)
    return (
        args,
        environment,
        resolve_request(args, root=tmp_path, engine=tmp_path, environ=environment),
        argv,
    )


def install_module(monkeypatch, name, **values):
    module = ModuleType(name)
    for key, value in values.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)


@pytest.mark.parametrize("mode", ["off", "idea1_only", "ordered_suffix_mvp"])
def test_actual_startup_arguments_match_frozen_main(tmp_path, monkeypatch, mode):
    args, environment, request, argv = fixture_request(tmp_path, mode)
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    events = []
    calls = []
    actions = SimpleNamespace(STOP=0)

    class Provider:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            reader = kwargs["task_reader"]
            events.append(
                (
                    "provider",
                    {
                        **kwargs,
                        "task_reader": (reader.func, reader.args, reader.keywords),
                    },
                )
            )

        def close(self):
            events.append(("close_provider",))

    class Habitat:
        def __init__(self, cfg):
            self.config = cfg
            self.episodes = ["0", "1", "2"]
            events.append(("habitat", deepcopy(cfg)))

        def close(self):
            events.append(("close_habitat",))

    evaluator = SimpleNamespace(top_down_map=lambda: None)

    def make_environment(habitat_env, config, **kwargs):
        events.append(("environment", list(habitat_env.episode_iterator)))
        return SimpleNamespace(evaluator=evaluator)

    def make_agent(config, seem_cfg, clip_cfg, settings=None):
        absolute = lambda value: str((tmp_path / value).resolve())
        events.append(
            (
                "agent",
                {key: absolute(value) for key, value in seem_cfg.items()},
                absolute(clip_cfg),
            )
        )
        return SimpleNamespace(
            settings=settings or request.settings,
            query_action_rerank_enabled=True,
            continuous_print_images=False,
            episode_final_snapshot=False,
            visualize=False,
        )

    def capture(**kwargs):
        calls.append(kwargs)
        Path(kwargs["artifacts"].directory).mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr("experiments.runner.run_episodes", capture)
    monkeypatch.setattr("runtime.mapping_client.MappingClient", Provider)
    install_module(
        monkeypatch, "adapters.habitat_config", get_config=lambda *args: config()
    )
    install_module(monkeypatch, "adapters.goat", create_environment=make_environment)
    install_module(monkeypatch, "adapters.lagmemo_agent", LagMemoAgent=make_agent)
    install_module(monkeypatch, "habitat.core.env", Env=Habitat)
    install_module(
        monkeypatch, "lagmemo.core.interfaces", DiscreteNavigationAction=actions
    )
    old = frozen_source()
    tree = ast.parse(old)
    ns = dict(
        argparse=argparse,
        hashlib=hashlib,
        json=json,
        os=os,
        time=time,
        Path=Path,
        partial=partial,
        _TDONAV_ROOT=tmp_path,
        _LAGMEMO_ROOT=tmp_path,
        _EARLY_DETIC_REGISTRATION_STUBS=(),
        MappingClient=Provider,
        load_runtime_tasks=__import__(
            "experiments.dataset", fromlist=["load_runtime_tasks"]
        ).load_runtime_tasks,
        get_config=lambda *args: config(),
        Env=Habitat,
        HabitatGoatEnv=make_environment,
        GLUEAgent=make_agent,
        CategorySwitchCooldown=CategorySwitchCooldown,
        parse_category_cooldowns=parse_category_cooldowns,
        CandidateActionReranker=CandidateActionReranker,
        CandidateRerankConfig=CandidateRerankConfig,
        to_jsonable=to_jsonable,
        DEFAULT_CANDIDATE_CAP=5,
        DEFAULT_SUFFIX_WEIGHT=0.35,
        DiscreteNavigationAction=actions,
    )
    helpers = ast.Module(
        body=[n for n in tree.body if isinstance(n, ast.FunctionDef)], type_ignores=[]
    )
    exec(compile(helpers, "<frozen-startup-helpers>", "exec"), ns)
    main = next(
        n
        for n in tree.body
        if isinstance(n, ast.If) and ast.unparse(n.test) == "__name__ == '__main__'"
    )
    monkeypatch.setattr(sys, "argv", ["old", *argv])
    exec(
        compile(
            ast.Module(body=main.body, type_ignores=[]), "<frozen-startup>", "exec"
        ),
        ns,
    )
    old_events = list(events)
    old_call = calls.pop()
    events.clear()
    run_navigation(request)
    new_call = calls.pop()
    # The native assembly selects and configures the same concrete services.
    assert events[:-2] == old_events[:-1]
    assert events[-2:] == [("close_habitat",), ("close_provider",)]
    for name in ("settings", "planning_settings"):
        assert asdict(new_call[name]) == asdict(old_call[name])
    assert new_call["reranker"].config == old_call["reranker"].config
    assert (
        new_call["category_cooldown"].__dict__ == old_call["category_cooldown"].__dict__
    )
    assert (
        new_call["planning"].project_goals.keywords
        == old_call["planning"].project_goals.keywords
    )
    assert new_call["sample_goal"].keywords == old_call["sample_goal"].keywords
    assert new_call["agent"].visualize == old_call["agent"].visualize == (mode != "off")
    assert (
        request.output / "results_scene/frozen_5/es_status.json"
    ).read_text() == "[]"


@pytest.mark.parametrize(
    "stage", ["config", "habitat", "environment", "agent", "runner"]
)
def test_actual_resource_construction_failure_closes_prepared_services(
    tmp_path, monkeypatch, stage
):
    _, _, request, _ = fixture_request(tmp_path, "ordered_suffix_mvp")
    events = []

    def fail(name):
        if stage == name:
            raise RuntimeError(name)

    class Provider:
        def __init__(self, **kw):
            events.append("provider")

        def close(self):
            events.append("close_provider")

    class Habitat:
        def __init__(self, cfg):
            fail("habitat")
            self.episodes = [0, 1]

        def close(self):
            events.append("close_habitat")

    def get_config(*args):
        fail("config")
        return config()

    def make_env(*args, **kwargs):
        fail("environment")
        return SimpleNamespace(evaluator=SimpleNamespace(top_down_map=lambda: None))

    def make_agent(**kwargs):
        fail("agent")
        return SimpleNamespace(
            continuous_print_images=False, episode_final_snapshot=False, visualize=False
        )

    def runner(**kwargs):
        fail("runner")

    monkeypatch.setattr("runtime.mapping_client.MappingClient", Provider)
    monkeypatch.setattr("experiments.runner.run_episodes", runner)
    install_module(monkeypatch, "adapters.habitat_config", get_config=get_config)
    install_module(monkeypatch, "adapters.goat", create_environment=make_env)
    install_module(monkeypatch, "adapters.lagmemo_agent", LagMemoAgent=make_agent)
    install_module(monkeypatch, "habitat.core.env", Env=Habitat)
    install_module(
        monkeypatch, "lagmemo.core.interfaces", DiscreteNavigationAction=object()
    )
    with pytest.raises(RuntimeError, match=stage):
        run_navigation(request)
    assert events == (
        ["provider", "close_provider"]
        if stage in ("config", "habitat")
        else ["provider", "close_habitat", "close_provider"]
    )


def test_episode_range_error_closes_habitat_before_perception(tmp_path, monkeypatch):
    _, _, request, _ = fixture_request(tmp_path, "ordered_suffix_mvp")
    request = replace(request, episode_end=4)
    events = []

    class Provider:
        def __init__(self, **kw):
            pass

        def close(self):
            events.append("close_provider")

    class Habitat:
        def __init__(self, cfg):
            self.episodes = [0, 1]

        def close(self):
            events.append("close_habitat")

    monkeypatch.setattr("runtime.mapping_client.MappingClient", Provider)
    install_module(
        monkeypatch, "adapters.habitat_config", get_config=lambda *a: config()
    )
    install_module(
        monkeypatch,
        "adapters.goat",
        create_environment=lambda *a, **kw: pytest.fail("perception must not load"),
    )
    install_module(monkeypatch, "adapters.lagmemo_agent", LagMemoAgent=object)
    install_module(monkeypatch, "habitat.core.env", Env=Habitat)
    install_module(
        monkeypatch, "lagmemo.core.interfaces", DiscreteNavigationAction=object()
    )
    with pytest.raises(ValueError, match="invalid episode range"):
        run_navigation(request)
    assert events == ["close_habitat", "close_provider"]


@pytest.mark.parametrize(
    "key,value,error",
    [
        ("LAGMEMO_SEMANTIC_SOURCE", "ground_truth", "Detic"),
        ("ES_GROUND_TRUTH_SEMANTICS", "1", "GROUND_TRUTH"),
        ("ES_SCSP_STRICT_GEODESIC", "0", "geodesic"),
        ("ES_SCSP_EUCLIDEAN_FALLBACK", "1", "geodesic"),
        ("ES_SCSP_ARM", "wrong", "arm/profile"),
        ("ES_SCSP_RUNTIME_MODE", "off", "mode"),
        ("ES_SCSP_EXPECTED_DATASET_MANIFEST_SHA256", "wrong", "hash mismatch"),
    ],
)
def test_identity_and_semantic_guards_before_any_model_loading(
    tmp_path, key, value, error
):
    args, environment, _, _ = fixture_request(tmp_path, "ordered_suffix_mvp")
    environment[key] = value
    with pytest.raises(ValueError, match=error):
        resolve_request(args, root=tmp_path, engine=tmp_path, environ=environment)


def test_native_import_is_light_and_does_not_modify_process_state(tmp_path):
    code = f"""import sys,os
sys.path.insert(0,{str(ROOT)!r})
before=dict(os.environ);cwd=os.getcwd();paths=list(sys.path)
import runtime.navigation_worker
assert dict(os.environ)==before and os.getcwd()==cwd and sys.path==paths
assert not any(name in sys.modules for name in ('torch','numpy','habitat','lagmemo'))
"""
    result = subprocess.run(
        [sys.executable, "-S", "-c", code], cwd=tmp_path, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_native_cli_help_needs_no_simulator_or_models(tmp_path):
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    result = subprocess.run(
        [sys.executable, "-m", "runtime.navigation_worker", "--help"],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--dataset_root" in result.stdout


@pytest.mark.parametrize("kind", ["config", "detic", "identities", "habitat"])
def test_moved_setup_definitions_preserve_frozen_executable_bodies(kind):
    def definitions(source):
        return {
            n.name: n
            for n in ast.parse(source).body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef))
        }

    def digest(node):
        node = deepcopy(node)
        for item in ast.walk(node):
            if (
                isinstance(item, (ast.FunctionDef, ast.ClassDef))
                and item.body
                and isinstance(item.body[0], ast.Expr)
                and isinstance(item.body[0].value, ast.Constant)
                and isinstance(item.body[0].value.value, str)
            ):
                item.body.pop(0)
        return ast.dump(node, include_attributes=False)

    original = definitions(frozen_source())
    if kind == "config":
        old = definitions(
            frozen_source("Navigation/runtime/evaluators/goat/config_utils.py")
        )
        new = definitions((ROOT / "adapters/habitat_config.py").read_text())
        assert digest(old["get_config"]) == digest(new["get_config"])
    elif kind == "detic":
        old = deepcopy(original["_install_detic_inference_import_shims"])
        old.name = "install_inference_import_shims"
        old.args.args = [ast.arg(arg="lagmemo_source")]

        class BindSource(ast.NodeTransformer):
            def visit_Name(self, node):
                if node.id == "_LAGMEMO_SRC":
                    node.id = "lagmemo_source"
                return node

        old = BindSource().visit(old)
        new = definitions((ROOT / "runtime/detic_bootstrap.py").read_text())
        assert digest(old) == digest(new["install_inference_import_shims"])
    elif kind == "identities":
        new = definitions((ROOT / "experiments/identities.py").read_text())
        for name, canonical in [
            ("_load_scsp_frozen_identities", "load_frozen_identities"),
            ("_load_scsp_identity_sidecar", "load_identity_sidecar"),
        ]:
            old = deepcopy(original[name])
            old.name = canonical
            if canonical == "load_frozen_identities":
                old.args.args.insert(0, ast.arg(arg="path"))
                old.body[1] = ast.parse("path = Path(path)").body[0]
            assert digest(old) == digest(new[canonical])
    else:
        old = definitions(frozen_source("runtime/legacy_habitat.py"))
        new = definitions((ROOT / "adapters/goat.py").read_text())
        for name in ("load_categories", "_ground_truth_mapping"):
            assert digest(old[name]) == digest(new[name])
        old_ctor = next(
            n
            for n in old["HabitatGoatEnv"].body
            if isinstance(n, ast.FunctionDef) and n.name == "__init__"
        )
        new_ctor = next(
            n
            for n in new["HabitatGoatEnv"].body
            if isinstance(n, ast.FunctionDef) and n.name == "__init__"
        )
        old_ctor = deepcopy(old_ctor)
        new_ctor = deepcopy(new_ctor)
        old_ctor.args = new_ctor.args
        old_ctor.body = [
            n
            for n in old_ctor.body
            if not (
                isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "settings"
            )
        ]
        for n in ast.walk(new_ctor):
            if (
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name)
                and n.func.id == "partial"
            ):
                n.keywords = [
                    k
                    for k in n.keywords
                    if k.arg not in ("config_file", "checkpoint_file")
                ]
        assert digest(old_ctor) == digest(new_ctor)


def test_worker_signal_unwinds_resources_and_restores_previous_handler(
    tmp_path, monkeypatch
):
    import signal
    import runtime.navigation_worker as worker
    from contextlib import ExitStack

    _, _, request, _ = fixture_request(tmp_path, "ordered_suffix_mvp")
    events = []
    previous = signal.getsignal(signal.SIGTERM)
    monkeypatch.setattr(
        "runtime.legacy_navigation_run.prepare_request", lambda argv: request
    )

    def run(_request):
        with ExitStack() as stack:
            stack.callback(events.append, "closed")
            os.kill(os.getpid(), signal.SIGTERM)

    monkeypatch.setattr(worker, "run_navigation", run)
    with pytest.raises(SystemExit) as exc:
        worker.main([])
    assert exc.value.code == 128 + signal.SIGTERM
    assert events == ["closed"] and signal.getsignal(signal.SIGTERM) == previous
