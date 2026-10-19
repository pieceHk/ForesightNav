"""Replay the original sensor and evaluator methods without loading Habitat."""

import ast
from enum import Enum
from functools import lru_cache
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from adapters.habitat import HabitatEnvironment, HabitatObservationSettings
from experiments.evaluator import GoatEvaluator
from runtime.legacy_habitat import load_categories, observation_settings_from_environment


ROOT = Path(__file__).resolve().parents[2]
SOURCE = "Navigation/runtime/lagmemo/src/lagmemo/lagmemo/env/habitat_lagmemo_env.py"
BASE = "Navigation/runtime/lagmemo/src/lagmemo/lagmemo/env/habitat_abstract_env.py"
CATEGORIES = ("bathroom_cabinet", "bed", "chair")
ACTIONS = {"stop": 0, "move_forward": 1, "turn_left": 2, "turn_right": 3}


class Action(Enum):
    STOP = 0
    MOVE_FORWARD = 1
    TURN_LEFT = 2
    TURN_RIGHT = 3


@lru_cache()
def frozen_environment():
    selected = {
        "_preprocess_obs", "_preprocess_semantic", "_preprocess_goals",
        "_preprocess_action", "_preprocess_depth", "get_subtask_path_lengths",
        "reset_subtask", "_preprocess_xy",
    }
    methods = []
    for path in (SOURCE, BASE):
        tree = ast.parse(subprocess.check_output(["git", "show", "422cb8e:" + path], cwd=ROOT))
        names = {"_preprocess_xy"} if path == BASE else selected - {"_preprocess_xy"}
        methods.extend(
            node for cls in tree.body if isinstance(cls, ast.ClassDef)
            for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names
        )
    for method in methods:
        method.returns = None
        for arg in (*method.args.posonlyargs, *method.args.args, *method.args.kwonlyargs):
            arg.annotation = None
    cls = ast.ClassDef(name="FrozenEnvironment", bases=[], keywords=[], body=methods, decorator_list=[])
    namespace = {
        "np": np, "all_ovon_categories": list(CATEGORIES), "cast": lambda _type, action: action,
        "MIN_DEPTH_REPLACEMENT_VALUE": 10000, "MAX_DEPTH_REPLACEMENT_VALUE": 10001,
        "HabitatSimActions": ACTIONS,
        "lagmemo": SimpleNamespace(core=SimpleNamespace(interfaces=SimpleNamespace(
            Observations=SimpleNamespace, DiscreteNavigationAction=Action,
        ))),
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), "<frozen-habitat>", "exec"), namespace)
    return namespace["FrozenEnvironment"]


def raw_observation():
    return {
        "rgb": np.arange(36, dtype=np.uint8).reshape(3, 4, 3),
        "depth": np.linspace(.5, 5, 12, dtype=np.float32).reshape(3, 4, 1),
        "gps": np.array([2.5, -1.25]), "compass": np.array([.3]),
        "globalpose": np.array([1., 0., 0., 0., 2., 1.5, -3.]),
        "semantic": np.arange(12).reshape(3, 4, 1),
        "multigoal": [
            {"category": "bed", "image": None, "description": ""},
            {"category": "bathroom cabinet", "image": np.ones((2, 2, 3), dtype=np.uint8), "description": ""},
            {"category": "chair", "image": None, "description": "chair by the wall"},
        ],
    }


class Simulator:
    def __init__(self, events=None):
        self.events = events if events is not None else []
        self.current_episode = SimpleNamespace(
            tasks=[("bed", "object"), ("bathroom cabinet", "image"), ("chair", "description")],
            scene_id="hm3d/scene-a/scene-a.basis.glb", episode_id="7",
        )
        self.episode_over = False
        self.number_of_episodes = 1

    def reset(self):
        self.events.append("reset")
        return raw_observation()

    def step(self, action):
        self.events.append(("step", action))
        return raw_observation()

    def get_metrics(self):
        raise AssertionError("Sensor translation must not read evaluation measurements")

    @property
    def sim(self):
        raise AssertionError("Predicted semantics must not read semantic annotations")


class Segmentation:
    def __init__(self, events):
        self.events = events

    def predict(self, obs):
        self.events.append("predict")
        obs.semantic = np.full(obs.rgb.shape[:2], 2, dtype=np.int64)
        obs.task_observations["instance_map"] = np.ones(obs.rgb.shape[:2], dtype=np.int64)
        return obs


def environment(source, *, events=None, observer=None):
    events = events if events is not None else []

    def factory(categories):
        assert tuple(categories) == CATEGORIES
        events.append("factory")
        return Segmentation(events)

    return HabitatEnvironment(
        Simulator(events), settings=HabitatObservationSettings(.5, 10., source),
        categories=CATEGORIES, observation_factory=SimpleNamespace, action_ids=ACTIONS,
        segmentation_factory=factory, semantic_mapping={1: 2, 3: 1, 7: 3}, observer=observer,
    )


def plain(value):
    if isinstance(value, np.ndarray):
        return {"shape": value.shape, "dtype": str(value.dtype), "values": value.tolist()}
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    if isinstance(value, SimpleNamespace):
        return plain(vars(value))
    return value


@pytest.mark.parametrize("source", ("disabled", "detic", "ground_truth"))
def test_sensor_arrays_and_ordered_goal_modalities_match_frozen(source):
    current = environment(source)
    current.current_episode = current.habitat_env.current_episode
    old = frozen_environment()()
    for name in ("semantic_source", "current_episode", "hm3d_mapping", "min_depth", "max_depth"):
        setattr(old, name, getattr(current, name))
    old.segmentation = Segmentation([])
    old.get_episode_metrics = lambda: {"goat_top_down_map": {"evaluation_only": True}}
    expected = old._preprocess_obs(raw_observation())
    expected.task_observations.pop("top_down_map")
    actual = current._preprocess_obs(raw_observation())
    assert plain(actual) == plain(expected)
    assert [goal["type"] for goal in actual.task_observations["tasks"]] == ["objectnav", "imagenav", "languagenav"]
    assert "top_down_map" not in actual.task_observations


@pytest.mark.parametrize("source", ("disabled", "detic"))
def test_predicted_and_disabled_modes_do_not_consume_semantic_sensor(source):
    current = environment(source)
    current.current_episode = current.habitat_env.current_episode
    raw = raw_observation()
    del raw["semantic"]
    obs = current._preprocess_obs(raw)
    assert obs.semantic.shape == (3, 4)


def test_reset_action_and_observer_order_preserve_episode_lifecycle():
    events = []
    observer = SimpleNamespace(
        reset=lambda episode: events.append(("observer_reset", episode.episode_id)),
        on_action=lambda info: events.append(("observer_action", info)),
    )
    current = environment("detic", events=events, observer=observer)
    obs = current.reset()
    assert current.get_observation() is obs
    current.apply_action(Action.TURN_LEFT, info={"frame": 1})
    assert events == [
        "factory", "reset", "factory", "predict", ("observer_reset", "7"),
        ("observer_action", {"frame": 1}), ("step", 2), "predict",
    ]
    assert "top_down_map" not in current.get_observation().task_observations


@pytest.mark.parametrize("action", (0, 3, Action.STOP, Action.MOVE_FORWARD, Action.TURN_LEFT, Action.TURN_RIGHT))
def test_discrete_action_mapping_matches_frozen(action):
    assert environment("disabled")._preprocess_action(action) == frozen_environment()()._preprocess_action(action)


def test_depth_helper_matches_frozen_and_sensor_path_keeps_metric_values():
    current = environment("disabled")
    old = frozen_environment()()
    old.min_depth, old.max_depth = current.min_depth, current.max_depth
    depth = np.array([0., .25, .75, 1.], dtype=np.float32).reshape(2, 2, 1)
    np.testing.assert_array_equal(current._preprocess_depth(depth), old._preprocess_depth(depth))
    current.current_episode = current.habitat_env.current_episode
    raw = raw_observation()
    np.testing.assert_array_equal(current._preprocess_obs(raw).depth, raw["depth"][:, :, -1])


@pytest.mark.parametrize("actual,shortest,success", (
    (0., 0., 0.), (2., 1., 1.), (2., float("inf"), 0.),
    (-1., 1., 0.), (float("nan"), 1., 0.), (float("inf"), 1., 0.),
    (None, 1., 0.), (2., None, 0.), (2., -1., 0.),
    (2., float("nan"), 0.), (2., float("-inf"), 0.),
    (2., float("inf"), 1.), (2., float("inf"), float("nan")),
))
def test_evaluation_path_state_and_errors_match_frozen(actual, shortest, success):
    measures = {
        "goat_sub-task_spl": SimpleNamespace(_agent_episode_distance=actual, _start_end_episode_distance=shortest),
        "goat_sub-task_success": SimpleNamespace(get_metric=lambda: success),
    }
    sim = SimpleNamespace(task=SimpleNamespace(measurements=SimpleNamespace(measures=measures)))
    current = GoatEvaluator(sim)
    old = frozen_environment()()
    old.habitat_env = sim
    outcomes = []
    for evaluator in (old, current):
        try:
            outcomes.append(evaluator.get_subtask_path_lengths())
        except RuntimeError as exc:
            outcomes.append((type(exc), str(exc)))
    assert outcomes[0] == outcomes[1]


@pytest.mark.parametrize("missing", ("goat_sub-task_spl", "goat_sub-task_success"))
def test_missing_evaluation_measure_matches_frozen(missing):
    measures = {
        "goat_sub-task_spl": SimpleNamespace(_agent_episode_distance=2., _start_end_episode_distance=float("inf")),
        "goat_sub-task_success": SimpleNamespace(get_metric=lambda: 0.),
    }
    del measures[missing]
    sim = SimpleNamespace(task=SimpleNamespace(measurements=SimpleNamespace(measures=measures)))
    old = frozen_environment()()
    old.habitat_env = sim
    for evaluator in (old, GoatEvaluator(sim)):
        with pytest.raises(RuntimeError, match="unavailable"):
            evaluator.get_subtask_path_lengths()


def test_distance_update_precedes_spl_and_map_reset_as_in_frozen():
    logs = []
    for cls in (frozen_environment(), GoatEvaluator):
        events = []
        measures = {}
        for name in ("goat_sub-task_spl", "unrelated", "goat_distance_to_sub-goal", "goat_top_down_map"):
            measures[name] = SimpleNamespace(
                uuid=name,
                update_metric=lambda name=name, **kw: events.append(("update", name, kw)),
                reset_metric=lambda name=name, **kw: events.append(("reset", name, kw)),
            )
        task = SimpleNamespace(measurements=SimpleNamespace(measures=measures))
        sim = SimpleNamespace(task=task, _task=task, current_episode="episode-7")
        evaluator = cls.__new__(cls)
        evaluator.habitat_env = sim
        evaluator.reset_subtask()
        logs.append([(kind, name, kw["episode"], kw["task"] is task) for kind, name, kw in events])
    assert logs[0] == logs[1]
    assert [item[:2] for item in logs[1]] == [
        ("update", "goat_distance_to_sub-goal"), ("reset", "goat_sub-task_spl"), ("reset", "goat_top_down_map"),
    ]


def test_legacy_category_loading_is_explicit_and_retains_original_sorting(tmp_path):
    path = tmp_path / "categories.json"
    path.write_text('["chair", "bed", "bathroom cabinet", "bed"]')
    assert load_categories(path) == CATEGORIES


def test_semantic_source_is_resolved_once_at_legacy_boundary(monkeypatch):
    config = SimpleNamespace(ENVIRONMENT=SimpleNamespace(min_depth=.5, max_depth=10.), GROUND_TRUTH_SEMANTICS=0)
    monkeypatch.setenv("LAGMEMO_SEMANTIC_SOURCE", "invalid inherited value")
    assert observation_settings_from_environment(config, {}).semantic_source == "detic"
    config.GROUND_TRUTH_SEMANTICS = 1
    assert observation_settings_from_environment(config, {}).semantic_source == "ground_truth"
    assert observation_settings_from_environment(config, {"LAGMEMO_SEMANTIC_SOURCE": "DISABLED"}).semantic_source == "disabled"


def test_sensor_adapter_import_does_not_load_simulator_or_perception(tmp_path):
    script = (
        "import sys; import adapters.habitat; import runtime.legacy_habitat; "
        "assert not any(n in sys.modules for n in ('habitat','lagmemo','detectron2','torch'))"
    )
    import os

    subprocess.run([sys.executable, "-c", script], cwd=tmp_path,
                   env={**os.environ, "PYTHONPATH": str(ROOT)}, check=True)
