"""Execute frozen and native loops with fixed observations and policy outputs."""

import ast
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from enum import Enum
from functools import lru_cache, partial
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from adapters.geometry import project_goals_to_navigable
from experiments import recorder
from experiments.metrics import DISTANCE_TO_GOAL_KEY, SPL_KEY, SUCCESS_KEY
from experiments.planning import PlanningBridge
from experiments.runner import PlanningRunSettings, RunArtifacts, RunSettings, run_episodes
from pbsp.agent import PBSPAgent
from pbsp.navigation.cooldown import CategorySwitchCooldown
from pbsp.navigation.ranking import CandidateActionReranker
from pbsp.types import CandidateBatch, CandidateRecord, GeodesicDistance
from runtime.config import load_experiment


ROOT = Path(__file__).resolve().parents[2]
SOURCE = "Navigation/runtime/evaluators/goat/entrypoint.py"
BASELINE = "43824ab"


class Action(Enum):
    STOP = 0
    MOVE_FORWARD = 1
    TURN_LEFT = 2
    TURN_RIGHT = 3


class Clock:
    def __init__(self):
        self.value = 0

    def perf_counter(self):
        self.value += 1
        return self.value * 0.001


@lru_cache()
def reference_code():
    tree = ast.parse(
        subprocess.check_output(["git", "show", BASELINE + ":" + SOURCE], cwd=ROOT)
    )
    names = {
        "_finite_metric_summary",
        "_project_goals_to_navigable",
        "_task_category",
        "_record_scsp_provider_event",
        "_scsp_shadow_fallback_event",
        "_compute_scsp_shadow_decision",
        "_record_scsp_shadow_decision",
        "_fallback_value",
        "_scsp_dispatch_gate",
        "_prepare_scsp_refresh",
        "_scsp_raw_rank_for_position",
        "_apply_scsp_runtime_order",
        "_finalize_scsp_dispatch_event",
        "_reset_scsp_task_state",
        "_active_candidate",
        "_write_timing_rows",
    }
    helpers = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(helpers) == len(names)
    loop = next(
        n for n in ast.walk(tree) if isinstance(n, ast.For) and ast.unparse(n.target) == "ep_i"
    )
    imports = [
        n
        for n in tree.body
        if isinstance(n, ast.ImportFrom)
        and n.module
        and n.module.startswith(("experiments.", "pbsp."))
    ]
    return tuple(
        compile(ast.Module(body=nodes, type_ignores=[]), filename, "exec")
        for nodes, filename in (
            (imports + helpers, "<frozen-runner-services>"),
            ([loop], "<frozen-complete-episode-loop>"),
        )
    )


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if is_dataclass(value):
        return plain(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    return value


class Visualizer:
    def __init__(self, events, name):
        self.events, self.name = events, name

    def set_vis_dir(self, *parts):
        self.events.append((self.name, *parts))


class Bar:
    def __init__(self, events, total):
        self.events = events
        events.append(("progress", total))

    def set_description(self, text):
        self.events.append(("description", text))

    def update(self, count):
        self.events.append(("progress_update", count))

    def reset(self):
        self.events.append(("progress_reset",))

    def close(self):
        self.events.append(("progress_close",))


class Environment:
    def __init__(self, categories, case, events):
        self.categories, self.case, self.events = categories, case, events
        self.index = -1
        self.task = SimpleNamespace(current_task_idx=0)
        self.pathfinder = SimpleNamespace(snap_point=lambda position: position)
        self.habitat_env = SimpleNamespace(
            task=self.task, sim=SimpleNamespace(pathfinder=self.pathfinder)
        )
        self.visualizer = Visualizer(events, "environment_vis")

    def reset(self):
        self.index += 1
        self.task.current_task_idx = 0
        self.steps = [0] * len(self.categories)
        self.frame = 0
        self.habitat_env.current_episode = SimpleNamespace(
            scene_id="hm3d/scene-a/scene-a.basis.glb",
            episode_id=str(self.index),
            start_position=[0.0, 0.0, 0.0],
            start_rotation=[0.0, 0.0, 0.0, 1.0],
        )
        self.events.append(("environment_reset", self.index))

    @property
    def episode_over(self):
        # Exercise Habitat's delayed episode-over flag at the final STOP.
        return self.case == "incomplete" and self.frame == 2

    def get_observation(self):
        self.events.append(("observation", self.index, self.frame))
        return SimpleNamespace(
            rgb=np.full((4, 4, 3), self.frame, dtype=np.uint8),
            depth=np.ones((4, 4), dtype=np.float32),
            globalpose=np.array([1.0, 0.0, 0.0, 0.0, self.frame * 0.1, 0.0, 0.0]),
            task_observations={
                "tasks": [
                    {"category": category, "type": "objectnav", "image": None}
                    for category in self.categories
                ]
            },
        )

    def apply_action(self, action, info=None):
        task = self.task.current_task_idx
        self.events.append(("apply_action", task, action.name, info))
        self.steps[task] += 1
        self.frame += 1
        if action is Action.STOP:
            self.stopped_task = task
            self.task.current_task_idx += 1


class Evaluator:
    def __init__(self, env):
        self.env = env

    def get_episode_metrics(self):
        task = self.env.stopped_task
        self.env.events.append(("evaluation", task))
        failed = self.env.case == "unreachable" and task == 0
        actual, shortest = self.env.steps[task] * 0.2, 1.0
        return {
            SUCCESS_KEY: float(not failed),
            SPL_KEY: 0.0 if failed else shortest / max(actual, shortest),
            DISTANCE_TO_GOAL_KEY: float("inf") if failed else 0.0,
            "goat_top_down_map": {"evaluation_only": object()},
        }

    def get_subtask_path_lengths(self):
        task = self.env.stopped_task
        self.env.events.append(("path_capture", task))
        return {
            "actual_path_length": self.env.steps[task] * 0.2,
            "shortest_path_length": (
                float("inf") if self.env.case == "unreachable" and task == 0 else 1.0
            ),
        }

    def reset_subtask(self):
        self.env.events.append(("measure_reset", self.env.stopped_task))


class Provider:
    def __init__(self, categories, events, *, live=True):
        self.categories, self.events, self._live_mode = categories, events, live

    def reset_episode(self, scene, episode):
        self.scene, self.episode, self.frame, self.serial = scene, episode, 0, 0
        self.batch = None
        self.events.append(("mapping_reset", scene, episode))

    def invalidate_cache(self):
        self.batch = None
        self.events.append(("invalidate", self.frame))

    def get_all_goals(self):
        self.events.append(("query", self.frame))
        if self.batch is None:
            self.serial += 1
            records = tuple(
                (
                    task,
                    tuple(
                        CandidateRecord(
                            task,
                            category,
                            (task * 4.0 + rank + 1.0, 0.0, rank + 1.0),
                            rank,
                            0.7 + rank * 0.1,
                            4,
                            0.8,
                            7,
                        )
                        for rank in range(2)
                    ),
                )
                for task, category in enumerate(self.categories)
            )
            self.batch = CandidateBatch(
                self.scene,
                self.episode,
                7,
                7,
                self.serial,
                self.categories,
                tuple(
                    (task, tuple(record.position for record in values))
                    for task, values in records
                ),
                records,
            )
        return {
            task: [list(point) for point in self.batch.positions_for_task(task)]
            for task in range(len(self.categories))
        }

    def get_candidate_batch(self):
        self.events.append(("candidate_batch", self.serial))
        return self.batch

    def get_ordered_categories(self):
        return self.categories

    def check_candidate_batch_status(self, batch, **expected):
        self.events.append(("version_gate", self.serial, expected))
        stale = batch is None or batch.version_token != self.batch.version_token
        return {
            "would_invalidate": stale,
            "reason": "stale_version_token" if stale else "none",
            "scene_id": self.scene,
            "episode_id": self.episode,
            "query_serial": self.serial,
            "batch_map_revision": 7,
            "status_last_fused_keyframes": 7,
            "status_rebuilding": False,
        }

    @property
    def current_snap(self):
        return self.frame

    def push_live_frame(self, rgb, depth, pose):
        self.frame += 1
        self.events.append(("push_live", int(rgb[0, 0, 0]), float(depth.mean()), tuple(pose)))

    def push_frame(self, index):
        self.frame += 1
        self.events.append(("push_frame", index))

    def status(self):
        self.events.append(("status", self.frame))
        return {"last_fused_keyframes": 7, "raw_frames": self.frame}


class Agent(PBSPAgent):
    def __init__(self, env, events):
        super().__init__(
            settings=load_experiment(
                ROOT / "configs/experiments/hm3d_4mon.yaml"
            ).method.navigation,
            actions=Action,
        )
        self.env, self.events = env, events
        self.planner = Visualizer(events, "planner_vis")
        self.imagenav_visualizer = Visualizer(events, "agent_vis")
        self.matching = Visualizer(events, "matching_vis")

    def reset(self, *, candidate_provider, **kwargs):
        self.events.append(("agent_reset", kwargs))
        self.current_task_idx = 0
        self.gs_poses = candidate_provider.get_all_goals()
        self.frontier_flag = {task: 0 for task in self.gs_poses}
        self.sub_task_timesteps = [[0] * len(self.env.categories)]
        self.visited_candidate_positions = {}
        self.candidate_evidence.reset()
        self.candidate_evidence_diagnostics = {}
        self.trapped_events = []

    def refresh_gs_poses(self, goals, **kwargs):
        self.events.append(("refresh", deepcopy(goals), deepcopy(kwargs)))
        return super().refresh_gs_poses(goals, **kwargs)

    def set_lightglue_vis_dir(self, directory):
        self.events.append(("matching_output", directory))

    def act(self, obs):
        task = self.current_task_idx
        self.sub_task_timesteps[0][task] += 1
        step = self.sub_task_timesteps[0][task]
        self.last_action_metadata = {}
        action = Action.MOVE_FORWARD
        if self.env.case == "illegal_stop" and step == 2:
            action = Action.STOP
            self.last_action_metadata["stop_reason"] = "planner_error"
        elif step == (12, 3, 11, 4)[task]:
            action = Action.STOP
            self.last_action_metadata["stop_reason"] = (
                "max_steps"
                if self.env.case == "unreachable" and task == 0
                else "verified_target"
            )
            self.current_task_idx += 1
        elif self.env.case == "recovery" and task == 0:
            if step == 2:
                event = {"step": step}
                self.trapped_events.append(event)
                self.last_action_metadata["trapped_event"] = event
            elif step == 3:
                self.last_action_metadata["random_exploration_requested"] = True
        self.events.append(("act", task, step, action.name))
        return action, None

    def set_navmesh_override_goal(self, goal, mode):
        self.events.append(("override", plain(goal), mode))

    def clear_local_navigation_state(self):
        self.events.append(("clear_navigation",))


def execute(
    tmp_path,
    monkeypatch,
    *,
    native,
    tasks=3,
    episodes=1,
    mode="ordered_suffix_mvp",
    case="normal",
    cadence=10,
    record_trajectory=False,
):
    output = tmp_path / ("native" if native else "frozen")
    events, clock = [], Clock()
    monkeypatch.setattr(recorder, "time", clock)
    categories = ("chair", "chair", "bed", "tv")[:tasks]
    env = Environment(categories, case, events)
    evaluator, provider, agent = (
        Evaluator(env),
        Provider(categories, events, live=case != "snapshot"),
        Agent(env, events),
    )
    cooldown = CategorySwitchCooldown({"chair": 10, "bed": 30, "tv": 10}, match_radius_m=1.0)
    reranker = CandidateActionReranker(enabled=True)
    identities = {
        ("scene-a", str(index)): {
            "dataset_manifest_sha256": "a" * 64,
            "scene_id": "scene-a",
            "runtime_episode_id": str(index),
            "source_episode_ids": [str(index * tasks + task) for task in range(tasks)],
            "ordered_categories": list(categories),
        }
        for index in range(episodes)
    }
    if case == "identity_mismatch":
        identities[("scene-a", "0")]["ordered_categories"] = list(reversed(categories))
    elif case == "missing_identity":
        identities.clear()
    settings = PlanningRunSettings(mode, mode, tasks, 5, 0.35, {}, {}, identities)
    artifacts = RunArtifacts(output)
    project = partial(project_goals_to_navigable, max_distance=1.5)

    def geodesic(pathfinder, start, end):
        events.append(("geodesic", tuple(start), tuple(end)))
        return GeodesicDistance.available(float(np.linalg.norm(np.asarray(end) - start)))

    def sample(pathfinder, origin, kind):
        events.append(("sample", tuple(origin), kind))
        return [4.0, 0.0, 2.0], {"mode": kind}

    progress = lambda total: Bar(events, total)
    error = None
    try:
        if native:
            observed_env = env
            if record_trajectory:
                from experiments.trajectory import TrajectoryEnvironment

                observed_env = TrajectoryEnvironment(
                    env, directory=output, identities=identities,
                    dataset_manifest_sha256="a" * 64, actions=Action,
                    action_metadata=lambda: agent.last_action_metadata,
                )
            run_episodes(
                env=observed_env,
                evaluator=evaluator,
                agent=agent,
                candidate_provider=provider,
                category_cooldown=cooldown,
                reranker=reranker,
                planning=PlanningBridge(project_goals=project, geodesic=geodesic, clock=clock),
                sample_goal=sample,
                actions=Action,
                settings=RunSettings(0, episodes, 12, True, cadence),
                planning_settings=settings,
                artifacts=artifacts,
                progress_factory=progress,
                clock=clock,
            )
        else:
            output.mkdir()
            namespace = dict(
                np=np,
                os=__import__("os"),
                json=json,
                Path=Path,
                time=clock,
                pprint=lambda value: None,
                tqdm=progress,
                DiscreteNavigationAction=Action,
                env=env,
                evaluator=evaluator,
                agent=agent,
                candidate_provider=provider,
                category_cooldown=cooldown,
                scsp_reranker=reranker,
                episode_start=0,
                episode_end=episodes,
                results_dir=str(output),
                args=SimpleNamespace(enable_timing=True),
                config=SimpleNamespace(AGENT=SimpleNamespace(max_steps=12)),
                metrics=artifacts.metrics,
                timing_rows=artifacts.timing_rows,
                es_status_log=artifacts.es_status_log,
                scsp_distance_cache=artifacts.distance_cache,
                scsp_runtime_mode=mode,
                scsp_enabled=mode != "off",
                scsp_provider_events_path=output / "scsp_provider_events.jsonl",
                scsp_decisions_path=output / "scsp_decisions.jsonl",
                scsp_arm=mode,
                scsp_ordered_subtasks=tasks,
                scsp_candidate_cap=5,
                scsp_suffix_weight=0.35,
                scsp_idea1_config={},
                scsp_provenance={},
                scsp_episode_identities=identities,
            )
            helpers, loop = reference_code()
            exec(helpers, namespace)
            namespace.update(_strict_geodesic_distance=geodesic, _sample_navmesh_goal=sample)
            exec(loop, namespace)
            artifacts.metrics = namespace["metrics"]
    except RuntimeError as exc:
        error = str(exc)
    files = {
        str(path.relative_to(output)): path.read_text()
        for path in output.rglob("*")
        if path.is_file()
    }
    return {
        "events": plain(events),
        "files": files,
        "metrics": artifacts.metrics,
        "timing": artifacts.timing_rows,
        "status": artifacts.es_status_log,
        "candidates": plain(agent.gs_poses) if hasattr(agent, "gs_poses") else None,
        "error": error,
    }


@pytest.mark.parametrize(
    "tasks,episodes,mode,case",
    (
        (2, 1, "ordered_suffix_mvp", "normal"),
        (3, 2, "ordered_suffix_mvp", "normal"),
        (4, 1, "ordered_suffix_mvp", "normal"),
        (3, 1, "idea1_only", "normal"),
        (3, 1, "shadow", "normal"),
        (3, 1, "off", "snapshot"),
        (3, 1, "ordered_suffix_mvp", "recovery"),
        (3, 1, "ordered_suffix_mvp", "unreachable"),
        (3, 1, "ordered_suffix_mvp", "identity_mismatch"),
        (3, 1, "ordered_suffix_mvp", "missing_identity"),
        (3, 1, "ordered_suffix_mvp", "illegal_stop"),
        (3, 1, "ordered_suffix_mvp", "incomplete"),
    ),
)
def test_complete_loop_actions_calls_and_artifacts_match_frozen(
    tmp_path, monkeypatch, tasks, episodes, mode, case
):
    old = execute(
        tmp_path,
        monkeypatch,
        native=False,
        tasks=tasks,
        episodes=episodes,
        mode=mode,
        case=case,
    )
    new = execute(
        tmp_path, monkeypatch, native=True, tasks=tasks, episodes=episodes, mode=mode, case=case
    )
    assert new == old
    if case in {"identity_mismatch", "missing_identity", "illegal_stop", "incomplete"}:
        assert new["error"]
    else:
        assert new["error"] is None
        assert len(new["metrics"]) == episodes
        assert all(len(record["metrics"]) == tasks for record in new["metrics"].values())
        assert sum(event[0] == "mapping_reset" for event in new["events"]) == episodes


def test_refresh_cadence_is_supplied_by_configuration(tmp_path, monkeypatch):
    result = execute(tmp_path, monkeypatch, native=True, cadence=3)
    assert result["error"] is None
    assert [event[1] for event in result["events"] if event[0] == "invalidate"] == list(
        range(0, 27, 3)
    )


@pytest.mark.parametrize("episodes,case", [(2, "normal"), (1, "recovery")])
def test_trajectory_observer_preserves_all_calls_actions_metrics_and_timing(tmp_path, monkeypatch, episodes, case):
    plain_run = execute(tmp_path / "plain", monkeypatch, native=True, episodes=episodes, case=case)
    recorded = execute(tmp_path / "recorded", monkeypatch, native=True, episodes=episodes,
                       case=case, record_trajectory=True)
    assert plain_run["error"] is recorded["error"] is None
    for key in set(plain_run) - {"files"}:
        assert recorded[key] == plain_run[key]
    assert {key: recorded["files"][key] for key in plain_run["files"]} == plain_run["files"]
    added = {f"episodes/scene-a/{episode}/{name}" for episode in range(episodes)
             for name in ("identity.json", "trajectory.jsonl")}
    assert set(recorded["files"]) - set(plain_run["files"]) == added
    all_rows = []
    for episode in range(episodes):
        identity = json.loads(recorded["files"][f"episodes/scene-a/{episode}/identity.json"])
        assert identity["identity"]["source_episode_ids"] == [str(episode * 3 + task) for task in range(3)]
        rows = [json.loads(line) for line in recorded["files"][f"episodes/scene-a/{episode}/trajectory.jsonl"].splitlines()]
        assert [row["frame_index"] for row in rows] == list(range(1, len(rows) + 1))
        assert [row["world_pose"][4] for row in rows] == pytest.approx([index * 0.1 for index in range(len(rows))])
        assert [row["subtask_index"] for row in rows if row["is_stop"]] == [0, 1, 2]
        all_rows.extend(rows)
    actions = [event for event in recorded["events"] if event[0] == "apply_action"]
    assert [(row["subtask_index"], row["action_name"]) for row in all_rows] == [(event[1], event[2]) for event in actions]


def test_runner_rejects_in_place_resume_before_using_services(tmp_path):
    (tmp_path / "per_episode_metrics.json").write_text("{}")
    with pytest.raises(RuntimeError, match="in-place resume is unsupported"):
        run_episodes(
            env=None,
            evaluator=None,
            agent=None,
            candidate_provider=None,
            category_cooldown=None,
            reranker=None,
            planning=None,
            sample_goal=None,
            actions=Action,
            settings=RunSettings(0, 1, 12, True, 10),
            planning_settings=PlanningRunSettings("off", "off", 0, 5, 0.35, {}, {}, {}),
            artifacts=RunArtifacts(tmp_path),
            progress_factory=lambda **kw: None,
        )


def test_running_loop_preserves_frozen_executable_structure():
    old = ast.parse(subprocess.check_output(["git", "show", BASELINE + ":" + SOURCE], cwd=ROOT))
    current = ast.parse((ROOT / "experiments/runner.py").read_text())

    def loop(tree):
        return next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.For) and ast.unparse(n.target) == "ep_i"
        )

    class ExplicitSettings(ast.NodeTransformer):
        def visit_Attribute(self, node):
            names = {
                "config.AGENT.max_steps": "settings.max_steps_per_task",
                "args.enable_timing": "settings.enable_timing",
            }
            if ast.unparse(node) in names:
                return ast.copy_location(
                    ast.parse(names[ast.unparse(node)], mode="eval").body, node
                )
            return self.generic_visit(node)

        def visit_BinOp(self, node):
            if (
                isinstance(node.op, ast.Mod)
                and ast.unparse(node.left) == "t"
                and isinstance(node.right, ast.Constant)
                and node.right.value == 10
            ):
                node.right = ast.parse("settings.query_refresh_cadence_steps", mode="eval").body
            return self.generic_visit(node)

    assert ast.dump(ExplicitSettings().visit(loop(old)), include_attributes=False) == ast.dump(
        loop(current), include_attributes=False
    )


def test_planning_bridges_preserve_frozen_executable_structure():
    from tests.replay.planning_sources import BRIDGE_METHODS, planning_function_source

    tree = ast.parse(
        subprocess.check_output(["git", "show", BASELINE + ":" + SOURCE], cwd=ROOT)
    )
    for name in BRIDGE_METHODS:
        old = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == name
        )
        new = ast.parse(planning_function_source(name)).body[0]
        assert ast.dump(old, include_attributes=False) == ast.dump(
            new, include_attributes=False
        )


def test_runner_import_does_not_load_model_or_simulator(tmp_path):
    import os

    code = (
        "import sys; import experiments.runner; import experiments.planning; "
        "assert not any(n in sys.modules for n in ('torch','habitat','lagmemo','detectron2'))"
    )
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        check=True,
    )


def test_legacy_geometry_controls_are_bound_once_per_instance(monkeypatch):
    from runtime.legacy_experiments import planning_bridge_from_environment

    monkeypatch.setenv("ES_NAV_PROJECTION_MAX_DISTANCE", "invalid inherited value")
    environment = {"ES_NAV_PROJECTION_MAX_DISTANCE": ".5"}
    strict = planning_bridge_from_environment(environment)
    environment["ES_NAV_PROJECTION_MAX_DISTANCE"] = "2.0"
    relaxed = planning_bridge_from_environment(environment)
    pathfinder = SimpleNamespace(snap_point=lambda point: np.asarray(point) + [1.0, 0.0, 0.0])
    assert strict.project_goals({0: [[0.0, 0.0, 0.0]]}, pathfinder) == {0: []}
    assert relaxed.project_goals({0: [[0.0, 0.0, 0.0]]}, pathfinder) == {0: [[1.0, 0.0, 0.0]]}


def test_planning_fallback_uses_the_injected_clock_without_a_global_patch():
    bridge = PlanningBridge(project_goals=None, geodesic=None, clock=Clock())
    event = bridge.compute_decision(
        phase="fixture", arm="ordered_suffix_mvp", provider_event={}, candidate_batch=None,
        projected_goals={}, projected_indices={}, current_task=0, origin=[0., 0., 0.],
        pathfinder=None, agent=None, reranker=None, distance_cache={}, candidate_cap=5,
        suffix_weight=.35, idea1_config={}, provenance={}, episode_identity=None,
    )
    assert event["fallback_reason"] == "invalid_candidate_batch"
    assert event["total_shadow_latency_ms"] == pytest.approx(1.)
