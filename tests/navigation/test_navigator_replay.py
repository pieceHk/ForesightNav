"""Replay observations and fixed perception/planner outputs through both kernels."""

import ast
from copy import deepcopy
from enum import Enum
from functools import lru_cache
import logging
import math
from pathlib import Path
import subprocess
from types import MethodType, SimpleNamespace
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation as R

from experiments.visualization import NavigationVisualization
from pbsp.navigation.evidence import (
    CandidateEvidenceBank,
    passes_dual_score_verification,
    select_independent_evidence,
)
from pbsp.navigation.navigator import Navigator
from pbsp.navigation.ranking import FrontierUtilityRanker
from pbsp.navigation.geometry import get_rel_pose_change
from pbsp.navigation.local_map import LocalMapState
from pbsp.navigation.verification import GoalVerifier


ROOT = Path(__file__).resolve().parents[2]
SOURCE = "Navigation/runtime/lagmemo/src/lagmemo/lagmemo/agent/lagmemo_agent/glue_agent.py"


class DiscreteNavigationAction(Enum):
    STOP = 0
    MOVE_FORWARD = 1
    TURN_LEFT = 2
    TURN_RIGHT = 3


class NativeAgent(NavigationVisualization, Navigator):
    pass


def test_matching_visualization_writes_a_readable_image(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    agent = NativeAgent(actions=DiscreteNavigationAction)
    agent.lightglue_vis_dir = str(tmp_path / "matches")
    agent.sub_task_timesteps = [[3]]
    agent.current_task_idx = 0
    goal = np.zeros((24, 32, 3), dtype=np.uint8)
    observation = np.full((24, 32, 3), 255, dtype=np.uint8)
    points = np.array([[5.0, 7.0], [12.0, 15.0]])

    agent.save_glue_visualization(goal, observation, points, points)

    saved = cv2.imread(str(tmp_path / "matches" / "3.png"))
    assert saved is not None
    assert saved.shape[0] > 24 and saved.shape[1] > 64
    assert np.any(saved[:, :, 1] > saved[:, :, 0])


@lru_cache()
def frozen_agent():
    tree = ast.parse(subprocess.check_output(["git", "show", "422cb8e:" + SOURCE], cwd=ROOT))
    original = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "GLUEAgent"
    )
    selected = {
        "act",
        "score_thresh",
        "reset_sub_episode",
        "_clear_seem_verification_state",
        "_clear_candidate_verification_evidence",
        "_mark_current_candidate_visited",
        "_reset_trapped_state",
        "_update_trapped_diagnostic",
        "_clear_navmesh_override_goal",
        "clear_local_navigation_state",
        "set_navmesh_override_goal",
        "mask_to_relative_2d_vectorized",
        "process_relative_poses",
        "lagmemo_score",
        "_preprocess_obs",
        "prepare_planner_inputs",
        "_prep_goal_map_input",
    }
    methods = [
        node
        for node in original.body
        if isinstance(node, ast.FunctionDef) and node.name in selected
    ]
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    functions.append(
        ast.ClassDef(name="FrozenAgent", bases=[], keywords=[], body=methods, decorator_list=[])
    )
    namespace = dict(
        np=np,
        torch=torch,
        cv2=cv2,
        math=math,
        R=R,
        logging=logging,
        LOGGER=logging.getLogger("frozen-navigation"),
        Observations=SimpleNamespace,
        DiscreteNavigationAction=DiscreteNavigationAction,
        Any=Any,
        Dict=Dict,
        List=List,
        Tuple=Tuple,
        passes_dual_score_verification=passes_dual_score_verification,
        select_independent_evidence=select_independent_evidence,
        pu=SimpleNamespace(get_rel_pose_change=get_rel_pose_change),
    )
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[])),
            "<frozen-navigation>",
            "exec",
        ),
        namespace,
    )
    return namespace["FrozenAgent"]


class Planner:
    def __init__(self, action=DiscreteNavigationAction.MOVE_FORWARD, error=None, no_path=False):
        self.action = action
        self.last_plan_error = error
        self.no_path = no_path
        self.calls = []
        self.resets = 0

    def plan(self, **inputs):
        self.calls.append(deepcopy(inputs))
        goal = np.zeros((40, 40), dtype=np.float32)
        goal[25, 25] = 1
        return self.action, goal, None, None, self.no_path, False

    def reset_subtask_state(self):
        self.resets += 1


class Module:
    def __init__(self):
        self.resets = 0

    def reset_sub_episode(self):
        self.resets += 1


class Visualizer:
    def __init__(self):
        self.frames = []

    def visualize(self, **info):
        self.frames.append(deepcopy(info))


def _preprocess(actor, observation, task_type):
    return (None,) * 10


def _map_inputs(actor, *args, **kwargs):
    return deepcopy(actor.map_inputs), deepcopy(actor.map_context)


def _perception(actor, observation, detic_type="interest"):
    actor.perception_calls += 1
    actor._last_verification_triggered = actor.perception_trigger
    actor._last_clip_score = actor.perception_clip
    actor._last_seem_score = actor.perception_seem
    mask = np.full((16, 16), int(actor.perception_trigger), dtype=np.uint8)
    return actor._last_seem_score, mask, None


def make_agent(
    cls,
    *,
    step=1,
    goals=(),
    planner_action=DiscreteNavigationAction.MOVE_FORWARD,
    planner_error=None,
    no_path=False,
    visualize=False,
):
    actor = cls.__new__(cls)
    actor.actions = DiscreteNavigationAction
    actor._es_test_gs_override = 3
    actor.exp_first = False
    actor.current_task_idx = 0
    actor.num_environments = 1
    actor.device = "cpu"
    actor.gs_poses = {0: list(goals), 1: []}
    actor.frontier_flag = {0: 0, 1: 0}
    actor.visited_candidate_positions = {}
    actor.candidate_match_radius = 1.0
    actor.candidate_evidence = CandidateEvidenceBank()
    actor.candidate_evidence_diagnostics = {}
    actor.last_external_ranking_metadata = None
    actor.frontier_blacklist = []
    actor.frontier_reranker = FrontierUtilityRanker(enabled=False)
    actor.frontier_rerank_diagnostics = {}
    actor.fully_explored = [False]
    actor.force_match_against_memory = False
    actor.reject_visited_targets = False
    actor.see_around = 0
    actor.navmesh_override_goal = actor.navmesh_override_task = actor.navmesh_override_mode = None
    actor.sub_task_timesteps = [[step, 0]]
    actor.total_timesteps = [step]
    actor.timesteps_before_goal_update = [0]
    actor.max_steps = 500
    actor.episode_panorama_start_steps = 0
    actor.random_exploration_warmup_steps = 30
    actor.use_dilation_for_stg = True
    actor.prev_position = None
    actor.ctr = actor.stuck_num = actor.stuck_flag = 0
    actor.rot_matrix_v2 = np.eye(3)
    actor.tran_v2 = np.zeros(3)
    actor.td_transform = np.eye(3)
    actor.td_goal_pos = None
    actor.runtime_scene_id, actor.runtime_episode_id = "scene-a", "0"
    actor.trapped_window_steps, actor.trapped_warmup_steps, actor.trapped_radius = 25, 30, 0.05
    actor.trapped_events = []
    actor.trapped_position_history = []
    actor.trapped_active = False
    actor.two_stage_verification = True
    actor.verification_window_steps, actor.verification_min_confirmations = 6, 2
    actor.verification_pose_radius, actor.verification_min_peak_seem = 1.5, 1.15
    actor.verification_dual_score_gate = True
    actor.verification_strong_peak_seem, actor.verification_min_peak_clip = 1.35, 0.12
    actor.verification_min_observer_translation = actor.verification_min_observer_yaw_deg = 0.0
    actor.stop_by_seem = False
    actor.seem_goal_pose = actor.seem_semantic = None
    actor.goal_vis_stage = 0
    actor.verification_history = []
    actor._last_verification_triggered = False
    actor._last_clip_score = actor._last_seem_score = 0.0
    actor.perception_trigger = False
    actor.perception_clip, actor.perception_seem = 0.05, 0.5
    actor.perception_calls = 0
    actor.goal_disk_data = None
    actor.found_goal = torch.zeros((1, 1), dtype=torch.bool)
    actor.goal_image = actor.goal_mask = actor.goal_image_keypoints = None
    actor.planner = Planner(planner_action, planner_error, no_path)
    actor._module = Module()
    actor.visualize = visualize
    actor.continuous_print_images = visualize
    actor.episode_final_snapshot = False
    actor.imagenav_visualizer = Visualizer()
    actor.map_inputs = [
        {
            "obstacle_map": np.zeros((40, 40), dtype=np.float32),
            "frontier_map": np.eye(40, dtype=np.float32),
            "goal_map": np.zeros((40, 40), dtype=np.float32),
            "sensor_pose": np.array([1.0, 1.0, 0.0, 0.0, 40.0, 0.0, 40.0]),
            "found_goal": False,
            "goal_pose": None,
        }
    ]
    actor.map_context = [
        {
            "explored_map": np.ones((40, 40), dtype=np.float32),
            "been_close_map": np.zeros((40, 40)),
            "frontiers": [np.array([25.0, 25.0])],
        }
    ]
    actor._preprocess_obs = MethodType(_preprocess, actor)
    actor.prepare_planner_inputs = MethodType(_map_inputs, actor)
    actor.lagmemo_score = MethodType(_perception, actor)
    return actor


def observation():
    return SimpleNamespace(
        rgb=np.zeros((16, 16, 3), dtype=np.uint8),
        depth=np.full((16, 16), 2.0, dtype=np.float32),
        semantic=np.zeros((16, 16), dtype=np.int32),
        gps=np.array([0.0, 0.0]),
        compass=np.array([0.0]),
        globalpose=np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        task_observations={
            "tasks": [
                {"type": "objectnav", "category": "bed", "image": None},
                {"type": "objectnav", "category": "chair", "image": None},
            ],
            "top_down_map": {"agent_map_coord": [20, 20]},
        },
    )


def plain(value):
    if callable(value):
        return "<callable>"
    if torch.is_tensor(value):
        return {"dtype": str(value.dtype), "values": value.detach().cpu().tolist()}
    if isinstance(value, np.ndarray):
        return {"dtype": str(value.dtype), "values": value.tolist()}
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if hasattr(value, "__dict__") and not isinstance(value, Enum):
        return plain(vars(value))
    return value


@pytest.mark.parametrize(
    "case",
    (
        "explore",
        "candidate",
        "verify_scan",
        "verified_stop",
        "unverified_stop",
        "max_steps",
        "planner_error",
        "random_exploration",
        "stuck",
        "visualization",
    ),
)
def test_action_and_persistent_state_match_frozen(case):
    options = {}
    if case == "candidate":
        options["goals"] = ([0.0, 0.0, -5.0],)
    if case == "verify_scan":
        options["goals"] = ([0.0, 0.0, -0.2],)
    if case in {"verified_stop", "unverified_stop"}:
        options["planner_action"] = DiscreteNavigationAction.STOP
    if case == "max_steps":
        options["step"] = 500
    if case == "planner_error":
        options.update(planner_error="no_local_path", no_path=True)
    if case == "random_exploration":
        options["step"] = 31
    if case == "visualization":
        options["visualize"] = True
    actors = [make_agent(cls, **options) for cls in (NativeAgent, frozen_agent())]
    for actor in actors:
        if case == "verified_stop":
            actor.stop_by_seem = True
            actor.seem_goal_pose = np.array([[0.1, 0.0]])
        if case == "random_exploration":
            actor.map_inputs[0]["frontier_map"][:] = 0
        if case == "stuck":
            actor.ctr, actor.stuck_flag = 20, 1
            actor.prev_position = np.array([0.0, 0.0])
    returned = [actor.act(observation()) for actor in actors]
    np.testing.assert_equal(plain(returned[0]), plain(returned[1]))
    np.testing.assert_equal(plain(actors[0]), plain(actors[1]))
    if case == "unverified_stop":
        assert returned[0][0] is DiscreteNavigationAction.TURN_RIGHT
        assert actors[0].last_action_metadata["planner_error"] == "unverified_planner_stop"
    if case == "verified_stop":
        assert actors[0].current_task_idx == 1
        assert actors[0]._module.resets == actors[0].planner.resets == 1
    if case == "max_steps":
        assert actors[0].last_action_metadata["stop_reason"] == "max_steps"
    if case == "random_exploration":
        assert actors[0].last_action_metadata["random_exploration_requested"]
    if case == "visualization":
        assert actors[0].imagenav_visualizer.frames[0]["goal_category"] == "bed"


@pytest.mark.parametrize("weak_scores", (False, True))
def test_temporal_confirmation_and_dual_score_veto_match_frozen(weak_scores):
    actors = [
        make_agent(cls, goals=([0.0, 0.0, -0.2],), planner_action=DiscreteNavigationAction.STOP)
        for cls in (NativeAgent, frozen_agent())
    ]
    for actor in actors:
        actor.perception_trigger = True
        actor.perception_clip = 0.05 if weak_scores else 0.15
        actor.perception_seem = 1.2 if weak_scores else 1.4
    for step in (1, 2):
        returned = []
        for actor in actors:
            actor.sub_task_timesteps[0][0] = actor.total_timesteps[0] = step
            returned.append(actor.act(observation()))
        np.testing.assert_equal(plain(returned[0]), plain(returned[1]))
        np.testing.assert_equal(plain(actors[0]), plain(actors[1]))
    if weak_scores:
        assert actors[0].current_task_idx == 0
        assert actors[0].last_action_metadata["verification_event"]["dual_score_rejection"]
    else:
        assert actors[0].current_task_idx == 1
        assert returned[0][0] is DiscreteNavigationAction.STOP


@pytest.mark.parametrize("camera_pose", (None, np.eye(4)))
def test_rgbd_semantics_and_pose_preparation_match_frozen(camera_pose):
    actors = [
        SimpleNamespace(
            device="cpu",
            current_task_idx=0,
            one_hot_encoding=torch.eye(3),
            last_poses=[np.zeros(3)],
            record_instance_ids=True,
            sub_task_timesteps=[[0]],
            force_match_against_memory=True,
        )
        for _ in range(2)
    ]
    frame = observation()
    frame.semantic[4:8, 4:8] = 2
    frame.camera_pose = camera_pose
    frame.task_observations["tasks"][0]["semantic_id"] = 1
    frame.depth[0, 0] = 0.5
    for position in ((0.2, -0.1, 0.3), (0.4, -0.2, -0.5)):
        frame.gps = np.array(position[:2])
        frame.compass = np.array([position[2]])
        outputs = [
            function(actor, frame, "imagenav")
            for function, actor in zip(
                (LocalMapState._preprocess_obs, frozen_agent()._preprocess_obs), actors
            )
        ]
        np.testing.assert_equal(plain(outputs[0]), plain(outputs[1]))
        np.testing.assert_equal(plain(actors[0]), plain(actors[1]))
    assert outputs[0][0].shape == (1, 7, 16, 16)
    assert outputs[0][0][0, 3, 0, 0] == 50.0


class MapState:
    def __init__(self):
        self.local_map = torch.zeros((1, 4, 8, 8))
        self.global_map = self.local_map.clone()
        self.local_pose = self.global_pose = self.origins = torch.zeros((1, 3))
        self.lmb = torch.tensor([[0, 8, 0, 8]])
        self.frontier = np.zeros((8, 8))
        self.goal = np.zeros((8, 8))

    def get_planner_pose_inputs(self, index):
        return np.array([0.0, 0.0, 0.0, 0.0, 8.0, 0.0, 8.0])

    def update_frontier_map(self, index, value):
        self.frontier = value

    def update_global_goal_for_env(self, index, value):
        self.goal = value

    def get_goal_map(self, index):
        return self.goal

    def get_frontier_map(self, index):
        return self.frontier

    def get_obstacle_map(self, index):
        return np.zeros((8, 8))

    def get_explored_map(self, index):
        return np.ones((8, 8))

    def get_been_close_map(self, index):
        return np.zeros((8, 8))


class MapModule:
    def __init__(self, found):
        self.found = found
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append(deepcopy((args, kwargs)))
        return (
            torch.ones((1, 1, 8, 8)),
            torch.tensor([[self.found]]),
            None,
            torch.ones((1, 1, 8, 8)),
            args[7].clone(),
            args[8].clone(),
            torch.zeros((1, 1, 3)),
            torch.zeros((1, 1, 3)),
            torch.tensor([[[0, 8, 0, 8]]]),
            torch.zeros((1, 1, 3)),
            [[np.array([2, 2])]],
        )


def map_actor(visualize, found):
    actor = SimpleNamespace(
        num_environments=1,
        timesteps_before_goal_update=[0],
        found_goal=torch.zeros((1, 1), dtype=torch.bool),
        goal_map=torch.zeros((1, 1, 8, 8)),
        semantic_map=MapState(),
        goal_update_steps=10,
        total_timesteps=[0],
        sub_task_timesteps=[[0]],
        current_task_idx=0,
        visualize=visualize,
        record_instance_ids=False,
        goal_filtering=False,
        module=MapModule(found),
    )
    actor._prep_goal_map_input = MethodType(LocalMapState._prep_goal_map_input, actor)
    return actor


@pytest.mark.parametrize("found", (False, True))
def test_local_map_updates_and_image_independence_match_frozen(found):
    actors = [map_actor(False, found), map_actor(True, found)]
    outputs = [
        function(
            actor,
            torch.zeros((1, 7, 8, 8)),
            torch.zeros((1, 3)),
            object_goal_category=torch.tensor([1]),
        )
        for function, actor in zip(
            (LocalMapState.prepare_planner_inputs, frozen_agent().prepare_planner_inputs), actors
        )
    ]
    np.testing.assert_equal(plain(outputs[0]), plain(outputs[1]))
    states = [plain(actor) for actor in actors]
    for state in states:
        state.pop("visualize")
    np.testing.assert_equal(states[0], states[1])
    assert outputs[0][1][0]["explored_map"].sum() == 64
    assert actors[0].total_timesteps == [1]
    assert actors[0].sub_task_timesteps == [[1]]


class SeemModel:
    def img2txt_seem_seg(self, image, text):
        return 1.4, np.ones((16, 16), dtype=np.uint8), None

    def img2img_seem_seg(self, image, reference, mask):
        return 1.5, mask, None


@pytest.mark.parametrize("task_type", ("objectnav", "languagenav", "imagenav"))
def test_image_language_and_object_verification_match_frozen(task_type):
    frame = observation()
    frame.task_observations["tasks"][0].update(type=task_type, image=frame.rgb, description="a bed")
    actors = [
        SimpleNamespace(
            current_task_idx=0,
            two_stage_verification=True,
            stop_by_seem=False,
            goal_vis_stage=0,
            goal_disk_data=None,
            seem_model=SeemModel(),
            mobileclip_score=lambda *args: 0.15,
            disk_preprocess=lambda image: "encoded",
            lightglue_score=lambda image: (201.0, None),
        )
        for _ in range(2)
    ]
    output = [
        function(actor, frame)
        for function, actor in zip(
            (GoalVerifier.lagmemo_score, frozen_agent().lagmemo_score), actors
        )
    ]
    np.testing.assert_equal(plain(output[0]), plain(output[1]))
    np.testing.assert_equal(plain(actors[0]), plain(actors[1]))
    assert actors[0].stop_by_seem is (task_type == "imagenav")
