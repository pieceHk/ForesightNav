"""Replay candidate, verification and recovery state against the frozen agent."""

import ast
from copy import deepcopy
from dataclasses import fields, is_dataclass
import json
from pathlib import Path
import subprocess
from typing import Mapping

import numpy as np
import pytest

from pbsp.navigation.candidates import CandidateTracking
from pbsp.navigation.evidence import CandidateEvidenceBank, select_independent_evidence
from pbsp.navigation.recovery import RecoveryState
from pbsp.navigation.verification import VerificationState, update_multiview_verification


ROOT = Path(__file__).resolve().parents[2]
SOURCE = "Navigation/runtime/lagmemo/src/lagmemo/lagmemo/agent/lagmemo_agent/glue_agent.py"


class NativeAgent(CandidateTracking, VerificationState, RecoveryState):
    pass


def _frozen():
    source = subprocess.check_output(["git", "show", "422cb8e:" + SOURCE], cwd=ROOT)
    tree = ast.parse(source)
    agent = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "GLUEAgent"
    )
    names = {
        name
        for cls in (CandidateTracking, VerificationState, RecoveryState)
        for name, value in vars(cls).items()
        if callable(value) or isinstance(value, staticmethod)
    }
    methods = [
        node for node in agent.body if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        in {
            "reconcile_spatial_candidates",
            "update_multiview_verification",
            "should_request_random_exploration",
        }
    ]
    functions.append(
        ast.ClassDef(name="FrozenAgent", bases=[], keywords=[], body=methods, decorator_list=[])
    )
    namespace = dict(
        np=np,
        deepcopy=deepcopy,
        Mapping=Mapping,
        json=json,
        select_independent_evidence=select_independent_evidence,
    )
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[])),
            "<frozen-agent-state>",
            "exec",
        ),
        namespace,
    )
    return namespace


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if is_dataclass(value):
        return {field.name: _plain(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if hasattr(value, "__dict__"):
        return _plain(vars(value))
    return value


def _candidate_agent(cls, rerank):
    agent = cls()
    agent.gs_poses = {}
    agent.frontier_flag = {}
    agent.visited_candidate_positions = {}
    agent.candidate_match_radius = 1.0
    agent.candidate_blacklist_radius = 1.5
    agent.query_action_rerank_enabled = rerank
    agent.current_task_idx = 0
    agent.candidate_evidence = CandidateEvidenceBank(enabled=True)
    agent.candidate_evidence_diagnostics = {}
    agent.stop_by_seem = True
    agent.verification_history = [{"step": 1, "pose": [0.0, 0.0]}]
    return agent


def _metadata(serial=3):
    return {
        "mode": "ordered_suffix_mvp",
        "source": "suffix",
        "task_idx": 0,
        "version_token": {
            "scene_id": "scene-a",
            "episode_id": "0",
            "map_revision": 7,
            "last_fused_keyframes": 7,
            "query_serial": serial,
        },
        "ordered_raw_ranks": [1, 0],
        "dispatch_version_gate": {
            "would_invalidate": False,
            "scene_id": "scene-a",
            "episode_id": "0",
            "batch_map_revision": 7,
            "status_last_fused_keyframes": 7,
            "query_serial": 3,
            "status_rebuilding": False,
        },
    }


@pytest.mark.parametrize("rerank", (False, True))
def test_candidate_refresh_and_transaction_replay_match_frozen(rerank):
    agents = [_candidate_agent(cls, rerank) for cls in (NativeAgent, _frozen()["FrozenAgent"])]
    first, second = [0.0, 0.0, 0.0], [3.0, 0.0, 0.0]
    for agent in agents:
        agent.refresh_gs_poses({0: [first, second], 1: [[6.0, 0.0, 0.0]]})
    np.testing.assert_equal(_plain(agents[0]), _plain(agents[1]))
    for agent in agents:
        agent.refresh_gs_poses(
            {0: [second, first], 1: [[6.1, 0.0, 0.0]]}, external_ranking_metadata=_metadata()
        )
        agent._mark_current_candidate_visited()
        agent.refresh_gs_poses({0: [first, second]}, protected_active_by_task={0: second})
    np.testing.assert_equal(_plain(agents[0]), _plain(agents[1]))
    for agent in agents:
        before = deepcopy(_plain(agent))
        with pytest.raises(ValueError, match="does not match"):
            agent.refresh_gs_poses(
                {0: [first, second]}, external_ranking_metadata=_metadata(serial=4)
            )
        np.testing.assert_equal(_plain(agent), before)


@pytest.mark.parametrize("independence", ((0.0, 0.0), (0.2, 45.0)))
def test_multiview_history_confirmation_expiry_and_reset_match_frozen(independence):
    original = _frozen()
    histories = [[], []]
    observations = (
        (0, True, [0.0, 0.0], [0.0, 0.0], 0.0, 1.2),
        (1, True, [0.1, 0.0], [0.1, 0.0], 0.0, 1.2),
        (2, True, [0.1, 0.0], [0.1, 0.0], np.pi / 3, 1.4),
        (10, False, [np.nan, 0.0], [0.0, 0.0], 0.0, 0.5),
    )
    for step, triggered, center, position, yaw, score in observations:
        results = []
        for index, function in enumerate(
            (update_multiview_verification, original["update_multiview_verification"])
        ):
            result = function(
                histories[index],
                step,
                triggered,
                center,
                0.15,
                score,
                observer_position=position,
                observer_yaw=yaw,
                min_observer_translation=independence[0],
                min_observer_yaw_deg=independence[1],
            )
            histories[index] = result[0]
            results.append(result)
        np.testing.assert_equal(results[0], results[1])
    assert histories == [[], []]
    agents = [cls() for cls in (NativeAgent, original["FrozenAgent"])]
    for agent in agents:
        agent._clear_seem_verification_state()
        agent._clear_candidate_verification_evidence()
    np.testing.assert_equal(_plain(agents[0]), _plain(agents[1]))


class Planner:
    def __init__(self):
        self.resets = 0

    def reset_subtask_state(self):
        self.resets += 1


def test_trap_release_reentry_and_override_reset_match_frozen():
    agents = [cls() for cls in (NativeAgent, _frozen()["FrozenAgent"])]
    for agent in agents:
        agent.current_task_idx = 0
        agent.sub_task_timesteps = [[0]]
        agent.trapped_window_steps = agent.trapped_warmup_steps = 3
        agent.trapped_radius = 0.05
        agent.runtime_scene_id, agent.runtime_episode_id = "scene-a", "0"
        agent.see_around = 0
        agent.trapped_events = []
        agent.planner = Planner()
        agent._reset_trapped_state()
    for step, position in enumerate(
        ([0.0, 0.0], [0.01, 0.0], [0.01, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]), 1
    ):
        events = []
        for agent in agents:
            agent.sub_task_timesteps[0][0] = step
            events.append(agent._update_trapped_diagnostic(position, "move_forward"))
        np.testing.assert_equal(events[0], events[1])
        np.testing.assert_equal(_plain(agents[0]), _plain(agents[1]))
    assert len(agents[0].trapped_events) == 2
    for agent in agents:
        agent.set_navmesh_override_goal([1.0, 0.0, 2.0], "escape")
        agent._clear_navmesh_override_goal()
    np.testing.assert_equal(_plain(agents[0]), _plain(agents[1]))
