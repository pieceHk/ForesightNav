"""Replay the evaluator's frozen execution blocks through the native Agent."""

import ast
from copy import deepcopy
from dataclasses import fields, is_dataclass
from functools import lru_cache
import json
from pathlib import Path
import subprocess
import textwrap

import numpy as np
import pytest

from pbsp.agent import PBSPAgent
from pbsp.navigation.cooldown import CategorySwitchCooldown
from pbsp.types import CandidateBatch, CandidateRecord, SCSPContractError
from runtime.config import load_experiment


ROOT = Path(__file__).resolve().parents[2]
SOURCE = "Navigation/runtime/evaluators/goat/entrypoint.py"


@lru_cache()
def frozen_code(initial):
    source = subprocess.check_output(
        ["git", "show", "422cb8e:" + SOURCE], cwd=ROOT, text=True
    )
    if initial:
        start = source.index(
            "                    ordered_ranks = initial_dispatch_metadata["
        )
        end = source.index(
            "                    initial_decision = _finalize_scsp_dispatch_event(",
            start,
        )
    else:
        start = source.index("                    cooldown_event = None")
        end = source.index(
            "                    if periodic_decision is not None:", start
        )
    helper_names = {
        "_active_candidate",
        "_scsp_raw_ranks_for_positions",
        "_scsp_raw_rank_for_position",
        "_task_category",
    }
    helpers = [
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name in helper_names
    ]
    assert len(helpers) == len(helper_names)
    return (
        compile(
            ast.Module(body=helpers, type_ignores=[]),
            "<frozen-dispatch-helpers>",
            "exec",
        ),
        compile(textwrap.dedent(source[start:end]), "<frozen-dispatch-block>", "exec"),
    )


def same(left, right):
    assert type(left) is type(right)
    if isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif is_dataclass(left):
        for item in fields(left):
            same(getattr(left, item.name), getattr(right, item.name))
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            same(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for old, new in zip(left, right):
            same(old, new)
    else:
        assert left == right


def batch():
    categories = ("chair", "chair", "bed")
    records = tuple(
        (
            task,
            tuple(
                CandidateRecord(
                    task,
                    category,
                    (rank * 3.0, 0.0, task * 5.0),
                    rank * 2,
                    0.8,
                    3,
                    0.7,
                    7,
                )
                for rank in range(2)
            ),
        )
        for task, category in enumerate(categories)
    )
    return CandidateBatch(
        "scene-a",
        "episode-1",
        7,
        7,
        3,
        categories,
        tuple(
            (task, tuple(record.position for record in values))
            for task, values in records
        ),
        records,
    )


class ProbeAgent(PBSPAgent):
    def __init__(self, current_batch):
        super().__init__(
            settings=load_experiment(
                ROOT / "configs/experiments/hm3d_4mon.yaml"
            ).method.navigation,
            actions="fixed action enum",
        )
        self.current_task_idx = 1
        self.runtime_scene_id = current_batch.scene_id
        self.runtime_episode_id = current_batch.episode_id
        self.gs_poses = {
            task: [list(point) for point in current_batch.positions_for_task(task)]
            for task in range(3)
        }
        self.frontier_flag = {task: 0 for task in range(3)}
        self.calls = []
        self.fail_refresh = False

    def refresh_gs_poses(self, goals, **kwargs):
        self.calls.append(deepcopy((goals, kwargs)))
        if self.fail_refresh:
            raise RuntimeError("refresh unavailable")
        return super().refresh_gs_poses(goals, **kwargs)


def fixture(variant):
    current = batch()
    agent = ProbeAgent(current)
    goals = {
        task: [list(point) for point in reversed(current.positions_for_task(task))]
        for task in range(3)
    }
    metadata = {
        "mode": "ordered_suffix_mvp",
        "source": "suffix",
        "task_idx": 1,
        "externally_ranked": True,
        "ordered_raw_ranks": [2, 0],
        "version_token": {
            "scene_id": "scene-a",
            "episode_id": "episode-1",
            "map_revision": 7,
            "last_fused_keyframes": 7,
            "query_serial": 3,
        },
        "dispatch_version_gate": {
            "would_invalidate": False,
            "reason": "none",
            "scene_id": "scene-a",
            "episode_id": "episode-1",
            "query_serial": 3,
            "batch_map_revision": 7,
            "status_last_fused_keyframes": 7,
            "status_rebuilding": False,
        },
    }
    cooldown = CategorySwitchCooldown(
        {"chair": 10} if variant != "disabled" else {}, match_radius_m=1.0
    )
    cooldown.observe_active(1, agent.gs_poses[1][0], 0)
    step = 5
    if variant == "expired":
        step = 30
    elif variant == "first_step":
        step = 0
    elif variant == "no_goals":
        goals = None
        metadata["ordered_raw_ranks"] = []
        metadata["externally_ranked"] = False
    elif variant == "missing_task":
        goals = {0: goals[0]}
        metadata["ordered_raw_ranks"] = []
        metadata["externally_ranked"] = False
    elif variant == "empty_candidates":
        goals[1] = []
        metadata["ordered_raw_ranks"] = []
    elif variant == "rank_mismatch":
        goals[1][0] = [20.0, 0.0, 5.0]
    elif variant == "stale_gate":
        metadata["dispatch_version_gate"]["would_invalidate"] = True
    elif variant == "refresh_error":
        agent.fail_refresh = True
    elif variant == "visited_active":
        agent.visited_candidate_positions[1] = [goals[1][1]]
    elif variant == "internal_order":
        metadata["ordered_raw_ranks"] = []
        metadata["externally_ranked"] = False
    elif variant == "no_metadata":
        metadata = None
    return agent, goals, metadata, cooldown, current, step


@pytest.mark.parametrize(
    "initial,variant",
    [
        *(
            (True, variant)
            for variant in (
                "normal",
                "internal_order",
                "no_goals",
                "stale_gate",
                "refresh_error",
                "visited_active",
            )
        ),
        *(
            (False, variant)
            for variant in (
                "normal",
                "expired",
                "first_step",
                "disabled",
                "no_goals",
                "missing_task",
                "empty_candidates",
                "rank_mismatch",
                "stale_gate",
                "refresh_error",
                "visited_active",
                "internal_order",
                "no_metadata",
            )
        ),
    ],
)
def test_candidate_execution_matches_original_block(initial, variant):
    old, old_goals, old_metadata, old_cooldown, current_batch, step = fixture(variant)
    new, new_goals, new_metadata, new_cooldown, _, _ = fixture(variant)
    namespace = dict(
        np=np,
        json=json,
        SCSPContractError=SCSPContractError,
        agent=old,
        current_task_idx=1,
        local_step=step,
        category_cooldown=old_cooldown,
        obs_tasks=[
            {"object_category": value, "type": "objectnav"}
            for value in current_batch.ordered_categories
        ],
        category_cooldown_events=[],
        projected_goals=old_goals,
        dispatch_metadata=old_metadata,
        periodic_decision={"candidate_batch": current_batch},
        initial_goals=old_goals,
        initial_dispatch_metadata=old_metadata,
        initial_decision={"candidate_batch": current_batch},
    )
    helpers, code = frozen_code(initial)
    exec(helpers, namespace)
    exec(code, namespace)
    if initial:
        result = new.dispatch_initial_candidates(
            new_goals, new_metadata, task_idx=1, candidate_batch=current_batch
        )
        same(result.projected_goals, namespace["initial_goals"])
        same(result.refresh_applied, namespace["initial_refresh_applied"])
        same(result.refresh_error, namespace["initial_refresh_error"])
        same(result.realized_active_position, namespace["initial_realized"])
        same(result.realized_raw_rank, namespace["initial_realized_rank"])
        same(result.metadata, namespace["initial_dispatch_metadata"])
        assert result.cooldown_event is None
    else:
        result = new.dispatch_candidates(
            new_goals,
            new_metadata,
            task_idx=1,
            category="chair",
            local_step=step,
            category_cooldown=new_cooldown,
            candidate_batch=current_batch,
        )
        same(result.projected_goals, namespace["projected_goals"])
        same(result.metadata, namespace["dispatch_metadata"])
        same(result.cooldown_event, namespace["cooldown_event"])
        same(result.refresh_applied, namespace["refresh_applied"])
        same(result.refresh_error, namespace["refresh_error"])
        old_active = namespace["_active_candidate"](old, 1)
        same(result.realized_active_position, old_active)
        same(
            result.realized_raw_rank,
            namespace["_scsp_raw_rank_for_position"](current_batch, 1, old_active),
        )
    same(old.__dict__, new.__dict__)
    same(old_cooldown, new_cooldown)
    assert len(old.calls) == len(new.calls) <= 1
