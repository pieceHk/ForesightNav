"""Compare planning kernels with directly executed, frozen evaluator functions."""

import ast
from copy import deepcopy
from dataclasses import fields, is_dataclass
from functools import lru_cache
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from adapters.geometry import project_goals_to_navigable, sample_navmesh_goal
from experiments.telemetry import to_jsonable
from pbsp.navigation import planning
from pbsp.navigation.ranking import CandidateActionReranker
from pbsp.navigation.suffix import Idea1Candidate
from pbsp.types import (
    CandidateBatch,
    CandidateRecord,
    FallbackReason,
    GeodesicDistance,
    GeodesicMatrix,
    GeodesicStatus,
    SCSPContractError,
    validate_candidate_batch,
)

ROOT = Path(__file__).resolve().parents[2]
SOURCE = "Navigation/runtime/evaluators/goat/entrypoint.py"


@lru_cache()
def frozen():
    tree = ast.parse(subprocess.check_output(["git", "show", "422cb8e:" + SOURCE], cwd=ROOT))
    names = {
        "_projected_candidate_batch",
        "_subtask_has_acted",
        "_active_candidate",
        "_rank_scsp_idea1_layers",
        "_build_scsp_geodesic_matrices",
        "_scsp_raw_ranks_for_positions",
        "_scsp_raw_rank_for_position",
        "_apply_scsp_runtime_order",
        "_fallback_value",
        "_reset_scsp_task_state",
        "_project_goals_to_navigable",
        "_sample_navmesh_goal",
    }
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(nodes) == len(names)
    namespace = {
        "np": np,
        "os": os,
        "CandidateBatch": CandidateBatch,
        "CandidateRecord": CandidateRecord,
        "FallbackReason": FallbackReason,
        "GeodesicDistance": GeodesicDistance,
        "GeodesicMatrix": GeodesicMatrix,
        "GeodesicStatus": GeodesicStatus,
        "SCSPContractError": SCSPContractError,
        "validate_candidate_batch": validate_candidate_batch,
        "Idea1Candidate": Idea1Candidate,
        "to_jsonable": to_jsonable,
        "_strict_geodesic_distance": lambda query, start, end: query(start, end),
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<frozen-planning>", "exec"), namespace)
    return namespace


def same(left, right):
    assert type(left) is type(right)
    if isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif is_dataclass(left):
        for field in fields(left):
            same(getattr(left, field.name), getattr(right, field.name))
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            same(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            same(a, b)
    else:
        assert left == right


def batch(serial=1):
    categories = ("chair", "tv", "chair")
    records = tuple(
        (
            task,
            tuple(
                CandidateRecord(
                    task,
                    category,
                    (task * 5 + rank + 1.0, 0.0, rank + 1.0),
                    rank * 2,
                    0.6 + rank * 0.2,
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
        scene_id="scene-a",
        episode_id="episode-0",
        map_revision=7,
        last_fused_keyframes=7,
        query_serial=serial,
        ordered_categories=categories,
        positions_by_task=tuple(
            (task, tuple(record.position for record in values)) for task, values in records
        ),
        records_by_task=records,
    )


@pytest.mark.parametrize("acted", [False, True])
@pytest.mark.parametrize("status", list(GeodesicStatus))
def test_ranked_layers_match_frozen_outputs_and_reranker_state(acted, status):
    current = batch()
    actor = SimpleNamespace(
        gs_poses={task: list(current.positions_for_task(task)) for task in range(3)},
        frontier_flag={0: 1, 1: 0, 2: 0},
        sub_task_timesteps=[[int(acted), 0, 0]],
        visited_candidate_positions={0: [current.positions_for_task(0)[0]]},
    )

    def geodesic(start, end):
        if status is GeodesicStatus.UNREACHABLE:
            return GeodesicDistance.unreachable()
        if status is GeodesicStatus.UNAVAILABLE:
            return GeodesicDistance.unavailable()
        return GeodesicDistance.available(abs(end[0] - start[0]) + 2.0)

    original_ranker = CandidateActionReranker(enabled=True)
    native_ranker = CandidateActionReranker(enabled=True)
    original_actor = deepcopy(actor)
    native_actor = deepcopy(actor)
    expected = frozen()["_rank_scsp_idea1_layers"](
        original_ranker, current, (0.0, 0.0, 0.0), geodesic, original_actor
    )
    actual = planning.rank_candidate_layers(
        native_ranker, current, (0.0, 0.0, 0.0), geodesic, native_actor
    )
    same(actual, expected)
    same(vars(native_ranker), vars(original_ranker))
    same(vars(native_actor), vars(original_actor))


@pytest.mark.parametrize("cap", [1, 2])
def test_distance_cache_reuse_and_version_eviction_match_frozen(cap):
    caches = ({}, {})
    calls = ([], [])
    for serial in (1, 1, 2):
        current = batch(serial)
        candidates = {
            task: tuple(Idea1Candidate(record, 1.0) for record in current.records_for_task(task))
            for task in range(3)
        }

        def query(index):
            def distance(start, end):
                calls[index].append((tuple(start), tuple(end)))
                if end[2] == 2.0:
                    return GeodesicDistance.unreachable()
                return GeodesicDistance.available(5.0)

            return distance

        expected = frozen()["_build_scsp_geodesic_matrices"](
            current, 0, candidates, query(0), caches[0], cap
        )
        actual = planning.build_geodesic_matrices(current, 0, candidates, query(1), caches[1], cap)
        same(actual, expected)
        same(caches[1], caches[0])
        same(calls[1], calls[0])
    assert actual[1]["evictions"] > 0


def test_projection_binding_preserves_duplicate_task_identity_and_raw_rank():
    current = batch()
    projected = {task: [list(current.positions_for_task(task)[1])] for task in range(3)}
    indices = {task: [1] for task in range(3)}
    expected = frozen()["_projected_candidate_batch"](current, projected, indices)
    actual = planning.bind_projected_candidates(current, projected, indices)
    same(actual, expected)
    assert actual.version_token == current.version_token
    assert [actual.records_for_task(task)[0].raw_rank for task in range(3)] == [2, 2, 2]
    assert actual.ordered_categories == ("chair", "tv", "chair")


@pytest.mark.parametrize(
    "mode,stale,fallback",
    [
        ("ordered_suffix_mvp", False, FallbackReason.NONE),
        ("ordered_suffix_mvp", True, FallbackReason.NONE),
        ("ordered_suffix_mvp", False, FallbackReason.GEODESIC_UNAVAILABLE),
        ("idea1_only", False, FallbackReason.NONE),
        ("shadow", False, FallbackReason.NONE),
    ],
)
def test_dispatch_permutation_and_metadata_match_frozen(mode, stale, fallback):
    current = batch()
    goals = {
        task: [list(position) for position in current.positions_for_task(task)] for task in range(3)
    }
    snapshot = deepcopy(goals)
    decision = {
        "status": "planned",
        "current_task": 0,
        "candidate_batch": current,
        "dispatch_version_gate": {"would_invalidate": stale, "reason": "none"},
        "idea1_order": ((0, 2), (0, 2), (0, 2)),
        "suffix_order": ((2, 0), (0, 2), (0, 2)),
        "suffix_plan": SimpleNamespace(fallback_reason=fallback, intervention=True),
    }
    same(
        planning.apply_runtime_order(goals, decision, mode),
        frozen()["_apply_scsp_runtime_order"](goals, decision, mode),
    )
    assert goals == snapshot


class Navmesh:
    def __init__(self):
        self.calls = []
        self.index = 0

    def snap_point(self, point):
        point = np.array(point, dtype=np.float32)
        if point[0] < 0:
            raise RuntimeError("no snap")
        point[1] = 0.0
        if point[0] > 80:
            point[0] += 3.0
        return point

    def get_island(self, point):
        return int(point[0] > 80)

    def _sample(self):
        rows = (
            (2.0, 0.0, 0.0),
            (4.0, 1.0, 0.0),
            (99.0, 0.0, 0.0),
            (0.5, 0.0, 0.0),
            (float("nan"), 0.0, 0.0),
            (4.0, 0.0, 0.0),
        )
        point = rows[self.index % len(rows)]
        self.index += 1
        return point

    def get_random_navigable_point_near(self, *args):
        self.calls.append(("near", args))
        return self._sample()

    def get_random_navigable_point(self, *args):
        self.calls.append(("global", args))
        return self._sample()


def test_projection_rejections_and_source_indices_match_frozen(monkeypatch):
    monkeypatch.setenv("ES_NAV_PROJECTION_MAX_DISTANCE", "1.5")
    goals = {0: [[1.0, 0.2, 0.0], [1.1, 0.2, 0.0], [-1.0, 0.0, 0.0], [99.0, 0.0, 0.0]], 1: []}
    same(
        project_goals_to_navigable(goals, Navmesh(), return_indices=True),
        frozen()["_project_goals_to_navigable"](goals, Navmesh(), return_indices=True),
    )


@pytest.mark.parametrize("mode", ["escape", "random"])
@pytest.mark.parametrize("custom", [False, True])
def test_recovery_rejects_other_floors_and_islands_like_frozen(mode, custom, monkeypatch):
    params = {
        "max_vertical_delta": 0.35,
        "sample_count": 32,
        "escape_radius_m": 3.0,
        "escape_min_distance_m": 1.25,
        "random_min_distance_m": 2.0,
    }
    names = (
        "ES_NAVMESH_MAX_VERTICAL_DELTA",
        "ES_NAVMESH_SAMPLE_COUNT",
        "ES_NAVMESH_ESCAPE_RADIUS",
        "ES_NAVMESH_ESCAPE_MIN_DISTANCE",
        "ES_NAVMESH_RANDOM_MIN_DISTANCE",
    )
    if custom:
        params.update(max_vertical_delta=0.25, sample_count=12, escape_radius_m=4.0)
    for name, value in zip(names, params.values()):
        monkeypatch.setenv(name, str(value))
    original, native = Navmesh(), Navmesh()
    expected = frozen()["_sample_navmesh_goal"](original, [0.0, 0.0, 0.0], mode)
    actual = sample_navmesh_goal(native, [0.0, 0.0, 0.0], mode, **params)
    same(actual, expected)
    same(native.calls, original.calls)
    assert actual[0].tolist() == [4.0, 0.0, 0.0]
    assert actual[1]["vertical_rejections"] > 0
    assert actual[1]["island_rejections"] > 0
