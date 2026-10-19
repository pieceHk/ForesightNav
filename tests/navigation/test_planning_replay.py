"""Compare planning kernels with directly executed, frozen evaluator functions."""

import ast
from copy import deepcopy
from dataclasses import fields, is_dataclass, replace
from functools import lru_cache
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from adapters.geometry import project_goals_to_navigable, sample_navmesh_goal
from experiments.telemetry import build_scsp_shadow_decision, to_jsonable
from pbsp.navigation import planning
from pbsp.navigation.ranking import CandidateActionReranker
from pbsp.navigation.suffix import Idea1Candidate, SuffixPlanningResult, plan_ordered_suffix
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
def frozen_tree():
    return ast.parse(subprocess.check_output(["git", "show", "422cb8e:" + SOURCE], cwd=ROOT))


def frozen():
    tree = frozen_tree()
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
        "_scsp_shadow_fallback_event",
        "_compute_scsp_shadow_decision",
        "_prepare_scsp_refresh",
    }
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(nodes) == len(names)
    namespace = {
        "np": np,
        "replace": replace,
        "SHADOW_SCHEMA_VERSION": "scsp_shadow_decision_v1",
        "SuffixPlanningResult": SuffixPlanningResult,
        "plan_ordered_suffix": plan_ordered_suffix,
        "build_scsp_shadow_decision": build_scsp_shadow_decision,
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


def batch(serial=1, task_count=3):
    categories = ("chair", "tv", "chair", "bed")[:task_count]
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


class Clock:
    def __init__(self):
        self.counter = 0

    def perf_counter(self):
        value = self.counter * 0.001
        self.counter += 1
        return value


def current_functions(namespace, names):
    from tests.replay.planning_sources import BRIDGE_METHODS, planning_function_source

    tree = ast.parse((ROOT / SOURCE).read_text())
    nodes = []
    for name in names:
        if name in BRIDGE_METHODS:
            nodes.extend(ast.parse(planning_function_source(name)).body)
        else:
            nodes.append(next(node for node in tree.body
                              if isinstance(node, ast.FunctionDef) and node.name == name))
    assert len(nodes) == len(names)
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), "<current-planning-bridge>", "exec"),
        namespace,
    )


@pytest.mark.parametrize(
    "task_count,weight,case",
    [
        (2, 0.35, "normal"),
        (3, 0.35, "normal"),
        (4, 0.35, "normal"),
        (3, 0.0, "zero_weight"),
        (3, 0.35, "last_task"),
        (3, 0.35, "empty_future"),
        (3, 0.35, "unavailable"),
        (3, 0.35, "unreachable"),
        (3, 0.35, "stale"),
        (3, 0.35, "invalid_projection"),
    ],
)
def test_complete_planning_event_and_state_match_frozen(task_count, weight, case):
    def run(native):
        namespace = frozen()
        namespace["time"] = Clock()
        if native:
            current_functions(namespace, {"_compute_scsp_shadow_decision"})
        current = batch(task_count=task_count)
        projected = {
            task: [list(point) for point in current.positions_for_task(task)]
            for task in range(task_count)
        }
        indices = {task: [0, 1] for task in range(task_count)}
        if case == "empty_future":
            projected[1], indices[1] = [], []
        if case == "invalid_projection":
            indices[0] = []
        actor = SimpleNamespace(
            sub_task_timesteps=[[0] * task_count],
            visited_candidate_positions={},
            gs_poses=deepcopy(projected),
            frontier_flag={task: 0 for task in range(task_count)},
        )
        reranker, cache, geometry_calls = CandidateActionReranker(enabled=True), {}, []

        def geodesic(start, end):
            geometry_calls.append((tuple(start), tuple(end)))
            if case == "unavailable":
                return GeodesicDistance.unavailable()
            if case == "unreachable":
                return GeodesicDistance.unreachable()
            return GeodesicDistance.available(abs(end[0] - start[0]) + abs(end[2] - start[2]) + 1.0)

        event = namespace["_compute_scsp_shadow_decision"](
            phase="fixture",
            arm="ordered_suffix_mvp",
            provider_event={
                "version_gate": {
                    "would_invalidate": case == "stale",
                    "reason": "rebuilding" if case == "stale" else "none",
                }
            },
            candidate_batch=current,
            projected_goals=projected,
            projected_indices=indices,
            current_task=task_count - 1 if case == "last_task" else 0,
            origin=[0.0, 0.0, 0.0],
            pathfinder=geodesic,
            agent=actor,
            reranker=reranker,
            distance_cache=cache,
            candidate_cap=5,
            suffix_weight=weight,
            idea1_config=to_jsonable(reranker.config),
            provenance={
                "profile_sha256": "a" * 64,
                "dataset_manifest_sha256": "b" * 64,
                "code_sha": "c" * 40,
            },
            episode_identity={
                "scene_id": "scene-a",
                "runtime_episode_id": "episode-0",
                "source_episode_ids": tuple(f"source-{i}" for i in range(task_count)),
                "ordered_categories": current.ordered_categories,
            },
        )
        return event, vars(reranker), cache, geometry_calls, vars(actor)

    actual, expected = run(True), run(False)
    same(actual, expected)
    if case in ("stale", "invalid_projection"):
        assert actual[0]["status"] == "fallback"
    else:
        assert actual[0]["status"] == "planned"
    if case == "zero_weight":
        assert actual[2] == {}


@pytest.mark.parametrize(
    "scenario",
    [
        "fresh",
        "requery",
        "still_stale",
        "query_error",
        "retry_plan_error",
        "initial_plan_error",
        "double_projection_error",
        "retry_record_error",
        "shadow_stale",
        "retry_on_reason",
        "keep_existing",
        "new_payload_after_missing",
        "empty_retry",
        "non_retry_fallback",
    ],
)
def test_refresh_transaction_matches_frozen_returns_and_call_order(scenario):
    def run(native):
        namespace = frozen()
        namespace["time"] = Clock()
        calls = []

        class Provider:
            serial = 1

            def invalidate_cache(self):
                calls.append(("invalidate",))

            def get_all_goals(self):
                calls.append(("query",))
                if scenario in ("query_error", "keep_existing"):
                    raise RuntimeError("query failed")
                self.serial = 2
                return {} if scenario == "empty_retry" else {0: [[2.0, 0.0, 0.0]]}

            def get_candidate_batch(self):
                calls.append(("batch", self.serial))
                return batch(serial=self.serial)

        provider = Provider()

        def project(goals, _pathfinder, return_indices=False):
            calls.append(("project", deepcopy(goals), return_indices))
            if scenario == "double_projection_error":
                raise RuntimeError("projection failed")
            value = {} if goals is None else deepcopy(goals)
            return (value, {0: [0]}) if return_indices else value

        def compute(**kwargs):
            calls.append(("plan", kwargs["candidate_batch"].query_serial, kwargs["suffix_weight"]))
            if scenario == "initial_plan_error" or (
                scenario == "retry_plan_error" and provider.serial == 2
            ):
                raise RuntimeError("plan failed")
            decision = {
                "status": "planned",
                "current_task": 0,
                "candidate_batch": kwargs["candidate_batch"],
            }
            if provider.serial == 1 and scenario in ("retry_on_reason", "non_retry_fallback"):
                decision.update(
                    status="fallback",
                    fallback_reason=(
                        FallbackReason.REBUILDING
                        if scenario == "retry_on_reason"
                        else FallbackReason.FUTURE_LAYER_EMPTY
                    ),
                )
            return decision

        def gate(_provider, decision, **kwargs):
            calls.append(("gate", decision["status"], kwargs))
            stale = scenario not in (
                "fresh",
                "initial_plan_error",
                "double_projection_error",
                "retry_on_reason",
                "non_retry_fallback",
            )
            stale = stale and (provider.serial == 1 or scenario == "still_stale")
            return {
                "would_invalidate": stale,
                "reason": "stale_version_token" if stale else "none",
                "query_serial": provider.serial,
            }

        def fallback(**kwargs):
            calls.append(
                ("fallback", kwargs["reason"], kwargs["failure_stage"], str(kwargs["error"]))
            )
            return {
                "status": "fallback",
                "current_task": kwargs["current_task"],
                "fallback_reason": kwargs["reason"],
            }

        def record(_provider, **kwargs):
            calls.append(("record", kwargs))
            if scenario == "retry_record_error":
                raise RuntimeError("snapshot failed")
            return {"version_gate": {"would_invalidate": False, "reason": "none"}}

        namespace.update(
            _project_goals_to_navigable=project,
            _compute_scsp_shadow_decision=compute,
            _scsp_dispatch_gate=gate,
            _scsp_shadow_fallback_event=fallback,
            _record_scsp_provider_event=record,
        )
        kwargs = dict(
            phase="fixture",
            runtime_mode="shadow" if scenario == "shadow_stale" else "ordered_suffix_mvp",
            arm="ordered_suffix_mvp",
            provider=provider,
            provider_event={"version_gate": {"would_invalidate": False, "reason": "none"}},
            raw_goals=(
                None
                if scenario in ("keep_existing", "new_payload_after_missing")
                else {0: [[1.0, 0.0, 0.0]]}
            ),
            current_task=0,
            origin=[0.0, 0.0, 0.0],
            pathfinder=object(),
            agent=object(),
            reranker=object(),
            distance_cache={},
            candidate_cap=5,
            suffix_weight=0.35,
            idea1_config={},
            provenance={},
            episode_identity=None,
            scene_id="scene-a",
            episode_id="episode-0",
            ordered_categories=("chair",),
            provider_events_path=None,
            preserve_existing_on_missing=scenario
            in ("keep_existing", "new_payload_after_missing", "empty_retry"),
        )
        if native:
            current_functions(namespace, {"_prepare_scsp_refresh"})
        result = namespace["_prepare_scsp_refresh"](**kwargs)
        return result, calls

    actual, expected = run(True), run(False)
    same(actual, expected)
    result, calls = actual
    assert sum(call[0] == "query" for call in calls) <= 1
    assert sum(call[0] == "invalidate" for call in calls) <= 1
    if scenario == "still_stale":
        assert result[1]["status"] == "fallback"
        assert result[1]["failure_stage"] == "pre_dispatch_version_gate"
    if scenario == "shadow_stale":
        assert result[1]["synchronous_requery_count"] == 0
    if scenario in ("keep_existing", "empty_retry"):
        assert result[0] is None
    if scenario == "new_payload_after_missing":
        assert result[0] == {0: [[2.0, 0.0, 0.0]]}
