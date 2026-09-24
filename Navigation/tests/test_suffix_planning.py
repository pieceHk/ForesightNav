import ast
import importlib
import math
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from tdonav_navigation.scsp_contracts import (
    BatchValidationStatus,
    CandidateBatch,
    CandidateRecord,
    FallbackReason,
    GeodesicDistance,
    GeodesicMatrix,
    GeodesicStatus,
    InterventionReason,
    VersionToken,
)
from tdonav_navigation.suffix_planning import (
    DEFAULT_CANDIDATE_CAP,
    DEFAULT_SUFFIX_WEIGHT,
    Idea1Candidate,
    SuffixPlanningError,
    plan_ordered_suffix,
)


TOKEN = VersionToken(
    scene_id="scene-a",
    episode_id="episode-1",
    map_revision=7,
    last_fused_keyframes=7,
    query_serial=3,
)


def _candidate(
    task_idx,
    category,
    raw_rank,
    utility,
    *,
    position=None,
    revision=7,
):
    if position is None:
        position = (float(task_idx * 10 + raw_rank), 0.0, float(raw_rank + 1))
    return Idea1Candidate(
        record=CandidateRecord(
            task_idx=task_idx,
            category=category,
            position=position,
            raw_rank=raw_rank,
            semantic_score=0.5,
            support_count=1,
            support_score=0.25,
            map_revision=revision,
        ),
        utility=utility,
    )


def _distance(value):
    if value is None:
        return GeodesicDistance.unavailable()
    if value == math.inf:
        return GeodesicDistance.unreachable()
    return GeodesicDistance.available(value)


def _matrix(source_task, sources, targets, values):
    return GeodesicMatrix(
        source_task_idx=source_task,
        target_task_idx=source_task + 1,
        source_candidate_identities=tuple(item.record.identity() for item in sources),
        target_candidate_identities=tuple(item.record.identity() for item in targets),
        distances=tuple(
            tuple(_distance(value) for value in row) for row in values
        ),
    )


def _batch(
    categories,
    by_task,
    *,
    status=BatchValidationStatus.VALID,
    reason=None,
):
    return CandidateBatch(
        scene_id=TOKEN.scene_id,
        episode_id=TOKEN.episode_id,
        map_revision=TOKEN.map_revision,
        last_fused_keyframes=TOKEN.last_fused_keyframes,
        query_serial=TOKEN.query_serial,
        ordered_categories=categories,
        positions_by_task=tuple(
            (
                task_idx,
                tuple(candidate.record.position for candidate in by_task[task_idx]),
            )
            for task_idx in range(len(categories))
        ),
        records_by_task=tuple(
            (
                task_idx,
                tuple(candidate.record for candidate in by_task[task_idx]),
            )
            for task_idx in range(len(categories))
        ),
        validation_status=status,
        validation_reason=reason,
    )


def _costs(result):
    return {
        item.raw_rank: item.cost.distance_m for item in result.plan.suffix_costs
    }


def _statuses(result):
    return {
        item.raw_rank: item.cost.status for item in result.plan.suffix_costs
    }


def test_three_task_hand_calculation_and_order_composition_are_deterministic():
    categories = ("chair", "table", "plant")
    task0 = (
        _candidate(0, "chair", 0, 1.0),
        _candidate(0, "chair", 1, 0.9),
    )
    task1 = (
        _candidate(1, "table", 4, 0.8),
        _candidate(1, "table", 2, 0.7),
    )
    task2 = (
        _candidate(2, "plant", 9, 0.6),
        _candidate(2, "plant", 3, 0.5),
    )
    first = _matrix(0, task0, task1, ((4.0, 1.0), (1.0, 5.0)))
    second = _matrix(1, task1, task2, ((5.0, 1.0), (2.0, 4.0)))
    by_task = {2: task2, 0: task0, 1: task1}
    snapshot = tuple((key, by_task[key]) for key in by_task)

    kwargs = {
        "candidate_batch": _batch(categories, by_task),
        "current_task": 0,
        "idea1_candidates_by_task": by_task,
        "strict_geodesic_matrices": (second, first),
    }
    result = plan_ordered_suffix(**kwargs)
    repeated = plan_ordered_suffix(**kwargs)
    reordered_mapping = plan_ordered_suffix(
        **dict(kwargs, idea1_candidates_by_task={0: task0, 1: task1, 2: task2})
    )

    assert result == repeated == reordered_mapping
    assert tuple((key, by_task[key]) for key in by_task) == snapshot
    assert _costs(result) == {0: 3.0, 1: 2.0}
    assert result.ordered_raw_ranks == (1, 0)
    assert result.plan.intervention
    assert result.plan.fallback_reason is FallbackReason.NONE
    assert result.plan.intervention_reason is InterventionReason.LOWER_SUFFIX_COST
    assert tuple(
        (matrix.source_task_idx, matrix.target_task_idx)
        for matrix in result.plan.strict_geodesic_matrices
    ) == ((0, 1), (1, 2))
    assert sum(
        len(row)
        for matrix in result.plan.strict_geodesic_matrices
        for row in matrix.distances
    ) == 8


def test_two_task_partial_unreachable_never_uses_infeasible_idea1_head():
    task0 = (
        _candidate(0, "chair", 8, 100.0),
        _candidate(0, "chair", 5, -100.0),
    )
    task1 = (_candidate(1, "table", 0, 0.0),)
    matrix = _matrix(0, task0, task1, ((math.inf,), (12.0,)))
    by_task = {0: task0, 1: task1}

    result = plan_ordered_suffix(
        candidate_batch=_batch(("chair", "table"), by_task),
        current_task=0,
        idea1_candidates_by_task=by_task,
        strict_geodesic_matrices=(matrix,),
    )

    assert result.ordered_raw_ranks == (5, 8)
    assert _statuses(result) == {
        8: GeodesicStatus.UNREACHABLE,
        5: GeodesicStatus.AVAILABLE,
    }
    assert result.plan.intervention_reason is InterventionReason.IDEA1_CANDIDATE_UNREACHABLE


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (((math.inf,), (math.inf,)), FallbackReason.ALL_EDGES_UNREACHABLE),
        (((None,), (math.inf,)), FallbackReason.GEODESIC_UNAVAILABLE),
    ],
)
def test_no_finite_suffix_is_a_reasoned_noop(values, expected):
    task0 = (
        _candidate(0, "chair", 4, 0.0),
        _candidate(0, "chair", 1, 100.0),
    )
    task1 = (_candidate(1, "table", 0, 0.0),)
    by_task = {0: task0, 1: task1}

    result = plan_ordered_suffix(
        candidate_batch=_batch(("chair", "table"), by_task),
        current_task=0,
        idea1_candidates_by_task=by_task,
        strict_geodesic_matrices=(_matrix(0, task0, task1, values),),
    )

    assert result.ordered_raw_ranks == (4, 1)
    assert not result.plan.intervention
    assert result.plan.fallback_reason is expected


def test_populated_layers_without_an_end_to_end_route_are_a_noop():
    task0 = (_candidate(0, "chair", 0, 0.0),)
    task1 = (
        _candidate(1, "table", 0, 0.0),
        _candidate(1, "table", 1, 0.0),
    )
    task2 = (_candidate(2, "plant", 0, 0.0),)
    first = _matrix(0, task0, task1, ((1.0, math.inf),))
    second = _matrix(1, task1, task2, ((math.inf,), (1.0,)))
    by_task = {0: task0, 1: task1, 2: task2}

    result = plan_ordered_suffix(
        candidate_batch=_batch(("chair", "table", "plant"), by_task),
        current_task=0,
        idea1_candidates_by_task=by_task,
        strict_geodesic_matrices=(first, second),
    )

    assert result.ordered_raw_ranks == (0,)
    assert result.plan.fallback_reason is FallbackReason.NO_FEASIBLE_SUFFIX
    assert result.plan.suffix_costs[0].cost.status is GeodesicStatus.UNREACHABLE


def test_equal_finite_suffix_costs_preserve_idea1_order():
    task0 = (
        _candidate(0, "chair", 7, -10.0),
        _candidate(0, "chair", 3, 10.0),
    )
    task1 = (_candidate(1, "table", 0, 0.0),)
    by_task = {0: task0, 1: task1}

    result = plan_ordered_suffix(
        candidate_batch=_batch(("chair", "table"), by_task),
        current_task=0,
        idea1_candidates_by_task=by_task,
        strict_geodesic_matrices=(_matrix(0, task0, task1, ((2.0,), (2.0,))),),
    )

    assert result.ordered_raw_ranks == (7, 3)
    assert result.plan.fallback_reason is FallbackReason.EQUAL_SUFFIX_COSTS


def test_equal_combined_utility_uses_idea1_order_as_the_first_tie_break():
    task0 = (
        _candidate(0, "chair", 9, DEFAULT_SUFFIX_WEIGHT),
        _candidate(0, "chair", 1, 0.0),
    )
    task1 = (_candidate(1, "table", 0, 0.0),)
    by_task = {0: task0, 1: task1}

    result = plan_ordered_suffix(
        candidate_batch=_batch(("chair", "table"), by_task),
        current_task=0,
        idea1_candidates_by_task=by_task,
        strict_geodesic_matrices=(_matrix(0, task0, task1, ((2.0,), (1.0,))),),
    )

    assert result.ordered_raw_ranks == (9, 1)
    assert result.plan.fallback_reason is FallbackReason.NONE


def test_finite_only_normalisation_keeps_nonfinite_candidates_last():
    task0 = (
        _candidate(0, "chair", 0, 0.6),
        _candidate(0, "chair", 1, 0.5),
        _candidate(0, "chair", 2, 1000.0),
    )
    task1 = (_candidate(1, "table", 0, 0.0),)
    by_task = {0: task0, 1: task1}

    result = plan_ordered_suffix(
        candidate_batch=_batch(("chair", "table"), by_task),
        current_task=0,
        idea1_candidates_by_task=by_task,
        strict_geodesic_matrices=(
            _matrix(0, task0, task1, ((4.0,), (2.0,), (math.inf,))),
        ),
    )

    assert result.ordered_raw_ranks == (1, 0, 2)
    assert _statuses(result)[2] is GeodesicStatus.UNREACHABLE


@pytest.mark.parametrize(
    ("categories", "current_task"),
    [(('chair',), 0), (("chair", "table", "plant"), 2)],
)
def test_last_task_has_zero_suffix_and_never_intervenes(categories, current_task):
    by_task = {
        task_idx: (
            _candidate(task_idx, category, 2, 0.0),
            _candidate(task_idx, category, 1, 100.0),
        )
        for task_idx, category in enumerate(categories)
    }

    result = plan_ordered_suffix(
        candidate_batch=_batch(categories, by_task),
        current_task=current_task,
        idea1_candidates_by_task=by_task,
        strict_geodesic_matrices=(),
    )

    assert result.ordered_raw_ranks == (2, 1)
    assert result.plan.fallback_reason is FallbackReason.LAST_TASK
    assert all(cost.cost.distance_m == 0.0 for cost in result.plan.suffix_costs)


def test_repeated_categories_and_empty_future_layer_are_task_indexed():
    task0 = (_candidate(0, "chair", 2, 0.0),)
    task2 = (_candidate(2, "chair", 1, 0.0),)
    by_task = {2: task2, 1: (), 0: task0}

    result = plan_ordered_suffix(
        candidate_batch=_batch(("chair", "chair", "chair"), by_task),
        current_task=0,
        idea1_candidates_by_task=by_task,
        strict_geodesic_matrices=(),
    )

    assert result.ordered_raw_ranks == (2,)
    assert result.plan.fallback_reason is FallbackReason.FUTURE_LAYER_EMPTY


def test_repeated_categories_support_a_complete_suffix_dp():
    task0 = (
        _candidate(0, "chair", 0, 0.5),
        _candidate(0, "chair", 1, 0.4),
    )
    task1 = (_candidate(1, "chair", 0, 0.0),)
    task2 = (_candidate(2, "chair", 0, 0.0),)
    by_task = {0: task0, 1: task1, 2: task2}

    result = plan_ordered_suffix(
        candidate_batch=_batch(("chair", "chair", "chair"), by_task),
        current_task=0,
        idea1_candidates_by_task=by_task,
        strict_geodesic_matrices=(
            _matrix(1, task1, task2, ((3.0,),)),
            _matrix(0, task0, task1, ((2.0,), (1.0,))),
        ),
    )

    assert _costs(result) == {0: 5.0, 1: 4.0}
    assert result.ordered_raw_ranks == (1, 0)


def test_empty_current_layer_has_no_selection():
    task1 = (_candidate(1, "table", 0, 0.0),)
    by_task = {0: (), 1: task1}
    result = plan_ordered_suffix(
        candidate_batch=_batch(("chair", "table"), by_task),
        current_task=0,
        idea1_candidates_by_task=by_task,
        strict_geodesic_matrices=(),
    )

    assert result.ordered_raw_ranks == ()
    assert result.plan.selected_raw_rank is None
    assert result.plan.fallback_reason is FallbackReason.NO_CANDIDATES


def test_default_k_cap_limits_edges_and_never_promotes_the_tail():
    assert DEFAULT_CANDIDATE_CAP == 5
    task0 = tuple(
        _candidate(0, "chair", rank, 1000.0 if rank == 5 else 0.0)
        for rank in range(6)
    )
    task1 = tuple(_candidate(1, "table", rank, 0.0) for rank in range(6))
    capped0 = task0[:DEFAULT_CANDIDATE_CAP]
    capped1 = task1[:DEFAULT_CANDIDATE_CAP]
    values = tuple(
        tuple(float(5 - source_idx) for _ in capped1)
        for source_idx, _ in enumerate(capped0)
    )
    by_task = {0: task0, 1: task1}

    result = plan_ordered_suffix(
        candidate_batch=_batch(("chair", "table"), by_task),
        current_task=0,
        idea1_candidates_by_task=by_task,
        strict_geodesic_matrices=(_matrix(0, capped0, capped1, values),),
    )

    assert result.ordered_raw_ranks[0] == 4
    assert result.ordered_raw_ranks[-1] == 5
    assert len(result.plan.suffix_costs) == DEFAULT_CANDIDATE_CAP
    matrix = result.plan.strict_geodesic_matrices[0]
    assert (len(matrix.distances), len(matrix.distances[0])) == (5, 5)


def test_candidate_cap_cannot_exceed_the_frozen_maximum():
    candidate = _candidate(0, "chair", 0, 0.0)
    by_task = {0: (candidate,)}
    with pytest.raises(SuffixPlanningError, match="candidate_cap must be <= 5"):
        plan_ordered_suffix(
            candidate_batch=_batch(("chair",), by_task),
            current_task=0,
            idea1_candidates_by_task=by_task,
            strict_geodesic_matrices=(),
            candidate_cap=6,
        )


def test_zero_weight_preserves_every_idea1_item_without_geodesic_work():
    raw_order = (5, 1, 9, 0, 2, 3, 4)
    task0 = tuple(
        _candidate(0, "chair", rank, float(index * 100))
        for index, rank in enumerate(raw_order)
    )
    task1 = (_candidate(1, "table", 0, 0.0),)
    by_task = {0: task0, 1: task1}

    result = plan_ordered_suffix(
        candidate_batch=_batch(("chair", "table"), by_task),
        current_task=0,
        idea1_candidates_by_task=by_task,
        strict_geodesic_matrices=(),
        suffix_weight=0.0,
    )

    assert result.ordered_raw_ranks == raw_order
    assert result.plan.fallback_reason is FallbackReason.WEIGHT_ZERO
    assert result.plan.strict_geodesic_matrices == ()


def test_mixed_candidate_revisions_fail_closed_without_matrix_access():
    batch_task0 = (
        _candidate(0, "chair", 4, 0.0),
        _candidate(0, "chair", 1, 100.0),
    )
    stale_task0 = (
        batch_task0[0],
        _candidate(0, "chair", 1, 100.0, revision=8),
    )
    task1 = (_candidate(1, "table", 0, 0.0),)
    batch_records = {0: batch_task0, 1: task1}
    stale_records = {1: task1, 0: stale_task0}

    result = plan_ordered_suffix(
        candidate_batch=_batch(("chair", "table"), batch_records),
        current_task=0,
        idea1_candidates_by_task=stale_records,
        strict_geodesic_matrices=(),
    )

    assert result.ordered_raw_ranks == (4, 1)
    assert result.plan.fallback_reason is FallbackReason.STALE_VERSION_TOKEN


def test_incomplete_idea1_permutation_abstains_instead_of_dispatching_partial_order():
    task0 = (
        _candidate(0, "chair", 4, 0.0),
        _candidate(0, "chair", 1, 100.0),
    )
    task1 = (_candidate(1, "table", 0, 0.0),)
    batch_records = {0: task0, 1: task1}
    incomplete = {0: (task0[1],), 1: task1}

    result = plan_ordered_suffix(
        candidate_batch=_batch(("chair", "table"), batch_records),
        current_task=0,
        idea1_candidates_by_task=incomplete,
        strict_geodesic_matrices=(),
    )

    assert result.ordered_raw_ranks == ()
    assert result.plan.selected_raw_rank is None
    assert result.plan.fallback_reason is FallbackReason.STALE_VERSION_TOKEN


def test_complete_idea1_permutation_may_differ_from_candidate_batch_raw_order():
    raw_task0 = (
        _candidate(0, "chair", 0, 0.0),
        _candidate(0, "chair", 1, 0.0),
    )
    idea1_task0 = tuple(reversed(raw_task0))
    task1 = (_candidate(1, "table", 0, 0.0),)
    batch_records = {0: raw_task0, 1: task1}
    idea1_records = {0: idea1_task0, 1: task1}

    result = plan_ordered_suffix(
        candidate_batch=_batch(("chair", "table"), batch_records),
        current_task=0,
        idea1_candidates_by_task=idea1_records,
        strict_geodesic_matrices=(
            _matrix(0, idea1_task0, task1, ((2.0,), (2.0,))),
        ),
    )

    assert result.ordered_raw_ranks == (1, 0)
    assert result.plan.fallback_reason is FallbackReason.EQUAL_SUFFIX_COSTS


def test_invalid_candidate_batch_fails_closed_before_matrix_access():
    task0 = (_candidate(0, "chair", 4, 0.0),)
    task1 = (_candidate(1, "table", 0, 0.0),)
    by_task = {0: task0, 1: task1}

    result = plan_ordered_suffix(
        candidate_batch=_batch(
            ("chair", "table"),
            by_task,
            status=BatchValidationStatus.INVALID,
            reason="provider alignment failed",
        ),
        current_task=0,
        idea1_candidates_by_task=by_task,
        strict_geodesic_matrices=(),
    )

    assert result.ordered_raw_ranks == ()
    assert result.plan.selected_raw_rank is None
    assert result.plan.fallback_reason is FallbackReason.INVALID_CANDIDATE_BATCH


def test_unavailable_edge_cannot_override_unreachable_downstream_suffix():
    task0 = (_candidate(0, "chair", 0, 0.0),)
    task1 = (_candidate(1, "table", 0, 0.0),)
    task2 = (_candidate(2, "plant", 0, 0.0),)
    by_task = {0: task0, 1: task1, 2: task2}

    result = plan_ordered_suffix(
        candidate_batch=_batch(("chair", "table", "plant"), by_task),
        current_task=0,
        idea1_candidates_by_task=by_task,
        strict_geodesic_matrices=(
            _matrix(0, task0, task1, ((None,),)),
            _matrix(1, task1, task2, ((math.inf,),)),
        ),
    )

    assert result.plan.suffix_costs[0].cost.status is GeodesicStatus.UNREACHABLE
    assert result.plan.fallback_reason is FallbackReason.NO_FEASIBLE_SUFFIX


def test_finite_path_sum_overflow_fails_closed_with_full_order_parity():
    task0 = (
        _candidate(0, "chair", 4, 0.0),
        _candidate(0, "chair", 1, 100.0),
    )
    task1 = (_candidate(1, "table", 0, 0.0),)
    task2 = (_candidate(2, "plant", 0, 0.0),)
    first = _matrix(0, task0, task1, ((1e308,), (1e308,)))
    second = _matrix(1, task1, task2, ((1e308,),))
    by_task = {0: task0, 1: task1, 2: task2}

    result = plan_ordered_suffix(
        candidate_batch=_batch(("chair", "table", "plant"), by_task),
        current_task=0,
        idea1_candidates_by_task=by_task,
        strict_geodesic_matrices=(first, second),
    )

    assert result.ordered_raw_ranks == (4, 1)
    assert not result.plan.intervention
    assert result.plan.fallback_reason is FallbackReason.PLANNER_ERROR


def test_combined_utility_overflow_fails_closed_with_full_order_parity():
    task0 = (
        _candidate(0, "chair", 4, -1e308),
        _candidate(0, "chair", 1, -1e308),
    )
    task1 = (_candidate(1, "table", 0, 0.0),)
    by_task = {0: task0, 1: task1}

    result = plan_ordered_suffix(
        candidate_batch=_batch(("chair", "table"), by_task),
        current_task=0,
        idea1_candidates_by_task=by_task,
        strict_geodesic_matrices=(
            _matrix(0, task0, task1, ((2.0,), (1.0,))),
        ),
        suffix_weight=1e308,
    )

    assert result.ordered_raw_ranks == (4, 1)
    assert result.plan.fallback_reason is FallbackReason.PLANNER_ERROR


def test_strict_matrix_validation_fails_closed_for_missing_or_misaligned_layers():
    task0 = (
        _candidate(0, "chair", 0, 0.0),
        _candidate(0, "chair", 1, 0.0),
    )
    task1 = (_candidate(1, "table", 0, 0.0),)
    task2 = (_candidate(2, "plant", 0, 0.0),)
    second = _matrix(1, task1, task2, ((1.0,),))
    by_task = {0: task0, 1: task1, 2: task2}
    common = {
        "candidate_batch": _batch(("chair", "table", "plant"), by_task),
        "current_task": 0,
        "idea1_candidates_by_task": by_task,
    }

    missing = plan_ordered_suffix(**common, strict_geodesic_matrices=(second,))
    assert missing.ordered_raw_ranks == (0, 1)
    assert missing.plan.fallback_reason is FallbackReason.GEODESIC_UNAVAILABLE

    reversed_source = _matrix(0, tuple(reversed(task0)), task1, ((1.0,), (1.0,)))
    misaligned = plan_ordered_suffix(
        **common, strict_geodesic_matrices=(reversed_source, second)
    )
    assert misaligned.ordered_raw_ranks == (0, 1)
    assert misaligned.plan.fallback_reason is FallbackReason.GEODESIC_UNAVAILABLE


def test_distinct_tiny_costs_are_not_hidden_by_an_unregistered_epsilon():
    task0 = (
        _candidate(0, "chair", 0, 0.1),
        _candidate(0, "chair", 1, 0.05),
    )
    task1 = (_candidate(1, "table", 0, 0.0),)
    by_task = {0: task0, 1: task1}

    result = plan_ordered_suffix(
        candidate_batch=_batch(("chair", "table"), by_task),
        current_task=0,
        idea1_candidates_by_task=by_task,
        strict_geodesic_matrices=(
            _matrix(0, task0, task1, ((5e-13,), (0.0,))),
        ),
    )

    assert result.ordered_raw_ranks == (1, 0)
    assert result.plan.fallback_reason is FallbackReason.NONE


def test_module_import_boundary_is_contract_only_and_gpu_free():
    module = importlib.import_module("tdonav_navigation.suffix_planning")
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])

    assert not imported & {
        "habitat",
        "habitat_sim",
        "numpy",
        "torch",
        "runtime",
        "mapping",
    }


def test_planner_values_are_frozen_and_reject_nonfinite_heuristics():
    assert DEFAULT_SUFFIX_WEIGHT == 0.35
    candidate = _candidate(0, "chair", 0, 1.0)
    with pytest.raises(FrozenInstanceError):
        candidate.utility = 2.0
    with pytest.raises(SuffixPlanningError, match="utility must be finite"):
        _candidate(0, "chair", 1, math.nan)
    with pytest.raises(SuffixPlanningError, match="suffix_weight must be finite"):
        by_task = {0: (candidate,)}
        plan_ordered_suffix(
            candidate_batch=_batch(("chair",), by_task),
            current_task=0,
            idea1_candidates_by_task=by_task,
            strict_geodesic_matrices=(),
            suffix_weight=math.inf,
        )
