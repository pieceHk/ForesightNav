"""Pure ordered-suffix planning over precomputed Idea-1 candidates.

This module deliberately depends only on the standard library and the frozen SCSP
contracts.  It does not query a mapper, a simulator, Habitat, or a pathfinder.  A
caller supplies strict geodesic matrices whose labels exactly match the capped
Idea-1 candidate layers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Mapping, Optional, Sequence, Tuple

from pbsp.types import (
    BatchValidationStatus,
    CandidateBatch,
    CandidateRecord,
    CandidateSuffixCost,
    FallbackReason,
    GeodesicDistance,
    GeodesicMatrix,
    GeodesicStatus,
    InterventionReason,
    SCSPContractError,
    SuffixPlan,
    VersionToken,
    validate_candidate_batch,
)


DEFAULT_CANDIDATE_CAP = 5
DEFAULT_SUFFIX_WEIGHT = 0.35


class SuffixPlanningError(ValueError):
    """Raised when planner inputs do not form one strict, aligned snapshot."""


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise SuffixPlanningError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise SuffixPlanningError(f"{name} must be >= 0")
    return result


def _finite_float(
    value: Any, name: str, *, minimum: Optional[float] = None
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise SuffixPlanningError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise SuffixPlanningError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise SuffixPlanningError(f"{name} must be >= {minimum}")
    return result


@dataclass(frozen=True)
class Idea1Candidate:
    """One candidate in an already-computed Idea-1 order.

    ``utility`` is the existing deterministic Idea-1 heuristic utility.  The
    planner does not derive it from, or reinterpret, semantic/support scores.
    """

    record: CandidateRecord
    utility: float

    def __post_init__(self) -> None:
        if not isinstance(self.record, CandidateRecord):
            raise SuffixPlanningError("record must be a CandidateRecord")
        object.__setattr__(
            self, "utility", _finite_float(self.utility, "utility")
        )


@dataclass(frozen=True)
class SuffixPlanningResult:
    """Deterministic plan plus the complete composed current-task order.

    An empty order with an explicit fallback is an abstention: the caller must keep
    its existing order because no complete, trusted Idea-1 permutation was supplied.
    """

    plan: SuffixPlan
    ordered_raw_ranks: Tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.plan, SuffixPlan):
            raise SuffixPlanningError("plan must be a SuffixPlan")
        try:
            ranks = tuple(
                _nonnegative_int(rank, f"ordered_raw_ranks[{index}]")
                for index, rank in enumerate(self.ordered_raw_ranks)
            )
        except TypeError as exc:
            raise SuffixPlanningError("ordered_raw_ranks must be a sequence") from exc
        if len(set(ranks)) != len(ranks):
            raise SuffixPlanningError("ordered_raw_ranks contains duplicate ranks")
        if ranks:
            if self.plan.selected_raw_rank != ranks[0]:
                raise SuffixPlanningError(
                    "selected_raw_rank must be the composed order head"
                )
        elif self.plan.selected_raw_rank is not None:
            raise SuffixPlanningError(
                "an empty composed order cannot have a selected_raw_rank"
            )
        object.__setattr__(self, "ordered_raw_ranks", ranks)


def _normalise_categories(value: Sequence[str]) -> Tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise SuffixPlanningError("ordered_categories must be a sequence")
    try:
        categories = tuple(value)
    except TypeError as exc:
        raise SuffixPlanningError("ordered_categories must be a sequence") from exc
    if not categories:
        raise SuffixPlanningError("ordered_categories must not be empty")
    for index, category in enumerate(categories):
        if not isinstance(category, str) or not category.strip():
            raise SuffixPlanningError(
                f"ordered_categories[{index}] must be a non-empty string"
            )
    return categories


def _normalise_candidates(
    value: Mapping[int, Sequence[Idea1Candidate]],
    categories: Tuple[str, ...],
    token: VersionToken,
) -> Tuple[Tuple[Tuple[Idea1Candidate, ...], ...], bool]:
    if not isinstance(value, Mapping):
        raise SuffixPlanningError("idea1_candidates_by_task must be a mapping")

    keyed = {}
    for raw_task_idx, candidates in value.items():
        task_idx = _nonnegative_int(raw_task_idx, "candidate task index")
        if task_idx in keyed:
            raise SuffixPlanningError(f"duplicate candidate task index {task_idx}")
        if isinstance(candidates, (str, bytes)):
            raise SuffixPlanningError(
                f"idea1_candidates_by_task[{task_idx}] must be a sequence"
            )
        try:
            keyed[task_idx] = tuple(candidates)
        except TypeError as exc:
            raise SuffixPlanningError(
                f"idea1_candidates_by_task[{task_idx}] must be a sequence"
            ) from exc

    expected = set(range(len(categories)))
    if set(keyed) != expected:
        raise SuffixPlanningError(
            "idea1_candidates_by_task must contain every task exactly once"
        )

    version_valid = True
    normalised = []
    for task_idx, category in enumerate(categories):
        candidates = keyed[task_idx]
        identities = set()
        raw_ranks = set()
        for order_idx, candidate in enumerate(candidates):
            if not isinstance(candidate, Idea1Candidate):
                raise SuffixPlanningError(
                    f"task {task_idx} candidate {order_idx} must be an Idea1Candidate"
                )
            record = candidate.record
            if record.task_idx != task_idx:
                raise SuffixPlanningError(
                    f"task {task_idx} candidate {order_idx} has task_idx={record.task_idx}"
                )
            if record.category != category:
                raise SuffixPlanningError(
                    f"task {task_idx} candidate {order_idx} category does not align"
                )
            if record.identity() in identities:
                raise SuffixPlanningError(
                    f"task {task_idx} contains a duplicate candidate identity"
                )
            if record.raw_rank in raw_ranks:
                raise SuffixPlanningError(
                    f"task {task_idx} contains duplicate raw_rank {record.raw_rank}"
                )
            identities.add(record.identity())
            raw_ranks.add(record.raw_rank)
            if record.map_revision != token.map_revision:
                version_valid = False
        normalised.append(candidates)
    return tuple(normalised), version_valid


def _matches_candidate_batch(
    candidates_by_task: Tuple[Tuple[Idea1Candidate, ...], ...],
    batch: CandidateBatch,
) -> bool:
    for task_idx, candidates in enumerate(candidates_by_task):
        records = batch.records_for_task(task_idx)
        if len(candidates) != len(records):
            return False
        expected = {record.identity(): record for record in records}
        for candidate in candidates:
            record = candidate.record
            if expected.get(record.identity()) != record:
                return False
    return True


def _has_complete_candidate_identities(
    candidates_by_task: Tuple[Tuple[Idea1Candidate, ...], ...],
    batch: CandidateBatch,
) -> bool:
    for task_idx, candidates in enumerate(candidates_by_task):
        supplied = {candidate.record.identity() for candidate in candidates}
        expected = {record.identity() for record in batch.records_for_task(task_idx)}
        if len(candidates) != len(expected) or supplied != expected:
            return False
    return True


def _unavailable_costs(
    candidates: Sequence[Idea1Candidate],
) -> Tuple[CandidateSuffixCost, ...]:
    return tuple(
        CandidateSuffixCost(
            raw_rank=candidate.record.raw_rank,
            cost=GeodesicDistance.unavailable(),
        )
        for candidate in candidates
    )


def _result(
    *,
    token: VersionToken,
    current_task: int,
    current_candidates: Tuple[Idea1Candidate, ...],
    ordered_raw_ranks: Tuple[int, ...],
    matrices: Tuple[GeodesicMatrix, ...],
    suffix_costs: Tuple[CandidateSuffixCost, ...],
    fallback_reason: FallbackReason,
    intervention_reason: InterventionReason = InterventionReason.NONE,
    compute_time_ms: float,
) -> SuffixPlanningResult:
    selected = ordered_raw_ranks[0] if ordered_raw_ranks else None
    idea1_head = current_candidates[0].record.raw_rank if current_candidates else None
    intervention = selected is not None and selected != idea1_head
    plan = SuffixPlan(
        input_token=token,
        current_task=current_task,
        strict_geodesic_matrices=matrices,
        suffix_costs=suffix_costs,
        selected_raw_rank=selected,
        intervention=intervention,
        fallback_reason=fallback_reason,
        intervention_reason=(
            intervention_reason if intervention else InterventionReason.NONE
        ),
        compute_time_ms=compute_time_ms,
    )
    return SuffixPlanningResult(plan=plan, ordered_raw_ranks=ordered_raw_ranks)


def _fallback(
    *,
    token: VersionToken,
    current_task: int,
    current_candidates: Tuple[Idea1Candidate, ...],
    capped_current: Tuple[Idea1Candidate, ...],
    fallback_reason: FallbackReason,
    compute_time_ms: float,
    matrices: Tuple[GeodesicMatrix, ...] = (),
    suffix_costs: Optional[Tuple[CandidateSuffixCost, ...]] = None,
) -> SuffixPlanningResult:
    return _result(
        token=token,
        current_task=current_task,
        current_candidates=current_candidates,
        ordered_raw_ranks=tuple(
            candidate.record.raw_rank for candidate in current_candidates
        ),
        matrices=matrices,
        suffix_costs=(
            _unavailable_costs(capped_current)
            if suffix_costs is None
            else suffix_costs
        ),
        fallback_reason=fallback_reason,
        compute_time_ms=compute_time_ms,
    )


def _strict_matrix_chain(
    matrices: Sequence[GeodesicMatrix],
    capped: Tuple[Tuple[Idea1Candidate, ...], ...],
    current_task: int,
) -> Tuple[GeodesicMatrix, ...]:
    try:
        supplied = tuple(matrices)
    except TypeError as exc:
        raise SuffixPlanningError(
            "strict_geodesic_matrices must be a sequence"
        ) from exc

    by_transition = {}
    for index, matrix in enumerate(supplied):
        if not isinstance(matrix, GeodesicMatrix):
            raise SuffixPlanningError(
                f"strict_geodesic_matrices[{index}] must be a GeodesicMatrix"
            )
        transition = (matrix.source_task_idx, matrix.target_task_idx)
        if transition in by_transition:
            raise SuffixPlanningError(
                f"duplicate geodesic transition layer {transition!r}"
            )
        by_transition[transition] = matrix

    expected_transitions = tuple(
        (task_idx, task_idx + 1)
        for task_idx in range(current_task, len(capped) - 1)
    )
    if set(by_transition) != set(expected_transitions):
        raise SuffixPlanningError(
            "strict_geodesic_matrices must contain exactly the required suffix chain"
        )

    chain = []
    for source_task, target_task in expected_transitions:
        matrix = by_transition[(source_task, target_task)]
        expected_sources = tuple(
            candidate.record.identity() for candidate in capped[source_task]
        )
        expected_targets = tuple(
            candidate.record.identity() for candidate in capped[target_task]
        )
        if matrix.source_candidate_identities != expected_sources:
            raise SuffixPlanningError(
                f"matrix {source_task}->{target_task} source labels must exactly "
                "match the capped Idea-1 order"
            )
        if matrix.target_candidate_identities != expected_targets:
            raise SuffixPlanningError(
                f"matrix {source_task}->{target_task} target labels must exactly "
                "match the capped Idea-1 order"
            )
        chain.append(matrix)
    return tuple(chain)


def _edge_value(distance: GeodesicDistance) -> float:
    if not isinstance(distance, GeodesicDistance):
        raise SuffixPlanningError("matrix edge must be a GeodesicDistance")
    if distance.status is GeodesicStatus.AVAILABLE:
        if distance.distance_m is None or not math.isfinite(distance.distance_m):
            raise SuffixPlanningError("available geodesic edge must be finite")
        if distance.distance_m < 0.0:
            raise SuffixPlanningError("available geodesic edge must be non-negative")
        return float(distance.distance_m)
    if distance.status in (GeodesicStatus.UNREACHABLE, GeodesicStatus.UNAVAILABLE):
        return math.inf
    raise SuffixPlanningError("matrix edge has an unknown geodesic status")


def _reverse_min_plus(
    matrices: Tuple[GeodesicMatrix, ...],
) -> Tuple[Tuple[float, ...], Tuple[GeodesicStatus, ...]]:
    final_count = len(matrices[-1].target_candidate_identities)
    downstream_costs = tuple(0.0 for _ in range(final_count))
    downstream_statuses = tuple(
        GeodesicStatus.AVAILABLE for _ in range(final_count)
    )

    for matrix in reversed(matrices):
        layer_costs = []
        layer_statuses = []
        for row in matrix.distances:
            options = []
            blocked_by_unavailable = False
            for edge, suffix_cost, suffix_status in zip(
                row, downstream_costs, downstream_statuses
            ):
                edge_cost = _edge_value(edge)
                if edge.status is GeodesicStatus.UNAVAILABLE:
                    if suffix_status is not GeodesicStatus.UNREACHABLE:
                        blocked_by_unavailable = True
                    continue
                if not math.isfinite(edge_cost):
                    continue
                if suffix_status is GeodesicStatus.UNAVAILABLE:
                    blocked_by_unavailable = True
                    continue
                if not math.isfinite(suffix_cost):
                    continue
                total = edge_cost + suffix_cost
                if not math.isfinite(total):
                    raise SuffixPlanningError("geodesic suffix cost overflowed")
                options.append(total)
            if options:
                layer_costs.append(min(options))
                layer_statuses.append(GeodesicStatus.AVAILABLE)
            else:
                layer_costs.append(math.inf)
                layer_statuses.append(
                    GeodesicStatus.UNAVAILABLE
                    if blocked_by_unavailable
                    else GeodesicStatus.UNREACHABLE
                )
        downstream_costs = tuple(layer_costs)
        downstream_statuses = tuple(layer_statuses)
    return downstream_costs, downstream_statuses


def _contract_costs(
    candidates: Tuple[Idea1Candidate, ...],
    costs: Tuple[float, ...],
    statuses: Tuple[GeodesicStatus, ...],
) -> Tuple[CandidateSuffixCost, ...]:
    result = []
    for candidate, cost, status in zip(candidates, costs, statuses):
        if status is GeodesicStatus.AVAILABLE:
            distance = GeodesicDistance.available(cost)
        elif status is GeodesicStatus.UNAVAILABLE:
            distance = GeodesicDistance.unavailable()
        else:
            distance = GeodesicDistance.unreachable()
        result.append(
            CandidateSuffixCost(raw_rank=candidate.record.raw_rank, cost=distance)
        )
    return tuple(result)


def _no_feasible_reason(
    matrices: Tuple[GeodesicMatrix, ...],
    statuses: Tuple[GeodesicStatus, ...],
) -> FallbackReason:
    if any(status is GeodesicStatus.UNAVAILABLE for status in statuses):
        return FallbackReason.GEODESIC_UNAVAILABLE
    edges = tuple(
        distance for matrix in matrices for row in matrix.distances for distance in row
    )
    if edges and all(
        distance.status is GeodesicStatus.UNREACHABLE for distance in edges
    ):
        return FallbackReason.ALL_EDGES_UNREACHABLE
    return FallbackReason.NO_FEASIBLE_SUFFIX


def _equal_finite_costs(
    costs: Tuple[float, ...], statuses: Tuple[GeodesicStatus, ...]
) -> bool:
    if not costs or any(
        status is not GeodesicStatus.AVAILABLE for status in statuses
    ):
        return False
    return max(costs) == min(costs)


def _compose_order(
    current_candidates: Tuple[Idea1Candidate, ...],
    capped_current: Tuple[Idea1Candidate, ...],
    costs: Tuple[float, ...],
    statuses: Tuple[GeodesicStatus, ...],
    suffix_weight: float,
) -> Tuple[int, ...]:
    finite_costs = tuple(
        cost
        for cost, status in zip(costs, statuses)
        if status is GeodesicStatus.AVAILABLE
    )
    lower = min(finite_costs)
    upper = max(finite_costs)
    span = upper - lower

    ranked = []
    for idea1_order, (candidate, cost, status) in enumerate(
        zip(capped_current, costs, statuses)
    ):
        record = candidate.record
        stable_tie = (idea1_order, record.raw_rank, record.position)
        if status is GeodesicStatus.AVAILABLE:
            normalised = 0.0 if span == 0.0 else (cost - lower) / span
            combined = candidate.utility - suffix_weight * normalised
            if not math.isfinite(combined):
                raise SuffixPlanningError("combined heuristic utility must be finite")
            key = (0, -combined) + stable_tie
        else:
            key = (1, 0.0) + stable_tie
        ranked.append((key, record.raw_rank))

    capped_order = tuple(rank for _, rank in sorted(ranked, key=lambda item: item[0]))
    tail = tuple(
        candidate.record.raw_rank
        for candidate in current_candidates[len(capped_current) :]
    )
    return capped_order + tail


def plan_ordered_suffix(
    *,
    candidate_batch: CandidateBatch,
    current_task: int,
    idea1_candidates_by_task: Mapping[int, Sequence[Idea1Candidate]],
    strict_geodesic_matrices: Sequence[GeodesicMatrix],
    suffix_weight: float = DEFAULT_SUFFIX_WEIGHT,
    candidate_cap: int = DEFAULT_CANDIDATE_CAP,
    compute_time_ms: float = 0.0,
) -> SuffixPlanningResult:
    """Compose the current Idea-1 order with a strict ordered-suffix heuristic.

    ``candidate_batch`` binds every record to one episode/query/map snapshot.
    ``idea1_candidates_by_task`` must be an exact per-task permutation of those
    records, with the desired Idea-1 order and its precomputed finite utility.
    Matrices must form the complete current-to-final task chain and label exactly
    the first ``candidate_cap`` Idea-1 records at each layer.

    Unavailable and unreachable edges are kept non-finite in the min-plus DP; no
    Euclidean substitute is permitted.  Operational matrix/calculation failures
    return the unchanged Idea-1 order with an explicit fallback.  Complexity is
    ``O(sum_r |C_r| * |C_(r+1)|)`` over the capped layers.  The returned matrix
    shapes let :class:`SCSPTelemetry` derive the exact edge count.

    The function has no clock or external side effects.  ``compute_time_ms`` is
    caller-supplied metadata so equal inputs produce equal results; a later runtime
    wrapper may measure latency around this call.
    """

    if not isinstance(candidate_batch, CandidateBatch):
        raise SuffixPlanningError("candidate_batch must be a CandidateBatch")
    input_token = candidate_batch.version_token
    categories = _normalise_categories(candidate_batch.ordered_categories)
    task_idx = _nonnegative_int(current_task, "current_task")
    if task_idx >= len(categories):
        raise SuffixPlanningError("current_task is outside ordered_categories")
    cap = _nonnegative_int(candidate_cap, "candidate_cap")
    if cap == 0:
        raise SuffixPlanningError("candidate_cap must be > 0")
    if cap > DEFAULT_CANDIDATE_CAP:
        raise SuffixPlanningError(
            f"candidate_cap must be <= {DEFAULT_CANDIDATE_CAP}"
        )
    weight = _finite_float(suffix_weight, "suffix_weight", minimum=0.0)
    latency = _finite_float(compute_time_ms, "compute_time_ms", minimum=0.0)

    candidates_by_task, version_valid = _normalise_candidates(
        idea1_candidates_by_task, categories, input_token
    )
    current_candidates = candidates_by_task[task_idx]
    capped = tuple(candidates[:cap] for candidates in candidates_by_task)
    capped_current = capped[task_idx]

    if candidate_batch.validation_status is BatchValidationStatus.INVALID:
        return _fallback(
            token=input_token,
            current_task=task_idx,
            current_candidates=(),
            capped_current=(),
            fallback_reason=FallbackReason.INVALID_CANDIDATE_BATCH,
            compute_time_ms=latency,
        )
    try:
        validate_candidate_batch(candidate_batch)
    except SCSPContractError:
        return _fallback(
            token=input_token,
            current_task=task_idx,
            current_candidates=(),
            capped_current=(),
            fallback_reason=FallbackReason.INVALID_CANDIDATE_BATCH,
            compute_time_ms=latency,
        )
    if not _has_complete_candidate_identities(candidates_by_task, candidate_batch):
        return _fallback(
            token=input_token,
            current_task=task_idx,
            current_candidates=(),
            capped_current=(),
            fallback_reason=FallbackReason.STALE_VERSION_TOKEN,
            compute_time_ms=latency,
        )
    if not version_valid:
        return _fallback(
            token=input_token,
            current_task=task_idx,
            current_candidates=current_candidates,
            capped_current=capped_current,
            fallback_reason=FallbackReason.STALE_VERSION_TOKEN,
            compute_time_ms=latency,
        )
    if not _matches_candidate_batch(candidates_by_task, candidate_batch):
        return _fallback(
            token=input_token,
            current_task=task_idx,
            current_candidates=(),
            capped_current=(),
            fallback_reason=FallbackReason.STALE_VERSION_TOKEN,
            compute_time_ms=latency,
        )
    if not current_candidates:
        return _fallback(
            token=input_token,
            current_task=task_idx,
            current_candidates=current_candidates,
            capped_current=capped_current,
            fallback_reason=FallbackReason.NO_CANDIDATES,
            compute_time_ms=latency,
        )
    if weight == 0.0:
        return _fallback(
            token=input_token,
            current_task=task_idx,
            current_candidates=current_candidates,
            capped_current=capped_current,
            fallback_reason=FallbackReason.WEIGHT_ZERO,
            compute_time_ms=latency,
        )
    if task_idx == len(categories) - 1:
        zero_costs = tuple(
            CandidateSuffixCost(
                raw_rank=candidate.record.raw_rank,
                cost=GeodesicDistance.available(0.0),
            )
            for candidate in capped_current
        )
        return _fallback(
            token=input_token,
            current_task=task_idx,
            current_candidates=current_candidates,
            capped_current=capped_current,
            fallback_reason=FallbackReason.LAST_TASK,
            compute_time_ms=latency,
            suffix_costs=zero_costs,
        )
    if any(not capped[future_task] for future_task in range(task_idx + 1, len(capped))):
        return _fallback(
            token=input_token,
            current_task=task_idx,
            current_candidates=current_candidates,
            capped_current=capped_current,
            fallback_reason=FallbackReason.FUTURE_LAYER_EMPTY,
            compute_time_ms=latency,
        )

    try:
        matrices = _strict_matrix_chain(strict_geodesic_matrices, capped, task_idx)
    except SuffixPlanningError:
        return _fallback(
            token=input_token,
            current_task=task_idx,
            current_candidates=current_candidates,
            capped_current=capped_current,
            fallback_reason=FallbackReason.GEODESIC_UNAVAILABLE,
            compute_time_ms=latency,
        )
    try:
        costs, statuses = _reverse_min_plus(matrices)
    except SuffixPlanningError:
        return _fallback(
            token=input_token,
            current_task=task_idx,
            current_candidates=current_candidates,
            capped_current=capped_current,
            fallback_reason=FallbackReason.PLANNER_ERROR,
            compute_time_ms=latency,
            matrices=matrices,
        )
    suffix_costs = _contract_costs(capped_current, costs, statuses)
    if not any(status is GeodesicStatus.AVAILABLE for status in statuses):
        return _fallback(
            token=input_token,
            current_task=task_idx,
            current_candidates=current_candidates,
            capped_current=capped_current,
            fallback_reason=_no_feasible_reason(matrices, statuses),
            compute_time_ms=latency,
            matrices=matrices,
            suffix_costs=suffix_costs,
        )
    if _equal_finite_costs(costs, statuses):
        return _fallback(
            token=input_token,
            current_task=task_idx,
            current_candidates=current_candidates,
            capped_current=capped_current,
            fallback_reason=FallbackReason.EQUAL_SUFFIX_COSTS,
            compute_time_ms=latency,
            matrices=matrices,
            suffix_costs=suffix_costs,
        )

    try:
        ordered_raw_ranks = _compose_order(
            current_candidates, capped_current, costs, statuses, weight
        )
    except SuffixPlanningError:
        return _fallback(
            token=input_token,
            current_task=task_idx,
            current_candidates=current_candidates,
            capped_current=capped_current,
            fallback_reason=FallbackReason.PLANNER_ERROR,
            compute_time_ms=latency,
            matrices=matrices,
            suffix_costs=suffix_costs,
        )
    selected = ordered_raw_ranks[0]
    original_status = statuses[0]
    intervention_reason = InterventionReason.NONE
    if selected != current_candidates[0].record.raw_rank:
        intervention_reason = (
            InterventionReason.IDEA1_CANDIDATE_UNREACHABLE
            if original_status is GeodesicStatus.UNREACHABLE
            else InterventionReason.LOWER_SUFFIX_COST
        )
    return _result(
        token=input_token,
        current_task=task_idx,
        current_candidates=current_candidates,
        ordered_raw_ranks=ordered_raw_ranks,
        matrices=matrices,
        suffix_costs=suffix_costs,
        fallback_reason=FallbackReason.NONE,
        intervention_reason=intervention_reason,
        compute_time_ms=latency,
    )


__all__ = [
    "DEFAULT_CANDIDATE_CAP",
    "DEFAULT_SUFFIX_WEIGHT",
    "Idea1Candidate",
    "SuffixPlanningError",
    "SuffixPlanningResult",
    "plan_ordered_suffix",
]
