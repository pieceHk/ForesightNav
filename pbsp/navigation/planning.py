"""Rank projected candidates, cache geodesics and dispatch the current task.

Geometry queries are supplied as callables returning GeodesicDistance. This module
contains no simulator imports, experiment metadata or file output.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Callable, Mapping, Optional, Protocol
import time
import numpy as np

from pbsp.types import (
    CandidateBatch,
    CandidateRecord,
    FallbackReason,
    GeodesicDistance,
    GeodesicMatrix,
    GeodesicStatus,
    SCSPContractError,
    SuffixPlan,
    validate_candidate_batch,
)
from .suffix import Idea1Candidate, SuffixPlanningResult, plan_ordered_suffix

GeodesicQuery = Callable[[object, object], GeodesicDistance]


class MappingProvider(Protocol):
    """Snapshot and refresh operations needed by the planning transaction."""

    def invalidate_cache(self) -> None: ...

    def get_all_goals(self) -> dict: ...

    def get_candidate_batch(self) -> Optional[CandidateBatch]: ...

    def check_candidate_batch_status(
        self, batch, *, scene_id, episode_id, ordered_categories
    ) -> Mapping[str, Any]: ...


def _record_payload(record: CandidateRecord) -> dict:
    payload = asdict(record)
    payload["position"] = list(record.position)
    return payload


def _version_payload(token) -> dict:
    return asdict(token)


def bind_projected_candidates(candidate_batch, projected, projected_indices):
    """Bind projected waypoints to the immutable mapper version token."""
    validate_candidate_batch(candidate_batch)
    positions_by_task = []
    records_by_task = []
    for task_idx, category in enumerate(candidate_batch.ordered_categories):
        positions = tuple(projected.get(task_idx, ()))
        accepted_indices = tuple(projected_indices.get(task_idx, ()))
        source_records = candidate_batch.records_for_task(task_idx)
        if len(positions) != len(accepted_indices):
            raise SCSPContractError(f"projected task {task_idx} index/position count mismatch")
        records = []
        for record_idx, (position, source_index) in enumerate(zip(positions, accepted_indices)):
            try:
                source_record = source_records[int(source_index)]
                records.append(
                    CandidateRecord(
                        task_idx=task_idx,
                        category=category,
                        position=position,
                        raw_rank=source_record.raw_rank,
                        semantic_score=source_record.semantic_score,
                        support_count=source_record.support_count,
                        support_score=source_record.support_score,
                        map_revision=source_record.map_revision,
                    )
                )
            except (IndexError, TypeError, ValueError) as exc:
                raise SCSPContractError(
                    f"invalid projected record {record_idx} for task {task_idx}: {exc}"
                ) from exc
        positions_by_task.append((task_idx, positions))
        records_by_task.append((task_idx, tuple(records)))
    return CandidateBatch(
        scene_id=candidate_batch.scene_id,
        episode_id=candidate_batch.episode_id,
        map_revision=candidate_batch.map_revision,
        last_fused_keyframes=candidate_batch.last_fused_keyframes,
        query_serial=candidate_batch.query_serial,
        ordered_categories=candidate_batch.ordered_categories,
        positions_by_task=tuple(positions_by_task),
        records_by_task=tuple(records_by_task),
    )


def subtask_has_acted(agent, task_idx):
    try:
        return int(agent.sub_task_timesteps[0][int(task_idx)]) > 0
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        return False


def active_candidate(agent, task_idx):
    candidates = getattr(agent, "gs_poses", {}).get(int(task_idx), [])
    active_index = getattr(agent, "frontier_flag", {}).get(int(task_idx), 0)
    if 0 <= active_index < len(candidates):
        return candidates[active_index]
    return None


def rank_candidate_layers(reranker, candidate_batch, origin, geodesic: GeodesicQuery, agent):
    """Compute the existing Idea-1 utility order without changing agent state."""
    candidates_by_task = {}
    telemetry_by_task = []
    diagnostics_by_task = []
    for task_idx in range(len(candidate_batch.ordered_categories)):
        has_acted = subtask_has_acted(agent, task_idx)
        if not has_acted:
            getattr(reranker, "_active", {}).pop(task_idx, None)
        source_records = candidate_batch.records_for_task(task_idx)
        record_by_rank = {record.raw_rank: record for record in source_records}
        payload = [_record_payload(record) for record in source_records]
        travel_by_position = {}
        travel_status_by_rank = []
        for record in source_records:
            distance = geodesic(origin, record.position)
            travel_by_position[tuple(record.position)] = (
                distance.distance_m if distance.status is GeodesicStatus.AVAILABLE else float("inf")
            )
            travel_status_by_rank.append(
                {
                    "raw_rank": record.raw_rank,
                    "status": distance.status.value,
                    "distance_m": (
                        distance.distance_m if distance.status is GeodesicStatus.AVAILABLE else None
                    ),
                }
            )

        def strict_travel_cost(start, end):
            del start
            return travel_by_position[tuple((float(value) for value in end))]

        (ranked, diagnostics) = reranker.rank(
            task_idx,
            payload,
            origin,
            travel_cost=strict_travel_cost,
            active_position=active_candidate(agent, task_idx) if has_acted else None,
            visited_positions=getattr(agent, "visited_candidate_positions", {}).get(task_idx, ()),
        )
        if not has_acted:
            getattr(reranker, "_active", {}).pop(task_idx, None)
        diagnostics = dict(diagnostics)
        diagnostics["strict_travel_by_raw_rank"] = travel_status_by_rank
        candidates = tuple(
            (
                Idea1Candidate(
                    record=record_by_rank[int(record["raw_rank"])], utility=record["utility"]
                )
                for record in ranked
            )
        )
        candidates_by_task[task_idx] = candidates
        telemetry_by_task.append(
            (
                task_idx,
                tuple(
                    (
                        {
                            "raw_rank": int(record["raw_rank"]),
                            "utility": float(record["utility"]),
                            "utility_components": record["utility_components"],
                        }
                        for record in ranked
                    )
                ),
            )
        )
        diagnostics_by_task.append((task_idx, diagnostics))
    return (candidates_by_task, tuple(telemetry_by_task), tuple(diagnostics_by_task))


def build_geodesic_matrices(
    candidate_batch: CandidateBatch,
    current_task,
    idea1_candidates_by_task,
    geodesic: GeodesicQuery,
    distance_cache,
    candidate_cap,
):
    """Build the capped suffix chain using a versioned pairwise cache."""
    token = candidate_batch.version_token
    entries_before = len(distance_cache)
    stale_keys = [key for key in distance_cache if key[0] != token]
    for key in stale_keys:
        del distance_cache[key]
    stats = {
        "hits": 0,
        "misses": 0,
        "evictions": len(stale_keys),
        "entries_before": entries_before,
        "entries_after_eviction": len(distance_cache),
        "entries_after": len(distance_cache),
    }
    task_count = len(candidate_batch.ordered_categories)
    capped = {
        task_idx: tuple(idea1_candidates_by_task[task_idx][:candidate_cap])
        for task_idx in range(task_count)
    }
    if current_task >= task_count - 1 or any(
        (not capped[task_idx] for task_idx in range(current_task, task_count))
    ):
        return ((), stats)
    matrices = []
    for source_task in range(current_task, task_count - 1):
        target_task = source_task + 1
        source = capped[source_task]
        target = capped[target_task]
        rows = []
        for source_candidate in source:
            row = []
            for target_candidate in target:
                key = (
                    token,
                    source_task,
                    target_task,
                    source_candidate.record.identity(),
                    target_candidate.record.identity(),
                )
                if key in distance_cache:
                    distance = distance_cache[key]
                    stats["hits"] += 1
                else:
                    distance = geodesic(
                        source_candidate.record.position, target_candidate.record.position
                    )
                    distance_cache[key] = distance
                    stats["misses"] += 1
                row.append(distance)
            rows.append(tuple(row))
        matrices.append(
            GeodesicMatrix(
                source_task_idx=source_task,
                target_task_idx=target_task,
                source_candidate_identities=tuple(
                    (candidate.record.identity() for candidate in source)
                ),
                target_candidate_identities=tuple(
                    (candidate.record.identity() for candidate in target)
                ),
                distances=tuple(rows),
            )
        )
    stats["entries_after"] = len(distance_cache)
    return (tuple(matrices), stats)


def raw_ranks_for_positions(candidate_batch, task_idx, positions):
    """Recover raw ranks from projected positions after cooldown reordering."""
    records = candidate_batch.records_for_task(int(task_idx))
    used = set()
    ordered = []
    for position in positions:
        point = np.asarray(position, dtype=np.float64)
        matches = [
            record
            for record in records
            if record.raw_rank not in used
            and np.allclose(
                point, np.asarray(record.position, dtype=np.float64), rtol=0.0, atol=1e-06
            )
        ]
        if len(matches) != 1:
            raise SCSPContractError(
                "dispatched position does not map to exactly one projected record"
            )
        used.add(matches[0].raw_rank)
        ordered.append(int(matches[0].raw_rank))
    if len(ordered) != len(records):
        raise SCSPContractError("dispatched raw-rank order is not a complete projected permutation")
    return tuple(ordered)


def raw_rank_for_position(candidate_batch, task_idx, position):
    if position is None or candidate_batch is None:
        return None
    try:
        point = np.asarray(position, dtype=np.float64)
        matches = [
            record.raw_rank
            for record in candidate_batch.records_for_task(int(task_idx))
            if np.allclose(
                point, np.asarray(record.position, dtype=np.float64), rtol=0.0, atol=1e-06
            )
        ]
        return int(matches[0]) if len(matches) == 1 else None
    except Exception:
        return None


def apply_runtime_order(projected_goals, decision, runtime_mode):
    """Apply a version-valid order to the current task and no future task."""
    goals = None if projected_goals is None else dict(projected_goals)
    current_task = int(decision.get("current_task", 0))
    gate = dict(decision.get("dispatch_version_gate", {}))
    metadata = {
        "mode": runtime_mode,
        "source": "legacy",
        "task_idx": current_task,
        "externally_ranked": False,
        "suffix_eligible": False,
        "suffix_intervention": False,
        "reason": "shadow_only" if runtime_mode == "shadow" else "fallback",
        "dispatch_version_gate": gate,
        "stale_suffix_dispatch": None,
        "ordered_raw_ranks": [],
        "version_token": None,
    }
    if goals is None:
        metadata["reason"] = "preserved_existing_candidates"
        return (None, metadata)
    if decision.get("status") == "planned":
        try:
            candidate_batch = decision["candidate_batch"]
            legacy_ranks = raw_ranks_for_positions(
                candidate_batch, current_task, goals.get(current_task, ())
            )
            metadata["ordered_raw_ranks"] = [int(rank) for rank in legacy_ranks]
            metadata["version_token"] = _version_payload(candidate_batch.version_token)
        except Exception as exc:
            metadata["legacy_order_error"] = f"{type(exc).__name__}: {exc}"
    if (
        runtime_mode not in {"idea1_only", "ordered_suffix_mvp"}
        or decision.get("status") != "planned"
        or gate.get("would_invalidate", True)
    ):
        return (goals, metadata)
    try:
        candidate_batch = decision["candidate_batch"]
        if runtime_mode == "idea1_only":
            source = "idea1"
            raw_ranks = tuple(decision["idea1_order"][current_task])
            metadata["reason"] = "version_valid_idea1"
        else:
            fallback_reason = fallback_value(decision["suffix_plan"].fallback_reason)
            if fallback_reason == FallbackReason.NONE.value:
                source = "suffix"
                raw_ranks = tuple(decision["suffix_order"][current_task])
                metadata["suffix_eligible"] = True
                metadata["suffix_intervention"] = bool(decision["suffix_plan"].intervention)
            else:
                source = "idea1_fallback"
                raw_ranks = tuple(decision["idea1_order"][current_task])
                metadata["reason"] = fallback_reason
        records = candidate_batch.records_for_task(current_task)
        record_by_rank = {int(record.raw_rank): record for record in records}
        if len(record_by_rank) != len(records) or set(raw_ranks) != set(record_by_rank):
            raise SCSPContractError("external order must contain every projected raw rank once")
        goals[current_task] = [
            [float(value) for value in record_by_rank[rank].position] for rank in raw_ranks
        ]
        metadata.update(
            {
                "source": source,
                "externally_ranked": True,
                "reason": "version_valid_suffix" if source == "suffix" else metadata["reason"],
                "ordered_raw_ranks": [int(rank) for rank in raw_ranks],
                "version_token": _version_payload(candidate_batch.version_token),
                "stale_suffix_dispatch": int(
                    source == "suffix" and gate.get("would_invalidate", True)
                ),
            }
        )
    except Exception as exc:
        goals = dict(projected_goals)
        metadata["suffix_eligible"] = False
        metadata["suffix_intervention"] = False
        metadata["stale_suffix_dispatch"] = None
        metadata["reason"] = "invalid_external_order"
        metadata["error"] = f"{type(exc).__name__}: {exc}"
    return (goals, metadata)


def fallback_value(value):
    return getattr(value, "value", value)


def reset_task_planning(reranker, distance_cache):
    """Clear all plan-local state when the ordered subtask advances."""
    reranker.reset()
    distance_cache.clear()


@dataclass(frozen=True)
class CandidatePlan:
    """A projected candidate snapshot, its ordered plan and computation diagnostics."""

    candidate_batch: CandidateBatch
    suffix_plan: SuffixPlan
    legacy_order: tuple
    idea1_order: tuple
    suffix_order: tuple
    idea1_candidates: tuple
    rerank_diagnostics: tuple
    timing_ms: Mapping[str, float]
    distance_cache: Mapping[str, int]


@dataclass(frozen=True)
class CandidateRefresh:
    """The final version gate and candidates after zero or one synchronous retry."""

    projected_goals: Optional[dict]
    decision: dict
    retry_provider_events: list


def compute_candidate_plan(
    *,
    candidate_batch: CandidateBatch,
    projected_goals,
    projected_indices,
    current_task,
    origin,
    geodesic: GeodesicQuery,
    agent,
    reranker,
    distance_cache,
    candidate_cap,
    suffix_weight,
    clock: Optional[Callable[[], float]] = None,
    started_at: Optional[float] = None,
) -> CandidatePlan:
    """Bind, rank and solve an ordered suffix on one mapper snapshot.

    Geometry uses Habitat world coordinates in metres and returns explicit
    available, unreachable or unavailable distance states.
    """
    if clock is None:
        clock = time.perf_counter
    if started_at is None:
        started_at = clock()
    planning_batch = bind_projected_candidates(candidate_batch, projected_goals, projected_indices)
    legacy_order = tuple(
        (
            tuple((record.raw_rank for record in planning_batch.records_for_task(task)))
            for task in range(len(planning_batch.ordered_categories))
        )
    )
    rerank_started = clock()
    (idea1_candidates_by_task, idea1_candidates, rerank_diagnostics) = rank_candidate_layers(
        reranker, planning_batch, origin, geodesic, agent
    )
    rerank_ms = (clock() - rerank_started) * 1000.0
    idea1_order = tuple(
        (
            tuple((candidate.record.raw_rank for candidate in candidates))
            for (_, candidates) in sorted(idea1_candidates_by_task.items())
        )
    )
    geodesic_started = clock()
    if float(suffix_weight) == 0.0:
        matrices = ()
        cache_stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "entries_before": len(distance_cache),
            "entries_after_eviction": len(distance_cache),
            "entries_after": len(distance_cache),
        }
    else:
        (matrices, cache_stats) = build_geodesic_matrices(
            planning_batch,
            int(current_task),
            idea1_candidates_by_task,
            geodesic,
            distance_cache,
            int(candidate_cap),
        )
    geodesic_ms = (clock() - geodesic_started) * 1000.0
    planner_started = clock()
    result = plan_ordered_suffix(
        candidate_batch=planning_batch,
        current_task=int(current_task),
        idea1_candidates_by_task=idea1_candidates_by_task,
        strict_geodesic_matrices=matrices,
        suffix_weight=float(suffix_weight),
        candidate_cap=int(candidate_cap),
        compute_time_ms=0.0,
    )
    planner_ms = (clock() - planner_started) * 1000.0
    result = SuffixPlanningResult(
        plan=replace(result.plan, compute_time_ms=planner_ms),
        ordered_raw_ranks=result.ordered_raw_ranks,
    )
    suffix_order = list(idea1_order)
    suffix_order[int(current_task)] = result.ordered_raw_ranks
    total_ms = (clock() - started_at) * 1000.0
    return CandidatePlan(
        candidate_batch=planning_batch,
        suffix_plan=result.plan,
        legacy_order=legacy_order,
        idea1_order=idea1_order,
        suffix_order=tuple(suffix_order),
        idea1_candidates=idea1_candidates,
        rerank_diagnostics=rerank_diagnostics,
        timing_ms={
            "idea1_rerank": rerank_ms,
            "geodesic_matrices": geodesic_ms,
            "suffix_dp": planner_ms,
            "total_shadow": total_ms,
        },
        distance_cache=cache_stats,
    )


def dispatch_version_gate(
    provider: MappingProvider, decision, *, scene_id, episode_id, ordered_categories
):
    """Read mapper status immediately before an active dispatch."""
    try:
        batch = decision.get("candidate_batch")
        if batch is None:
            batch = provider.get_candidate_batch()
        return provider.check_candidate_batch_status(
            batch, scene_id=scene_id, episode_id=episode_id, ordered_categories=ordered_categories
        )
    except Exception as exc:
        return {
            "would_invalidate": True,
            "reason": FallbackReason.PROVIDER_STATUS_UNAVAILABLE.value,
            "error": f"{type(exc).__name__}: {exc}",
        }


def prepare_candidate_refresh(
    *,
    runtime_mode,
    provider: MappingProvider,
    provider_event,
    raw_goals,
    plan_once: Callable[[Optional[dict], Mapping], tuple],
    project_fallback: Callable[[Optional[dict]], dict],
    make_fallback: Callable[[Exception], dict],
    read_dispatch_gate: Callable[[dict], Mapping],
    read_retry_event: Callable[[], Mapping],
    preserve_existing_on_missing=False,
) -> CandidateRefresh:
    """Plan against a final status gate, re-querying at most once.

    ``plan_once`` projects and computes a plan, then reads its dispatch gate.
    The remaining callbacks supply geometry, fallback diagnostics and the new
    provider snapshot. File output and experiment provenance belong to callers.
    """
    active = runtime_mode in {"idea1_only", "ordered_suffix_mvp"}
    retry_provider_events = []
    attempts = []
    refresh_payload = raw_goals
    try:
        (projected_goals, decision, gate) = plan_once(raw_goals, provider_event)
    except Exception as exc:
        try:
            projected_goals = project_fallback(raw_goals)
        except Exception:
            projected_goals = {}
        decision = make_fallback(exc)
        gate = read_dispatch_gate(decision)
    attempts.append(
        {
            "attempt": 0,
            "query_serial": gate.get("query_serial"),
            "version_gate": dict(gate),
            "decision_status": decision.get("status"),
            "fallback_reason": fallback_value(decision.get("fallback_reason")),
        }
    )
    requery_error = None
    retryable_fallbacks = {
        FallbackReason.REBUILDING.value,
        FallbackReason.EPISODE_MISMATCH.value,
        FallbackReason.PROVIDER_TOKEN_CHANGED.value,
        FallbackReason.PROVIDER_STATUS_UNAVAILABLE.value,
        FallbackReason.INVALID_CANDIDATE_BATCH.value,
        FallbackReason.STALE_VERSION_TOKEN.value,
    }
    if active and (
        gate.get("would_invalidate", True)
        or fallback_value(decision.get("fallback_reason")) in retryable_fallbacks
    ):
        try:
            provider.invalidate_cache()
            retry_goals = provider.get_all_goals()
            refresh_payload = retry_goals
            retry_event = read_retry_event()
            retry_provider_events.append(retry_event)
            (projected_goals, decision, gate) = plan_once(retry_goals, retry_event)
            attempts.append(
                {
                    "attempt": 1,
                    "query_serial": gate.get("query_serial"),
                    "version_gate": dict(gate),
                    "decision_status": decision.get("status"),
                    "fallback_reason": fallback_value(decision.get("fallback_reason")),
                }
            )
        except Exception as exc:
            requery_error = f"{type(exc).__name__}: {exc}"
            gate = {
                "would_invalidate": True,
                "reason": FallbackReason.PROVIDER_STATUS_UNAVAILABLE.value,
                "error": requery_error,
            }
            attempts.append(
                {
                    "attempt": 1,
                    "query_serial": None,
                    "version_gate": dict(gate),
                    "decision_status": "fallback",
                    "fallback_reason": FallbackReason.PROVIDER_STATUS_UNAVAILABLE.value,
                }
            )
    decision = dict(decision)
    decision["mode"] = runtime_mode
    decision["shadow_only"] = runtime_mode == "shadow"
    decision["dispatch_version_gate"] = dict(gate)
    decision["version_attempts"] = attempts
    decision["synchronous_requery_count"] = len(attempts) - 1
    decision["synchronous_requery_succeeded"] = bool(
        len(attempts) == 2
        and (not gate.get("would_invalidate", True))
        and (decision.get("status") == "planned")
    )
    if requery_error is not None:
        decision["synchronous_requery_error"] = requery_error
    if active and gate.get("would_invalidate", True):
        decision["computed_status"] = decision.get("status")
        decision["status"] = "fallback"
        decision["fallback_reason"] = gate.get(
            "reason", FallbackReason.PROVIDER_STATUS_UNAVAILABLE.value
        )
        decision["failure_stage"] = "pre_dispatch_version_gate"
        decision["dispatch_order_source"] = "legacy"
    if preserve_existing_on_missing and (not refresh_payload):
        projected_goals = None
        decision["preserved_existing_candidates"] = True
    else:
        decision["preserved_existing_candidates"] = False
    return CandidateRefresh(projected_goals, decision, retry_provider_events)
