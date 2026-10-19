"""Rank projected candidates, cache geodesics and dispatch the current task.

Geometry queries are supplied as callables returning GeodesicDistance. This module
contains no simulator imports, experiment metadata or file output.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Callable
import numpy as np

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
from .suffix import Idea1Candidate

GeodesicQuery = Callable[[object, object], GeodesicDistance]


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
    candidate_batch,
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
