"""Provider/dispatch records and per-step experiment timing output."""

import csv
import json
import os
import time
from pathlib import Path
import numpy as np

from pbsp.types import (
    BatchValidationStatus,
    FallbackReason,
    SCSPContractError,
    validate_candidate_batch,
)
from .metrics import DISTANCE_TO_GOAL_KEY, DISTANCE_TO_GOAL_STATUS_KEY, SHORTEST_PATH_UNREACHABLE
from .telemetry import SHADOW_SCHEMA_VERSION, dumps_json_strict, to_jsonable

TIMING_THRESHOLDS = [("≥10 FPS", 100), ("≥ 5 FPS", 200), ("≥ 1 FPS", 1000), ("≥0.2 FPS", 5000)]


def record_provider_event(
    provider,
    *,
    phase,
    current_task,
    scene_id,
    episode_id,
    ordered_categories=None,
    output_path=None,
    query_error=None,
):
    """Validate and record provider metadata without changing candidate use."""
    try:
        batch = provider.get_candidate_batch()
        validation_status = "missing"
        validation_reason = "provider has no complete candidate batch"
        if batch is not None:
            validation_status = batch.validation_status.value
            validation_reason = batch.validation_reason
            if batch.validation_status is BatchValidationStatus.VALID:
                try:
                    validate_candidate_batch(batch)
                except SCSPContractError as exc:
                    validation_status = BatchValidationStatus.INVALID.value
                    validation_reason = str(exc)
        expected_categories = (
            provider.get_ordered_categories() if ordered_categories is None else ordered_categories
        )
        gate = provider.check_candidate_batch_status(
            batch, scene_id=scene_id, episode_id=episode_id, ordered_categories=expected_categories
        )
        event = {
            "phase": str(phase),
            "scene_id": str(scene_id),
            "episode_id": str(episode_id),
            "current_task": int(current_task),
            "batch_validation_status": validation_status,
            "batch_validation_reason": validation_reason,
            "candidate_batch": None if batch is None else to_jsonable(batch),
            "version_gate": gate,
        }
    except Exception as exc:
        validation_status = BatchValidationStatus.INVALID.value
        validation_reason = f"provider event recording failed: {type(exc).__name__}: {exc}"
        gate = {"would_invalidate": True, "reason": "provider_status_unavailable"}
        event = {
            "phase": str(phase),
            "scene_id": str(scene_id),
            "episode_id": str(episode_id),
            "current_task": int(current_task),
            "batch_validation_status": validation_status,
            "batch_validation_reason": validation_reason,
            "candidate_batch": None,
            "version_gate": gate,
        }
    event["recorded_to_jsonl"] = False
    if query_error is not None:
        event["query_error"] = str(query_error)
    if output_path is not None:
        try:
            persisted_event = dict(event)
            persisted_event["recorded_to_jsonl"] = True
            with Path(output_path).open("a", encoding="utf-8") as stream:
                stream.write(dumps_json_strict(persisted_event) + "\n")
            event = persisted_event
        except Exception as exc:
            event["recording_error"] = f"{type(exc).__name__}: {exc}"
            print(
                f"[SCSP_PROVIDER] telemetry write warning: {event['recording_error']}", flush=True
            )
    print(
        "[SCSP_PROVIDER] "
        + json.dumps(
            {
                "phase": event["phase"],
                "current_task": event["current_task"],
                "batch_validation_status": validation_status,
                "batch_validation_reason": validation_reason,
                "version_gate": gate,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return event


def planning_fallback_event(
    *,
    phase,
    arm,
    current_task,
    reason,
    version_gate,
    candidate_batch,
    origin,
    started_at,
    error=None,
    failure_stage="shadow_compute",
    provenance=None,
    candidate_cap=None,
    suffix_weight=None,
    idea1_config=None,
    episode_identity=None,
):
    event = {
        "schema_version": SHADOW_SCHEMA_VERSION,
        "phase": str(phase),
        "arm": str(arm),
        "mode": "shadow",
        "shadow_only": True,
        "status": "fallback",
        "dispatch_order_source": "legacy",
        "current_task": int(current_task),
        "fallback_reason": str(reason),
        "failure_stage": str(failure_stage),
        "version_gate": dict(version_gate or {}),
        "candidate_batch": candidate_batch,
        "origin_position": [float(value) for value in origin],
        "total_shadow_latency_ms": (time.perf_counter() - started_at) * 1000.0,
    }
    if isinstance(provenance, dict):
        event.update(
            {
                "profile_sha256": provenance.get("profile_sha256"),
                "dataset_manifest_sha256": provenance.get("dataset_manifest_sha256"),
                "code_sha": provenance.get("code_sha"),
                "identity_sidecar_sha256": provenance.get("identity_sidecar_sha256"),
            }
        )
    event["planner_config"] = {
        "candidate_cap": candidate_cap,
        "suffix_weight": suffix_weight,
        "suffix_strict_geodesic": True,
        "suffix_euclidean_fallback": False,
    }
    event["idea1_config"] = dict(idea1_config or {})
    if isinstance(episode_identity, dict):
        event["episode_identity"] = {
            "dataset_manifest_sha256": event.get("dataset_manifest_sha256"),
            **episode_identity,
        }
        event["episode_identity_status"] = "complete"
    else:
        event["episode_identity"] = None
        event["episode_identity_status"] = "source_ids_not_in_frozen_manifest"
    if error is not None:
        event["error"] = {"type": type(error).__name__, "message": str(error)}
    return event


def finalize_dispatch_event(
    decision,
    dispatch_metadata,
    *,
    cooldown_event,
    refresh_applied,
    realized_active_position,
    realized_raw_rank,
    refresh_error=None,
):
    """Attach the post-cooldown, post-refresh outcome before JSONL persistence."""
    event = dict(decision)
    metadata = dict(dispatch_metadata)
    event["planning_schema_version"] = event.get("schema_version")
    event["schema_version"] = "scsp_runtime_dispatch_v1"
    stale_suffix_dispatch = int(
        bool(refresh_applied)
        and metadata.get("source") == "suffix"
        and metadata.get("dispatch_version_gate", {}).get("would_invalidate", True)
    )
    metadata["stale_suffix_dispatch"] = stale_suffix_dispatch
    metadata["applied"] = bool(refresh_applied)
    metadata["realized_raw_rank"] = realized_raw_rank
    if refresh_error is not None:
        metadata["refresh_error"] = str(refresh_error)
    if stale_suffix_dispatch:
        raise RuntimeError("SCSP attempted a stale suffix dispatch")
    event["mode"] = metadata["mode"]
    event["shadow_only"] = metadata["mode"] == "shadow"
    event["proposed_order_source"] = metadata["source"]
    event["dispatch_order_source"] = metadata["source"] if refresh_applied else "not_applied"
    event["dispatch"] = metadata
    event["cooldown"] = cooldown_event
    event["refresh_applied"] = bool(refresh_applied)
    event["realized_active_position"] = (
        None
        if realized_active_position is None
        else [float(value) for value in realized_active_position]
    )
    selected = dict(event.get("selected_raw_ranks", {}))
    selected["dispatched"] = realized_raw_rank if refresh_applied else None
    event["selected_raw_ranks"] = selected
    if metadata.get("reason") == "invalid_external_order":
        event["computed_status"] = event.get("status")
        event["status"] = "fallback"
        event["fallback_reason"] = FallbackReason.PLANNER_ERROR.value
        event["failure_stage"] = "external_order_validation"
    if not refresh_applied:
        event.setdefault("computed_status", event.get("status"))
        if event.get("status") != "fallback":
            event["status"] = "fallback"
            event["fallback_reason"] = FallbackReason.PLANNER_ERROR.value
            if metadata.get("reason") != "invalid_external_order":
                event["failure_stage"] = "atomic_refresh"
        if refresh_error is not None:
            event["refresh_failure"] = str(refresh_error)
    return event


def finite_metric_summary(subtasks, metric):
    """Summarize finite values while auditing nullable unreachable distances."""
    values = []
    missing_count = 0
    for subtask in subtasks:
        raw_value = subtask.get(metric)
        if raw_value is None:
            if (
                metric != DISTANCE_TO_GOAL_KEY
                or subtask.get(DISTANCE_TO_GOAL_STATUS_KEY) != SHORTEST_PATH_UNREACHABLE
            ):
                raise RuntimeError(f"unexpected null {metric}")
            missing_count += 1
            continue
        value = float(raw_value)
        if not np.isfinite(value):
            raise RuntimeError(f"non-finite {metric}: {value!r}")
        values.append(value)
    return {
        "mean": None if not values else round(float(np.mean(values)), 4),
        "median": None if not values else round(float(np.median(values)), 4),
        "finite_count": len(values),
        "unreachable_count": missing_count,
    }


def write_timing_rows(rows, results_dir):
    if not rows:
        return
    csv_path = Path(results_dir) / "timing_log.csv"
    temporary = csv_path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, csv_path)


def print_timing_summary(rows, results_dir, save_csv=True):
    """Print per-step timing summary and write CSV."""
    if not rows:
        return
    if save_csv:
        write_timing_rows(rows, results_dir)
        csv_path = os.path.join(results_dir, "timing_log.csv")
        print(f"\n[Timing] Saved {len(rows)} rows → {csv_path}")
    totals = np.array([r["dt_total_ms"] for r in rows])
    push_ms = np.array([r["dt_push_ms"] for r in rows])
    goals_ms = np.array([r["dt_goals_ms"] for r in rows])
    act_ms = np.array([r["dt_act_ms"] for r in rows])
    upgrades = np.array([r["is_upgrade_step"] for r in rows], dtype=bool)
    print("\n" + "=" * 62)
    print(" Online Navigation Timing Report (Phase 4 OnlineESProvider)")
    print("=" * 62)
    total_wall = totals.sum() / 1000.0
    n = len(totals)
    print(f"  Total steps      : {n}")
    print(f"  Total wall time  : {total_wall:.1f} s ({total_wall / 60:.1f} min)")
    print(f"  Overall FPS      : {n / total_wall:.3f}")
    print(f"  Mean step latency: {totals.mean():.0f} ms")
    print(f"  Median step lat  : {np.median(totals):.0f} ms")
    print(f"  Max  step lat    : {totals.max():.0f} ms")
    print()
    n_upg = upgrades.sum()
    n_norm = (~upgrades).sum()
    print(
        f"  Upgrade steps  : {n_upg} ({100 * n_upg / n:.0f}%)  [every 10 steps — snapshot reload]"
    )
    if n_upg:
        upg_t = totals[upgrades]
        print(
            f"    Mean: {upg_t.mean():.0f} ms   Median: {np.median(upg_t):.0f} ms   Max: {upg_t.max():.0f} ms"
        )
    print(f"  Normal  steps  : {n_norm} ({100 * n_norm / n:.0f}%)")
    if n_norm:
        norm_t = totals[~upgrades]
        print(
            f"    Mean: {norm_t.mean():.0f} ms   Median: {np.median(norm_t):.0f} ms   Max: {norm_t.max():.0f} ms"
        )
    print()
    print("  Step latency breakdown (mean across ALL steps):")
    print(f"    push_frame IPC   : {push_ms.mean():.1f} ms")
    print(f"    get_all_goals    : {goals_ms.mean():.1f} ms  (non-zero only at upgrade steps)")
    print(f"    agent.act        : {act_ms.mean():.1f} ms")
    print(f"    other (obs/misc) : {(totals - push_ms - goals_ms - act_ms).mean():.1f} ms")
    print()
    print("  Online standard compliance:")
    print(f"    {'Threshold':<20} {'All steps':>10} {'Normal only':>12}")
    print(f"    {'-' * 20} {'-' * 10} {'-' * 12}")
    for label, budget_ms in TIMING_THRESHOLDS:
        pct_all = 100.0 * (totals <= budget_ms).sum() / n
        pct_norm = 100.0 * ((totals[~upgrades] <= budget_ms).sum() / n_norm if n_norm else 0)
        flag = "✓" if pct_all > 90 else "△" if pct_all > 50 else "✗"
        print(f"    {label:<20} {pct_all:>9.1f}%  {pct_norm:>11.1f}%  {flag}")
    print()
    norm_ok = (totals[~upgrades] <= 1000).sum() / n_norm * 100 if n_norm else 0
    upg_ok = (totals[upgrades] <= 1000).sum() / n_upg * 100 if n_upg else 0
    print("  Verdict:")
    if n_upg and upg_ok < 10:
        print(
            "  ✗ Snapshot upgrade steps (~{:.0f}s avg) VIOLATE all real-time".format(
                totals[upgrades].mean() / 1000
            )
        )
        print("    thresholds. Async preloading needed for real deployment.")
    if norm_ok > 90:
        print(f"  ✓ Normal (non-upgrade) steps meet ≥1 FPS standard ({norm_ok:.0f}% ≤ 1000 ms).")
    print("=" * 62 + "\n")


def record_planning_decision(
    event, *, output_path=None, current_task_hint=0, serialize=dumps_json_strict
):
    """Write a completed dispatch while retaining its truth if serialization fails."""
    event["recorded_to_jsonl"] = False
    if output_path is not None:
        try:
            persisted_event = dict(event)
            persisted_event["recorded_to_jsonl"] = True
            with Path(output_path).open("a", encoding="utf-8") as stream:
                stream.write(serialize(persisted_event) + "\n")
            event = persisted_event
        except Exception as exc:
            original_event = dict(event)
            event = dict(original_event)
            event.update(
                {
                    "recorded_to_jsonl": False,
                    "recording_error": f"{type(exc).__name__}: {exc}",
                }
            )
            diagnostic = {
                "schema_version": original_event.get(
                    "schema_version", SHADOW_SCHEMA_VERSION
                ),
                "phase": str(original_event.get("phase", "unknown")),
                "arm": str(original_event.get("arm", "unknown")),
                "mode": str(original_event.get("mode", "shadow")),
                "shadow_only": bool(original_event.get("shadow_only", True)),
                "status": "telemetry_error",
                "computed_status": original_event.get("status"),
                "dispatch_order_source": str(
                    original_event.get("dispatch_order_source", "not_applied")
                ),
                "current_task": int(
                    original_event.get("current_task", current_task_hint)
                ),
                "fallback_reason": FallbackReason.PLANNER_ERROR.value,
                "failure_stage": "serialization",
                "profile_sha256": original_event.get("profile_sha256"),
                "dataset_manifest_sha256": original_event.get(
                    "dataset_manifest_sha256"
                ),
                "code_sha": original_event.get("code_sha"),
                "planner_config": original_event.get("planner_config"),
                "idea1_config": original_event.get("idea1_config"),
                "episode_identity": original_event.get("episode_identity"),
                "episode_identity_status": original_event.get(
                    "episode_identity_status"
                ),
                "selected_raw_ranks": dict(
                    original_event.get("selected_raw_ranks", {})
                ),
                "recording_error": f"{type(exc).__name__}: {exc}",
            }
            try:
                with Path(output_path).open("a", encoding="utf-8") as stream:
                    diagnostic["recorded_to_jsonl"] = True
                    stream.write(serialize(diagnostic) + "\n")
                event["recorded_to_jsonl"] = True
                event["recording_degraded"] = True
            except Exception as fallback_exc:
                event["fallback_recording_error"] = (
                    f"{type(fallback_exc).__name__}: {fallback_exc}"
                )
            print(
                "[SCSP_SHADOW] telemetry write warning: " f'{event["recording_error"]}',
                flush=True,
            )
    selected = event.get("selected_raw_ranks", {})
    fallback_reason = event.get("fallback_reason")
    if fallback_reason is None and event.get("suffix_plan") is not None:
        fallback_reason = event["suffix_plan"].fallback_reason.value
    print(
        "[SCSP_RUNTIME] "
        + json.dumps(
            {
                "phase": event["phase"],
                "current_task": event["current_task"],
                "status": event["status"],
                "fallback_reason": fallback_reason,
                "legacy_selected": selected.get("legacy"),
                "idea1_selected": selected.get("idea1"),
                "suffix_selected": selected.get("suffix"),
                "dispatched_selected": selected.get("dispatched"),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return event
