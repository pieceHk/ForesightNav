"""Strict, analysis-ready JSON telemetry for SCSP decisions."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

from .scsp_contracts import (
    BatchValidationStatus,
    CandidateBatch,
    CandidateRecord,
    EpisodeIdentity,
    ExperimentArm,
    GeodesicDistance,
    GeodesicStatus,
    SCSPContractError,
    SuffixPlan,
    validate_candidate_batch,
)


SCHEMA_VERSION = "scsp_telemetry_v1"
SHADOW_SCHEMA_VERSION = "scsp_shadow_decision_v1"


class SCSPTelemetryError(ValueError):
    """Raised when telemetry would be ambiguous or invalid JSON."""


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SCSPTelemetryError(f"{name} must be a non-empty string")
    return value


def _hex_digest(value: Any, name: str, length: int) -> str:
    digest = _nonempty_string(value, name).lower()
    if len(digest) != length or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise SCSPTelemetryError(
            f"{name} must be a {length}-character hexadecimal digest"
        )
    return digest


def _raw_rank(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise SCSPTelemetryError(f"{name} must be an integer, got {value!r}")
    result = int(value)
    if result < 0:
        raise SCSPTelemetryError(f"{name} must be >= 0, got {result!r}")
    return result


def _normalise_idea1_order(value: Any) -> Tuple[Tuple[int, ...], ...]:
    if isinstance(value, (str, bytes)):
        raise SCSPTelemetryError("idea1_order must be task-aligned rank sequences")
    try:
        task_orders = tuple(value)
    except TypeError as exc:
        raise SCSPTelemetryError(
            "idea1_order must be task-aligned rank sequences"
        ) from exc

    normalised = []
    for task_idx, task_order in enumerate(task_orders):
        if isinstance(task_order, (str, bytes)):
            raise SCSPTelemetryError(f"idea1_order[{task_idx}] must be a sequence")
        try:
            ranks = tuple(
                _raw_rank(rank, f"idea1_order[{task_idx}][{rank_idx}]")
                for rank_idx, rank in enumerate(task_order)
            )
        except TypeError as exc:
            raise SCSPTelemetryError(
                f"idea1_order[{task_idx}] must be a sequence"
            ) from exc
        if len(ranks) != len(set(ranks)):
            raise SCSPTelemetryError(
                f"idea1_order[{task_idx}] contains duplicate raw ranks"
            )
        normalised.append(ranks)
    return tuple(normalised)


def _candidate_records(
    batch: CandidateBatch,
) -> Tuple[Tuple[CandidateRecord, ...], ...]:
    return tuple(records for _, records in batch.records_by_task)


def _find_selected_candidate(
    batch: CandidateBatch, plan: SuffixPlan
) -> Optional[CandidateRecord]:
    if plan.selected_raw_rank is None:
        return None
    try:
        records = batch.records_for_task(plan.current_task)
    except KeyError:
        return None
    for record in records:
        if record.raw_rank == plan.selected_raw_rank:
            return record
    return None


@dataclass(frozen=True)
class SCSPTelemetry:
    """One immutable planner decision and all inputs needed to audit it."""

    episode_identity: EpisodeIdentity
    candidate_batch: CandidateBatch
    suffix_plan: SuffixPlan
    arm: ExperimentArm
    profile_sha256: str
    code_sha: str
    idea1_order: Tuple[Tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.episode_identity, EpisodeIdentity):
            raise SCSPTelemetryError("episode_identity must be an EpisodeIdentity")
        if not isinstance(self.candidate_batch, CandidateBatch):
            raise SCSPTelemetryError("candidate_batch must be a CandidateBatch")
        if not isinstance(self.suffix_plan, SuffixPlan):
            raise SCSPTelemetryError("suffix_plan must be a SuffixPlan")
        try:
            arm = (
                self.arm
                if isinstance(self.arm, ExperimentArm)
                else ExperimentArm(self.arm)
            )
        except (TypeError, ValueError) as exc:
            allowed = ", ".join(repr(item.value) for item in ExperimentArm)
            raise SCSPTelemetryError(
                f"arm must be one of {allowed}, got {self.arm!r}"
            ) from exc
        object.__setattr__(self, "arm", arm)
        object.__setattr__(
            self,
            "profile_sha256",
            _hex_digest(self.profile_sha256, "profile_sha256", 64),
        )
        object.__setattr__(self, "code_sha", _hex_digest(self.code_sha, "code_sha", 40))

        idea1_order = _normalise_idea1_order(self.idea1_order)
        object.__setattr__(self, "idea1_order", idea1_order)
        batch = self.candidate_batch
        plan = self.suffix_plan
        identity = self.episode_identity

        if identity.scene_id != batch.scene_id:
            raise SCSPTelemetryError(
                "episode identity scene_id does not match candidate batch"
            )
        if identity.runtime_episode_id != batch.episode_id:
            raise SCSPTelemetryError(
                "episode identity runtime_episode_id does not match candidate batch"
            )
        if identity.ordered_categories != batch.ordered_categories:
            raise SCSPTelemetryError(
                "episode identity ordered_categories do not match candidate batch"
            )

        if plan.input_token != batch.version_token:
            raise SCSPTelemetryError(
                "suffix plan version token does not match candidate batch"
            )
        if plan.current_task >= len(batch.ordered_categories):
            raise SCSPTelemetryError(
                f"current_task {plan.current_task} is outside ordered_categories"
            )
        if len(idea1_order) != len(batch.ordered_categories):
            raise SCSPTelemetryError(
                "Idea-1 order/task mismatch: "
                f"orders={len(idea1_order)} tasks={len(batch.ordered_categories)}"
            )

        if batch.validation_status is BatchValidationStatus.VALID:
            try:
                validate_candidate_batch(batch)
            except SCSPContractError as exc:
                raise SCSPTelemetryError(str(exc)) from exc
            records_by_task = _candidate_records(batch)
            for task_idx, (ordered_ranks, records) in enumerate(
                zip(idea1_order, records_by_task)
            ):
                record_ranks = tuple(record.raw_rank for record in records)
                if len(ordered_ranks) != len(record_ranks) or set(ordered_ranks) != set(
                    record_ranks
                ):
                    raise SCSPTelemetryError(
                        f"Idea-1 order for task {task_idx} must contain every raw rank once"
                    )

            current_ranks = {
                record.raw_rank for record in records_by_task[plan.current_task]
            }
            cost_ranks = {item.raw_rank for item in plan.suffix_costs}
            if not cost_ranks <= current_ranks:
                raise SCSPTelemetryError(
                    "suffix costs contain a candidate outside the current task"
                )
            for matrix_index, matrix in enumerate(plan.strict_geodesic_matrices):
                if matrix.target_task_idx >= len(records_by_task):
                    raise SCSPTelemetryError(
                        f"geodesic matrix {matrix_index} targets a task outside the batch"
                    )
                source_identities = {
                    record.identity()
                    for record in records_by_task[matrix.source_task_idx]
                }
                target_identities = {
                    record.identity()
                    for record in records_by_task[matrix.target_task_idx]
                }
                if not set(matrix.source_candidate_identities) <= source_identities:
                    raise SCSPTelemetryError(
                        f"geodesic matrix {matrix_index} has unknown source candidate labels"
                    )
                if not set(matrix.target_candidate_identities) <= target_identities:
                    raise SCSPTelemetryError(
                        f"geodesic matrix {matrix_index} has unknown target candidate labels"
                    )
            if (
                plan.selected_raw_rank is not None
                and _find_selected_candidate(batch, plan) is None
            ):
                raise SCSPTelemetryError(
                    "selected_raw_rank does not identify a current-task candidate"
                )
            if plan.selected_raw_rank is not None:
                idea1_head = idea1_order[plan.current_task][0]
                expected_intervention = plan.selected_raw_rank != idea1_head
                if plan.intervention != expected_intervention:
                    raise SCSPTelemetryError(
                        "intervention must equal whether selected_raw_rank differs "
                        "from the current task's Idea-1 head"
                    )
        elif plan.fallback_reason.value == "none":
            raise SCSPTelemetryError(
                "an invalid candidate batch requires an explicit fallback reason"
            )

    @property
    def candidate_counts(self) -> Tuple[int, ...]:
        return tuple(
            len(records) for records in _candidate_records(self.candidate_batch)
        )

    @property
    def geodesic_status(
        self,
    ) -> Tuple[Tuple[Tuple[str, ...], ...], ...]:
        return tuple(
            tuple(
                tuple(distance.status.value for distance in row)
                for row in matrix.distances
            )
            for matrix in self.suffix_plan.strict_geodesic_matrices
        )

    @property
    def selected_candidate(self) -> Optional[CandidateRecord]:
        return _find_selected_candidate(self.candidate_batch, self.suffix_plan)

    def to_dict(self) -> dict[str, Any]:
        """Return the stable, flat-on-purpose telemetry schema."""

        batch = self.candidate_batch
        plan = self.suffix_plan
        matrices = plan.strict_geodesic_matrices
        return {
            "schema_version": SCHEMA_VERSION,
            "episode_identity": self.episode_identity,
            "scene_id": self.episode_identity.scene_id,
            "runtime_episode_id": self.episode_identity.runtime_episode_id,
            "source_episode_ids": self.episode_identity.source_episode_ids,
            "ordered_categories": self.episode_identity.ordered_categories,
            "dataset_manifest_sha256": self.episode_identity.dataset_manifest_sha256,
            "arm": self.arm,
            "profile_sha256": self.profile_sha256,
            "code_sha": self.code_sha,
            "version_token": batch.version_token,
            "candidate_counts": self.candidate_counts,
            "candidate_records": _candidate_records(batch),
            "batch_validation_status": batch.validation_status,
            "batch_validation_reason": batch.validation_reason,
            "geodesic_status": self.geodesic_status,
            "strict_geodesic_matrices": matrices,
            "idea1_order": self.idea1_order,
            "suffix_costs": plan.suffix_costs,
            "current_task": plan.current_task,
            "selected_raw_rank": plan.selected_raw_rank,
            "selected_candidate": self.selected_candidate,
            "intervention": plan.intervention,
            "intervention_reason": plan.intervention_reason,
            "fallback_reason": plan.fallback_reason,
            "planner_latency_ms": plan.compute_time_ms,
            "planner_complexity": {
                "matrix_layers": len(matrices),
                "layer_shapes": tuple(
                    (
                        len(matrix.source_candidate_identities),
                        len(matrix.target_candidate_identities),
                    )
                    for matrix in matrices
                ),
                "geodesic_edges": sum(
                    len(row) for matrix in matrices for row in matrix.distances
                ),
                "current_candidates": len(plan.suffix_costs),
            },
        }


def build_scsp_shadow_decision(
    *,
    phase: str,
    arm: str,
    candidate_batch: CandidateBatch,
    suffix_plan: SuffixPlan,
    legacy_order: Any,
    idea1_order: Any,
    suffix_order: Any,
    idea1_candidates: Any,
    rerank_diagnostics: Any,
    version_gate: Mapping[str, Any],
    origin_position: Any,
    timing_ms: Mapping[str, Any],
    distance_cache: Mapping[str, Any],
    candidate_cap: int,
    suffix_weight: float,
    idea1_config: Mapping[str, Any],
    provenance: Mapping[str, Any],
    episode_identity: Optional[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build one auditable shadow decision without implying active dispatch.

    The runtime intentionally records all three orders while declaring legacy as
    the sole dispatch source.  Candidate utilities and strict-geodesic matrices
    remain in the event so the suffix decision can be recomputed offline.
    """

    phase_name = _nonempty_string(phase, "phase")
    arm_name = _nonempty_string(arm, "arm")
    if not isinstance(candidate_batch, CandidateBatch):
        raise SCSPTelemetryError("candidate_batch must be a CandidateBatch")
    try:
        validate_candidate_batch(candidate_batch)
    except SCSPContractError as exc:
        raise SCSPTelemetryError(str(exc)) from exc
    if not isinstance(suffix_plan, SuffixPlan):
        raise SCSPTelemetryError("suffix_plan must be a SuffixPlan")
    if suffix_plan.input_token != candidate_batch.version_token:
        raise SCSPTelemetryError(
            "suffix plan version token does not match candidate batch"
        )
    if not isinstance(version_gate, Mapping) or version_gate.get(
        "would_invalidate"
    ) is not False:
        raise SCSPTelemetryError(
            "a planned shadow decision requires a valid version gate"
        )
    cap = _raw_rank(candidate_cap, "candidate_cap")
    if not 1 <= cap <= 5:
        raise SCSPTelemetryError("candidate_cap must be in [1, 5]")
    if isinstance(suffix_weight, bool) or not isinstance(suffix_weight, Real):
        raise SCSPTelemetryError("suffix_weight must be numeric")
    weight = float(suffix_weight)
    if not math.isfinite(weight) or weight < 0.0:
        raise SCSPTelemetryError("suffix_weight must be finite and non-negative")

    if not isinstance(provenance, Mapping):
        raise SCSPTelemetryError("provenance must be a mapping")
    profile_sha256 = _hex_digest(
        provenance.get("profile_sha256"), "profile_sha256", 64
    )
    dataset_manifest_sha256 = _hex_digest(
        provenance.get("dataset_manifest_sha256"),
        "dataset_manifest_sha256",
        64,
    )
    code_sha = _hex_digest(provenance.get("code_sha"), "code_sha", 40)
    if episode_identity is None:
        complete_episode_identity = None
        source_episode_ids = None
        episode_identity_status = "source_ids_not_in_frozen_manifest"
    else:
        if not isinstance(episode_identity, Mapping):
            raise SCSPTelemetryError("episode_identity must be a mapping")
        scene_id = _nonempty_string(
            episode_identity.get("scene_id"), "episode_identity.scene_id"
        )
        runtime_episode_id = _nonempty_string(
            episode_identity.get("runtime_episode_id"),
            "episode_identity.runtime_episode_id",
        )
        try:
            source_episode_ids = tuple(
                _nonempty_string(value, "episode_identity.source_episode_ids")
                for value in episode_identity.get("source_episode_ids", ())
            )
            identity_categories = tuple(
                _nonempty_string(value, "episode_identity.ordered_categories")
                for value in episode_identity.get("ordered_categories", ())
            )
        except TypeError as exc:
            raise SCSPTelemetryError(
                "episode_identity sequences are malformed"
            ) from exc
        if len(source_episode_ids) != 3 or len(identity_categories) != 3:
            raise SCSPTelemetryError(
                "a frozen episode identity requires exactly three subtasks"
            )
        if (
            scene_id != candidate_batch.scene_id
            or runtime_episode_id != candidate_batch.episode_id
            or identity_categories != candidate_batch.ordered_categories
        ):
            raise SCSPTelemetryError(
                "frozen episode identity does not match candidate batch"
            )
        complete_episode_identity = {
            "dataset_manifest_sha256": dataset_manifest_sha256,
            "scene_id": scene_id,
            "runtime_episode_id": runtime_episode_id,
            "source_episode_ids": source_episode_ids,
            "ordered_categories": identity_categories,
        }
        episode_identity_status = "complete"

    component_names = (
        "semantic",
        "support",
        "recurrence",
        "travel",
        "novelty",
        "revisit",
        "failure",
    )
    config_names = component_names + ("switch_margin", "match_radius_m")
    if not isinstance(idea1_config, Mapping) or set(idea1_config) != set(
        config_names
    ):
        raise SCSPTelemetryError("idea1_config has incomplete utility fields")
    normalised_idea1_config = {}
    for name in config_names:
        value = idea1_config[name]
        if isinstance(value, bool) or not isinstance(value, Real):
            raise SCSPTelemetryError(f"idea1_config.{name} must be numeric")
        number = float(value)
        if not math.isfinite(number) or number < 0.0:
            raise SCSPTelemetryError(
                f"idea1_config.{name} must be finite and non-negative"
            )
        normalised_idea1_config[name] = number

    task_count = len(candidate_batch.ordered_categories)
    current_task = suffix_plan.current_task
    if current_task >= task_count:
        raise SCSPTelemetryError("current_task is outside candidate batch")
    legacy = _normalise_idea1_order(legacy_order)
    idea1 = _normalise_idea1_order(idea1_order)
    suffix = _normalise_idea1_order(suffix_order)
    if not (len(legacy) == len(idea1) == len(suffix) == task_count):
        raise SCSPTelemetryError("shadow orders must align with every task")

    try:
        idea1_groups = tuple(idea1_candidates)
    except TypeError as exc:
        raise SCSPTelemetryError("idea1_candidates must be task-indexed") from exc
    if len(idea1_groups) != task_count:
        raise SCSPTelemetryError("idea1_candidates must include every task")
    for expected_task, group in enumerate(idea1_groups):
        try:
            task_idx, candidates = group
            candidates = tuple(candidates)
        except (TypeError, ValueError) as exc:
            raise SCSPTelemetryError(
                f"idea1_candidates[{expected_task}] must be a task pair"
            ) from exc
        if task_idx != expected_task:
            raise SCSPTelemetryError("idea1_candidates task indices are misaligned")
        candidate_ranks = []
        for candidate_idx, candidate in enumerate(candidates):
            if not isinstance(candidate, Mapping):
                raise SCSPTelemetryError("Idea-1 candidate telemetry must be a mapping")
            rank = _raw_rank(
                candidate.get("raw_rank"),
                f"idea1_candidates[{task_idx}][{candidate_idx}].raw_rank",
            )
            utility = candidate.get("utility")
            if isinstance(utility, bool) or not isinstance(utility, Real):
                raise SCSPTelemetryError("Idea-1 utility must be numeric")
            utility = float(utility)
            if not math.isfinite(utility):
                raise SCSPTelemetryError("Idea-1 utility must be finite")
            components = candidate.get("utility_components")
            if not isinstance(components, Mapping) or set(components) != set(
                component_names
            ):
                raise SCSPTelemetryError(
                    "Idea-1 utility_components are incomplete"
                )
            values = {}
            for name in component_names:
                value = components[name]
                if isinstance(value, bool) or not isinstance(value, Real):
                    raise SCSPTelemetryError(
                        f"Idea-1 utility component {name} must be numeric"
                    )
                number = float(value)
                if not math.isfinite(number) or not 0.0 <= number <= 1.0:
                    raise SCSPTelemetryError(
                        f"Idea-1 utility component {name} must be in [0, 1]"
                    )
                values[name] = number
            recomputed_utility = (
                normalised_idea1_config["semantic"] * values["semantic"]
                + normalised_idea1_config["support"] * values["support"]
                + normalised_idea1_config["recurrence"] * values["recurrence"]
                + normalised_idea1_config["novelty"] * values["novelty"]
                - normalised_idea1_config["travel"] * values["travel"]
                - normalised_idea1_config["revisit"] * values["revisit"]
                - normalised_idea1_config["failure"] * values["failure"]
            )
            if not math.isclose(
                utility, recomputed_utility, rel_tol=1e-9, abs_tol=1e-9
            ):
                raise SCSPTelemetryError(
                    "Idea-1 utility does not match its recorded components"
                )
            candidate_ranks.append(rank)
        if tuple(candidate_ranks) != idea1[expected_task]:
            raise SCSPTelemetryError(
                f"Idea-1 candidate telemetry/order mismatch for task {expected_task}"
            )

    for task_idx, records in enumerate(_candidate_records(candidate_batch)):
        expected = {record.raw_rank for record in records}
        for label, order in (
            ("legacy", legacy[task_idx]),
            ("idea1", idea1[task_idx]),
            ("suffix", suffix[task_idx]),
        ):
            if len(order) != len(expected) or set(order) != expected:
                raise SCSPTelemetryError(
                    f"{label} order for task {task_idx} must contain every raw rank once"
                )
        if task_idx != current_task and suffix[task_idx] != idea1[task_idx]:
            raise SCSPTelemetryError("suffix may only reorder the current task")
    if suffix_plan.selected_raw_rank != (
        suffix[current_task][0] if suffix[current_task] else None
    ):
        raise SCSPTelemetryError(
            "suffix selected_raw_rank must match the recorded suffix order"
        )
    current_ranks = {
        record.raw_rank for record in candidate_batch.records_for_task(current_task)
    }
    if not {
        item.raw_rank for item in suffix_plan.suffix_costs
    } <= current_ranks:
        raise SCSPTelemetryError("suffix costs reference an unknown current candidate")
    for matrix_idx, matrix in enumerate(suffix_plan.strict_geodesic_matrices):
        if matrix.target_task_idx >= task_count:
            raise SCSPTelemetryError(
                f"geodesic matrix {matrix_idx} targets a task outside the batch"
            )
        source_identities = {
            record.identity()
            for record in candidate_batch.records_for_task(matrix.source_task_idx)
        }
        target_identities = {
            record.identity()
            for record in candidate_batch.records_for_task(matrix.target_task_idx)
        }
        if not set(matrix.source_candidate_identities) <= source_identities:
            raise SCSPTelemetryError("geodesic matrix has unknown source candidates")
        if not set(matrix.target_candidate_identities) <= target_identities:
            raise SCSPTelemetryError("geodesic matrix has unknown target candidates")
        if (
            len(matrix.source_candidate_identities) > cap
            or len(matrix.target_candidate_identities) > cap
        ):
            raise SCSPTelemetryError("geodesic matrix exceeds candidate_cap")

    expected_intervention = (
        suffix[current_task][0] if suffix[current_task] else None
    ) != (
        idea1[current_task][0] if idea1[current_task] else None
    )
    if suffix_plan.intervention != expected_intervention:
        raise SCSPTelemetryError(
            "suffix intervention does not match the Idea-1/suffix heads"
        )

    def selected(order: Tuple[Tuple[int, ...], ...]) -> Optional[int]:
        return order[current_task][0] if order[current_task] else None

    legacy_selected = selected(legacy)
    idea1_selected = selected(idea1)
    suffix_selected = selected(suffix)
    return {
        "schema_version": SHADOW_SCHEMA_VERSION,
        "phase": phase_name,
        "arm": arm_name,
        "profile_sha256": profile_sha256,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "code_sha": code_sha,
        "scene_id": candidate_batch.scene_id,
        "runtime_episode_id": candidate_batch.episode_id,
        "ordered_categories": candidate_batch.ordered_categories,
        "source_episode_ids": source_episode_ids,
        "episode_identity": complete_episode_identity,
        "episode_identity_status": episode_identity_status,
        "mode": "shadow",
        "shadow_only": True,
        "dispatch_order_source": "legacy",
        "legacy_order_stage": "post_projection_pre_cooldown",
        "current_task": current_task,
        "origin_position": origin_position,
        "version_token": candidate_batch.version_token,
        "version_gate": dict(version_gate),
        "candidate_batch": candidate_batch,
        "legacy_order": legacy,
        "idea1_order": idea1,
        "suffix_order": suffix,
        "idea1_candidates": idea1_candidates,
        "rerank_diagnostics": rerank_diagnostics,
        "suffix_plan": suffix_plan,
        "planner_config": {
            "candidate_cap": cap,
            "suffix_weight": weight,
            "suffix_strict_geodesic": True,
            "suffix_euclidean_fallback": False,
        },
        "idea1_config": normalised_idea1_config,
        "idea1_travel_contract": {
            "primary": "strict_habitat_geodesic",
            "fallback": "legacy_xz_euclidean",
            "fallback_is_explicit_in": (
                "rerank_diagnostics.geodesic_fallbacks"
            ),
        },
        "selected_raw_ranks": {
            "legacy": legacy_selected,
            "idea1": idea1_selected,
            "suffix": suffix_selected,
        },
        "changed": {
            "idea1_from_legacy": idea1_selected != legacy_selected,
            "suffix_from_idea1": suffix_selected != idea1_selected,
            "suffix_from_legacy": suffix_selected != legacy_selected,
        },
        "timing_ms": dict(timing_ms),
        "distance_cache": dict(distance_cache),
    }


def _json_value(value: Any, path: str) -> Any:
    if isinstance(value, GeodesicDistance):
        distance = (
            value.distance_m if value.status is GeodesicStatus.AVAILABLE else None
        )
        return {"status": value.status.value, "distance_m": distance}
    if isinstance(value, Enum):
        return _json_value(value.value, path)
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _json_value(getattr(value, field.name), f"{path}.{field.name}")
            for field in fields(value)
        }
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        if not math.isfinite(number):
            raise SCSPTelemetryError(f"{path} must be finite, got {number!r}")
        return number
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise SCSPTelemetryError(
                    f"{path} contains non-string JSON object key {key!r}"
                )
            result[key] = _json_value(item, f"{path}.{key}")
        return result
    if isinstance(value, (tuple, list)):
        return [
            _json_value(item, f"{path}[{index}]") for index, item in enumerate(value)
        ]
    raise SCSPTelemetryError(
        f"{path} contains unsupported JSON value {type(value).__name__}"
    )


def to_jsonable(payload: Any) -> Any:
    """Convert supported contracts to JSON values, rejecting ambiguity."""

    if isinstance(payload, SCSPTelemetry):
        payload = payload.to_dict()
    return _json_value(payload, "$")


def dumps_json_strict(payload: Any, *, indent: Optional[int] = None) -> str:
    """Serialize without ever emitting the non-standard NaN/Infinity literals."""

    return json.dumps(
        to_jsonable(payload),
        allow_nan=False,
        indent=indent,
        sort_keys=True,
    )


def write_json_strict(path: Path, payload: Any) -> None:
    """Atomically write strict JSON, leaving no partial file on failure."""

    output = Path(path)
    temporary = output.with_suffix(output.suffix + ".tmp")
    try:
        temporary.write_text(
            dumps_json_strict(payload, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


__all__ = [
    "SCHEMA_VERSION",
    "SHADOW_SCHEMA_VERSION",
    "SCSPTelemetry",
    "SCSPTelemetryError",
    "build_scsp_shadow_decision",
    "dumps_json_strict",
    "to_jsonable",
    "write_json_strict",
]
