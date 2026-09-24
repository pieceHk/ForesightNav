import json
import math
from dataclasses import FrozenInstanceError, replace

import pytest

from tdonav_navigation.scsp_contracts import (
    BatchValidationStatus,
    CandidateBatch,
    CandidateRecord,
    CandidateSuffixCost,
    EpisodeIdentity,
    ExperimentArm,
    FallbackReason,
    GeodesicDistance,
    GeodesicMatrix,
    InterventionReason,
    SuffixPlan,
    VersionToken,
)
from tdonav_navigation.scsp_telemetry import (
    SCHEMA_VERSION,
    SCSPTelemetry,
    SCSPTelemetryError,
    dumps_json_strict,
    to_jsonable,
    write_json_strict,
)


def _record(task_idx, category, position, raw_rank):
    return CandidateRecord(
        task_idx=task_idx,
        category=category,
        position=position,
        raw_rank=raw_rank,
        semantic_score=0.9 - raw_rank * 0.1,
        support_count=4,
        support_score=0.7,
        map_revision=9,
    )


def _batch(status=BatchValidationStatus.VALID, reason=None):
    records = (
        (
            0,
            (
                _record(0, "chair", (1.0, 0.0, 2.0), 0),
                _record(0, "chair", (2.0, 0.0, 3.0), 1),
            ),
        ),
        (1, (_record(1, "table", (4.0, 0.0, 5.0), 0),)),
        (2, (_record(2, "plant", (6.0, 0.0, 7.0), 0),)),
    )
    positions = tuple(
        (task_idx, tuple(record.position for record in task_records))
        for task_idx, task_records in records
    )
    return CandidateBatch(
        scene_id="scene-a",
        episode_id="runtime-22",
        map_revision=9,
        last_fused_keyframes=9,
        query_serial=3,
        ordered_categories=("chair", "table", "plant"),
        positions_by_task=positions,
        records_by_task=records,
        validation_status=status,
        validation_reason=reason,
    )


def _identity(**overrides):
    values = {
        "dataset_manifest_sha256": "a" * 64,
        "scene_id": "scene-a",
        "runtime_episode_id": "runtime-22",
        "source_episode_ids": ("source-a", "source-b", "source-c"),
        "ordered_categories": ("chair", "table", "plant"),
    }
    values.update(overrides)
    return EpisodeIdentity(**values)


def _plan(token=None, **overrides):
    batch = _batch()
    task0 = tuple(record.identity() for record in batch.records_for_task(0))
    task1 = tuple(record.identity() for record in batch.records_for_task(1))
    task2 = tuple(record.identity() for record in batch.records_for_task(2))
    values = {
        "input_token": token or _batch().version_token,
        "current_task": 0,
        "strict_geodesic_matrices": (
            GeodesicMatrix(
                source_task_idx=0,
                target_task_idx=1,
                source_candidate_identities=task0,
                target_candidate_identities=task1,
                distances=(
                    (GeodesicDistance.unreachable(),),
                    (GeodesicDistance.available(1.0),),
                ),
            ),
            GeodesicMatrix(
                source_task_idx=1,
                target_task_idx=2,
                source_candidate_identities=task1,
                target_candidate_identities=task2,
                distances=((GeodesicDistance.unavailable(),),),
            ),
        ),
        "suffix_costs": (
            CandidateSuffixCost(0, GeodesicDistance.unreachable()),
            CandidateSuffixCost(1, GeodesicDistance.available(3.0)),
        ),
        "selected_raw_rank": 1,
        "intervention": False,
        "fallback_reason": FallbackReason.NONE,
        "intervention_reason": InterventionReason.NONE,
        "compute_time_ms": 4.25,
    }
    values.update(overrides)
    return SuffixPlan(**values)


def _telemetry(**overrides):
    batch = overrides.pop("candidate_batch", _batch())
    values = {
        "episode_identity": _identity(),
        "candidate_batch": batch,
        "suffix_plan": _plan(batch.version_token),
        "arm": ExperimentArm.ORDERED_SUFFIX_MVP,
        "profile_sha256": "b" * 64,
        "code_sha": "c" * 40,
        "idea1_order": ((1, 0), (0,), (0,)),
    }
    values.update(overrides)
    return SCSPTelemetry(**values)


def test_complete_telemetry_schema_is_frozen_and_analysis_ready():
    telemetry = _telemetry(profile_sha256="B" * 64, code_sha="C" * 40)
    payload = telemetry.to_dict()
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["scene_id"] == "scene-a"
    assert payload["runtime_episode_id"] == "runtime-22"
    assert payload["source_episode_ids"] == ("source-a", "source-b", "source-c")
    assert payload["ordered_categories"] == ("chair", "table", "plant")
    assert payload["dataset_manifest_sha256"] == "a" * 64
    assert payload["arm"] is ExperimentArm.ORDERED_SUFFIX_MVP
    assert payload["profile_sha256"] == "b" * 64
    assert payload["code_sha"] == "c" * 40
    assert payload["version_token"] == _batch().version_token
    assert payload["candidate_counts"] == (2, 1, 1)
    assert payload["candidate_records"][0][1].raw_rank == 1
    assert payload["geodesic_status"] == (
        (("unreachable",), ("available",)),
        (("unavailable",),),
    )
    assert payload["idea1_order"] == ((1, 0), (0,), (0,))
    assert payload["current_task"] == 0
    assert payload["selected_raw_rank"] == 1
    assert payload["selected_candidate"].raw_rank == 1
    assert payload["intervention"] is False
    assert payload["fallback_reason"] is FallbackReason.NONE
    assert payload["planner_latency_ms"] == 4.25
    assert payload["planner_complexity"]["geodesic_edges"] == 3
    with pytest.raises(FrozenInstanceError):
        telemetry.arm = "baseline"


@pytest.mark.parametrize(
    "field,value",
    [("profile_sha256", "short"), ("code_sha", "g" * 40)],
)
def test_telemetry_rejects_noncanonical_identity_digests(field, value):
    with pytest.raises(SCSPTelemetryError, match="hexadecimal"):
        _telemetry(**{field: value})


def test_strict_json_encodes_unreachable_and_unavailable_as_null_with_status():
    encoded = dumps_json_strict(_telemetry())
    assert "NaN" not in encoded
    assert "Infinity" not in encoded
    payload = json.loads(encoded)

    unreachable = payload["strict_geodesic_matrices"][0]["distances"][0][0]
    unavailable = payload["strict_geodesic_matrices"][1]["distances"][0][0]
    suffix_unreachable = payload["suffix_costs"][0]["cost"]
    assert unreachable == {"distance_m": None, "status": "unreachable"}
    assert unavailable == {"distance_m": None, "status": "unavailable"}
    assert suffix_unreachable == {"distance_m": None, "status": "unreachable"}
    assert payload["geodesic_status"][0][0][0] == "unreachable"


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_strict_json_rejects_nonfinite_values_outside_distance_contract(value):
    with pytest.raises(SCSPTelemetryError, match="finite"):
        dumps_json_strict({"nested": [0.0, {"bad": value}]})


def test_strict_json_requires_string_object_keys_and_supported_values():
    with pytest.raises(SCSPTelemetryError, match="non-string"):
        dumps_json_strict({1: "ambiguous"})
    with pytest.raises(SCSPTelemetryError, match="unsupported"):
        dumps_json_strict({"bad": object()})


def test_distance_contract_is_the_only_nonfinite_json_escape_hatch():
    assert to_jsonable(GeodesicDistance.unreachable()) == {
        "status": "unreachable",
        "distance_m": None,
    }
    with pytest.raises(SCSPTelemetryError):
        to_jsonable({"status": "unreachable", "distance_m": math.inf})


def test_strict_writer_round_trips_and_leaves_no_temporary_file(tmp_path):
    output = tmp_path / "decision.json"
    write_json_strict(output, _telemetry())
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["candidate_counts"] == [2, 1, 1]
    assert payload["episode_identity"]["source_episode_ids"] == [
        "source-a",
        "source-b",
        "source-c",
    ]

    bad_output = tmp_path / "bad.json"
    with pytest.raises(SCSPTelemetryError):
        write_json_strict(bad_output, {"bad": math.nan})
    assert not bad_output.exists()
    assert not (tmp_path / "bad.json.tmp").exists()


def test_telemetry_rejects_version_identity_and_order_mismatches():
    batch = _batch()
    wrong_token = VersionToken("scene-a", "runtime-22", 9, 9, 4)
    with pytest.raises(SCSPTelemetryError, match="version token"):
        _telemetry(candidate_batch=batch, suffix_plan=_plan(wrong_token))
    with pytest.raises(SCSPTelemetryError, match="runtime_episode_id"):
        _telemetry(episode_identity=_identity(runtime_episode_id="other"))
    with pytest.raises(SCSPTelemetryError, match="ordered_categories"):
        _telemetry(
            episode_identity=_identity(ordered_categories=("chair", "plant", "table"))
        )
    with pytest.raises(SCSPTelemetryError, match="contain every raw rank"):
        _telemetry(idea1_order=((0,), (0,), ()))


def test_telemetry_allows_k_capped_matrix_and_suffix_cost_subsets():
    batch = _batch()
    default_plan = _plan(batch.version_token)
    source_identity = (batch.records_for_task(0)[1].identity(),)
    first = GeodesicMatrix(
        source_task_idx=0,
        target_task_idx=1,
        source_candidate_identities=source_identity,
        target_candidate_identities=default_plan.strict_geodesic_matrices[
            0
        ].target_candidate_identities,
        distances=((GeodesicDistance.available(1.0),),),
    )
    plan = replace(
        default_plan,
        strict_geodesic_matrices=(
            first,
            default_plan.strict_geodesic_matrices[1],
        ),
        suffix_costs=(CandidateSuffixCost(1, GeodesicDistance.available(3.0)),),
    )
    telemetry = _telemetry(candidate_batch=batch, suffix_plan=plan)
    assert telemetry.to_dict()["planner_complexity"]["geodesic_edges"] == 2


def test_telemetry_rejects_matrix_labels_not_present_in_candidate_batch():
    batch = _batch()
    default_plan = _plan(batch.version_token)
    unknown_first = GeodesicMatrix(
        source_task_idx=0,
        target_task_idx=1,
        source_candidate_identities=((1, (9999, 0, 9999)),),
        target_candidate_identities=default_plan.strict_geodesic_matrices[
            0
        ].target_candidate_identities,
        distances=((GeodesicDistance.available(1.0),),),
    )
    plan = replace(
        default_plan,
        strict_geodesic_matrices=(
            unknown_first,
            default_plan.strict_geodesic_matrices[1],
        ),
        suffix_costs=(CandidateSuffixCost(1, GeodesicDistance.available(3.0)),),
    )
    with pytest.raises(SCSPTelemetryError, match="unknown source candidate labels"):
        _telemetry(candidate_batch=batch, suffix_plan=plan)


@pytest.mark.parametrize(
    "plan",
    [
        replace(_plan(), selected_raw_rank=0),
        replace(
            _plan(),
            intervention=True,
            intervention_reason=InterventionReason.LOWER_SUFFIX_COST,
        ),
    ],
)
def test_telemetry_rejects_intervention_flag_that_disagrees_with_idea1(plan):
    with pytest.raises(SCSPTelemetryError, match="intervention must equal"):
        _telemetry(suffix_plan=plan)


def test_telemetry_rejects_unknown_experiment_arm():
    with pytest.raises(SCSPTelemetryError, match="arm must be one of"):
        _telemetry(arm="scsp")


def test_invalid_empty_batch_is_serializable_only_with_explicit_fallback():
    batch = _batch(
        status=BatchValidationStatus.INVALID,
        reason="provider episode mismatch",
    )
    empty_pairs = ((0, ()), (1, ()), (2, ()))
    batch = replace(
        batch,
        positions_by_task=empty_pairs,
        records_by_task=empty_pairs,
    )
    plan = _plan(
        batch.version_token,
        strict_geodesic_matrices=(),
        suffix_costs=(),
        selected_raw_rank=None,
        fallback_reason=FallbackReason.EPISODE_MISMATCH,
    )
    telemetry = _telemetry(
        candidate_batch=batch,
        suffix_plan=plan,
        idea1_order=((), (), ()),
    )
    payload = json.loads(dumps_json_strict(telemetry))
    assert payload["batch_validation_status"] == "invalid"
    assert payload["batch_validation_reason"] == "provider episode mismatch"
    assert payload["fallback_reason"] == "episode_mismatch"
    assert payload["candidate_counts"] == [0, 0, 0]

    with pytest.raises(SCSPTelemetryError, match="fallback reason"):
        no_fallback_plan = _plan(
            batch.version_token,
            strict_geodesic_matrices=(),
            suffix_costs=(CandidateSuffixCost(0, GeodesicDistance.available(0.0)),),
            selected_raw_rank=0,
            fallback_reason=FallbackReason.NONE,
        )
        _telemetry(
            candidate_batch=batch,
            suffix_plan=no_fallback_plan,
            idea1_order=((), (), ()),
        )
