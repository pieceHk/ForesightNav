import math
from dataclasses import FrozenInstanceError

import pytest

from tdonav_navigation.scsp_contracts import (
    BatchValidationStatus,
    CandidateBatch,
    CandidateRecord,
    CandidateSuffixCost,
    EpisodeIdentity,
    EvidenceState,
    ExperimentArm,
    FallbackReason,
    GeodesicDistance,
    GeodesicMatrix,
    GeodesicStatus,
    InterventionReason,
    SCSPContractError,
    SuffixPlan,
    VersionToken,
    quantize_xyz,
    validate_candidate_batch,
)


def _record(
    task_idx=0,
    category="chair",
    position=(1.0, 0.0, 2.0),
    raw_rank=0,
    semantic_score=0.8,
    support_count=3,
    support_score=0.6,
    map_revision=7,
):
    return CandidateRecord(
        task_idx=task_idx,
        category=category,
        position=position,
        raw_rank=raw_rank,
        semantic_score=semantic_score,
        support_count=support_count,
        support_score=support_score,
        map_revision=map_revision,
    )


def _batch(**overrides):
    values = {
        "scene_id": "scene-a",
        "episode_id": "episode-4",
        "map_revision": 7,
        "last_fused_keyframes": 7,
        "query_serial": 2,
        "ordered_categories": ("chair", "table"),
        "positions_by_task": (
            (0, ((1.0, 0.0, 2.0), (2.0, 0.0, 3.0))),
            (1, ((4.0, 0.0, 5.0),)),
        ),
        "records_by_task": (
            (0, (_record(), _record(position=(2.0, 0.0, 3.0), raw_rank=1))),
            (
                1,
                (
                    _record(
                        task_idx=1,
                        category="table",
                        position=(4.0, 0.0, 5.0),
                    ),
                ),
            ),
        ),
    }
    values.update(overrides)
    return CandidateBatch(**values)


def test_record_is_deeply_immutable_and_has_revision_local_identity():
    record = _record(position=[1.0004, -0.0, 2.0004])
    assert record.position == (1.0004, -0.0, 2.0004)
    assert record.identity() == (0, (1000, 0, 2000))
    assert quantize_xyz(record.position, 0.01) == (100, 0, 200)
    with pytest.raises(FrozenInstanceError):
        record.raw_rank = 9
    with pytest.raises(TypeError):
        record.position[0] = 9.0


@pytest.mark.parametrize(
    "field,value",
    [
        ("task_idx", True),
        ("raw_rank", 1.5),
        ("support_count", -1),
        ("map_revision", False),
        ("semantic_score", math.nan),
        ("semantic_score", math.inf),
        ("support_score", -math.inf),
    ],
)
def test_record_rejects_invalid_numeric_fields(field, value):
    kwargs = {field: value}
    with pytest.raises(SCSPContractError):
        _record(**kwargs)


@pytest.mark.parametrize(
    "position",
    [(), (1.0, 2.0), (1.0, 2.0, 3.0, 4.0), (math.nan, 0.0, 0.0), (0.0, math.inf, 0.0)],
)
def test_record_rejects_non_xyz_or_nonfinite_position(position):
    with pytest.raises(SCSPContractError, match="position"):
        _record(position=position)


def test_complete_candidate_batch_is_aligned_and_versioned():
    batch = _batch()
    assert validate_candidate_batch(batch) is batch
    assert batch.positions_for_task(0)[1] == (2.0, 0.0, 3.0)
    assert batch.records_for_task(1)[0].category == "table"
    assert batch.version_token == VersionToken("scene-a", "episode-4", 7, 7, 2)
    assert isinstance(batch.positions_by_task, tuple)
    assert isinstance(batch.records_by_task[0][1], tuple)


def test_episode_identity_freezes_three_ordered_sources_and_categories():
    identity = EpisodeIdentity(
        dataset_manifest_sha256="A" * 64,
        scene_id="scene-a",
        runtime_episode_id="episode-4",
        source_episode_ids=("source-1", "source-2", "source-3"),
        ordered_categories=("chair", "table", "plant"),
    )
    assert identity.dataset_manifest_sha256 == "a" * 64
    assert identity.source_episode_ids == ("source-1", "source-2", "source-3")
    with pytest.raises(FrozenInstanceError):
        identity.scene_id = "other"
    with pytest.raises(SCSPContractError, match="exactly three"):
        EpisodeIdentity(
            dataset_manifest_sha256="a" * 64,
            scene_id="scene-a",
            runtime_episode_id="episode-4",
            source_episode_ids=("source-1", "source-2"),
            ordered_categories=("chair", "table"),
        )
    with pytest.raises(SCSPContractError, match="SHA-256"):
        EpisodeIdentity(
            dataset_manifest_sha256="not-a-sha256",
            scene_id="scene-a",
            runtime_episode_id="episode-4",
            source_episode_ids=("source-1", "source-2", "source-3"),
            ordered_categories=("chair", "table", "plant"),
        )


def test_version_token_and_batch_require_one_coherent_mapper_revision():
    with pytest.raises(SCSPContractError, match="map_revision must equal"):
        VersionToken("scene-a", "episode-4", 7, 8, 2)
    with pytest.raises(SCSPContractError, match="map_revision must equal"):
        _batch(last_fused_keyframes=8)


@pytest.mark.parametrize("column", ["positions_by_task", "records_by_task"])
def test_candidate_batch_rejects_duplicate_task_index(column):
    batch = _batch()
    duplicate = getattr(batch, column) + (getattr(batch, column)[0],)
    with pytest.raises(SCSPContractError, match="duplicate task index"):
        _batch(**{column: duplicate})


def test_candidate_batch_rejects_missing_or_out_of_order_task_groups():
    with pytest.raises(SCSPContractError, match="every task exactly once"):
        _batch(positions_by_task=((1, ((4.0, 0.0, 5.0),)),))


def test_candidate_batch_rejects_record_position_count_mismatch():
    records = _batch().records_by_task
    with pytest.raises(SCSPContractError, match="count mismatch"):
        _batch(
            positions_by_task=((0, ((1.0, 0.0, 2.0),)), _batch().positions_by_task[1]),
            records_by_task=records,
        )


def test_candidate_batch_rejects_record_position_alignment_mismatch():
    positions = (
        (0, ((9.0, 0.0, 9.0), (2.0, 0.0, 3.0))),
        _batch().positions_by_task[1],
    )
    with pytest.raises(SCSPContractError, match="alignment mismatch"):
        _batch(positions_by_task=positions)


def test_candidate_batch_rejects_mixed_revisions_even_when_marked_invalid():
    mixed = _record(position=(2.0, 0.0, 3.0), raw_rank=1, map_revision=8)
    records = ((0, (_record(), mixed)), _batch().records_by_task[1])
    with pytest.raises(SCSPContractError, match="mixed map revisions"):
        _batch(
            records_by_task=records,
            validation_status=BatchValidationStatus.INVALID,
            validation_reason="mixed revisions from provider",
        )


def test_candidate_batch_rejects_revision_local_candidate_identity_collision():
    first = _record(position=(1.0001, 0.0, 2.0001), raw_rank=0)
    collision = _record(position=(1.0004, 0.0, 2.0004), raw_rank=0)
    positions = (
        (0, (first.position, collision.position)),
        _batch().positions_by_task[1],
    )
    records = ((0, (first, collision)), _batch().records_by_task[1])
    with pytest.raises(SCSPContractError, match="duplicate candidate identity"):
        _batch(positions_by_task=positions, records_by_task=records)


def test_invalid_empty_sentinel_requires_reason_and_is_rejected_by_consumer():
    empty_positions = ((0, ()), (1, ()))
    empty_records = ((0, ()), (1, ()))
    with pytest.raises(SCSPContractError, match="requires validation_reason"):
        _batch(
            positions_by_task=empty_positions,
            records_by_task=empty_records,
            validation_status=BatchValidationStatus.INVALID,
        )

    batch = _batch(
        positions_by_task=empty_positions,
        records_by_task=empty_records,
        validation_status=BatchValidationStatus.INVALID,
        validation_reason="episode mismatch",
    )
    with pytest.raises(SCSPContractError, match="episode mismatch"):
        validate_candidate_batch(batch)


def test_evidence_state_vocabulary_is_exact_and_reserved_negative_exists():
    assert {item.value for item in EvidenceState} == {
        "verified_positive",
        "certified_negative",
        "unknown",
    }


def test_experiment_arm_vocabulary_matches_the_three_frozen_profiles():
    assert {item.value for item in ExperimentArm} == {
        "exact_474a752",
        "idea1_only",
        "ordered_suffix_mvp",
    }


def test_geodesic_distance_requires_explicit_status():
    assert GeodesicDistance.available(1.25).distance_m == 1.25
    assert GeodesicDistance.unreachable().distance_m == math.inf
    assert GeodesicDistance.unavailable().distance_m is None
    with pytest.raises(SCSPContractError):
        GeodesicDistance(GeodesicStatus.AVAILABLE, math.inf)
    with pytest.raises(SCSPContractError):
        GeodesicDistance(GeodesicStatus.UNREACHABLE, 12.0)
    with pytest.raises(SCSPContractError):
        GeodesicDistance(GeodesicStatus.UNAVAILABLE, math.nan)
    with pytest.raises(SCSPContractError, match="must be None"):
        GeodesicDistance(GeodesicStatus.UNAVAILABLE, math.inf)


def _geodesic_matrix():
    batch = _batch()
    source = tuple(record.identity() for record in batch.records_for_task(0))
    target = tuple(record.identity() for record in batch.records_for_task(1))
    return GeodesicMatrix(
        source_task_idx=0,
        target_task_idx=1,
        source_candidate_identities=source,
        target_candidate_identities=target,
        distances=(
            (GeodesicDistance.available(1.0),),
            (GeodesicDistance.unreachable(),),
        ),
    )


def test_suffix_plan_freezes_labeled_matrix_costs_and_decision_reasons():
    token = _batch().version_token
    matrix = _geodesic_matrix()
    plan = SuffixPlan(
        input_token=token,
        current_task=0,
        strict_geodesic_matrices=(matrix,),
        suffix_costs=(
            CandidateSuffixCost(0, GeodesicDistance.available(5.0)),
            CandidateSuffixCost(1, GeodesicDistance.available(3.0)),
        ),
        selected_raw_rank=1,
        intervention=True,
        fallback_reason=FallbackReason.NONE,
        intervention_reason=InterventionReason.LOWER_SUFFIX_COST,
        compute_time_ms=2.5,
    )
    assert plan.version_token == token
    assert plan.planner_latency_ms == 2.5
    assert plan.strict_geodesic_matrices == (matrix,)
    assert matrix.source_candidate_identities[0] == _record().identity()
    with pytest.raises(FrozenInstanceError):
        plan.current_task = 1


def test_geodesic_matrix_rejects_shape_and_label_errors():
    matrix = _geodesic_matrix()
    with pytest.raises(SCSPContractError, match="row count"):
        GeodesicMatrix(
            source_task_idx=0,
            target_task_idx=1,
            source_candidate_identities=matrix.source_candidate_identities,
            target_candidate_identities=matrix.target_candidate_identities,
            distances=((GeodesicDistance.available(1.0),),),
        )
    with pytest.raises(SCSPContractError, match="row 0 length"):
        GeodesicMatrix(
            source_task_idx=0,
            target_task_idx=1,
            source_candidate_identities=matrix.source_candidate_identities,
            target_candidate_identities=matrix.target_candidate_identities,
            distances=((), ()),
        )
    with pytest.raises(SCSPContractError, match="duplicate candidate identities"):
        GeodesicMatrix(
            source_task_idx=0,
            target_task_idx=1,
            source_candidate_identities=(
                matrix.source_candidate_identities[0],
                matrix.source_candidate_identities[0],
            ),
            target_candidate_identities=matrix.target_candidate_identities,
            distances=(
                (GeodesicDistance.available(1.0),),
                (GeodesicDistance.available(1.0),),
            ),
        )
    with pytest.raises(SCSPContractError, match="consecutive task"):
        GeodesicMatrix(
            source_task_idx=0,
            target_task_idx=2,
            source_candidate_identities=(),
            target_candidate_identities=(),
            distances=(),
        )


def test_suffix_plan_rejects_broken_layer_chain_and_misaligned_labels():
    first = _geodesic_matrix()
    second_target = ((0, (6000, 0, 7000)),)
    second = GeodesicMatrix(
        source_task_idx=1,
        target_task_idx=2,
        source_candidate_identities=first.target_candidate_identities,
        target_candidate_identities=second_target,
        distances=((GeodesicDistance.available(2.0),),),
    )
    base = {
        "input_token": _batch().version_token,
        "current_task": 0,
        "strict_geodesic_matrices": (first, second),
        "suffix_costs": (
            CandidateSuffixCost(0, GeodesicDistance.available(2.0)),
            CandidateSuffixCost(1, GeodesicDistance.available(3.0)),
        ),
        "selected_raw_rank": 0,
        "intervention": False,
        "fallback_reason": FallbackReason.NONE,
        "intervention_reason": InterventionReason.NONE,
        "compute_time_ms": 1.0,
    }
    wrong_source = GeodesicMatrix(
        source_task_idx=1,
        target_task_idx=2,
        source_candidate_identities=((0, (9999, 0, 9999)),),
        target_candidate_identities=second_target,
        distances=((GeodesicDistance.available(2.0),),),
    )
    with pytest.raises(SCSPContractError, match="misaligned candidate identities"):
        SuffixPlan(**{**base, "strict_geodesic_matrices": (first, wrong_source)})
    with pytest.raises(SCSPContractError, match="starting at current_task"):
        SuffixPlan(
            **{
                **base,
                "current_task": 1,
                "strict_geodesic_matrices": (first,),
            }
        )


def test_suffix_plan_rejects_duplicate_costs_and_nonfinite_time():
    base = {
        "input_token": _batch().version_token,
        "current_task": 0,
        "strict_geodesic_matrices": (),
        "suffix_costs": (CandidateSuffixCost(0, GeodesicDistance.available(2.0)),),
        "selected_raw_rank": 0,
        "intervention": False,
        "fallback_reason": FallbackReason.NONE,
        "intervention_reason": InterventionReason.NONE,
        "compute_time_ms": 1.0,
    }
    with pytest.raises(SCSPContractError, match="duplicate raw_rank"):
        SuffixPlan(
            **{
                **base,
                "suffix_costs": (
                    CandidateSuffixCost(0, GeodesicDistance.available(2.0)),
                    CandidateSuffixCost(0, GeodesicDistance.available(3.0)),
                ),
            }
        )
    with pytest.raises(SCSPContractError, match="finite"):
        SuffixPlan(**{**base, "compute_time_ms": math.nan})


def test_suffix_plan_intervention_flag_and_reason_cannot_disagree():
    base = {
        "input_token": _batch().version_token,
        "current_task": 0,
        "strict_geodesic_matrices": (),
        "suffix_costs": (CandidateSuffixCost(0, GeodesicDistance.available(0.0)),),
        "selected_raw_rank": 0,
        "fallback_reason": FallbackReason.NONE,
        "compute_time_ms": 0.0,
    }
    with pytest.raises(SCSPContractError, match="requires an explicit"):
        SuffixPlan(
            **base,
            intervention=True,
            intervention_reason=InterventionReason.NONE,
        )
    with pytest.raises(SCSPContractError, match="requires intervention_reason='none'"):
        SuffixPlan(
            **base,
            intervention=False,
            intervention_reason=InterventionReason.LOWER_SUFFIX_COST,
        )
    with pytest.raises(SCSPContractError, match="requires a selected_raw_rank"):
        SuffixPlan(
            **{**base, "selected_raw_rank": None},
            intervention=True,
            intervention_reason=InterventionReason.LOWER_SUFFIX_COST,
        )


def test_fallback_vocabulary_covers_planned_noop_gates():
    values = {reason.value for reason in FallbackReason}
    assert {
        "last_task",
        "future_layer_empty",
        "all_edges_unreachable",
        "equal_suffix_costs",
        "rebuilding",
        "episode_mismatch",
        "provider_token_changed",
        "provider_status_unavailable",
        "weight_zero",
        "no_candidates",
    } <= values
