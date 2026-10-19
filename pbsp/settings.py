"""Explicit method parameters, independent of datasets and experiment launchers."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Mapping


def _validate_scalars(value):
    for field in fields(value):
        item = getattr(value, field.name)
        kind = field.type
        if kind == "bool":
            valid = type(item) is bool
        elif kind == "int":
            valid = type(item) is int and item >= 0
        elif kind == "float":
            valid = type(item) in (int, float) and math.isfinite(item) and item >= 0
        elif kind == "str":
            valid = isinstance(item, str) and bool(item.strip())
        else:
            continue
        if not valid:
            raise ValueError(f"{type(value).__name__}.{field.name}: invalid {kind} value {item!r}")


@dataclass(frozen=True)
class MappingSettings:
    top_k: int
    keyframe_threshold: float
    pixel_delta: float
    warmup_keyframes: int
    rebuild_interval: int
    query_similarity: float
    query_support_radius_m: float
    query_minimum_support: int
    query_nms_radius_m: float
    query_pool_size: int
    query_support_rerank_weight: float
    raw_frame_buffer: int
    frame_compare_height: int
    frame_compare_width: int
    max_gaussians: int
    max_keyframes: int
    offload_fastsam_after_rebuild: bool
    clip_coverage_diagnostic: bool

    def __post_init__(self):
        _validate_scalars(self)
        if min(self.top_k, self.warmup_keyframes, self.rebuild_interval, self.query_pool_size,
               self.frame_compare_height, self.frame_compare_width) < 1:
            raise ValueError("Mapping counts must be positive")
        if self.keyframe_threshold > 1 or self.pixel_delta > 1:
            raise ValueError("Frame-change thresholds must be within [0, 1]")

@dataclass(frozen=True)
class NavigationSettings:
    semantic_source: str
    projection_tolerance_m: float
    candidate_merge_radius_m: float
    candidate_track_radius_m: float
    candidate_blacklist_radius_m: float
    candidate_evidence_radius_m: float
    candidate_promotion_margin: int
    candidate_evidence_ranking: bool
    query_action_rerank_enabled: bool
    query_refresh_cadence_steps: int
    query_w_semantic: float
    query_w_support: float
    query_w_recurrence: float
    query_w_travel: float
    query_w_novelty: float
    query_w_revisit: float
    query_w_failure: float
    query_switch_margin: float
    frontier_rerank_enabled: bool
    trapped_warmup_steps: int
    trapped_window_steps: int
    trapped_radius_m: float
    random_exploration_warmup_steps: int
    recovery_history_size: int
    recovery_novelty_weight: float
    two_stage_verification: bool
    verification_window_steps: int
    verification_min_confirmations: int
    verification_pose_radius_m: float
    verification_min_peak_seem: float
    verification_dual_score_gate: bool
    verification_strong_peak_seem: float
    verification_min_peak_clip: float
    verification_min_observer_translation_m: float
    verification_min_observer_yaw_deg: float
    approach_ring: bool
    candidate_exhaustion_exploration: bool
    verify_es_agreement: bool
    frontier_w_gain: float
    frontier_w_clearance: float
    frontier_w_travel: float
    frontier_w_novelty: float
    frontier_w_revisit: float
    frontier_w_failure: float
    frontier_switch_margin: float

    def __post_init__(self):
        _validate_scalars(self)

@dataclass(frozen=True)
class SuffixSettings:
    mode: str
    candidate_cap: int
    candidate_identity_quantum_m: float
    suffix_weight: float
    strict_geodesic: bool
    euclidean_fallback: bool
    mvp_evidence_states: tuple[str, ...]

    def __post_init__(self):
        _validate_scalars(self)
        if self.mode not in {"ordered_suffix_mvp", "idea1_only"}:
            raise ValueError("Unknown suffix mode")
        if not 1 <= self.candidate_cap <= 5:
            raise ValueError("candidate_cap must be in [1, 5]")
        if not self.strict_geodesic or self.euclidean_fallback:
            raise ValueError("PBSP requires strict geodesic distances")
        if self.candidate_identity_quantum_m <= 0:
            raise ValueError("Candidate identity quantum must be positive")
        states = tuple(self.mvp_evidence_states)
        if states != ("verified_positive", "unknown"):
            raise ValueError("Unsupported evidence-state producer contract")
        object.__setattr__(self, "mvp_evidence_states", states)


@dataclass(frozen=True)
class MethodSettings:
    name: str
    ground_truth_semantics: bool
    mapping: MappingSettings
    navigation: NavigationSettings
    suffix: SuffixSettings
    query_switch_cooldown_by_category: Mapping[str, int]

    def __post_init__(self):
        _validate_scalars(self)
        for category, count in self.query_switch_cooldown_by_category.items():
            if not isinstance(category, str) or not category.strip() or type(count) is not int or count < 0:
                raise ValueError("Category cooldowns require non-empty names and non-negative integers")
        if self.query_switch_cooldown_by_category and not self.navigation.query_action_rerank_enabled:
            raise ValueError("Category cooldowns require candidate action reranking")
