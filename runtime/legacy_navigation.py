"""Read controls used by the retained launcher at its compatibility boundary."""

import os
from pathlib import Path

from pbsp.settings import NavigationSettings
from runtime.config import read_yaml


ENVIRONMENT_FIELDS = {
    "candidate_track_radius_m": "ES_CANDIDATE_TRACK_RADIUS",
    "candidate_blacklist_radius_m": "ES_CANDIDATE_BLACKLIST_RADIUS",
    "candidate_evidence_radius_m": "ES_CANDIDATE_EVIDENCE_RADIUS",
    "candidate_promotion_margin": "ES_CANDIDATE_PROMOTION_MARGIN",
    "candidate_evidence_ranking": "ES_CANDIDATE_EVIDENCE_RANKING",
    "query_action_rerank_enabled": "ES_QUERY_ACTION_RERANK_ENABLED",
    "two_stage_verification": "HM3D_EXP_TWO_STAGE_VERIFY",
    "verification_window_steps": "HM3D_EXP_VERIFY_WINDOW_STEPS",
    "verification_min_confirmations": "HM3D_EXP_VERIFY_MIN_CONFIRMATIONS",
    "verification_pose_radius_m": "HM3D_EXP_VERIFY_POSE_RADIUS",
    "verification_min_peak_seem": "HM3D_EXP_VERIFY_MIN_PEAK_SEEM",
    "verification_dual_score_gate": "HM3D_EXP_VERIFY_DUAL_SCORE_GATE",
    "verification_strong_peak_seem": "HM3D_EXP_VERIFY_STRONG_PEAK_SEEM",
    "verification_min_peak_clip": "HM3D_EXP_VERIFY_MIN_PEAK_CLIP",
    "verification_min_observer_translation_m": "HM3D_EXP_VERIFY_MIN_OBSERVER_TRANSLATION",
    "verification_min_observer_yaw_deg": "HM3D_EXP_VERIFY_MIN_OBSERVER_YAW_DEG",
    "frontier_rerank_enabled": "ES_FRONTIER_RERANK_ENABLED",
    "frontier_w_gain": "ES_FRONTIER_W_GAIN",
    "frontier_w_clearance": "ES_FRONTIER_W_CLEARANCE",
    "frontier_w_travel": "ES_FRONTIER_W_TRAVEL",
    "frontier_w_novelty": "ES_FRONTIER_W_NOVELTY",
    "frontier_w_revisit": "ES_FRONTIER_W_REVISIT",
    "frontier_w_failure": "ES_FRONTIER_W_FAILURE",
    "frontier_switch_margin": "ES_FRONTIER_SWITCH_MARGIN",
    "trapped_warmup_steps": "ES_TRAPPED_WARMUP_STEPS",
    "trapped_window_steps": "ES_TRAPPED_WINDOW_STEPS",
    "trapped_radius_m": "ES_TRAPPED_RADIUS",
    "random_exploration_warmup_steps": "ES_RANDOM_EXPLORATION_WARMUP_STEPS",
}


def navigation_settings_from_environment(environ=None) -> NavigationSettings:
    """Preserve the original constructor's defaults for legacy entrypoints."""
    environment = os.environ if environ is None else environ
    root = Path(__file__).resolve().parents[1]
    values = read_yaml(root / "configs/methods/full_pbsp.yaml")["navigation"]
    values.update(
        candidate_blacklist_radius_m=values["candidate_track_radius_m"],
        two_stage_verification=False,
        verification_dual_score_gate=False,
        query_action_rerank_enabled=False,
    )
    for name, variable in ENVIRONMENT_FIELDS.items():
        if variable in environment:
            raw = environment[variable]
            values[name] = (
                raw == "1" if type(values[name]) is bool else type(values[name])(raw)
            )
    if "ES_CANDIDATE_BLACKLIST_RADIUS" not in environment:
        values["candidate_blacklist_radius_m"] = values["candidate_track_radius_m"]
    return NavigationSettings(**values)
