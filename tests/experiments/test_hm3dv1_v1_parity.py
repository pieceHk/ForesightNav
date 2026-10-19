import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Exp" / "scripts"))

from _runtime import load_toml, navigation_environment


PROFILE_PATH = ROOT / "Exp" / "profiles" / "hm3dv1_objectnav_val.toml"
CANDIDATE_PROFILE_PATH = (
    ROOT / "Exp" / "profiles" / "hm3dv1_objectnav_val_query_candidate.toml"
)
RUNTIME_PATH = ROOT / "Exp" / "runtime.local.example.toml"


def test_hm3dv1_profile_matches_v1_controls():
    profile = load_toml(PROFILE_PATH)
    mapping = profile["mapping"]
    navigation = profile["navigation"]

    assert mapping["query_support_rerank_weight"] == 0.0025
    assert navigation["candidate_track_radius_m"] == 1.0
    assert navigation["candidate_blacklist_radius_m"] == 1.5
    assert navigation["two_stage_verification"] is True
    assert navigation["verification_window_steps"] == 6
    assert navigation["verification_min_confirmations"] == 2
    assert navigation["verification_pose_radius_m"] == 1.5
    assert navigation["verification_min_peak_seem"] == 1.15

    assert navigation["approach_ring"] is False
    assert navigation["candidate_exhaustion_exploration"] is False
    assert navigation["verify_es_agreement"] is False
    assert navigation["query_action_rerank_enabled"] is False
    assert navigation["frontier_rerank_enabled"] is False


def test_navigation_environment_overrides_inherited_controls(monkeypatch):
    for name in (
        "HM3D_EXP_TWO_STAGE_VERIFY",
        "HM3D_EXP_APPROACH_RING",
        "HM3D_EXP_CANDIDATE_EXHAUSTION_EXPLORE",
        "HM3D_EXP_VERIFY_ES_AGREEMENT",
    ):
        monkeypatch.setenv(name, "unexpected")

    profile = load_toml(PROFILE_PATH)
    runtime = load_toml(RUNTIME_PATH)
    env = navigation_environment(runtime, profile, nav_gpu=3)

    expected = {
        "ES_NAV_PROJECTION_MAX_DISTANCE": "1.5",
        "ES_CANDIDATE_TRACK_RADIUS": "1.0",
        "ES_CANDIDATE_BLACKLIST_RADIUS": "1.5",
        "HM3D_EXP_TWO_STAGE_VERIFY": "1",
        "HM3D_EXP_VERIFY_WINDOW_STEPS": "6",
        "HM3D_EXP_VERIFY_MIN_CONFIRMATIONS": "2",
        "HM3D_EXP_VERIFY_POSE_RADIUS": "1.5",
        "HM3D_EXP_VERIFY_MIN_PEAK_SEEM": "1.15",
        "HM3D_EXP_APPROACH_RING": "0",
        "HM3D_EXP_CANDIDATE_EXHAUSTION_EXPLORE": "0",
        "HM3D_EXP_VERIFY_ES_AGREEMENT": "0",
        "ES_QUERY_ACTION_RERANK_ENABLED": "0",
        "ES_FRONTIER_RERANK_ENABLED": "0",
        "MALLOC_ARENA_MAX": "2",
        "MALLOC_TRIM_THRESHOLD_": "131072",
    }
    assert {name: env[name] for name in expected} == expected


def test_candidate_profile_enables_only_candidate_reranking():
    profile = load_toml(CANDIDATE_PROFILE_PATH)
    runtime = load_toml(RUNTIME_PATH)
    env = navigation_environment(runtime, profile, nav_gpu=3)

    assert env["ES_QUERY_ACTION_RERANK_ENABLED"] == "1"
    assert env["ES_FRONTIER_RERANK_ENABLED"] == "0"
    assert env["ES_CANDIDATE_EVIDENCE_RANKING"] == "1"
    assert env["HM3D_EXP_TWO_STAGE_VERIFY"] == "1"
