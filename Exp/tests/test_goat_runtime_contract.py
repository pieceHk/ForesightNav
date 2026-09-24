from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Exp" / "scripts"))

from _runtime import load_toml, navigation_environment


def test_goat_profile_declares_shared_navigation_contract():
    profile = load_toml(ROOT / "Exp" / "profiles" / "goat_val_seen.toml")
    navigation = profile["navigation"]

    assert navigation["candidate_track_radius_m"] == 1.0
    assert navigation["candidate_blacklist_radius_m"] == 1.0
    assert navigation["two_stage_verification"] is False
    assert navigation["candidate_exhaustion_exploration"] is False


def test_goat_navigation_environment_uses_explicit_baseline_controls():
    runtime = load_toml(ROOT / "Exp" / "runtime.local.toml")
    profile = load_toml(ROOT / "Exp" / "profiles" / "goat_val_seen.toml")
    environment = navigation_environment(runtime, profile, nav_gpu=1)

    expected = {
        "ES_CANDIDATE_TRACK_RADIUS": "1.0",
        "ES_CANDIDATE_BLACKLIST_RADIUS": "1.0",
        "HM3D_EXP_TWO_STAGE_VERIFY": "0",
        "HM3D_EXP_APPROACH_RING": "0",
        "HM3D_EXP_CANDIDATE_EXHAUSTION_EXPLORE": "0",
        "HM3D_EXP_VERIFY_ES_AGREEMENT": "0",
    }
    assert {name: environment[name] for name in expected} == expected
