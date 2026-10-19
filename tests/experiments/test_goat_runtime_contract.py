from pathlib import Path
import sys
import json
import hashlib
import time
from collections import Counter
from copy import deepcopy
from dataclasses import replace
import ast
from types import SimpleNamespace
import types

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Exp" / "scripts"))
sys.path.insert(0, str(ROOT / "Navigation" / "src"))

from _runtime import (
    load_toml,
    navigation_environment,
    resolve_scsp_profile_text,
    resolve_scsp_runtime_mode,
)
from pbsp.navigation.cooldown import CategorySwitchCooldown
from pbsp.navigation.ranking import (
    CandidateActionReranker,
    CandidateRerankConfig,
)
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
from experiments.telemetry import (
    build_scsp_shadow_decision,
    dumps_json_strict,
    to_jsonable,
)
from pbsp.navigation.suffix import (
    Idea1Candidate,
    SuffixPlanningResult,
    plan_ordered_suffix,
)


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


def test_mon_category_cooldown_profile_only_changes_the_cooldown_contract():
    base = load_toml(ROOT / "Exp/profiles/hm3dv1_mon_val_query_candidate.toml")
    candidate = load_toml(
        ROOT / "Exp/profiles/hm3dv1_mon_val_query_category_cooldown.toml"
    )

    expected_cooldowns = {
        "bed": 30,
        "couch": 20,
        "toilet": 10,
        "chair": 10,
        "plant": 20,
        "tv": 10,
    }
    assert candidate["query_switch_cooldown_by_category"] == expected_cooldowns

    base_without_identity = deepcopy(base)
    candidate_without_contract = deepcopy(candidate)
    base_without_identity.pop("name")
    candidate_without_contract.pop("name")
    candidate_without_contract.pop("query_switch_cooldown_by_category")
    assert candidate_without_contract == base_without_identity


def test_mon_category_cooldown_is_exported_to_goat_runtime():
    profile = load_toml(
        ROOT / "Exp/profiles/hm3dv1_mon_val_query_category_cooldown.toml"
    )
    environment = navigation_environment({}, profile, nav_gpu=1)
    assert json.loads(environment["ES_QUERY_SWITCH_COOLDOWN_BY_CATEGORY"]) == (
        profile["query_switch_cooldown_by_category"]
    )


SCSP_PROFILES = {
    "exact_474a752": ROOT / "Exp/profiles/hm3dv1_mon_scsp_common.toml",
    "idea1_only": ROOT / "Exp/profiles/hm3dv1_mon_scsp_idea1.toml",
    "ordered_suffix_mvp": ROOT / "Exp/profiles/hm3dv1_mon_scsp_suffix.toml",
}
SCSP_2MON_PROFILE = ROOT / "Exp/profiles/hm3dv1_2mon_scsp_suffix.toml"
FROZEN_PAIRS = ROOT / "Exp/experiments/scsp_idea2_frozen_pairs_v1.json"
DATASET_MANIFEST_SHA256 = (
    "c3245c4bc0322d618b12e2845f138842c5a2553c2b0f4bfbdfffba641c415165"
)
FROZEN_SELECTION_SHA256 = (
    "3ed10a3fc880fbd177af27ee7c501fc61eb998a30ab1d24578afcdcc06b2e8cf"
)


def _changed_leaf_paths(left, right, prefix=()):
    if isinstance(left, dict) and isinstance(right, dict):
        assert set(left) == set(right), prefix
        changed = set()
        for key in left:
            changed.update(_changed_leaf_paths(left[key], right[key], (*prefix, key)))
        return changed
    return {".".join(prefix)} if left != right else set()


def _sha256_bytes(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value):
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_frozen_pairs():
    def reject_constant(value):
        raise ValueError(f"non-strict JSON constant: {value}")

    return json.loads(FROZEN_PAIRS.read_text(encoding="utf-8"), parse_constant=reject_constant)


def test_scsp_profiles_are_resolved_and_only_arm_mode_differ():
    profiles = {arm: load_toml(path) for arm, path in SCSP_PROFILES.items()}
    baseline = profiles["exact_474a752"]
    common_text = SCSP_PROFILES["exact_474a752"].read_text(encoding="utf-8")

    for arm, profile in profiles.items():
        expected_changes = set() if arm == "exact_474a752" else {
            "scsp.arm",
            "scsp.mode",
        }
        assert _changed_leaf_paths(baseline, profile) == expected_changes
        assert profile["scsp"]["arm"] == arm
        assert resolve_scsp_profile_text(
            common_text,
            arm=arm,
            mode=profile["scsp"]["mode"],
        ) == SCSP_PROFILES[arm].read_text(encoding="utf-8")
        assert profile["name"] == "hm3dv1_mon_scsp"
        assert profile["dataset"] == "hm3dv1_mon_val"
        assert profile["semantic_source"] == "detic"
        assert profile["seed"] == 1
        assert profile["ground_truth_semantics"] is False
        assert profile["max_steps_per_subtask"] == 500
        assert profile["mapping"]["top_k"] == 10
        assert profile["navigation"]["query_refresh_cadence_steps"] == 10
        assert profile["scsp"]["candidate_cap"] == 5
        assert profile["scsp"]["candidate_identity_quantum_m"] == 0.001
        assert profile["scsp"]["suffix_weight"] == 0.35
        assert profile["scsp"]["strict_geodesic"] is True
        assert profile["scsp"]["euclidean_fallback"] is False
        assert profile["scsp"]["mvp_evidence_states"] == [
            "verified_positive",
            "unknown",
        ]
        assert profile["visualization"] == {
            "per_keyframe_snapshots": False,
            "print_images": False,
        }


def test_batch4_exports_active_modes_and_supports_only_safe_rollbacks(
    monkeypatch,
):
    expected_runtime_modes = {
        "exact_474a752": "off",
        "idea1_only": "idea1_only",
        "ordered_suffix_mvp": "ordered_suffix_mvp",
    }
    for arm, path in SCSP_PROFILES.items():
        profile = load_toml(path)
        if arm == "exact_474a752":
            try:
                navigation_environment({}, profile, nav_gpu=5)
            except ValueError as exc:
                assert "must run from detached exact commit" in str(exc)
            else:
                raise AssertionError("new code must not impersonate exact baseline")
            continue
        environment = navigation_environment({}, profile, nav_gpu=5)
        assert environment["ES_SCSP_ARM"] == arm
        assert environment["ES_SCSP_PROFILE_MODE"] == profile["scsp"]["mode"]
        assert environment["ES_SCSP_RUNTIME_MODE"] == expected_runtime_modes[arm]
        assert environment["ES_SCSP_CANDIDATE_CAP"] == "5"
        assert environment["ES_SCSP_SUFFIX_WEIGHT"] == "0.35"
        assert environment["ES_SCSP_STRICT_GEODESIC"] == "1"
        assert environment["ES_SCSP_EUCLIDEAN_FALLBACK"] == "0"
        assert environment["ES_GROUND_TRUTH_SEMANTICS"] == "0"
        assert environment["ES_SCSP_PROFILE_SHA256"] == _sha256_bytes(path)
        assert environment["ES_SCSP_EXPECTED_DATASET_MANIFEST_SHA256"] == (
            DATASET_MANIFEST_SHA256
        )
        assert len(environment["ES_SCSP_CODE_SHA"]) == 40
        assert set(environment["ES_SCSP_CODE_SHA"]) <= set("0123456789abcdef")

    assert resolve_scsp_runtime_mode("idea1_only", "shadow") == "shadow"
    assert (
        resolve_scsp_runtime_mode("ordered_suffix_mvp", "idea1_only")
        == "idea1_only"
    )
    for profile_mode, unsafe in (
        ("exact_baseline", "shadow"),
        ("exact_baseline", "ordered_suffix_mvp"),
        ("idea1_only", "ordered_suffix_mvp"),
    ):
        try:
            resolve_scsp_runtime_mode(profile_mode, unsafe)
        except ValueError as exc:
            assert "safe realization/rollback" in str(exc)
        else:
            raise AssertionError("unsafe SCSP activation must fail closed")

    monkeypatch.setenv("ES_SCSP_RUNTIME_MODE_OVERRIDE", "shadow")
    suffix = load_toml(SCSP_PROFILES["ordered_suffix_mvp"])
    environment = navigation_environment({}, suffix, nav_gpu=5)
    assert environment["ES_SCSP_RUNTIME_MODE"] == "shadow"


def test_batch3_rejects_a_semantically_equal_nonfrozen_profile_path(tmp_path):
    source = SCSP_PROFILES["ordered_suffix_mvp"]
    copied = tmp_path / source.name
    copied.write_bytes(source.read_bytes())
    profile = load_toml(copied)

    try:
        navigation_environment({}, profile, nav_gpu=5)
    except ValueError as exc:
        assert "profile path differs" in str(exc)
    else:
        raise AssertionError("SCSP profile path must be frozen")


def test_batch3_rejects_mutation_after_loading_a_frozen_profile():
    profile = load_toml(SCSP_PROFILES["ordered_suffix_mvp"])
    profile["scsp"]["suffix_weight"] = 0.5

    try:
        navigation_environment({}, profile, nav_gpu=5)
    except ValueError as exc:
        assert "mutated after frozen loading" in str(exc)
    else:
        raise AssertionError("mutated SCSP profile must fail closed")


def test_portable_2mon_profile_changes_only_dataset_identity_fields():
    frozen = load_toml(SCSP_PROFILES["ordered_suffix_mvp"])
    portable = load_toml(SCSP_2MON_PROFILE)
    normalized = deepcopy(portable)
    normalized["scsp"].pop("portable_dataset")
    normalized["scsp"].pop("dataset_manifest_sha256")
    assert _changed_leaf_paths(frozen, normalized) == {
        "name",
        "dataset",
        "scsp.ordered_subtasks",
    }
    environment = navigation_environment({}, portable, nav_gpu=5)
    assert environment["ES_SCSP_ORDERED_SUBTASKS"] == "2"
    assert environment["ES_SCSP_EXPECTED_DATASET_MANIFEST_SHA256"] == (
        portable["scsp"]["dataset_manifest_sha256"]
    )


def test_portable_2mon_profile_rejects_algorithm_changes():
    profile = load_toml(SCSP_2MON_PROFILE)
    profile["scsp"]["suffix_weight"] = 0.5
    try:
        navigation_environment({}, profile, nav_gpu=5)
    except ValueError as exc:
        assert "differs outside dataset identity fields" in str(exc)
    else:
        raise AssertionError("portable profile must not change SCSP behavior")


def test_scsp_frozen48_identity_selection_and_profile_hashes():
    frozen = _load_frozen_pairs()
    assert frozen["schema_version"] == "scsp_frozen_pairs_v1"
    assert frozen["dataset"]["dataset_manifest_sha256"] == DATASET_MANIFEST_SHA256
    assert frozen["selection"]["algorithm"] == "scene_stratified_sha256_lowest_v1"
    assert frozen["selection"]["canonicalization"] == (
        "UTF-8 JSON, ensure_ascii=false, sort_keys=true, "
        "separators=(',', ':'), allow_nan=false"
    )
    assert frozen["selection"]["historical_policy_results_used"] is False
    assert frozen["selection"]["selection_sha256"] == FROZEN_SELECTION_SHA256
    assert frozen["protocol"]["profile_diff_allowlist"] == [
        "scsp.arm",
        "scsp.mode",
    ]
    assert frozen["protocol"]["mapping_gpu"] == 4
    assert frozen["protocol"]["navigation_gpu"] == 5
    assert frozen["analysis"]["paired_bootstrap"] == {
        "unit": "episode",
        "rng_seed": 1,
        "resamples": 10000,
        "frozen48_interval": "one_sided_90_percent",
    }
    gates = frozen["analysis"]["frozen48_go_no_go"]
    assert gates == {
        "integrity": {
            "paired_complete_episodes": 48,
            "paired_complete_subtasks": 144,
            "duplicate_identities": 0,
            "missing_identities": 0,
            "ground_truth_accesses": 0,
            "image_artifacts": 0,
            "stale_suffix_dispatches": 0,
            "non_strict_json_records": 0,
        },
        "safety": {
            "sr_c_minus_a_min": 0.0,
            "psr_c_minus_a_min": 0.0,
            "noninferiority_lower_bound_min": -0.01,
            "confidence": 0.90,
            "sided": "one_sided",
        },
        "efficiency": {
            "spl_c_minus_a_min": 0.01,
            "pespl_c_minus_a_min": 0.01,
            "pespl_c_minus_a_lower_bound_strictly_gt": 0.0,
            "confidence": 0.90,
            "sided": "one_sided",
        },
        "suffix_attribution": {
            "spl_c_minus_b_strictly_gt": 0.0,
            "pespl_c_minus_b_strictly_gt": 0.0,
        },
        "runtime": {
            "mean_frame_time_c_vs_b_relative_max": 0.05,
            "p95_frame_time_c_vs_b_relative_max": 0.10,
            "suffix_planner_p95_ms_max": 20.0,
            "sustained_cache_growth_allowed": False,
        },
    }

    episodes = frozen["episodes"]
    assert len(episodes) == 48
    assert [row["evaluation_index"] for row in episodes] == list(range(48))
    identities = [row["identity"] for row in episodes]
    identity_keys = {
        (
            identity["dataset_manifest_sha256"],
            identity["scene_id"],
            identity["runtime_episode_id"],
            tuple(identity["source_episode_ids"]),
            tuple(identity["ordered_categories"]),
        )
        for identity in identities
    }
    assert len(identity_keys) == 48
    for row in episodes:
        identity = row["identity"]
        assert identity["dataset_manifest_sha256"] == DATASET_MANIFEST_SHA256
        assert len(identity["source_episode_ids"]) == 3
        assert len(identity["ordered_categories"]) == 3
        assert row["subtask_count"] == 3
        assert row["identity_sha256"] == _canonical_sha256(identity)
        assert row["stratified_hash"] == _canonical_sha256(
            {
                "kind": "episode",
                "salt": frozen["selection"]["salt"],
                "identity": identity,
            }
        )
    assert _canonical_sha256(identities) == FROZEN_SELECTION_SHA256
    assert len(
        {
            (identity["scene_id"], source_id)
            for identity in identities
            for source_id in identity["source_episode_ids"]
        }
    ) == 144

    scene_counts = Counter(identity["scene_id"] for identity in identities)
    assert len(scene_counts) == 20
    assert Counter(scene_counts.values()) == {2: 12, 3: 8}
    assert sorted(
        scene for scene, count in scene_counts.items() if count == 3
    ) == sorted(frozen["selection"]["extra_scenes"])
    expected_extra_scenes = sorted(
        scene_counts,
        key=lambda scene: (
            _canonical_sha256(
                {
                    "kind": "scene",
                    "salt": frozen["selection"]["salt"],
                    "scene_id": scene,
                }
            ),
            scene,
        ),
    )[:8]
    assert frozen["selection"]["extra_scenes"] == expected_extra_scenes

    arms = {row["arm"]: row for row in frozen["arms"]}
    assert set(arms) == set(SCSP_PROFILES)
    assert arms["exact_474a752"]["code_sha"] == (
        "474a752a22867b4af34812bf12c50c8019872fcc"
    )
    for arm, path in SCSP_PROFILES.items():
        assert arms[arm]["profile_sha256"] == _sha256_bytes(path)
        assert arms[arm]["mode"] == load_toml(path)["scsp"]["mode"]


GOAT_ENTRYPOINT = (
    ROOT / "Navigation/runtime/evaluators/goat/entrypoint.py"
)


def _function_source(path, function_name):
    recorders = {
        "_record_scsp_provider_event": "record_provider_event",
        "_scsp_shadow_fallback_event": "planning_fallback_event",
        "_finalize_scsp_dispatch_event": "finalize_dispatch_event",
        "_finite_metric_summary": "finite_metric_summary",
        "_write_timing_rows": "write_timing_rows",
        "_print_timing_summary": "print_timing_summary",
    }
    canonical = path == GOAT_ENTRYPOINT and function_name in recorders
    target_name = recorders[function_name] if canonical else function_name
    if canonical:
        path = ROOT / "experiments/recorder.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
            node.name == target_name
        ):
            if canonical:
                # Keep isolated fault-injection fixtures bound to the canonical
                # function body while retaining their compatibility names.
                names = {value: key for key, value in recorders.items()}
                names["TIMING_THRESHOLDS"] = "_THRESHOLDS"

                class CompatibilityNames(ast.NodeTransformer):
                    def visit_Name(self, item):
                        item.id = names.get(item.id, item.id)
                        return item

                node = CompatibilityNames().visit(node)
                node.name = function_name
                return ast.unparse(node)
            return ast.get_source_segment(source, node)
    raise AssertionError(f"missing function {function_name}")


def test_goat_entrypoint_has_version_gated_active_dispatch_without_gt_access():
    source = GOAT_ENTRYPOINT.read_text(encoding="utf-8")
    planning_source = (ROOT / "pbsp/navigation/planning.py").read_text(encoding="utf-8")
    agent_source = (ROOT / "pbsp/agent.py").read_text(encoding="utf-8")
    recorder = _function_source(GOAT_ENTRYPOINT, "_record_scsp_provider_event")

    assert source.count("candidate_provider.invalidate_cache()") == 2
    assert "candidate_provider._goals_cache = {}" not in source
    assert source.count("_record_scsp_provider_event(") == 5
    assert "metrics[scene_ep_id]['scsp_provider_event_summary']" in source
    assert "scsp_provider_events.jsonl" in source
    assert "dumps_json_strict(persisted_event)" in recorder
    assert "get_candidate_batch()" in recorder
    assert "validate_candidate_batch(batch)" in recorder
    assert "check_candidate_batch_status(" in recorder
    assert "to_jsonable(batch)" in recorder
    assert "refresh_gs_poses" not in recorder
    assert "compute_candidate_plan(" in source
    assert "plan_ordered_suffix(" in planning_source
    assert "_record_scsp_shadow_decision(" in source
    assert "_prepare_scsp_refresh(" in source
    assert "_apply_scsp_runtime_order(" in source
    assert "_scsp_dispatch_gate(" in source
    assert "prepare_candidate_refresh(" in source
    assert "synchronous_requery_count" in planning_source
    assert "external_ranking_metadata" in agent_source
    assert "scsp_decisions.jsonl" in source
    assert "SCSP active/shadow execution requires a complete" in source
    assert "agent.dispatch_initial_candidates(" in source
    assert "agent.dispatch_candidates(" in source
    assert agent_source.count("if projected_goals is not None:") == 2
    assert "SCSP initial candidate refresh failed closed" in source
    gt_gate = source.index("SCSP runtime requires GROUND_TRUTH_SEMANTICS=0")
    assert gt_gate < source.index("env = HabitatGoatEnv(")
    gt_guard = source.rfind("if scsp_runtime_mode != 'off':", 0, gt_gate)
    assert gt_guard != -1
    assert "SCSP arm/profile mismatch" in source
    for forbidden_gt_access in (
        "_goal_viewpoints",
        ".view_points",
        ".goals",
        "semantic_scene",
        "goal.position",
    ):
        assert forbidden_gt_access not in source
    assert ".sort(" not in recorder
    assert "sorted(" not in recorder
    assert "except Exception as exc:" in recorder
    assert source.index("phase='initial_query'") < source.index("initial_goals =")
    assert source.index("phase='runtime_context'") < source.index("action, info = agent.act(obs)")
    identity_gate = source.index("SCSP episode identity does not match runtime tasks")
    initial_active = source.index("phase='initial_runtime_dispatch'", identity_gate)
    assert identity_gate < initial_active < source.index("action, info = agent.act(obs)")
    assert source.index("refresh_phase = (") < source.index("if scsp_enabled or new_goals:")


def test_pruned_detic_vendor_uses_inference_only_registration_stubs(tmp_path):
    helper = _function_source(
        GOAT_ENTRYPOINT, "_install_detic_inference_import_shims"
    )
    fake_sys = SimpleNamespace(modules={})
    namespace = {
        "_LAGMEMO_SRC": tmp_path,
        "sys": fake_sys,
        "types": types,
    }
    exec(compile(helper, "<detic-registration-stubs>", "exec"), namespace)
    installed = namespace["_install_detic_inference_import_shims"]()

    assert "centernet.data.datasets.objects365" in installed
    assert (
        fake_sys.modules[
            "centernet.data.datasets.objects365"
        ].categories_v1
        == ()
    )
    assert (
        fake_sys.modules["centernet.data.datasets.coco"]._PREDEFINED_SPLITS_COCO
        == {}
    )
    assert (
        "lagmemo.perception.detection.detic.Detic.detic.data.datasets.lvis_v1"
        in installed
    )
    assert all(
        getattr(fake_sys.modules[name], "__scsp_inference_stub__", False)
        for name in installed
    )

    center_data = (
        tmp_path
        / "lagmemo/perception/detection/detic/Detic/third_party/CenterNet2/centernet/data"
    )
    detic_data = (
        tmp_path / "lagmemo/perception/detection/detic/Detic/detic/data"
    )
    center_data.mkdir(parents=True)
    detic_data.mkdir(parents=True)
    real_sys = SimpleNamespace(modules={})
    real_namespace = {
        "_LAGMEMO_SRC": tmp_path,
        "sys": real_sys,
        "types": types,
    }
    exec(compile(helper, "<detic-real-data>", "exec"), real_namespace)
    real_installed = real_namespace[
        "_install_detic_inference_import_shims"
    ]()
    assert not any(".data" in name for name in real_installed)

    source = GOAT_ENTRYPOINT.read_text(encoding="utf-8")
    assert "if exc.name != 'timm.models.convnext':" in source
    assert source.index("_EARLY_DETIC_REGISTRATION_STUBS") < source.index(
        "from lagmemo.agent.lagmemo_agent.glue_agent import GLUEAgent"
    )


def test_scsp_no_image_runtime_materializes_planner_maps_without_writers():
    helper = _function_source(
        GOAT_ENTRYPOINT, "_enable_scsp_no_image_planning_state"
    )
    namespace = {}
    exec(compile(helper, "<scsp-no-image-planning>", "exec"), namespace)
    enable = namespace["_enable_scsp_no_image_planning_state"]

    agent = SimpleNamespace(
        visualize=False,
        continuous_print_images=False,
        episode_final_snapshot=False,
    )
    assert enable(agent, scsp_enabled=True, print_images=False) is True
    assert agent.visualize is True
    assert agent.continuous_print_images is False
    assert agent.episode_final_snapshot is False

    inactive = SimpleNamespace(
        visualize=False,
        continuous_print_images=False,
        episode_final_snapshot=False,
    )
    assert enable(inactive, scsp_enabled=False, print_images=False) is False
    assert inactive.visualize is False

    image_run = SimpleNamespace(
        visualize=False,
        continuous_print_images=True,
        episode_final_snapshot=False,
    )
    assert enable(image_run, scsp_enabled=True, print_images=True) is False
    assert image_run.visualize is False


def test_goat_provider_recorder_is_best_effort_for_runtime_parity(tmp_path):
    recorder = _function_source(GOAT_ENTRYPOINT, "_record_scsp_provider_event")
    namespace = {
        "BatchValidationStatus": SimpleNamespace(
            VALID=SimpleNamespace(value="valid"),
            INVALID=SimpleNamespace(value="invalid"),
        ),
        "Path": Path,
        "dumps_json_strict": lambda value: json.dumps(value),
        "json": json,
    }
    exec(compile(recorder, "<provider-recorder>", "exec"), namespace)

    class BrokenTelemetryProvider:
        def get_candidate_batch(self):
            raise RuntimeError("telemetry unavailable")

    output = tmp_path / "provider-error.jsonl"
    event = namespace["_record_scsp_provider_event"](
        BrokenTelemetryProvider(),
        phase="fixture",
        current_task=0,
        scene_id="scene-a",
        episode_id="episode-1",
        output_path=output,
    )

    assert event["batch_validation_status"] == "invalid"
    assert event["version_gate"] == {
        "would_invalidate": True,
        "reason": "provider_status_unavailable",
    }
    assert event["recorded_to_jsonl"] is True
    assert json.loads(output.read_text(encoding="utf-8")) == event


def test_goat_provider_recorder_appends_strict_jsonl(tmp_path):
    recorder = _function_source(GOAT_ENTRYPOINT, "_record_scsp_provider_event")
    valid = SimpleNamespace(value="valid")
    namespace = {
        "BatchValidationStatus": SimpleNamespace(
            VALID=valid,
            INVALID=SimpleNamespace(value="invalid"),
        ),
        "SCSPContractError": ValueError,
        "Path": Path,
        "dumps_json_strict": lambda value: json.dumps(
            value, allow_nan=False, sort_keys=True
        ),
        "json": json,
        "to_jsonable": lambda batch: {"query_serial": batch.query_serial},
        "validate_candidate_batch": lambda batch: batch,
    }
    exec(compile(recorder, "<provider-recorder>", "exec"), namespace)

    batch = SimpleNamespace(
        validation_status=valid,
        validation_reason=None,
        query_serial=4,
    )

    class Provider:
        def get_candidate_batch(self):
            return batch

        def get_ordered_categories(self):
            return ("chair", "table", "chair")

        def check_candidate_batch_status(self, candidate_batch, **kwargs):
            assert candidate_batch is batch
            return {"would_invalidate": False, "reason": "none"}

    output = tmp_path / "provider.jsonl"
    event = namespace["_record_scsp_provider_event"](
        Provider(),
        phase="fixture",
        current_task=0,
        scene_id="scene-a",
        episode_id="episode-1",
        output_path=output,
    )

    assert json.loads(output.read_text(encoding="utf-8")) == event


def test_goat_provider_sink_error_preserves_valid_gate_semantics(tmp_path):
    recorder = _function_source(GOAT_ENTRYPOINT, "_record_scsp_provider_event")
    valid = SimpleNamespace(value="valid")
    namespace = {
        "BatchValidationStatus": SimpleNamespace(
            VALID=valid,
            INVALID=SimpleNamespace(value="invalid"),
        ),
        "SCSPContractError": ValueError,
        "Path": Path,
        "dumps_json_strict": lambda value: json.dumps(value),
        "json": json,
        "to_jsonable": lambda batch: {"query_serial": batch.query_serial},
        "validate_candidate_batch": lambda batch: batch,
    }
    exec(compile(recorder, "<provider-recorder>", "exec"), namespace)
    batch = SimpleNamespace(
        validation_status=valid,
        validation_reason=None,
        query_serial=5,
    )

    class Provider:
        def get_candidate_batch(self):
            return batch

        def get_ordered_categories(self):
            return ("chair",)

        def check_candidate_batch_status(self, candidate_batch, **kwargs):
            return {"would_invalidate": False, "reason": "none"}

    event = namespace["_record_scsp_provider_event"](
        Provider(),
        phase="fixture",
        current_task=0,
        scene_id="scene-a",
        episode_id="episode-1",
        output_path=tmp_path / "missing-parent" / "provider.jsonl",
    )

    assert event["batch_validation_status"] == "valid"
    assert event["version_gate"] == {
        "would_invalidate": False,
        "reason": "none",
    }
    assert event["recorded_to_jsonl"] is False
    assert "recording_error" in event


def test_goat_refresh_is_atomic_after_scsp_order_and_cooldown():
    source = GOAT_ENTRYPOINT.read_text(encoding="utf-8")
    agent_source = (ROOT / "pbsp/agent.py").read_text(encoding="utf-8")
    for owner, expected in ((source, ["initial_goals"]),
                            (agent_source, ["projected_goals", "projected_goals"])):
        refresh_arguments = []
        for node in ast.walk(ast.parse(owner)):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr == "refresh_gs_poses" and node.args:
                refresh_arguments.append(ast.unparse(node.args[0]))
        assert refresh_arguments == expected
    periodic_order = source.index(
        "_apply_scsp_runtime_order(", source.index("if scsp_enabled or new_goals:")
    )
    dispatch = source.index("agent.dispatch_candidates(", periodic_order)
    persist = source.index("decision_event=periodic_decision", dispatch)
    assert periodic_order < dispatch < persist
    cooldown = agent_source.index("category_cooldown.apply(")
    refresh = agent_source.index("self.refresh_gs_poses(", cooldown)
    realized = agent_source.index("category_cooldown.observe_active(", refresh)
    assert cooldown < refresh < realized


def test_shadow_identity_lookup_uses_only_the_frozen_manifest():
    helper = _function_source(GOAT_ENTRYPOINT, "_load_scsp_frozen_identities")
    namespace = {"_TDONAV_ROOT": ROOT, "json": json}
    exec(compile(helper, "<frozen-identities>", "exec"), namespace)

    identities = namespace["_load_scsp_frozen_identities"](
        DATASET_MANIFEST_SHA256
    )
    assert len(identities) == 48
    assert identities[("4ok3usBNeis", "1")] == {
        "scene_id": "4ok3usBNeis",
        "runtime_episode_id": "1",
        "source_episode_ids": ("23", "76", "67"),
        "ordered_categories": ("chair", "toilet", "bed"),
    }
    assert namespace["_load_scsp_frozen_identities"](
        "0" * 64, 2
    ) == {}


def test_nonfrozen_identity_sidecar_rejects_nonidentity_fields(tmp_path):
    helper = _function_source(GOAT_ENTRYPOINT, "_load_scsp_identity_sidecar")
    for forbidden in ("baseline_success", "candidate_success", "outcome"):
        assert forbidden not in helper
    namespace = {"Path": Path, "json": json, "hashlib": hashlib}
    exec(compile(helper, "<identity-sidecar>", "exec"), namespace)
    payload = {
        "schema_version": "scsp_identity_sidecar_v1",
        "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
        "episodes": [
            {
                "scene_id": "4ok3usBNeis",
                "runtime_episode_id": "2",
                "source_episode_ids": ["59", "74", "46"],
                "ordered_categories": ["chair", "tv", "chair"],
            },
            {
                "scene_id": "4ok3usBNeis",
                "runtime_episode_id": "3",
                "source_episode_ids": ["94", "54", "87"],
                "ordered_categories": ["couch", "toilet", "couch"],
            },
        ],
    }
    sidecar = tmp_path / "identities.json"
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    identities, digest = namespace["_load_scsp_identity_sidecar"](
        sidecar, DATASET_MANIFEST_SHA256
    )
    assert len(identities) == 2
    assert identities[("4ok3usBNeis", "2")]["source_episode_ids"] == (
        "59",
        "74",
        "46",
    )
    assert digest == hashlib.sha256(sidecar.read_bytes()).hexdigest()

    two_task_payload = deepcopy(payload)
    for row in two_task_payload["episodes"]:
        row["source_episode_ids"] = row["source_episode_ids"][:2]
        row["ordered_categories"] = row["ordered_categories"][:2]
    sidecar.write_text(json.dumps(two_task_payload), encoding="utf-8")
    identities, _ = namespace["_load_scsp_identity_sidecar"](
        sidecar, DATASET_MANIFEST_SHA256, 2
    )
    assert len(identities) == 2

    sidecar.write_text(json.dumps(payload), encoding="utf-8")

    payload["episodes"][0]["outcome"] = "must-not-enter-runtime"
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    try:
        namespace["_load_scsp_identity_sidecar"](
            sidecar, DATASET_MANIFEST_SHA256
        )
    except ValueError as exc:
        assert "non-identity fields" in str(exc)
    else:
        raise AssertionError("identity sidecar must reject policy outcomes")

    top_level = deepcopy(payload)
    top_level["episodes"][0].pop("outcome")
    top_level["outcome"] = "must-not-enter-runtime"
    sidecar.write_text(json.dumps(top_level), encoding="utf-8")
    try:
        namespace["_load_scsp_identity_sidecar"](
            sidecar, DATASET_MANIFEST_SHA256
        )
    except ValueError as exc:
        assert "non-identity fields" in str(exc)
    else:
        raise AssertionError("sidecar top level must be identity-only")

    non_strict = deepcopy(top_level)
    non_strict.pop("outcome")
    non_strict["episodes"][0]["runtime_episode_id"] = float("nan")
    sidecar.write_text(json.dumps(non_strict), encoding="utf-8")
    try:
        namespace["_load_scsp_identity_sidecar"](
            sidecar, DATASET_MANIFEST_SHA256
        )
    except ValueError as exc:
        assert "non-strict JSON constant" in str(exc)
    else:
        raise AssertionError("sidecar must reject NaN")

    wrong_type = deepcopy(payload)
    wrong_type["episodes"][0].pop("outcome")
    wrong_type["episodes"][0]["source_episode_ids"][0] = 59
    sidecar.write_text(json.dumps(wrong_type), encoding="utf-8")
    try:
        namespace["_load_scsp_identity_sidecar"](
            sidecar, DATASET_MANIFEST_SHA256
        )
    except ValueError as exc:
        assert "non-empty strings" in str(exc)
    else:
        raise AssertionError("sidecar IDs must not be coerced")


def _shadow_batch(query_serial=1):
    categories = ("chair",)
    records = (
        CandidateRecord(0, "chair", (1.0, 0.0, 0.0), 0, 0.8, 2, 0.4, 7),
        CandidateRecord(0, "chair", (3.0, 0.0, 0.0), 1, 0.9, 3, 0.6, 7),
    )
    return CandidateBatch(
        scene_id="scene-a",
        episode_id="episode-1",
        map_revision=7,
        last_fused_keyframes=7,
        query_serial=query_serial,
        ordered_categories=categories,
        positions_by_task=((0, tuple(record.position for record in records)),),
        records_by_task=((0, records),),
    )


def _three_task_shadow_batch(query_serial=1):
    categories = ("chair", "table", "plant")
    positions = (
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        ((0.0, 0.0, 2.0), (2.0, 0.0, 2.0)),
        ((0.0, 0.0, 4.0), (3.0, 0.0, 4.0)),
    )
    records_by_task = []
    for task_idx, (category, task_positions) in enumerate(
        zip(categories, positions)
    ):
        records_by_task.append(
            (
                task_idx,
                tuple(
                    CandidateRecord(
                        task_idx,
                        category,
                        position,
                        raw_rank,
                        0.7 + 0.1 * raw_rank,
                        raw_rank + 1,
                        0.2 + 0.1 * raw_rank,
                        7,
                    )
                    for raw_rank, position in enumerate(task_positions)
                ),
            )
        )
    return CandidateBatch(
        scene_id="scene-a",
        episode_id="episode-1",
        map_revision=7,
        last_fused_keyframes=7,
        query_serial=query_serial,
        ordered_categories=categories,
        positions_by_task=tuple(
            (task_idx, task_positions)
            for task_idx, task_positions in enumerate(positions)
        ),
        records_by_task=tuple(records_by_task),
    )


def _two_task_shadow_batch(query_serial=1):
    three_task = _three_task_shadow_batch(query_serial)
    return CandidateBatch(
        scene_id=three_task.scene_id,
        episode_id=three_task.episode_id,
        map_revision=three_task.map_revision,
        last_fused_keyframes=three_task.last_fused_keyframes,
        query_serial=three_task.query_serial,
        ordered_categories=three_task.ordered_categories[:2],
        positions_by_task=three_task.positions_by_task[:2],
        records_by_task=three_task.records_by_task[:2],
    )


def test_projection_indices_preserve_legacy_payload_and_raw_ranks():
    projector = _function_source(GOAT_ENTRYPOINT, "_project_goals_to_navigable")
    batch_builder = _function_source(GOAT_ENTRYPOINT, "_projected_candidate_batch")
    namespace = {"np": np, "os": __import__("os")}
    exec(compile(projector, "<projector>", "exec"), namespace)
    batch_namespace = {
        "CandidateBatch": CandidateBatch,
        "CandidateRecord": CandidateRecord,
        "SCSPContractError": SCSPContractError,
        "validate_candidate_batch": validate_candidate_batch,
    }
    exec(compile(batch_builder, "<projected-batch>", "exec"), batch_namespace)

    class Pathfinder:
        @staticmethod
        def snap_point(point):
            result = np.asarray(point, dtype=np.float32).copy()
            result[1] = 0.0
            return result

    goals = {
        0: [[0.0, 0.2, 0.0], [0.1, 0.2, 0.1], [3.0, 0.2, 0.0]]
    }
    records = tuple(
        CandidateRecord(0, "chair", point, rank, 1.0, 1, 1.0, 7)
        for rank, point in enumerate(goals[0])
    )
    batch = CandidateBatch(
        scene_id="scene-a",
        episode_id="episode-1",
        map_revision=7,
        last_fused_keyframes=7,
        query_serial=1,
        ordered_categories=("chair",),
        positions_by_task=((0, tuple(record.position for record in records)),),
        records_by_task=((0, records),),
    )
    original = deepcopy(goals)
    legacy = namespace["_project_goals_to_navigable"](goals, Pathfinder())
    projected, indices = namespace["_project_goals_to_navigable"](
        goals, Pathfinder(), return_indices=True
    )
    planning_batch = batch_namespace["_projected_candidate_batch"](
        batch, projected, indices
    )

    assert projected == legacy
    assert goals == original
    assert indices[0] == [0, 2]
    assert [record.raw_rank for record in planning_batch.records_for_task(0)] == [0, 2]
    assert list(planning_batch.positions_for_task(0)) == [
        tuple(point) for point in projected[0]
    ]


def test_strict_geodesic_distinguishes_unreachable_and_unavailable(monkeypatch):
    helper = _function_source(GOAT_ENTRYPOINT, "_strict_geodesic_distance")
    namespace = {"np": np, "GeodesicDistance": GeodesicDistance}
    exec(compile(helper, "<strict-geodesic>", "exec"), namespace)

    class ShortestPath:
        geodesic_distance = 0.0

    monkeypatch.setitem(
        sys.modules, "habitat_sim", SimpleNamespace(ShortestPath=ShortestPath)
    )

    class Pathfinder:
        def __init__(self, outcome, distance=0.0):
            self.outcome = outcome
            self.distance = distance

        def find_path(self, path):
            if self.outcome == "error":
                raise RuntimeError("pathfinder unavailable")
            path.geodesic_distance = self.distance
            return self.outcome == "found"

    available = namespace["_strict_geodesic_distance"](
        Pathfinder("found", 4.25), [0, 0, 0], [1, 0, 1]
    )
    unreachable = namespace["_strict_geodesic_distance"](
        Pathfinder("missing"), [0, 0, 0], [1, 0, 1]
    )
    unavailable = namespace["_strict_geodesic_distance"](
        Pathfinder("error"), [0, 0, 0], [1, 0, 1]
    )

    assert available == GeodesicDistance.available(4.25)
    assert unreachable == GeodesicDistance.unreachable()
    assert unavailable == GeodesicDistance.unavailable()


def test_idea1_shadow_records_legacy_euclidean_fallback_explicitly():
    helper = _function_source(GOAT_ENTRYPOINT, "_rank_scsp_idea1_layers")
    unavailable_status = SimpleNamespace(value="unreachable")
    namespace = {
        "GeodesicStatus": SimpleNamespace(AVAILABLE=object()),
        "_strict_geodesic_distance": lambda *_args: SimpleNamespace(
            status=unavailable_status, distance_m=None
        ),
        "to_jsonable": to_jsonable,
        "_active_candidate": lambda _agent, _task_idx: None,
        "_subtask_has_acted": lambda _agent, _task_idx: False,
        "Idea1Candidate": Idea1Candidate,
    }
    exec(compile(helper, "<idea1-shadow>", "exec"), namespace)

    candidates, _telemetry, diagnostics = namespace[
        "_rank_scsp_idea1_layers"
    ](
        CandidateActionReranker(enabled=True),
        _shadow_batch(),
        [0.0, 0.0, 0.0],
        object(),
        SimpleNamespace(visited_candidate_positions={}),
    )

    assert len(candidates[0]) == 2
    task_diagnostics = dict(diagnostics)[0]
    assert task_diagnostics["geodesic_fallbacks"] == 2
    assert task_diagnostics["strict_travel_by_raw_rank"] == [
        {"raw_rank": 0, "status": "unreachable", "distance_m": None},
        {"raw_rank": 1, "status": "unreachable", "distance_m": None},
    ]


def test_initial_and_future_layers_do_not_inherit_fake_active_hysteresis():
    helper = _function_source(GOAT_ENTRYPOINT, "_rank_scsp_idea1_layers")
    namespace = {
        "GeodesicStatus": GeodesicStatus,
        "_strict_geodesic_distance": lambda *_args: GeodesicDistance.available(
            1.0
        ),
        "to_jsonable": to_jsonable,
        "_active_candidate": lambda agent, task_idx: agent.gs_poses[task_idx][0],
        "_subtask_has_acted": lambda agent, task_idx: (
            agent.sub_task_timesteps[0][task_idx] > 0
        ),
        "Idea1Candidate": Idea1Candidate,
    }
    exec(compile(helper, "<idea1-no-fake-active>", "exec"), namespace)
    batch = _shadow_batch()
    reranker = CandidateActionReranker(
        enabled=True,
        config=CandidateRerankConfig(
            semantic=1.0,
            support=0.0,
            recurrence=0.0,
            travel=0.0,
            novelty=0.0,
            revisit=0.0,
            failure=0.0,
            switch_margin=10.0,
        ),
    )
    fake_rank_zero = list(batch.records_for_task(0)[0].position)
    agent = SimpleNamespace(
        gs_poses={0: [fake_rank_zero]},
        frontier_flag={0: 0},
        sub_task_timesteps=[[0]],
        visited_candidate_positions={},
    )
    candidates, _, _ = namespace["_rank_scsp_idea1_layers"](
        reranker,
        batch,
        [0.0, 0.0, 0.0],
        object(),
        agent,
    )
    assert [candidate.record.raw_rank for candidate in candidates[0]] == [1, 0]
    assert 0 not in reranker._active


def test_pairwise_cache_is_keyed_by_version_and_candidate_identity():
    helper = _function_source(GOAT_ENTRYPOINT, "_build_scsp_geodesic_matrices")
    calls = []

    def strict_distance(_pathfinder, start, end):
        calls.append((tuple(start), tuple(end)))
        return GeodesicDistance.available(2.0)

    namespace = {
        "GeodesicMatrix": GeodesicMatrix,
        "_strict_geodesic_distance": strict_distance,
    }
    exec(compile(helper, "<matrix-cache>", "exec"), namespace)

    def three_task_batch(query_serial):
        categories = ("chair", "table", "plant")
        records = tuple(
            CandidateRecord(
                task_idx,
                category,
                (float(task_idx), 0.0, 0.0),
                0,
                1.0,
                1,
                1.0,
                7,
            )
            for task_idx, category in enumerate(categories)
        )
        batch = CandidateBatch(
            scene_id="scene-a",
            episode_id="episode-1",
            map_revision=7,
            last_fused_keyframes=7,
            query_serial=query_serial,
            ordered_categories=categories,
            positions_by_task=tuple(
                (idx, (record.position,)) for idx, record in enumerate(records)
            ),
            records_by_task=tuple(
                (idx, (record,)) for idx, record in enumerate(records)
            ),
        )
        idea1 = {
            idx: (Idea1Candidate(record, 1.0),)
            for idx, record in enumerate(records)
        }
        return batch, idea1

    cache = {}
    first_batch, first_idea1 = three_task_batch(1)
    first, first_stats = namespace["_build_scsp_geodesic_matrices"](
        first_batch, 0, first_idea1, object(), cache, 5
    )
    repeated, repeated_stats = namespace["_build_scsp_geodesic_matrices"](
        first_batch, 0, first_idea1, object(), cache, 5
    )
    next_batch, next_idea1 = three_task_batch(2)
    _, next_stats = namespace["_build_scsp_geodesic_matrices"](
        next_batch, 0, next_idea1, object(), cache, 5
    )

    assert len(first) == len(repeated) == 2
    assert first_stats["misses"] == 2
    assert repeated_stats["hits"] == 2
    assert next_stats["evictions"] == 2
    assert next_stats["misses"] == 2
    assert len(cache) == 2
    assert len(calls) == 4


@pytest.mark.parametrize("task_count", [2, 3])
def test_ordered_mon_shadow_decision_is_strict_and_recomputable(task_count):
    function_names = (
        "_projected_candidate_batch",
        "_active_candidate",
        "_subtask_has_acted",
        "_rank_scsp_idea1_layers",
        "_build_scsp_geodesic_matrices",
        "_scsp_shadow_fallback_event",
        "_compute_scsp_shadow_decision",
    )

    def strict_distance(_pathfinder, start, end):
        distance = float(
            np.linalg.norm(np.asarray(end, dtype=float) - np.asarray(start, dtype=float))
        )
        return GeodesicDistance.available(distance)

    namespace = {
        "np": np,
        "time": time,
        "replace": replace,
        "CandidateBatch": CandidateBatch,
        "CandidateRecord": CandidateRecord,
        "FallbackReason": FallbackReason,
        "GeodesicMatrix": GeodesicMatrix,
        "GeodesicStatus": GeodesicStatus,
        "SCSPContractError": SCSPContractError,
        "SHADOW_SCHEMA_VERSION": "scsp_shadow_decision_v1",
        "SuffixPlanningResult": SuffixPlanningResult,
        "Idea1Candidate": Idea1Candidate,
        "validate_candidate_batch": validate_candidate_batch,
        "to_jsonable": to_jsonable,
        "_strict_geodesic_distance": strict_distance,
        "plan_ordered_suffix": plan_ordered_suffix,
        "build_scsp_shadow_decision": build_scsp_shadow_decision,
    }
    for function_name in function_names:
        source = _function_source(GOAT_ENTRYPOINT, function_name)
        exec(compile(source, f"<{function_name}>", "exec"), namespace)

    batch = (
        _two_task_shadow_batch()
        if task_count == 2
        else _three_task_shadow_batch()
    )
    projected = {
        task_idx: [list(position) for position in batch.positions_for_task(task_idx)]
        for task_idx in range(task_count)
    }
    indices = {task_idx: [0, 1] for task_idx in range(task_count)}
    agent = SimpleNamespace(
        gs_poses=deepcopy(projected),
        frontier_flag={task_idx: 0 for task_idx in range(task_count)},
        visited_candidate_positions={},
    )
    reranker = CandidateActionReranker(enabled=True)
    cache = {}
    event = namespace["_compute_scsp_shadow_decision"](
        phase="fixture",
        arm="ordered_suffix_mvp",
        provider_event={
            "version_gate": {"would_invalidate": False, "reason": "none"}
        },
        candidate_batch=batch,
        projected_goals=projected,
        projected_indices=indices,
        current_task=0,
        origin=[0.0, 0.0, 0.0],
        pathfinder=object(),
        agent=agent,
        reranker=reranker,
        distance_cache=cache,
        candidate_cap=5,
        suffix_weight=0.35,
        idea1_config=to_jsonable(reranker.config),
        provenance={
            "profile_sha256": "a" * 64,
            "dataset_manifest_sha256": "b" * 64,
            "code_sha": "c" * 40,
        },
        episode_identity={
            "scene_id": "scene-a",
            "runtime_episode_id": "episode-1",
            "source_episode_ids": tuple(
                ("source-a", "source-b", "source-c")[:task_count]
            ),
            "ordered_categories": batch.ordered_categories,
        },
    )
    payload = json.loads(dumps_json_strict(event))

    assert payload["status"] == "planned"
    expected_edges = 4 * (task_count - 1)
    assert len(payload["suffix_plan"]["strict_geodesic_matrices"]) == (
        task_count - 1
    )
    assert payload["distance_cache"]["misses"] == expected_edges
    assert payload["distance_cache"]["entries_after"] == expected_edges
    assert len(payload["idea1_candidates"]) == task_count
    assert all(len(group[1]) == 2 for group in payload["idea1_candidates"])
    assert payload["episode_identity_status"] == "complete"
    assert payload["source_episode_ids"] == [
        "source-a",
        "source-b",
        "source-c",
    ][:task_count]
    assert projected == agent.gs_poses


def _shadow_idea1_config():
    return {
        "semantic": 1.0,
        "support": 0.0,
        "recurrence": 0.0,
        "travel": 0.0,
        "novelty": 0.0,
        "revisit": 0.0,
        "failure": 0.0,
        "switch_margin": 0.0,
        "match_radius_m": 0.75,
    }


def _shadow_components(semantic):
    return {
        "semantic": float(semantic),
        "support": 0.0,
        "recurrence": 0.0,
        "travel": 0.0,
        "novelty": 0.0,
        "revisit": 0.0,
        "failure": 0.0,
    }


def test_shadow_telemetry_records_three_heads_but_dispatches_legacy():
    batch = _shadow_batch()
    records = batch.records_for_task(0)
    idea1 = {
        0: (
            Idea1Candidate(records[1], 1.0),
            Idea1Candidate(records[0], 0.0),
        )
    }
    result = plan_ordered_suffix(
        candidate_batch=batch,
        current_task=0,
        idea1_candidates_by_task=idea1,
        strict_geodesic_matrices=(),
    )
    event = build_scsp_shadow_decision(
        phase="fixture",
        arm="ordered_suffix_mvp",
        candidate_batch=batch,
        suffix_plan=result.plan,
        legacy_order=((0, 1),),
        idea1_order=((1, 0),),
        suffix_order=(result.ordered_raw_ranks,),
        idea1_candidates=(
            (
                0,
                (
                    {
                        "raw_rank": 1,
                        "utility": 1.0,
                        "utility_components": _shadow_components(1.0),
                    },
                    {
                        "raw_rank": 0,
                        "utility": 0.0,
                        "utility_components": _shadow_components(0.0),
                    },
                ),
            ),
        ),
        rerank_diagnostics=((0, {"reordered": 1}),),
        version_gate={"would_invalidate": False, "reason": "none"},
        origin_position=[0.0, 0.0, 0.0],
        timing_ms={"suffix_dp": 0.1},
        distance_cache={"hits": 0, "misses": 0, "entries_after": 0},
        candidate_cap=5,
        suffix_weight=0.35,
        idea1_config=_shadow_idea1_config(),
        provenance={
            "profile_sha256": "a" * 64,
            "dataset_manifest_sha256": "b" * 64,
            "code_sha": "c" * 40,
        },
        episode_identity=None,
    )
    payload = json.loads(dumps_json_strict(event))

    assert payload["selected_raw_ranks"] == {
        "legacy": 0,
        "idea1": 1,
        "suffix": 1,
    }
    assert payload["changed"] == {
        "idea1_from_legacy": True,
        "suffix_from_idea1": False,
        "suffix_from_legacy": True,
    }
    assert payload["shadow_only"] is True
    assert payload["dispatch_order_source"] == "legacy"
    assert payload["legacy_order_stage"] == "post_projection_pre_cooldown"
    assert "dispatched" not in payload["selected_raw_ranks"]
    assert payload["planner_config"] == {
        "candidate_cap": 5,
        "suffix_weight": 0.35,
        "suffix_strict_geodesic": True,
        "suffix_euclidean_fallback": False,
    }
    assert payload["profile_sha256"] == "a" * 64
    assert payload["dataset_manifest_sha256"] == "b" * 64
    assert payload["code_sha"] == "c" * 40
    assert len(payload["idea1_candidates"][0][1]) == 2
    assert payload["episode_identity_status"] == (
        "source_ids_not_in_frozen_manifest"
    )
    assert payload["source_episode_ids"] is None


def test_version_gate_fallback_retains_provenance_and_planner_config():
    fallback = _function_source(GOAT_ENTRYPOINT, "_scsp_shadow_fallback_event")
    compute = _function_source(GOAT_ENTRYPOINT, "_compute_scsp_shadow_decision")
    namespace = {
        "time": time,
        "FallbackReason": FallbackReason,
        "SHADOW_SCHEMA_VERSION": "scsp_shadow_decision_v1",
    }
    exec(compile(fallback, "<shadow-fallback>", "exec"), namespace)
    exec(compile(compute, "<shadow-compute>", "exec"), namespace)
    provenance = {
        "profile_sha256": "a" * 64,
        "dataset_manifest_sha256": "b" * 64,
        "code_sha": "c" * 40,
    }

    event = namespace["_compute_scsp_shadow_decision"](
        phase="fixture",
        arm="ordered_suffix_mvp",
        provider_event={
            "version_gate": {"would_invalidate": True, "reason": "rebuilding"}
        },
        candidate_batch=_shadow_batch(),
        projected_goals={},
        projected_indices={},
        current_task=0,
        origin=[0.0, 0.0, 0.0],
        pathfinder=object(),
        agent=SimpleNamespace(),
        reranker=object(),
        distance_cache={},
        candidate_cap=5,
        suffix_weight=0.35,
        idea1_config=_shadow_idea1_config(),
        provenance=provenance,
        episode_identity=None,
    )

    assert event["status"] == "fallback"
    assert event["fallback_reason"] == "rebuilding"
    assert event["failure_stage"] == "version_gate"
    assert event["profile_sha256"] == "a" * 64
    assert event["dataset_manifest_sha256"] == "b" * 64
    assert event["code_sha"] == "c" * 40
    assert event["planner_config"]["candidate_cap"] == 5
    assert event["planner_config"]["suffix_weight"] == 0.35


def test_shadow_off_on_fake_refresh_replay_has_identical_actions_and_rng(monkeypatch):
    from pbsp.navigation import planning

    fallback_source = _function_source(GOAT_ENTRYPOINT, "_scsp_shadow_fallback_event")
    compute_source = _function_source(GOAT_ENTRYPOINT, "_compute_scsp_shadow_decision")

    def rank_layers(_reranker, batch, _origin, _pathfinder, _agent):
        records = tuple(reversed(batch.records_for_task(0)))
        candidates = tuple(
            Idea1Candidate(record, float(1 - idx)) for idx, record in enumerate(records)
        )
        telemetry = (
            (
                0,
                tuple(
                    {
                        "raw_rank": candidate.record.raw_rank,
                        "utility": candidate.utility,
                        "utility_components": _shadow_components(candidate.utility),
                    }
                    for candidate in candidates
                ),
            ),
        )
        return {0: candidates}, telemetry, ((0, {"reordered": 1}),)

    def projected_batch(batch, _goals, indices):
        if indices == "broken-sidecar":
            raise ValueError("sidecar alignment failed")
        return batch

    namespace = {
        "time": time,
        "FallbackReason": FallbackReason,
        "SHADOW_SCHEMA_VERSION": "scsp_shadow_decision_v1",
        "SCSPContractError": ValueError,
        "_projected_candidate_batch": projected_batch,
        "_rank_scsp_idea1_layers": rank_layers,
        "_build_scsp_geodesic_matrices": lambda *_args: (
            (),
            {"hits": 0, "misses": 0, "entries_after": 0},
        ),
        "plan_ordered_suffix": plan_ordered_suffix,
        "SuffixPlanningResult": SuffixPlanningResult,
        "replace": replace,
        "build_scsp_shadow_decision": build_scsp_shadow_decision,
    }
    monkeypatch.setattr(planning, "bind_projected_candidates", projected_batch)
    monkeypatch.setattr(planning, "rank_candidate_layers", rank_layers)
    monkeypatch.setattr(
        planning,
        "build_geodesic_matrices",
        namespace["_build_scsp_geodesic_matrices"],
    )
    exec(compile(fallback_source, "<shadow-fallback>", "exec"), namespace)
    exec(compile(compute_source, "<shadow-compute>", "exec"), namespace)

    stream = [
        {
            0: [
                [float(index + 1), 0.0, 0.0],
                [float(index) if index > 0 else 9.0, 0.0, 0.0],
            ]
        }
        for index in range(3)
    ]

    class FakeAgent:
        def __init__(self, seed):
            self.rng = np.random.RandomState(seed)
            self.gs_poses = {}
            self.refresh_inputs = []

        def refresh_gs_poses(self, goals):
            self.refresh_inputs.append(deepcopy(goals))
            self.gs_poses = deepcopy(goals)

        def act(self):
            return tuple(self.gs_poses[0][0]), float(self.rng.rand())

    off_agent = FakeAgent(17)
    on_agent = FakeAgent(17)
    cooldown = CategorySwitchCooldown({"chair": 10}, match_radius_m=0.25)
    off_actions = []
    on_actions = []
    cooldown_reasons = []
    for index, goals in enumerate(stream, start=1):
        original = deepcopy(goals)
        event = namespace["_compute_scsp_shadow_decision"](
            phase="fixture",
            arm="ordered_suffix_mvp",
            provider_event={"version_gate": {"would_invalidate": False, "reason": "none"}},
            candidate_batch=_shadow_batch(query_serial=index),
            projected_goals=goals,
            projected_indices=("broken-sidecar" if index == len(stream) else {}),
            current_task=0,
            origin=[0.0, 0.0, 0.0],
            pathfinder=object(),
            agent=on_agent,
            reranker=object(),
            distance_cache={},
            candidate_cap=5,
            suffix_weight=0.35,
            idea1_config=_shadow_idea1_config(),
            provenance={
                "profile_sha256": "a" * 64,
                "dataset_manifest_sha256": "b" * 64,
                "code_sha": "c" * 40,
            },
            episode_identity=None,
        )
        if index == len(stream):
            assert event["status"] == "fallback"
        else:
            assert event["status"] == "planned"
            assert event["selected_raw_ranks"]["legacy"] == 0
            assert "dispatched" not in event["selected_raw_ranks"]
        assert goals == original
        dispatch_goals = deepcopy(goals)
        active_position = (
            off_agent.gs_poses.get(0, [None])[0] if off_agent.gs_poses.get(0) else None
        )
        dispatch_goals[0], cooldown_event = cooldown.apply(
            task_idx=0,
            category="chair",
            step=index - 1,
            active_position=active_position,
            latest_ranked_candidates=dispatch_goals[0],
        )
        cooldown_reasons.append(cooldown_event["reason"])
        off_agent.refresh_gs_poses(dispatch_goals)
        on_agent.refresh_gs_poses(dispatch_goals)
        cooldown.observe_active(0, dispatch_goals[0][0], index - 1)
        off_actions.append(off_agent.act())
        on_actions.append(on_agent.act())

    assert on_actions == off_actions
    assert on_agent.refresh_inputs == off_agent.refresh_inputs
    assert cooldown_reasons[1] == "KEEP_ACTIVE_COOLDOWN"


def test_active_serialization_failure_preserves_realized_dispatch_truth(tmp_path):
    recorder = _function_source(GOAT_ENTRYPOINT, "_record_scsp_shadow_decision")
    serialization_calls = 0

    def fail_once(payload):
        nonlocal serialization_calls
        serialization_calls += 1
        if serialization_calls == 1:
            raise ValueError("serialization failed")
        return dumps_json_strict(payload)

    namespace = {
        "_compute_scsp_shadow_decision": lambda **_kwargs: None,
        "Path": Path,
        "SHADOW_SCHEMA_VERSION": "scsp_shadow_decision_v1",
        "FallbackReason": FallbackReason,
        "dumps_json_strict": fail_once,
        "json": json,
    }
    exec(compile(recorder, "<shadow-recorder>", "exec"), namespace)

    active = {
        "phase": "fixture",
        "arm": "ordered_suffix_mvp",
        "mode": "ordered_suffix_mvp",
        "shadow_only": False,
        "current_task": 0,
        "status": "planned",
        "dispatch_order_source": "suffix",
        "selected_raw_ranks": {
            "legacy": 0,
            "idea1": 1,
            "suffix": 2,
            "dispatched": 2,
        },
        "suffix_plan": SimpleNamespace(
            fallback_reason=SimpleNamespace(value="none")
        ),
    }
    event = namespace["_record_scsp_shadow_decision"](
        output_path=tmp_path / "active.jsonl",
        decision_event=active,
    )

    assert event["recorded_to_jsonl"] is True
    assert event["recording_degraded"] is True
    assert event["status"] == "planned"
    assert event["mode"] == "ordered_suffix_mvp"
    assert event["shadow_only"] is False
    assert event["dispatch_order_source"] == "suffix"
    assert event["selected_raw_ranks"]["dispatched"] == 2
    assert event["recording_error"] == "ValueError: serialization failed"
    diagnostic = json.loads(
        (tmp_path / "active.jsonl").read_text(encoding="utf-8")
    )
    assert diagnostic["failure_stage"] == "serialization"
    assert diagnostic["mode"] == "ordered_suffix_mvp"
    assert diagnostic["dispatch_order_source"] == "suffix"


def test_unexpected_shadow_compute_failure_is_structured_and_nonfatal(tmp_path):
    fallback = _function_source(GOAT_ENTRYPOINT, "_scsp_shadow_fallback_event")
    recorder = _function_source(GOAT_ENTRYPOINT, "_record_scsp_shadow_decision")
    namespace = {
        "_compute_scsp_shadow_decision": lambda **_kwargs: (_ for _ in ()).throw(
            RuntimeError("unexpected compute failure")
        ),
        "FallbackReason": FallbackReason,
        "SHADOW_SCHEMA_VERSION": "scsp_shadow_decision_v1",
        "time": time,
        "Path": Path,
        "dumps_json_strict": dumps_json_strict,
        "json": json,
    }
    exec(compile(fallback, "<shadow-fallback>", "exec"), namespace)
    exec(compile(recorder, "<shadow-recorder>", "exec"), namespace)

    output = tmp_path / "shadow.jsonl"
    event = namespace["_record_scsp_shadow_decision"](
        output_path=output,
        phase="fixture",
        arm="ordered_suffix_mvp",
        current_task=1,
        origin=[0.0, 0.0, 0.0],
        provider_event={
            "version_gate": {"would_invalidate": False, "reason": "none"}
        },
    )

    assert event["status"] == "fallback"
    assert event["fallback_reason"] == "planner_error"
    assert event["recorded_to_jsonl"] is True
    assert json.loads(output.read_text(encoding="utf-8"))["error"] == {
        "type": "RuntimeError",
        "message": "unexpected compute failure",
    }


def test_double_telemetry_fallback_never_references_unset_state():
    recorder = _function_source(GOAT_ENTRYPOINT, "_record_scsp_shadow_decision")
    namespace = {
        "_compute_scsp_shadow_decision": lambda **_kwargs: (_ for _ in ()).throw(
            RuntimeError("compute failed")
        ),
        "_scsp_shadow_fallback_event": lambda **_kwargs: (_ for _ in ()).throw(
            RuntimeError("fallback failed")
        ),
        "FallbackReason": FallbackReason,
        "SHADOW_SCHEMA_VERSION": "scsp_shadow_decision_v1",
        "time": time,
        "Path": Path,
        "dumps_json_strict": dumps_json_strict,
        "json": json,
    }
    exec(compile(recorder, "<shadow-recorder>", "exec"), namespace)
    event = namespace["_record_scsp_shadow_decision"](
        phase="fixture",
        runtime_mode="ordered_suffix_mvp",
    )
    assert event["status"] == "fallback"
    assert event["mode"] == "ordered_suffix_mvp"
    assert event["dispatch_order_source"] == "not_applied"
    assert event["fallback_construction_error"]["message"] == "fallback failed"


def test_shadow_candidate_batch_getter_failure_is_guarded(tmp_path):
    fallback = _function_source(GOAT_ENTRYPOINT, "_scsp_shadow_fallback_event")
    recorder = _function_source(GOAT_ENTRYPOINT, "_record_scsp_shadow_decision")

    class BrokenProvider:
        @staticmethod
        def get_candidate_batch():
            raise RuntimeError("candidate batch unavailable")

    namespace = {
        "_compute_scsp_shadow_decision": lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("compute must not run without a candidate batch")
        ),
        "FallbackReason": FallbackReason,
        "SHADOW_SCHEMA_VERSION": "scsp_shadow_decision_v1",
        "time": time,
        "Path": Path,
        "dumps_json_strict": dumps_json_strict,
        "json": json,
    }
    exec(compile(fallback, "<shadow-fallback>", "exec"), namespace)
    exec(compile(recorder, "<shadow-recorder>", "exec"), namespace)

    output = tmp_path / "shadow.jsonl"
    event = namespace["_record_scsp_shadow_decision"](
        output_path=output,
        provider=BrokenProvider(),
        phase="fixture",
        arm="ordered_suffix_mvp",
        current_task=0,
        origin=[0.0, 0.0, 0.0],
        provider_event={
            "version_gate": {"would_invalidate": False, "reason": "none"}
        },
    )

    assert event["status"] == "fallback"
    assert event["recorded_to_jsonl"] is True
    assert event["error"] == {
        "type": "RuntimeError",
        "message": "candidate batch unavailable",
    }


def _active_dispatch_batch():
    records = (
        (
            0,
            (
                CandidateRecord(0, "chair", (1.0, 0.0, 0.0), 0, 0.5, 1, 0.5, 7),
                CandidateRecord(0, "chair", (3.0, 0.0, 0.0), 2, 0.7, 1, 0.5, 7),
            ),
        ),
        (
            1,
            (CandidateRecord(1, "table", (4.0, 0.0, 0.0), 4, 0.5, 1, 0.5, 7),),
        ),
        (
            2,
            (CandidateRecord(2, "plant", (5.0, 0.0, 0.0), 8, 0.5, 1, 0.5, 7),),
        ),
    )
    return CandidateBatch(
        scene_id="scene-a",
        episode_id="episode-1",
        map_revision=7,
        last_fused_keyframes=7,
        query_serial=3,
        ordered_categories=("chair", "table", "plant"),
        positions_by_task=tuple(
            (task_idx, tuple(record.position for record in task_records))
            for task_idx, task_records in records
        ),
        records_by_task=records,
    )


def test_active_order_maps_projected_raw_ranks_and_only_changes_current_task():
    fallback = _function_source(GOAT_ENTRYPOINT, "_fallback_value")
    helper = _function_source(GOAT_ENTRYPOINT, "_apply_scsp_runtime_order")
    namespace = {
        "FallbackReason": FallbackReason,
        "SCSPContractError": SCSPContractError,
        "to_jsonable": to_jsonable,
    }
    exec(compile(fallback, "<fallback-value>", "exec"), namespace)
    exec(compile(helper, "<active-order>", "exec"), namespace)
    batch = _active_dispatch_batch()
    projected = {
        task_idx: [list(position) for position in batch.positions_for_task(task_idx)]
        for task_idx in range(3)
    }
    decision = {
        "status": "planned",
        "current_task": 0,
        "candidate_batch": batch,
        "dispatch_version_gate": {"would_invalidate": False, "reason": "none"},
        "idea1_order": ((0, 2), (4,), (8,)),
        "suffix_order": ((2, 0), (4,), (8,)),
        "suffix_plan": SimpleNamespace(
            fallback_reason=FallbackReason.NONE,
            intervention=True,
        ),
    }

    ordered, metadata = namespace["_apply_scsp_runtime_order"](
        projected, decision, "ordered_suffix_mvp"
    )
    assert ordered[0] == [[3.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    assert ordered[1] is projected[1]
    assert ordered[2] is projected[2]
    assert metadata["source"] == "suffix"
    assert metadata["ordered_raw_ranks"] == [2, 0]
    assert metadata["suffix_eligible"] is True
    assert metadata["stale_suffix_dispatch"] == 0

    idea1, idea1_metadata = namespace["_apply_scsp_runtime_order"](
        projected, decision, "idea1_only"
    )
    assert idea1[0] == projected[0]
    assert idea1_metadata["source"] == "idea1"


def test_suffix_fallback_uses_idea1_and_invalid_gate_uses_legacy():
    fallback = _function_source(GOAT_ENTRYPOINT, "_fallback_value")
    helper = _function_source(GOAT_ENTRYPOINT, "_apply_scsp_runtime_order")
    namespace = {
        "FallbackReason": FallbackReason,
        "SCSPContractError": SCSPContractError,
        "to_jsonable": to_jsonable,
    }
    exec(compile(fallback, "<fallback-value>", "exec"), namespace)
    exec(compile(helper, "<active-order>", "exec"), namespace)
    batch = _active_dispatch_batch()
    projected = {
        task_idx: [list(position) for position in batch.positions_for_task(task_idx)]
        for task_idx in range(3)
    }
    decision = {
        "status": "planned",
        "current_task": 0,
        "candidate_batch": batch,
        "dispatch_version_gate": {"would_invalidate": False, "reason": "none"},
        "idea1_order": ((2, 0), (4,), (8,)),
        "suffix_order": ((0, 2), (4,), (8,)),
        "suffix_plan": SimpleNamespace(
            fallback_reason=FallbackReason.WEIGHT_ZERO,
            intervention=False,
        ),
    }
    ordered, metadata = namespace["_apply_scsp_runtime_order"](
        projected, decision, "ordered_suffix_mvp"
    )
    assert ordered[0][0] == [3.0, 0.0, 0.0]
    assert metadata["source"] == "idea1_fallback"
    assert metadata["reason"] == "weight_zero"
    assert metadata["suffix_eligible"] is False

    decision["dispatch_version_gate"] = {
        "would_invalidate": True,
        "reason": "stale_version_token",
    }
    ordered, metadata = namespace["_apply_scsp_runtime_order"](
        projected, decision, "ordered_suffix_mvp"
    )
    assert ordered == projected
    assert metadata["source"] == "legacy"
    assert metadata["externally_ranked"] is False


def test_active_dispatch_requeries_once_then_never_dispatches_stale_suffix():
    helper = _function_source(GOAT_ENTRYPOINT, "_prepare_scsp_refresh")

    class Provider:
        def __init__(self):
            self.invalidations = 0
            self.queries = 0
            self.batch = SimpleNamespace(query_serial=1)

        def invalidate_cache(self):
            self.invalidations += 1

        def get_all_goals(self):
            self.queries += 1
            self.batch = SimpleNamespace(query_serial=2)
            return {0: [[2.0, 0.0, 0.0]]}

        def get_candidate_batch(self):
            return self.batch

    def run(gates):
        observed_gates = iter(gates)
        namespace = {
            "FallbackReason": FallbackReason,
            "_fallback_value": lambda value: getattr(value, "value", value),
            "_project_goals_to_navigable": (
                lambda goals, _pathfinder, return_indices=False: (goals, {0: [0]})
            ),
            "_compute_scsp_shadow_decision": lambda **kwargs: {
                "status": "planned",
                "current_task": 0,
                "candidate_batch": kwargs["candidate_batch"],
            },
            "_scsp_dispatch_gate": lambda *_args, **_kwargs: dict(
                next(observed_gates)
            ),
            "_record_scsp_provider_event": lambda *_args, **_kwargs: {
                "version_gate": {"would_invalidate": False, "reason": "none"}
            },
            "_scsp_shadow_fallback_event": lambda **kwargs: {
                "status": "fallback",
                "fallback_reason": kwargs["reason"],
                "current_task": kwargs["current_task"],
            },
            "time": time,
        }
        exec(compile(helper, "<prepare-refresh>", "exec"), namespace)
        provider = Provider()
        projected, decision, provider_events = namespace[
            "_prepare_scsp_refresh"
        ](
            phase="fixture",
            runtime_mode="ordered_suffix_mvp",
            arm="ordered_suffix_mvp",
            provider=provider,
            provider_event={
                "version_gate": {"would_invalidate": False, "reason": "none"}
            },
            raw_goals={0: [[1.0, 0.0, 0.0]]},
            current_task=0,
            origin=[0.0, 0.0, 0.0],
            pathfinder=object(),
            agent=object(),
            reranker=object(),
            distance_cache={},
            candidate_cap=5,
            suffix_weight=0.35,
            idea1_config={},
            provenance={},
            episode_identity=None,
            scene_id="scene-a",
            episode_id="episode-1",
            ordered_categories=("chair",),
        )
        return provider, projected, decision, provider_events

    stale = {"would_invalidate": True, "reason": "stale_version_token"}
    valid = {"would_invalidate": False, "reason": "none", "query_serial": 2}
    provider, projected, decision, provider_events = run((stale, valid))
    assert provider.invalidations == provider.queries == 1
    assert projected == {0: [[2.0, 0.0, 0.0]]}
    assert decision["synchronous_requery_count"] == 1
    assert decision["synchronous_requery_succeeded"] is True
    assert len(provider_events) == 1

    provider, _, decision, _ = run((stale, stale))
    assert provider.invalidations == provider.queries == 1
    assert decision["status"] == "fallback"
    assert decision["fallback_reason"] == "stale_version_token"
    assert decision["failure_stage"] == "pre_dispatch_version_gate"
    assert decision["synchronous_requery_count"] == 1


def test_failed_query_and_retry_preserve_existing_candidates():
    helper = _function_source(GOAT_ENTRYPOINT, "_prepare_scsp_refresh")
    apply_order = _function_source(GOAT_ENTRYPOINT, "_apply_scsp_runtime_order")
    finalize = _function_source(GOAT_ENTRYPOINT, "_finalize_scsp_dispatch_event")

    class Provider:
        def invalidate_cache(self):
            pass

        def get_all_goals(self):
            raise RuntimeError("retry query failed")

        def get_candidate_batch(self):
            return None

    namespace = {
        "FallbackReason": FallbackReason,
        "_fallback_value": lambda value: getattr(value, "value", value),
        "_project_goals_to_navigable": (
            lambda goals, _pathfinder, return_indices=False: ({}, {})
            if return_indices
            else {}
        ),
        "_compute_scsp_shadow_decision": lambda **kwargs: {
            "status": "planned",
            "current_task": 0,
            "candidate_batch": kwargs["candidate_batch"],
        },
        "_scsp_dispatch_gate": lambda *_args, **_kwargs: {
            "would_invalidate": True,
            "reason": "provider_status_unavailable",
        },
        "_record_scsp_provider_event": lambda *_args, **_kwargs: {},
        "_scsp_shadow_fallback_event": lambda **kwargs: {
            "status": "fallback",
            "fallback_reason": kwargs["reason"],
            "current_task": kwargs["current_task"],
        },
        "time": time,
    }
    exec(compile(helper, "<prepare-refresh>", "exec"), namespace)
    exec(compile(apply_order, "<apply-order>", "exec"), namespace)
    exec(compile(finalize, "<finalize-dispatch>", "exec"), namespace)
    projected, decision, _ = namespace["_prepare_scsp_refresh"](
        phase="periodic_refresh",
        runtime_mode="ordered_suffix_mvp",
        arm="ordered_suffix_mvp",
        provider=Provider(),
        provider_event={"version_gate": {}},
        raw_goals=None,
        current_task=0,
        origin=[0.0, 0.0, 0.0],
        pathfinder=object(),
        agent=object(),
        reranker=object(),
        distance_cache={},
        candidate_cap=5,
        suffix_weight=0.35,
        idea1_config={},
        provenance={},
        episode_identity=None,
        scene_id="scene-a",
        episode_id="episode-1",
        ordered_categories=("chair",),
        preserve_existing_on_missing=True,
    )
    assert projected is None
    assert decision["preserved_existing_candidates"] is True
    assert decision["status"] == "fallback"
    assert decision["fallback_reason"] == "provider_status_unavailable"
    assert decision["synchronous_requery_succeeded"] is False
    ordered, metadata = namespace["_apply_scsp_runtime_order"](
        projected, decision, "ordered_suffix_mvp"
    )
    assert ordered is None
    assert metadata["reason"] == "preserved_existing_candidates"
    event = namespace["_finalize_scsp_dispatch_event"](
        decision,
        metadata,
        cooldown_event=None,
        refresh_applied=False,
        realized_active_position=[9.0, 0.0, 9.0],
        realized_raw_rank=None,
    )
    assert event["fallback_reason"] == "provider_status_unavailable"
    assert event["dispatch_order_source"] == "not_applied"


def test_scsp_plan_state_is_cleared_at_each_subtask_boundary():
    helper = _function_source(GOAT_ENTRYPOINT, "_reset_scsp_task_state")
    namespace = {}
    exec(compile(helper, "<task-reset>", "exec"), namespace)

    class Reranker:
        resets = 0

        def reset(self):
            self.resets += 1

    reranker = Reranker()
    cache = {"old-plan": object()}
    namespace["_reset_scsp_task_state"](reranker, cache)
    assert reranker.resets == 1
    assert cache == {}

    source = GOAT_ENTRYPOINT.read_text(encoding="utf-8")
    assert "current_task_idx != scsp_last_task_idx" in source
    assert "subtask_state_reset_count" in source
    assert "scsp_boundary_replan_pending = scsp_runtime_mode in" in source
    assert "'subtask_boundary'" in source
    assert "subtask_boundary_replan_count" in source
    boundary_cache = source.index(
        "if scsp_boundary_replan_pending and not periodic_refresh_due:",
        source.index("new_goals = None"),
    )
    periodic_query = source.index("else:", boundary_cache)
    boundary_source = source[boundary_cache:periodic_query]
    assert "get_candidate_batch()" in boundary_source
    assert "invalidate_cache()" not in boundary_source
    assert "if periodic_refresh_due" in source
    assert "category_cooldown.enabled and local_step > 0" in source
    agent_source = (ROOT / "pbsp/agent.py").read_text(encoding="utf-8")
    assert "if local_step == 0:" in agent_source


def test_final_dispatch_event_uses_realized_rank_and_computed_stale_guard():
    helper = _function_source(GOAT_ENTRYPOINT, "_finalize_scsp_dispatch_event")
    namespace = {"FallbackReason": FallbackReason}
    exec(compile(helper, "<dispatch-finalizer>", "exec"), namespace)
    decision = {
        "schema_version": "scsp_shadow_decision_v1",
        "status": "planned",
        "selected_raw_ranks": {"legacy": 0, "idea1": 1, "suffix": 2},
    }
    metadata = {
        "mode": "ordered_suffix_mvp",
        "source": "suffix",
        "reason": "version_valid_suffix",
        "dispatch_version_gate": {"would_invalidate": False},
    }
    event = namespace["_finalize_scsp_dispatch_event"](
        decision,
        metadata,
        cooldown_event=None,
        refresh_applied=True,
        realized_active_position=[3.0, 0.0, 0.0],
        realized_raw_rank=2,
    )
    assert event["schema_version"] == "scsp_runtime_dispatch_v1"
    assert event["planning_schema_version"] == "scsp_shadow_decision_v1"
    assert event["dispatch_order_source"] == "suffix"
    assert event["selected_raw_ranks"]["dispatched"] == 2
    assert event["dispatch"]["stale_suffix_dispatch"] == 0

    stale = deepcopy(metadata)
    stale["dispatch_version_gate"]["would_invalidate"] = True
    try:
        namespace["_finalize_scsp_dispatch_event"](
            decision,
            stale,
            cooldown_event=None,
            refresh_applied=True,
            realized_active_position=[3.0, 0.0, 0.0],
            realized_raw_rank=2,
        )
    except RuntimeError as exc:
        assert "stale suffix dispatch" in str(exc)
    else:
        raise AssertionError("stale suffix dispatch must fail closed")

    provider_fallback = {
        "status": "fallback",
        "fallback_reason": "provider_status_unavailable",
        "failure_stage": "pre_dispatch_version_gate",
    }
    not_applied = {
        "mode": "ordered_suffix_mvp",
        "source": "legacy",
        "reason": "fallback",
        "dispatch_version_gate": {
            "would_invalidate": True,
            "reason": "provider_status_unavailable",
        },
    }
    event = namespace["_finalize_scsp_dispatch_event"](
        provider_fallback,
        not_applied,
        cooldown_event=None,
        refresh_applied=False,
        realized_active_position=None,
        realized_raw_rank=None,
    )
    assert event["fallback_reason"] == "provider_status_unavailable"
    assert event["failure_stage"] == "pre_dispatch_version_gate"
