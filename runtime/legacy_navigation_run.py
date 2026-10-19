"""Resolve retained launcher flags and environment controls into a native request."""

import argparse
from dataclasses import replace
import hashlib
import os
from pathlib import Path
import sys
import warnings


def parse_args(argv, root):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--habitat_config_path",
        type=str,
        default="project/config/habitat/lagmemo_hm3d.yaml",
    )
    parser.add_argument(
        "--baseline_config_path",
        type=str,
        default="project/config/agent/hm3d_eval.yaml",
    )
    parser.add_argument("--scene_idx", type=int, default=0)
    parser.add_argument("--scenes", type=str, default="5cdEh9F2hJL")
    parser.add_argument("--episode_start", type=int, default=0)
    parser.add_argument("--episode_end", type=int, default=None)
    parser.add_argument(
        "--seem_ckpt_path",
        type=str,
        default="./src/third_party/seem/checkpoints/seem_focall_v0.pt",
    )
    parser.add_argument(
        "--seem_yaml_path",
        type=str,
        default="./src/third_party/seem/configs/seem/focall_unicl_lang_demo.yaml",
    )
    parser.add_argument(
        "--mobileclip_ckpt_path",
        type=str,
        default="./src/third_party/ml-mobileclip/checkpoints/mobileclip_s0.pt",
    )
    parser.add_argument(
        "--output_path", type=str, default=str(root / "Exp" / "runs" / "p4_demo")
    )
    parser.add_argument("--input_data", type=str, default="3_episode_data")
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=None,
        help="Absolute GOAT dataset root containing val_seen.json.gz and content/.",
    )
    parser.add_argument("--socket_path", type=str, default="/tmp/es_mapper.sock")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument(
        "--max_steps_per_subtask",
        type=int,
        default=None,
        help="Override config.AGENT.max_steps for each ordered subtask.",
    )
    parser.add_argument(
        "--live_mode", action="store_true", help="Phase 5: stream live obs to ES server"
    )
    parser.add_argument(
        "--enable_timing",
        action="store_true",
        help="Record per-step timing and print compliance report",
    )
    parser.set_defaults(print_images=True)
    parser.add_argument(
        "--print-images",
        dest="print_images",
        action="store_true",
        help="Write the shared 3x2 snapshot at each planner keyframe",
    )
    parser.add_argument(
        "--no-print-images",
        dest="print_images",
        action="store_false",
        help="Disable per-keyframe snapshots for throughput-only runs",
    )
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    return args


def resolve_request(args, *, root, engine, environ):
    from experiments.identities import load_frozen_identities, load_identity_sidecar
    from experiments.runner import PlanningRunSettings
    from experiments.telemetry import to_jsonable
    from pbsp.navigation.cooldown import parse_category_cooldowns
    from pbsp.navigation.ranking import CandidateRerankConfig
    from runtime.legacy_navigation import navigation_settings_from_environment
    from runtime.navigation_worker import NavigationRequest, NavigationResources

    def resource(value):
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (engine / path).resolve()

    mode = environ.get("ES_SCSP_RUNTIME_MODE", "off")
    profile_mode = environ.get("ES_SCSP_PROFILE_MODE", "exact_baseline")
    compatible = {
        "exact_baseline": {"off"},
        "idea1_only": {"shadow", "idea1_only"},
        "ordered_suffix_mvp": {"shadow", "idea1_only", "ordered_suffix_mvp"},
    }
    if profile_mode not in compatible or mode not in compatible[profile_mode]:
        raise ValueError(
            f"SCSP runtime/profile mode mismatch: {mode!r}/{profile_mode!r}"
        )
    arm = environ.get("ES_SCSP_ARM", "legacy")
    if (
        arm
        != {
            "exact_baseline": "legacy",
            "idea1_only": "idea1_only",
            "ordered_suffix_mvp": "ordered_suffix_mvp",
        }[profile_mode]
    ):
        raise ValueError(f"SCSP arm/profile mismatch: {arm!r}/{profile_mode!r}")
    if mode != "off":
        if environ.get("LAGMEMO_SEMANTIC_SOURCE") != "detic":
            raise ValueError("SCSP runtime requires Detic semantics")
        if environ.get("ES_GROUND_TRUTH_SEMANTICS") != "0":
            raise ValueError("SCSP runtime requires GROUND_TRUTH_SEMANTICS=0")
        if (
            environ.get("ES_SCSP_STRICT_GEODESIC", "1") != "1"
            or environ.get("ES_SCSP_EUCLIDEAN_FALLBACK", "0") != "0"
        ):
            raise ValueError("SCSP requires strict geodesics without fallback")
    settings = navigation_settings_from_environment(environ)
    weights = {
        name: float(environ.get("ES_" + name.upper(), str(getattr(settings, name))))
        for name in (
            "query_w_semantic",
            "query_w_support",
            "query_w_recurrence",
            "query_w_travel",
            "query_w_novelty",
            "query_w_revisit",
            "query_w_failure",
            "query_switch_margin",
        )
    }
    settings = replace(
        settings,
        **weights,
        projection_tolerance_m=float(
            environ.get("ES_NAV_PROJECTION_MAX_DISTANCE", "1.5")
        ),
    )
    rerank_config = CandidateRerankConfig(
        semantic=settings.query_w_semantic,
        support=settings.query_w_support,
        recurrence=settings.query_w_recurrence,
        travel=settings.query_w_travel,
        novelty=settings.query_w_novelty,
        revisit=settings.query_w_revisit,
        failure=settings.query_w_failure,
        switch_margin=settings.query_switch_margin,
        match_radius_m=settings.candidate_evidence_radius_m,
    )
    dataset = (
        resource(args.dataset_root)
        if args.dataset_root
        else engine / "data/datasets/goat/hm3d" / args.input_data
    )
    if mode != "off":
        ordered = int(environ.get("ES_SCSP_ORDERED_SUBTASKS", "0"))
        if ordered < 1:
            raise ValueError("SCSP requires ES_SCSP_ORDERED_SUBTASKS to be positive")
        cap = int(environ.get("ES_SCSP_CANDIDATE_CAP", "5"))
        weight = float(environ.get("ES_SCSP_SUFFIX_WEIGHT", "0.35"))
        digest = hashlib.sha256(
            (dataset / "dataset_manifest.json").read_bytes()
        ).hexdigest()
        if digest != environ.get("ES_SCSP_EXPECTED_DATASET_MANIFEST_SHA256", ""):
            raise ValueError("SCSP dataset manifest hash mismatch")
        identities = load_frozen_identities(
            root / "Exp/experiments/scsp_idea2_frozen_pairs_v1.json", digest, ordered
        )
        extra, sidecar_digest = load_identity_sidecar(
            environ.get("ES_SCSP_IDENTITY_SIDECAR", ""), digest, ordered
        )
        overlap = set(identities) & set(extra)
        if overlap:
            raise ValueError(
                f"SCSP identity sidecar overlaps frozen identities: {overlap}"
            )
        identities.update(extra)
        provenance = {
            "profile_sha256": environ.get("ES_SCSP_PROFILE_SHA256", ""),
            "code_sha": environ.get("ES_SCSP_CODE_SHA", ""),
            "dataset_manifest_sha256": digest,
            "identity_sidecar_sha256": sidecar_digest,
        }
        idea1 = to_jsonable(rerank_config)
    else:
        ordered, cap, weight, identities, provenance, idea1 = 0, 5, 0.35, {}, {}, {}
    resources = NavigationResources(
        habitat_config=resource(args.habitat_config_path),
        agent_config=resource(args.baseline_config_path),
        categories=engine / "data/goal_list.json",
        seem_config=resource(args.seem_yaml_path),
        seem_checkpoint=resource(args.seem_ckpt_path),
        mobileclip_checkpoint=resource(args.mobileclip_ckpt_path),
    )
    return NavigationRequest(
        resources=resources,
        dataset=dataset,
        output=Path(args.output_path).expanduser().resolve(),
        socket_path=Path(args.socket_path),
        scene=args.scenes,
        scene_index=args.scene_idx,
        episode_start=args.episode_start,
        episode_end=args.episode_end,
        max_steps_per_task=args.max_steps_per_subtask,
        print_images=args.print_images,
        enable_timing=args.enable_timing,
        live_mode=args.live_mode,
        top_k=args.top_k,
        settings=settings,
        category_cooldowns=parse_category_cooldowns(
            environ.get("ES_QUERY_SWITCH_COOLDOWN_BY_CATEGORY", "{}")
        ),
        planning_settings=PlanningRunSettings(
            mode, arm, ordered, cap, weight, idea1, provenance, identities
        ),
        max_vertical_delta=float(environ.get("ES_NAVMESH_MAX_VERTICAL_DELTA", "0.35")),
        navmesh_samples=int(environ.get("ES_NAVMESH_SAMPLE_COUNT", "32")),
        escape_radius_m=float(environ.get("ES_NAVMESH_ESCAPE_RADIUS", "3.0")),
        escape_min_distance_m=float(
            environ.get("ES_NAVMESH_ESCAPE_MIN_DISTANCE", "1.25")
        ),
        random_min_distance_m=float(
            environ.get("ES_NAVMESH_RANDOM_MIN_DISTANCE", "2.0")
        ),
    )


def prepare_request(argv=None):
    root = Path(
        os.environ.get("TDONAV_ROOT", Path(__file__).resolve().parents[1])
    ).resolve()
    args = parse_args(argv, root)
    engine = Path(
        os.environ.get("TDONAV_LAGMEMO_ROOT", root / "Navigation/runtime/lagmemo")
    ).resolve()
    source = engine / "src/lagmemo"
    third_party = engine / "src/third_party"
    for path in reversed(
        (
            source,
            third_party / "seem/src",
            third_party / "detectron2",
            third_party / "ml-mobileclip",
            third_party / "frontier_exploration",
        )
    ):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    os.chdir(engine)
    warnings.filterwarnings("ignore")
    os.environ["MAGNUM_LOG"] = "quiet"
    os.environ["HABITAT_SIM_LOG"] = "quiet"
    from runtime.bootstrap import configure_runtime_threads

    configure_runtime_threads()
    if os.environ.get("ES_SCSP_RUNTIME_MODE", "off") != "off":
        from runtime.detic_bootstrap import install_inference_import_shims

        installed = install_inference_import_shims(source)
        print(
            f"[SCSP_RUNTIME] inference-only Detic registration stubs={list(installed)}",
            flush=True,
        )
    return resolve_request(args, root=root, engine=engine, environ=os.environ)
