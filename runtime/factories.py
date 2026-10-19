"""Compose default mapping and navigation services from a resolved run plan."""

from dataclasses import asdict
import math


def benchmark_intrinsic(camera):
    import numpy as np

    focal = camera["width"] / (
        2.0 * math.tan(math.radians(camera["hfov_degrees"] / 2.0))
    )
    return np.array(
        [
            [focal / camera["width"], 0.0, 0.5],
            [0.0, focal / camera["height"], 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def create_mapper(plan, *, device="cuda"):
    from adapters.embodied_splat import load_model
    from adapters.embodied_splat_query import create_query_backend
    from experiments.visualization import MappingArtifacts, write_fusion_debug_render
    from pbsp.mapping.mapper import OnlineMapper
    from pbsp.mapping.memory import release_host_memory
    from pbsp.mapping.model import ModelInputSpec

    resources, method = plan.resources, plan.experiment.method
    specification = plan.model_specification
    model = load_model(
        engine_root=resources.mapping,
        checkpoint=resources.weights / "freesplatpp_scannet_gtdepth.ckpt",
        work_directory=plan.run_directory / "artifacts/mapping_tmp",
        input_spec=ModelInputSpec(**specification["input"]),
        device=device,
    )
    try:
        model.encoder.fusion_debug_sink = write_fusion_debug_render
        queries = create_query_backend(
            model.encoder, settings=method.query, device=device
        )
        return OnlineMapper(
            settings=method.mapping,
            model=model,
            query_backend=queries,
            device=device,
            artifacts=MappingArtifacts(
                final_directory=plan.run_directory / "artifacts/mapping_final"
            ),
            host_memory_hook=release_host_memory,
            camera_intrinsic=benchmark_intrinsic(specification["camera"]),
        )
    except BaseException:
        model.close()
        raise


def navigation_request(plan, scene):
    from experiments.runner import PlanningRunSettings
    from pbsp.navigation.ranking import CandidateRerankConfig
    from runtime.navigation_worker import NavigationRequest

    config = plan.experiment
    nav, suffix = config.method.navigation, config.method.suffix
    selected = next((item for item in plan.scenes if item.scene_id == scene), None)
    if selected is None:
        raise ValueError(f"Scene is absent from this run: {scene}")
    reranker = CandidateRerankConfig(
        semantic=nav.query_w_semantic,
        support=nav.query_w_support,
        recurrence=nav.query_w_recurrence,
        travel=nav.query_w_travel,
        novelty=nav.query_w_novelty,
        revisit=nav.query_w_revisit,
        failure=nav.query_w_failure,
        switch_margin=nav.query_switch_margin,
        match_radius_m=nav.candidate_evidence_radius_m,
    )
    identities = {
        (item.scene_id, item.runtime_episode_id): {
            "scene_id": item.scene_id,
            "runtime_episode_id": item.runtime_episode_id,
            "source_episode_ids": item.source_episode_ids,
            "ordered_categories": item.ordered_categories,
        }
        for item in plan.identities
        if item.scene_id == scene
    }
    return NavigationRequest(
        resources=plan.resources.navigation_resources(),
        dataset=plan.resources.data / config.dataset.root,
        output=plan.run_directory,
        socket_path=plan.socket_path,
        scene=scene,
        scene_index=selected.scene_index,
        episode_start=0,
        episode_end=None,
        episode_ids=selected.episode_ids,
        max_steps_per_task=config.evaluation.max_steps_per_task,
        print_images=config.evaluation.visualize,
        enable_timing=True,
        live_mode=True,
        top_k=config.method.query.top_k,
        settings=nav,
        category_cooldowns=config.method.query_switch_cooldown_by_category,
        seed=config.seed,
        planning_settings=PlanningRunSettings(
            mode=suffix.mode,
            arm=suffix.mode,
            ordered_subtasks=config.dataset.expected_tasks_per_episode,
            candidate_cap=suffix.candidate_cap,
            suffix_weight=suffix.suffix_weight,
            idea1_config=asdict(reranker),
            provenance={
                "profile_sha256": plan.configuration_sha256,
                "code_sha": plan.source_commit,
                "dataset_manifest_sha256": config.dataset.manifest_sha256,
                "identity_sidecar_sha256": plan.identity_sha256,
            },
            episode_identities=identities,
        ),
    )
