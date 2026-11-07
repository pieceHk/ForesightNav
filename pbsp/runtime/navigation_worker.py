"""Assemble navigation services and run complete ordered episodes."""

from __future__ import annotations

# Portions adapted from LagMemo, Copyright (c) Meta Platforms, Inc. and affiliates.
# Upstream MIT terms and provenance are retained in THIRD_PARTY_NOTICES.md.

from contextlib import ExitStack
from dataclasses import dataclass
from functools import partial
import json
import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Optional

from pbsp.settings import NavigationSettings
from pbsp.adapters.model_resources import MatchingResources

if TYPE_CHECKING:
    from experiments.runner import PlanningRunSettings


@dataclass(frozen=True)
class NavigationResources:
    habitat_config: Path
    agent_config: Path
    categories: Path
    seem_config: Path
    seem_checkpoint: Path
    mobileclip_checkpoint: Path
    detic_config: Optional[Path] = None
    detic_checkpoint: Optional[Path] = None
    scene_directory: Optional[Path] = None
    scene_dataset: Optional[Path] = None
    seem_tokenizer: Optional[Path] = None
    detic_category_frequency: Optional[Path] = None
    matching: Optional[MatchingResources] = None


@dataclass(frozen=True)
class NavigationRequest:
    resources: NavigationResources
    dataset: Path
    output: Path
    socket_path: Path
    scene: str
    scene_index: int
    episode_start: int
    episode_end: Optional[int]
    max_steps_per_task: Optional[int]
    print_images: bool
    enable_timing: bool
    live_mode: bool
    top_k: int
    settings: NavigationSettings
    category_cooldowns: Mapping[str, int]
    planning_settings: PlanningRunSettings
    episode_ids: Optional[tuple[str, ...]] = None
    seed: Optional[int] = None
    record_decisions: bool = True
    max_vertical_delta: float = 0.35
    navmesh_samples: int = 32
    escape_radius_m: float = 3.0
    escape_min_distance_m: float = 1.25
    random_min_distance_m: float = 2.0
    dataset_format: str = 'goat_mon'

    def __post_init__(self):
        if self.dataset_format not in {'goat_mon', 'objectnav', 'mfmon'}:
            raise ValueError('Unknown native navigation dataset format')
        if self.top_k < 1 or self.episode_start < 0:
            raise ValueError("Expected positive top_k and a non-negative episode start")
        if self.episode_end is not None and self.episode_end <= self.episode_start:
            raise ValueError("Expected a nonempty episode range")
        if self.max_steps_per_task is not None and self.max_steps_per_task < 1:
            raise ValueError("Step budget must be positive")
        if (
            self.planning_settings.mode != "off"
            and self.settings.semantic_source != "detic"
        ):
            raise ValueError(
                "SCSP runtime requires Detic semantics without ground truth"
            )
        if self.category_cooldowns and not self.settings.query_action_rerank_enabled:
            raise ValueError(
                "category cooldown requires query_action_rerank_enabled=true"
            )

        if self.episode_ids is not None:
            if (
                not self.episode_ids
                or len(set(self.episode_ids)) != len(self.episode_ids)
                or any(
                    not isinstance(item, str) or not item for item in self.episode_ids
                )
            ):
                raise ValueError("Episode IDs must be distinct non-empty strings")
            if self.episode_start != 0 or self.episode_end is not None:
                raise ValueError("Select episodes by IDs or by index range")
        if self.seed is not None and (type(self.seed) is not int or self.seed < 0):
            raise ValueError("Seed must be a non-negative integer")


@dataclass
class NavigationServices:
    env: Any
    evaluator: Any
    agent: Any
    candidate_provider: Any
    actions: Any
    config: Any
    episode_end: int
    metric_protocol: Any = None


def enable_planning_maps(agent, *, scsp_enabled, print_images):
    """Keep planner-only maps available without enabling image writes."""

    if not scsp_enabled or bool(print_images):
        return False
    if bool(getattr(agent, "continuous_print_images", False)) or bool(
        getattr(agent, "episode_final_snapshot", False)
    ):
        raise RuntimeError(
            "SCSP no-image planning requires every image write gate disabled"
        )
    # GLUEAgent historically uses ``visualize`` both to materialize planner
    # inputs (explored map/frontiers) and to prepare optional render payloads.
    # Its writers remain guarded by the two flags checked above, while its
    # planner retained config.VISUALIZE=False during construction.
    agent.visualize = True
    return True


def configure_experiment(config, request):
    """Bind dataset, output and sensor-independent experiment controls."""
    config.GROUND_TRUTH_SEMANTICS = int(
        request.settings.semantic_source == "ground_truth"
    )
    if request.max_steps_per_task is not None:
        config.AGENT.max_steps = request.max_steps_per_task
    config.EXP_FIRST = False
    config.habitat.environment.iterator_options.shuffle = False
    config.DUMP_LOCATION = str(request.output)
    native = getattr(request, 'dataset_format', 'goat_mon') == 'objectnav'
    config.habitat.dataset.data_path = str(request.dataset / ('val.json.gz' if native else 'val_seen.json.gz'))
    if native:
        config.habitat.dataset.split = 'val'
        config.habitat.environment.max_episode_steps = request.max_steps_per_task
        config.ENVIRONMENT.max_num_sub_task_episodes = 1
        config.AGENT.radius = config.habitat.simulator.agents.main_agent.radius
    if request.resources.scene_directory is not None:
        config.habitat.dataset.scenes_dir = str(request.resources.scene_directory)
    if request.resources.scene_dataset is not None:
        config.habitat.simulator.scene_dataset = str(request.resources.scene_dataset)
    if request.seed is not None:
        config.seed = request.seed
        config.habitat.seed = request.seed
    all_scenes = sorted(
        path.name.removesuffix(".json.gz")
        for path in (request.dataset / "content").glob("*.json.gz")
    )
    config.habitat.dataset.content_scenes = (
        all_scenes if request.scene == "all" else [request.scene]
    )
    config.NUM_ENVIRONMENTS = 1
    config.PRINT_IMAGES = int(request.print_images)
    config.EXP_NAME = f"{config.EXP_NAME}_{request.scene_index}"
    return config


def build_services(request, stack):
    """Load default model adapters; register resources before later construction."""
    if request.dataset_format == 'mfmon':
        from experiments.mfmon import build_services as build_mfmon_services
        return build_mfmon_services(request, stack)
    from pbsp.adapters.habitat_config import get_config
    from habitat.core.env import Env
    from pbsp.adapters.goat import create_environment
    from pbsp.adapters.lagmemo_agent import LagMemoAgent
    from lagmemo.core.interfaces import DiscreteNavigationAction
    from experiments.dataset import load_runtime_tasks
    from pbsp.runtime.mapping_client import MappingClient

    native = getattr(request, 'dataset_format', 'goat_mon') == 'objectnav'
    metric_protocol = None
    if native:
        from pbsp.adapters.objectnav import create_environment, policy_category
        from experiments.objectnav import ObjectNavEvaluator, ObjectNavMetricProtocol, validate_episode_identity
        manifest_path = request.dataset/'dataset_manifest.json'
        raw = manifest_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != request.planning_settings.provenance['dataset_manifest_sha256']:
            raise ValueError('Native ObjectNav source manifest changed after planning')
        rows = json.loads(raw)['entries']
        native_identities = {(Path(row['scene_id']).name.split('.')[0], str(row['content_index'])): row for row in rows}
        if len(native_identities) != len(rows):
            raise ValueError('Native ObjectNav identities are ambiguous')

        def task_reader(scene, episode):
            return [[policy_category(native_identities[(str(scene), str(episode))]['category']), 'object']]

        metric_protocol = ObjectNavMetricProtocol(native_identities)
    else:
        task_reader = partial(load_runtime_tasks, str(request.dataset / 'content'))

    for path in (request.dataset / ('val.json.gz' if native else 'val_seen.json.gz'), request.dataset / 'content'):
        if not path.exists():
            raise FileNotFoundError(path)
    candidate_provider = MappingClient(
        socket_path=str(request.socket_path),
        top_k=request.top_k,
        connect_timeout=120.0,
        live_mode=request.live_mode,
        task_reader=task_reader,
    )
    stack.callback(candidate_provider.close)
    config = configure_experiment(
        get_config(
            str(request.resources.habitat_config), str(request.resources.agent_config)
        ),
        request,
    )
    habitat_env = Env(config)
    stack.callback(habitat_env.close)
    selected, episode_end = select_habitat_episodes(habitat_env.episodes, request)
    if native:
        for episode in selected:
            key = (Path(episode.scene_id).name.split('.')[0], str(episode.episode_id))
            validate_episode_identity(episode, native_identities[key])
    habitat_env.episode_iterator = iter(selected)
    env = create_environment(
        habitat_env,
        config=config,
        resources=request.resources,
        semantic_source=request.settings.semantic_source,
    )
    evaluator = ObjectNavEvaluator(habitat_env) if native else env.evaluator
    agent = LagMemoAgent(
        config=config,
        seem_cfg={
            "conf_path": str(request.resources.seem_config),
            "ckpt_path": str(request.resources.seem_checkpoint),
            **({"tokenizer_path": str(request.resources.seem_tokenizer)}
               if request.resources.seem_tokenizer else {}),
        },
        clip_cfg=str(request.resources.mobileclip_checkpoint),
        settings=request.settings,
        **({"matching_resources": request.resources.matching}
           if request.resources.matching else {}),
    )
    stack.callback(agent.close)
    agent.top_down_map_reader = getattr(evaluator, 'top_down_map', lambda: None)
    enable_planning_maps(
        agent,
        scsp_enabled=request.planning_settings.mode != "off",
        print_images=request.print_images,
    )
    return NavigationServices(
        env,
        evaluator,
        agent,
        candidate_provider,
        DiscreteNavigationAction,
        config,
        episode_end,
        metric_protocol,
    )


def select_habitat_episodes(episodes, request):
    if request.episode_ids is not None:
        by_id = {str(item.episode_id): item for item in episodes}
        if len(by_id) != len(episodes) or not set(request.episode_ids) <= set(by_id):
            raise ValueError("Selected episode IDs are missing or ambiguous in Habitat")
        return [by_id[key] for key in request.episode_ids], len(request.episode_ids)
    total = len(episodes)
    end = total if request.episode_end is None else request.episode_end
    if not 0 <= request.episode_start < end <= total:
        raise ValueError(
            f"invalid episode range [{request.episode_start}, {end}) for {total} episodes"
        )
    return episodes[request.episode_start : end], end


def run_navigation(request, *, services_factory=build_services):
    """Execute the native Runner and close both services on success or failure."""
    from pbsp.adapters.geometry import (
        project_goals_to_navigable,
        sample_navmesh_goal,
        strict_geodesic_distance,
    )
    from experiments.planning import PlanningBridge
    from experiments.recorder import print_timing_summary
    from experiments.runner import RunArtifacts, RunSettings, run_episodes
    from pbsp.navigation.cooldown import CategorySwitchCooldown
    from pbsp.navigation.ranking import CandidateActionReranker, CandidateRerankConfig

    settings = request.settings
    reranker = CandidateActionReranker(
        enabled=request.planning_settings.mode != "off",
        config=(
            CandidateRerankConfig(
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
            if request.planning_settings.mode != "off"
            else CandidateRerankConfig()
        ),
    )
    cooldown = CategorySwitchCooldown(
        request.category_cooldowns, match_radius_m=settings.candidate_track_radius_m
    )
    planning = PlanningBridge(
        project_goals=partial(
            project_goals_to_navigable, max_distance=settings.projection_tolerance_m
        ),
        geodesic=strict_geodesic_distance,
    )
    sampler = partial(
        sample_navmesh_goal,
        max_vertical_delta=request.max_vertical_delta,
        sample_count=request.navmesh_samples,
        escape_radius_m=request.escape_radius_m,
        escape_min_distance_m=request.escape_min_distance_m,
        random_min_distance_m=request.random_min_distance_m,
    )
    with ExitStack() as stack:
        services = services_factory(request, stack)
        results = request.output / f"results_{request.scene}" / services.config.EXP_NAME
        artifacts = RunArtifacts(results)
        from experiments.trajectory import TrajectoryEnvironment

        recorded_environment = TrajectoryEnvironment(
            services.env, directory=results,
            identities=request.planning_settings.episode_identities,
            dataset_manifest_sha256=request.planning_settings.provenance.get("dataset_manifest_sha256"),
            actions=services.actions,
            action_metadata=lambda: getattr(services.agent, "last_action_metadata", {}),
        )
        run_episodes(
            env=recorded_environment,
            evaluator=services.evaluator,
            agent=services.agent,
            candidate_provider=services.candidate_provider,
            category_cooldown=cooldown,
            reranker=reranker,
            planning=planning,
            sample_goal=sampler,
            actions=services.actions,
            settings=RunSettings(
                request.episode_start,
                services.episode_end,
                services.config.AGENT.max_steps,
                request.enable_timing,
                settings.query_refresh_cadence_steps,
                record_decisions=request.record_decisions,
            ),
            planning_settings=request.planning_settings,
            artifacts=artifacts,
            metric_protocol=services.metric_protocol,
        )
        with (results / "es_status.json").open("w") as stream:
            json.dump(artifacts.es_status_log, stream, indent=4)
        cache = getattr(getattr(services.agent, 'navigation_backend', None),
                        'distance_cache', None)
        if cache is not None and all(
            hasattr(cache, field) for field in ('hits', 'misses', 'bytes', 'max_bytes')
        ):
            with (results / 'planner_cache.json').open('w') as stream:
                json.dump({'hits': cache.hits, 'misses': cache.misses,
                           'bytes': cache.bytes, 'max_bytes': cache.max_bytes}, stream)
        if request.enable_timing and artifacts.timing_rows:
            print_timing_summary(artifacts.timing_rows, str(results), save_csv=True)
        return artifacts


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-plan", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    args = parser.parse_args(argv)
    from pbsp.runtime.plan import load_plan
    from pbsp.runtime.bootstrap import configure_run_process
    from pbsp.runtime.factories import navigation_request
    from pbsp.runtime.environment import record_environment

    plan = load_plan(args.run_plan)
    configure_run_process(plan, "navigation")
    record_environment(plan, "navigation", scene=args.scene)
    request = navigation_request(plan, args.scene)
    import signal

    print(
        f"[NavigationWorker] scene={request.scene} planning={request.planning_settings.mode} "
        f"semantic_source={request.settings.semantic_source}",
        flush=True,
    )

    def stop(signum, _frame):
        raise SystemExit(128 + signum)

    previous = {
        name: signal.signal(name, stop) for name in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        return run_navigation(request)
    finally:
        for name, handler in previous.items():
            signal.signal(name, handler)


if __name__ == "__main__":
    main()
