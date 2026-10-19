"""Assemble the default Habitat sensor and Detic adapters."""

# Portions adapted from LagMemo, Copyright (c) Meta Platforms, Inc. and affiliates.
# Upstream MIT terms and provenance are retained in THIRD_PARTY_NOTICES.md.

from functools import partial
import json
from pathlib import Path

from adapters.detic import create_segmentation
from adapters.habitat import HabitatEnvironment, HabitatObservationSettings
from experiments.evaluator import GoatEvaluator


def load_categories(path):
    categories = json.loads(Path(path).read_text())
    return tuple("_".join(name.split(" ")) for name in sorted(set(categories)))


def _ground_truth_mapping(habitat_env, categories):
    from lagmemo.perception.constants import df

    mapping = {}
    for obj in habitat_env.sim.semantic_scene.objects:
        main_category = df[df["raw_category"] == obj.category.name()]
        if len(main_category) == 0:
            continue
        category = "_".join(main_category["category"].item().split(" "))
        if category in categories:
            mapping[int(obj.id.split("_")[-1])] = categories.index(category) + 1
    return mapping


class HabitatGoatEnv(HabitatEnvironment):
    def __init__(
        self,
        habitat_env,
        config,
        *,
        settings,
        categories_path,
        detic_config=None,
        detic_checkpoint=None,
    ):
        from habitat.sims.habitat_simulator.actions import HabitatSimActions
        from lagmemo.core.interfaces import Observations
        from lagmemo.env.visualizer import Visualizer
        from experiments.visualization import HabitatVisualization

        self.config = config
        self.ground_truth_semantics = config.GROUND_TRUTH_SEMANTICS
        self.episodes_data_path = config.habitat.dataset.data_path
        assert (
            "hm3d" in self.episodes_data_path.lower()
            or "hm3d" in str(config.habitat.simulator.scene_dataset).lower()
        ), "only HM3D scenes supported for now."
        categories = load_categories(categories_path)
        self.evaluator = GoatEvaluator(habitat_env)
        self.visualizer = Visualizer(config)
        observer = HabitatVisualization(self.visualizer, habitat_env, self.evaluator)
        mapping = (
            _ground_truth_mapping(habitat_env, categories)
            if settings.semantic_source == "ground_truth"
            else None
        )
        super().__init__(
            habitat_env,
            settings=settings,
            categories=categories,
            observation_factory=Observations,
            action_ids=HabitatSimActions,
            segmentation_factory=partial(
                create_segmentation,
                gpu_id=(-1 if config.NO_GPU else habitat_env.sim.gpu_device),
                config_file=str(detic_config) if detic_config else None,
                checkpoint_file=str(detic_checkpoint) if detic_checkpoint else None,
            ),
            semantic_mapping=mapping,
            observer=observer,
        )
        print(f"[HabitatGoatEnv] semantic_source={self.semantic_source}", flush=True)

    def get_episode_metrics(self):
        return self.evaluator.get_episode_metrics()

    def get_subtask_path_lengths(self):
        return self.evaluator.get_subtask_path_lengths()

    def reset_subtask(self):
        return self.evaluator.reset_subtask()


def create_environment(habitat_env, *, config, resources, semantic_source):
    return HabitatGoatEnv(
        habitat_env,
        config=config,
        settings=HabitatObservationSettings(
            min_depth=config.ENVIRONMENT.min_depth,
            max_depth=config.ENVIRONMENT.max_depth,
            semantic_source=semantic_source,
        ),
        categories_path=resources.categories,
        detic_config=resources.detic_config,
        detic_checkpoint=resources.detic_checkpoint,
    )
