"""Bind retained environment controls to the native Habitat adapter."""

import os

from adapters.goat import (
    HabitatGoatEnv as _HabitatGoatEnv,
    load_categories,
    _ground_truth_mapping,
)
from adapters.habitat import HabitatObservationSettings


def observation_settings_from_environment(config, environ=None):
    environment = os.environ if environ is None else environ
    source = environment.get("LAGMEMO_SEMANTIC_SOURCE", "auto").lower()
    if source == "auto":
        source = "ground_truth" if config.GROUND_TRUTH_SEMANTICS else "detic"
    return HabitatObservationSettings(
        min_depth=config.ENVIRONMENT.min_depth,
        max_depth=config.ENVIRONMENT.max_depth,
        semantic_source=source,
    )


class HabitatGoatEnv(_HabitatGoatEnv):
    def __init__(
        self,
        habitat_env,
        config,
        *,
        settings=None,
        categories_path="data/goal_list.json",
    ):
        super().__init__(
            habitat_env,
            config=config,
            settings=settings or observation_settings_from_environment(config),
            categories_path=categories_path,
        )
