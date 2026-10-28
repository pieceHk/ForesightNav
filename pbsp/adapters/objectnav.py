"""Translate native ObjectNav targets and RGB-D sensors for the PBSP policy."""

from pbsp.adapters.habitat import HabitatEnvironment


def policy_category(category):
    """Translate official targets to the existing PBSP vocabulary."""
    if not isinstance(category, str) or not category.strip():
        raise ValueError('ObjectNav requires a target category')
    return {'tv_monitor': 'tv', 'sofa': 'couch'}.get(category, category)


class HabitatObjectNavEnvironment(HabitatEnvironment):
    """A native episode has one target; evaluation labels remain in Habitat."""

    def __init__(self, habitat_env, **kwargs):
        if kwargs['settings'].semantic_source == 'ground_truth':
            raise ValueError('ObjectNav PBSP evaluation uses predicted semantics')
        super().__init__(habitat_env, **kwargs)

    @property
    def task_index(self):
        return 0

    @property
    def action_end_pose(self):
        return self._last_obs.globalpose

    def _preprocess_obs(self, habitat_obs):
        category = policy_category(self.current_episode.object_category)
        if category not in self.categories:
            raise ValueError('ObjectNav category is absent from the policy vocabulary: ' + category)
        observation = self.observation_factory(
            rgb=habitat_obs['rgb'],
            depth=habitat_obs['depth'][:, :, -1],
            compass=habitat_obs['compass'],
            gps=self._preprocess_xy(habitat_obs['gps']),
            task_observations={'tasks': [{
                'category': category, 'description': '', 'image': None,
                'semantic_id': self.categories.index(category) + 1, 'type': 'objectnav',
            }]},
            camera_pose=None,
            third_person_image=None,
            globalpose=habitat_obs['globalpose'],
        )
        return self._preprocess_semantic(observation, None)


def create_environment(habitat_env, *, config, resources, semantic_source):
    from functools import partial
    from habitat.sims.habitat_simulator.actions import HabitatSimActions
    from lagmemo.core.interfaces import Observations
    from pbsp.adapters.detic import create_segmentation
    from pbsp.adapters.goat import load_categories
    from pbsp.adapters.habitat import HabitatObservationSettings

    return HabitatObjectNavEnvironment(habitat_env,
        settings=HabitatObservationSettings(config.ENVIRONMENT.min_depth,
                                          config.ENVIRONMENT.max_depth, semantic_source),
        categories=load_categories(resources.categories), observation_factory=Observations,
        action_ids=HabitatSimActions,
        segmentation_factory=partial(create_segmentation,
            gpu_id=(-1 if config.NO_GPU else habitat_env.sim.gpu_device),
            config_file=str(resources.detic_config), checkpoint_file=str(resources.detic_checkpoint),
            category_frequency=resources.detic_category_frequency,
            clip_checkpoint=resources.matching.clip if resources.matching else None))
