"""Translate Habitat sensor frames, task queries and discrete actions."""

from dataclasses import dataclass
import operator

import numpy as np

# Portions adapted from LagMemo, Copyright (c) Meta Platforms, Inc. and affiliates.
# Upstream MIT terms and provenance are retained in THIRD_PARTY_NOTICES.md.


def task_index(environment):
    """Read task progress from the adapter, including native single-target tasks."""
    index = getattr(environment, 'task_index', None)
    if index is None:
        index = environment.habitat_env.task.current_task_idx
    if isinstance(index, (bool, np.bool_)):
        raise ValueError('Task index must be a nonnegative integer')
    index = operator.index(index)
    if index < 0:
        raise ValueError('Task index must be a nonnegative integer')
    return index


@dataclass(frozen=True)
class HabitatObservationSettings:
    min_depth: float
    max_depth: float
    semantic_source: str

    def __post_init__(self):
        if not 0 <= self.min_depth < self.max_depth:
            raise ValueError("Expected 0 <= min_depth < max_depth")
        if self.semantic_source not in {"ground_truth", "detic", "disabled"}:
            raise ValueError("Unknown Habitat semantic source")


class HabitatEnvironment:
    """Keep evaluation measurements outside policy observations.

    Depth is in metres; globalpose is [qw,qx,qy,qz,x,y,z]. GPS changes from
    Habitat (x,y) to the navigator's (x,-y). The observation factory and action
    IDs are supplied by the assembly layer; importing this adapter needs no
    Habitat installation.
    """

    def __init__(
        self,
        habitat_env,
        *,
        settings: HabitatObservationSettings,
        categories,
        observation_factory,
        action_ids,
        segmentation_factory=None,
        semantic_mapping=None,
        observer=None,
    ):
        self.habitat_env = habitat_env
        self.min_depth = settings.min_depth
        self.max_depth = settings.max_depth
        self.semantic_source = settings.semantic_source
        self.categories = tuple(categories)
        self.observation_factory = observation_factory
        self.action_ids = action_ids
        self.segmentation_factory = segmentation_factory
        self.hm3d_mapping = semantic_mapping if semantic_mapping is not None else {}
        self.observer = observer
        self.current_episode = None
        self._last_obs = None
        if self.semantic_source == "detic":
            self.init_perception_module()

    def init_perception_module(self, vocabulary=None):
        if self.segmentation_factory is None:
            raise ValueError("Detic semantics requires a segmentation factory")
        self.segmentation = self.segmentation_factory(self.categories)

    def reset(self):
        habitat_obs = self.habitat_env.reset()
        self.current_episode = self.habitat_env.current_episode
        self.active_task_idx = 0
        if self.semantic_source == "detic" and not getattr(
            self.segmentation, "reuse_across_episodes", False
        ):
            self.init_perception_module()
        self._last_obs = self._preprocess_obs(habitat_obs)
        if self.observer is not None:
            self.observer.reset(self.current_episode)
        return self._last_obs

    def apply_action(self, action, info=None, prev_obs=None):
        if info is not None:
            self._process_info(info)
        habitat_action = self._preprocess_action(action)
        habitat_obs = self.habitat_env.step(habitat_action)
        self._last_obs = self._preprocess_obs(habitat_obs)

    def get_observation(self):
        return self._last_obs

    def _preprocess_xy(self, xy):
        return np.array([xy[0], -1 * xy[1]])

    def _preprocess_obs(self, habitat_obs):
        depth = habitat_obs["depth"][:, :, -1]
        goals = self._preprocess_goals(self.current_episode.tasks, habitat_obs)
        obs = self.observation_factory(
            rgb=habitat_obs["rgb"],
            depth=depth,
            compass=habitat_obs["compass"],
            gps=self._preprocess_xy(habitat_obs["gps"]),
            task_observations={"tasks": goals},
            camera_pose=None,
            third_person_image=None,
            globalpose=habitat_obs["globalpose"],
        )
        habitat_semantic = (
            habitat_obs["semantic"] if self.semantic_source == "ground_truth" else None
        )
        return self._preprocess_semantic(obs, habitat_semantic)

    def _preprocess_semantic(self, obs, habitat_semantic, vocabulary=None):
        if self.semantic_source == "disabled":
            obs.semantic = np.zeros(obs.rgb.shape[:2], dtype=np.int64)
            obs.task_observations["instance_map"] = np.zeros(
                obs.rgb.shape[:2], dtype=np.int64
            )
        elif self.semantic_source == "ground_truth":
            if habitat_semantic is None:
                raise ValueError("Ground-truth semantic observation is missing")
            obs.semantic = np.vectorize(lambda x: self.hm3d_mapping.get(x, 0))(
                habitat_semantic
            )[..., 0]
            obs.task_observations["instance_map"] = habitat_semantic[:, :, -1] + 1
            obs.task_observations["semantic_frame"] = obs.rgb
        elif self.semantic_source == "detic":
            obs = self.segmentation.predict(obs)
        obs.task_observations["semantic_frame"] = np.concatenate(
            [obs.rgb, obs.semantic[:, :, np.newaxis]], axis=2
        ).astype(np.uint8)
        return obs

    def _preprocess_depth(self, depth):
        rescaled_depth = self.min_depth + depth * (self.max_depth - self.min_depth)
        rescaled_depth[depth == 0.0] = 10000
        rescaled_depth[depth == 1.0] = 10001
        return rescaled_depth[:, :, -1]

    def _preprocess_goals(self, tasks, habitat_obs):
        goals = habitat_obs["multigoal"]
        for index, goal in enumerate(goals):
            goal["semantic_id"] = self.categories.index(
                "_".join(goal["category"].split(" "))
            ) + 1
            if goal["image"] is not None and tasks[index][1] == "image":
                goal["type"] = "imagenav"
            elif goal["description"]:
                goal["type"] = "languagenav"
            else:
                goal["type"] = "objectnav"
        return goals

    def _preprocess_action(self, action):
        if type(action) == int:
            return action
        return self.action_ids[action.name.lower()]

    def _process_info(self, info):
        if self.observer is not None:
            self.observer.on_action(info)

    @property
    def task_index(self):
        return self.habitat_env.task.current_task_idx

    @property
    def episode_over(self):
        return self.habitat_env.episode_over

    @property
    def number_of_episodes(self):
        return self.habitat_env.number_of_episodes

    @property
    def observation_space(self):
        return self.habitat_env.observation_space

    @property
    def action_space(self):
        return self.habitat_env.action_space

    @property
    def original_action_space(self):
        return self.habitat_env.original_action_space

    def seed(self, seed=None):
        return self.habitat_env.seed(seed)

    def close(self):
        return self.habitat_env.close()
