"""Observation preparation and persistent local-map updates."""

from __future__ import annotations

# Portions adapted from LagMemo, Copyright (c) Meta Platforms, Inc. and affiliates.
# Upstream MIT terms and provenance are retained in THIRD_PARTY_NOTICES.md.

import logging
from typing import Any, Dict, List, Tuple
import numpy as np
import torch
import scipy
from sklearn.cluster import DBSCAN
from .geometry import get_rel_pose_change
from .types import NavigationObservation

LOGGER = logging.getLogger(__name__)


class LocalMapState:

    @torch.no_grad()
    def prepare_planner_inputs(
        self,
        obs: torch.Tensor,
        pose_delta: torch.Tensor,
        object_goal_category: torch.Tensor = None,
        camera_pose: torch.Tensor = None,
        reject_visited_targets: bool = False,
        blacklist_target: bool = False,
        matches=None,
        confidence=None,
        local_instance_ids=None,
        all_matches=None,
        all_confidences=None,
        instance_ids=None,
        score_thresh=0.0,
        obstacle_locations: torch.Tensor = None,
        free_locations: torch.Tensor = None,
    ) -> Tuple[List[dict], List[dict]]:
        """Build planner inputs and the explored-map context needed for waypoint projection.

        RGB-D and relative pose are batched by environment. Map context is computed
        for every step, including runs with image recording disabled.
        """
        dones = torch.tensor([False] * self.num_environments)
        update_global = torch.tensor(
            [self.timesteps_before_goal_update[e] == 0 for e in range(self.num_environments)]
        )
        if obstacle_locations is not None:
            obstacle_locations = obstacle_locations.unsqueeze(1)
        if free_locations is not None:
            free_locations = free_locations.unsqueeze(1)
        if object_goal_category is not None:
            object_goal_category = object_goal_category.unsqueeze(1)
        (
            self.goal_map,
            self.found_goal,
            self.goal_pose,
            frontier_map,
            self.semantic_map.local_map,
            self.semantic_map.global_map,
            seq_local_pose,
            seq_global_pose,
            seq_lmb,
            seq_origins,
            frontiers,
        ) = self.module(
            obs.unsqueeze(1),
            pose_delta.unsqueeze(1),
            dones.unsqueeze(1),
            update_global.unsqueeze(1),
            camera_pose,
            self.found_goal,
            self.goal_map,
            self.semantic_map.local_map,
            self.semantic_map.global_map,
            self.semantic_map.local_pose,
            self.semantic_map.global_pose,
            self.semantic_map.lmb,
            self.semantic_map.origins,
            seq_object_goal_category=object_goal_category,
            reject_visited_targets=reject_visited_targets,
            blacklist_target=blacklist_target,
            matches=matches,
            confidence=confidence,
            local_instance_ids=local_instance_ids,
            all_matches=all_matches,
            all_confidences=all_confidences,
            instance_ids=instance_ids,
            score_thresh=score_thresh,
            seq_obstacle_locations=obstacle_locations,
            seq_free_locations=free_locations,
            camera_pose_yaw=self.semantic_map.get_planner_pose_inputs(0)[2],
        )
        self.semantic_map.local_pose = seq_local_pose[:, -1]
        self.semantic_map.global_pose = seq_global_pose[:, -1]
        self.semantic_map.lmb = seq_lmb[:, -1]
        self.semantic_map.origins = seq_origins[:, -1]
        goal_map = self.goal_map.squeeze(1).cpu().numpy()
        if self.found_goal[0].item():
            goal_map = self._prep_goal_map_input()
        for e in range(self.num_environments):
            if frontier_map is not None:
                self.semantic_map.update_frontier_map(e, frontier_map[e][0].cpu().numpy())
            if self.found_goal[e] or self.timesteps_before_goal_update[e] == 0:
                self.semantic_map.update_global_goal_for_env(e, goal_map[e])
                if self.timesteps_before_goal_update[e] == 0:
                    self.timesteps_before_goal_update[e] = self.goal_update_steps
            self.total_timesteps[e] = self.total_timesteps[e] + 1
            counters = self.sub_task_timesteps[e]
            if 0 <= self.current_task_idx < len(counters):
                counters[self.current_task_idx] += 1
            else:
                LOGGER.warning(
                    "Skipping invalid subtask counter: environment=%d task=%d slots=%d",
                    e,
                    self.current_task_idx,
                    len(counters),
                )
            self.timesteps_before_goal_update[e] = self.timesteps_before_goal_update[e] - 1
        planner_inputs = [
            {
                "obstacle_map": self.semantic_map.get_obstacle_map(e),
                "goal_map": self.semantic_map.get_goal_map(e),
                "frontier_map": self.semantic_map.get_frontier_map(e),
                "sensor_pose": self.semantic_map.get_planner_pose_inputs(e),
                "found_goal": self.found_goal[e].item(),
                "goal_pose": self.goal_pose[e] if self.goal_pose is not None else None,
            }
            for e in range(self.num_environments)
        ]
        vis_inputs = [
            {
                "explored_map": self.semantic_map.get_explored_map(e),
                "been_close_map": self.semantic_map.get_been_close_map(e),
                "timestep": self.total_timesteps[e],
                "frontiers": frontiers,
            }
            for e in range(self.num_environments)
        ]
        for e in range(self.num_environments):
            vis_inputs[e]["semantic_map"] = np.zeros_like(vis_inputs[e]["explored_map"])
        if self.record_instance_ids:
            for e in range(self.num_environments):
                vis_inputs[e]["instance_map"] = np.zeros_like(vis_inputs[e]["explored_map"])
        return (planner_inputs, vis_inputs)

    def reset_vectorized(self):
        """Initialize agent state."""
        self.total_timesteps = [0] * self.num_environments
        self.sub_task_timesteps = [[0] * self.max_num_sub_task_episodes] * self.num_environments
        self.timesteps_before_goal_update = [0] * self.num_environments
        self.last_poses = [np.zeros(3)] * self.num_environments
        self.semantic_map.init_map_and_pose()
        self.episode_panorama_start_steps = self.panorama_start_steps
        self.reached_goal_panorama_rotate_steps = self.panorama_rotate_steps
        if self.instance_memory is not None:
            self.instance_memory.reset()
        self.reject_visited_targets = False
        self.blacklist_target = False
        self.current_task_idx = 0
        self.fully_explored = [False] * self.num_environments
        self._clear_seem_verification_state()
        self.force_match_against_memory = False
        self.visited_candidate_positions = {}
        self.candidate_evidence.reset()
        self.candidate_evidence_diagnostics = {}
        self.last_external_ranking_metadata = None
        self.frontier_reranker.reset()
        self.frontier_rerank_diagnostics = {}
        self.trapped_events = []
        self._reset_trapped_state()
        self._clear_navmesh_override_goal()
        if self.imagenav_visualizer is not None:
            self.imagenav_visualizer.reset()
        self.goal_image = None
        self.goal_mask = None
        self.goal_image_keypoints = None
        self.found_goal[:] = False
        self.goal_map[:] *= 0
        self.prev_task_type = None
        self.planner.reset()
        self._module.reset()

    def reset_vectorized_for_env(self, e: int):
        """Initialize agent state for a specific environment."""
        self.total_timesteps[e] = 0
        self.sub_task_timesteps[e] = [0] * self.max_num_sub_task_episodes
        self.timesteps_before_goal_update[e] = 0
        self.last_poses[e] = np.zeros(3)
        self.semantic_map.init_map_and_pose_for_env(e)
        self.episode_panorama_start_steps = self.panorama_start_steps
        self.reached_goal_panorama_rotate_steps = self.panorama_rotate_steps
        if self.instance_memory is not None:
            self.instance_memory.reset_for_env(e)
        self.reject_visited_targets = False
        self.blacklist_target = False
        self.current_task_idx = 0
        self.fully_explored = [False] * self.num_environments
        self._clear_seem_verification_state()
        self.visited_candidate_positions = {}
        self.candidate_evidence.reset()
        self.candidate_evidence_diagnostics = {}
        self.last_external_ranking_metadata = None
        self.frontier_reranker.reset()
        self.frontier_rerank_diagnostics = {}
        self._reset_trapped_state()
        self._clear_navmesh_override_goal()
        self.planner.reset()
        self._module.reset()
        self.goal_image = None
        self.goal_image_keypoints = None
        self.goal_mask = None

    def _preprocess_obs(self, obs: NavigationObservation, task_type: str):
        """Take a home-robot observation, preprocess it to put it into the correct format for the
        semantic map."""
        rgb = torch.from_numpy(obs.rgb).to(self.device)
        depth = torch.from_numpy(obs.depth).unsqueeze(-1).to(self.device) * 100.0
        current_task = obs.task_observations["tasks"][self.current_task_idx]
        current_goal_semantic_id = current_task["semantic_id"]
        semantic = obs.semantic
        instance_ids = None
        (
            matches,
            confidences,
            keypoints,
            local_instance_ids,
            all_matches,
            all_confidences,
            all_rgb_keypoints,
            instance_ids,
        ) = (None, None, None, None, [], [], [], [])
        semantic = self.one_hot_encoding[torch.from_numpy(semantic).to(self.device)]
        obs_preprocessed = torch.cat([rgb, depth, semantic], dim=-1)
        obs_preprocessed = obs_preprocessed.unsqueeze(0).permute(0, 3, 1, 2)
        curr_pose = np.array([obs.gps[0], obs.gps[1], obs.compass[0]])
        pose_delta = torch.tensor(get_rel_pose_change(curr_pose, self.last_poses[0])).unsqueeze(0)
        self.last_poses[0] = curr_pose
        object_goal_category = torch.tensor(current_goal_semantic_id).unsqueeze(0)
        camera_pose = obs.camera_pose
        if camera_pose is not None:
            camera_pose = torch.tensor(np.asarray(camera_pose)).unsqueeze(0)
        if (
            task_type in ["languagenav", "imagenav"]
            and self.record_instance_ids
            and (
                self.sub_task_timesteps[0][self.current_task_idx] == 0
                or self.force_match_against_memory
            )
        ):
            if self.force_match_against_memory:
                print("Force a match against the memory")
            self.force_match_against_memory = False
        return (
            obs_preprocessed,
            pose_delta,
            object_goal_category,
            camera_pose,
            matches,
            confidences,
            local_instance_ids,
            all_matches,
            all_confidences,
            instance_ids,
        )

    def _prep_goal_map_input(self) -> None:
        """
        Perform optional clustering of the goal channel to mitigate noisy projection
        splatter.
        """
        goal_map = self.goal_map.squeeze(1).cpu().numpy()
        if not self.goal_filtering:
            return goal_map
        for e in range(goal_map.shape[0]):
            if not self.found_goal[e]:
                continue
            try:
                c = DBSCAN(eps=4, min_samples=1)
                data = np.array(goal_map[e].nonzero()).T
                c.fit(data)
                mode = scipy.stats.mode(c.labels_, keepdims=False).mode.item()
                mode_mask = (c.labels_ != mode).nonzero()
                x = data[mode_mask]
                goal_map_ = np.copy(goal_map[e])
                goal_map_[x] = 0.0
                if goal_map_.sum() > 0:
                    goal_map[e] = goal_map_
            except Exception as e:
                print(e)
                return goal_map
        return goal_map
