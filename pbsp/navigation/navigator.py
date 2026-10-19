"""Execute online waypoints, verify goals and advance ordered tasks."""

from __future__ import annotations

# Portions adapted from LagMemo, Copyright (c) Meta Platforms, Inc. and affiliates.
# Upstream MIT terms and provenance are retained in THIRD_PARTY_NOTICES.md.

import logging
import math
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import torch

from .candidates import CandidateTracking
from .geometry import (
    GoalProjection,
    compute_transform_params_v2,
    draw_goal_circle_to_explored,
    global_to_gps2,
)
from .local_map import LocalMapState
from .recovery import RecoveryState, should_request_random_exploration
from .verification import GoalVerifier, update_multiview_verification
from .evidence import passes_dual_score_verification
from .types import NavigationObservation
from .backends import NavigationFrame, NavigationRequest, navigation_backend

LOGGER = logging.getLogger(__name__)


class Navigator(CandidateTracking, GoalVerifier, RecoveryState, LocalMapState, GoalProjection):
    """Run the navigation policy with models and an action enum supplied by its adapter."""

    def __init__(self, *, actions, navigation_backend=None):
        self.actions = actions
        self.navigation_backend = navigation_backend
        self._es_test_gs_override = 3

    def _write_navigation_snapshot(self, info, action, *, observation):
        """Optional observer supplied by the experiment adapter."""
        return None

    def _update_navigation_diagnostics(self, obs, goal, *, initialize_transform):
        """Optional visualization observer; its output does not select actions."""
        return None

    def set_runtime_context(self, scene_id, episode_id):
        self.runtime_scene_id = scene_id
        self.runtime_episode_id = episode_id

    def reset_sub_episode(self) -> None:
        """Reset for a new sub-episode since pre-processing is temporally dependent."""
        self.goal_image = None
        self.goal_image_keypoints = None
        self.goal_mask = None
        self.fully_explored = [False] * self.num_environments
        self.force_match_against_memory = False
        self.frontier_blacklist = []
        self.prev_position = None
        self.ctr = 0
        self.stuck_num = 0
        self.see_around = 0
        self._clear_seem_verification_state()
        self.candidate_evidence.reset()
        self.candidate_evidence_diagnostics = {}
        self.last_external_ranking_metadata = None
        self.frontier_reranker.reset()
        self.frontier_rerank_diagnostics = {}
        self._reset_trapped_state()
        self._clear_navmesh_override_goal()
        navigation_backend(self).reset_task()
        self._module.reset_sub_episode()

    def reset(
        self, start_position=None, start_rotation=None, lagmemo_goals=None, candidate_provider=None
    ):
        """Initialize agent state."""
        self.reset_vectorized()
        navigation_backend(self).reset_episode()
        self.goal_image = None
        self.goal_mask = None
        self.goal_image_keypoints = None
        self.prev_position = None
        self.ctr = 0
        init_local_pose = np.array([0, 0, 0])
        (qx, qy, qz, qw) = start_rotation
        init_global_pose = np.array(
            [qx, qy, qz, qw, start_position[0], start_position[1], start_position[2]]
        )
        (self.rot_matrix_v2, self.tran_v2) = compute_transform_params_v2(
            init_global_pose, init_local_pose
        )
        self.td_transform = None
        if candidate_provider is not None:
            self.candidate_provider = candidate_provider
            self.gs_poses = candidate_provider.get_all_goals()
        else:
            self.candidate_provider = None
            self.gs_poses = lagmemo_goals
        self.frontier_flag = {int(i): 0 for i in self.gs_poses}
        if self.exp_first:
            gs_poses = {0: self.gs_poses[0]}
            for i in range(1, len(self.gs_poses) + 1):
                gs_poses[i] = self.gs_poses[i - 1]
            self.gs_poses = gs_poses
            self.frontier_flag = {i: 0 for i in range(len(self.gs_poses))}

    def score_thresh(self, task_type):
        if self.fully_explored[0]:
            return 0.0
        if task_type == "languagenav":
            return self.goal_policy_config.score_thresh_lang
        elif task_type == "imagenav":
            return self.goal_policy_config.score_thresh_image
        else:
            return 0.0

    def act(self, obs: NavigationObservation) -> Tuple[Any, Optional[dict]]:
        """Update the map, select a waypoint, verify it and return one local action."""
        current_task = obs.task_observations["tasks"][self.current_task_idx]
        task_type = current_task["type"]
        stop_reason = None
        planner_error = None
        info = None
        random_exploration_requested = False
        trapped_event = None
        verification_event = None
        frontier_rerank_applied = False
        self.frontier_rerank_diagnostics = {}
        compass = obs.compass
        (
            obs_preprocessed,
            pose_delta,
            object_goal_category,
            camera_pose,
            matches,
            confidence,
            local_instance_ids,
            all_matches,
            all_confidences,
            instance_ids,
        ) = self._preprocess_obs(obs, task_type)
        (planner_inputs, vis_inputs) = self.prepare_planner_inputs(
            obs_preprocessed,
            pose_delta,
            object_goal_category=object_goal_category,
            camera_pose=camera_pose,
            reject_visited_targets=self.reject_visited_targets,
            matches=matches,
            confidence=confidence,
            local_instance_ids=local_instance_ids,
            all_matches=all_matches,
            all_confidences=all_confidences,
            instance_ids=instance_ids,
            score_thresh=self.score_thresh(task_type),
        )
        stop_by_clip = False
        try:
            sp = planner_inputs[0]["sensor_pose"][3:].astype(np.int32)
            for e in range(self.num_environments):
                if self.frontier_blacklist:
                    mask = np.zeros_like(planner_inputs[e]["frontier_map"], dtype=np.uint8)
                    for frontier in self.frontier_blacklist:
                        (x, y) = (int(frontier[0] - sp[2]), int(frontier[1] - sp[0]))
                        if x < 0 or x >= mask.shape[1] or y < 0 or (y >= mask.shape[0]):
                            continue
                        mask[y, x] = 1
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
                    dilated_mask = cv2.dilate(mask, kernel, iterations=1)
                    planner_inputs[e]["frontier_map"][dilated_mask == 1] = 0
                    filtered_frontiers = []
                    for frontier in vis_inputs[0]["frontiers"]:
                        is_near_blacklist = any(
                            (
                                np.linalg.norm(np.array(frontier) - np.array(blacklist_point)) < 5
                                for blacklist_point in self.frontier_blacklist
                            )
                        )
                        if not is_near_blacklist:
                            filtered_frontiers.append(frontier)
                    vis_inputs[0]["frontiers"] = filtered_frontiers
        except (IndexError, KeyError, TypeError, ValueError) as error:
            LOGGER.warning("Frontier blacklist filtering skipped: %s", error, exc_info=True)
        match_score = 0
        goal_global_pose = None
        frontier_global_pose = None
        global_pose = obs.globalpose
        local_pose = obs.gps
        (qx, qy, qz, qw, x_global, _, y_global) = global_pose
        test_point_v2 = global_pose[4:]
        converted_point_v2 = global_to_gps2(test_point_v2, self.rot_matrix_v2, self.tran_v2)
        cur_self_pixel_pose = planner_inputs[0]["sensor_pose"][:2] * 20 - np.array(
            [planner_inputs[0]["sensor_pose"][5], planner_inputs[0]["sensor_pose"][3]]
        )
        current_task = obs.task_observations["tasks"][self.current_task_idx]
        task_type = current_task["type"]
        goal_global_pose = None
        frontier_global_pose = None
        override_active = (
            self.navmesh_override_goal is not None
            and self.navmesh_override_task == self.current_task_idx
        )
        if self.see_around > 12:
            self.see_around = 0
            self._clear_candidate_verification_evidence()
            if override_active:
                self._clear_navmesh_override_goal()
                override_active = False
            else:
                self._mark_current_candidate_visited()
                self.frontier_flag[self.current_task_idx] += 1
            pass
        if (
            self.current_task_idx not in self.frontier_flag
            or self.current_task_idx not in self.gs_poses
        ):
            pass
        elif self.frontier_flag[self.current_task_idx] < len(self.gs_poses[self.current_task_idx]):
            frontier_global_pose = self.gs_poses[self.current_task_idx][
                self.frontier_flag[self.current_task_idx]
            ]
        else:
            pass
        if override_active:
            override_goal = self.navmesh_override_goal
            override_distance = np.linalg.norm(
                np.asarray(obs.globalpose)[[4, 6]] - override_goal[[0, 2]]
            )
            if self.navmesh_override_mode == "escape" and override_distance <= 0.6:
                self._clear_navmesh_override_goal()
            else:
                frontier_global_pose = override_goal
        if frontier_global_pose is None and self.frontier_reranker.enabled:
            (ranked_frontier, self.frontier_rerank_diagnostics) = self.frontier_reranker.rank(
                planner_inputs[0]["frontier_map"],
                planner_inputs[0]["obstacle_map"],
                vis_inputs[0].get("explored_map", planner_inputs[0]["obstacle_map"]),
                [cur_self_pixel_pose[1], cur_self_pixel_pose[0]],
            )
            if self.frontier_rerank_diagnostics.get("components", 0):
                planner_inputs[0]["frontier_map"] = ranked_frontier
                frontier_rerank_applied = True
                if self.frontier_rerank_diagnostics.get("reordered"):
                    print(f"[FRONTIER_RERANK] {self.frontier_rerank_diagnostics}", flush=True)
        if should_request_random_exploration(
            frontier_global_pose is not None,
            planner_inputs[0]["frontier_map"],
            vis_inputs[0].get("explored_map", planner_inputs[0]["obstacle_map"]),
            self.sub_task_timesteps[0][self.current_task_idx],
            self.random_exploration_warmup_steps,
            self.fully_explored[0],
        ):
            self.fully_explored[0] = True
            self.force_match_against_memory = True
            random_exploration_requested = True
            self.see_around = 1
            print(
                f"[RANDOM_EXPLORATION] task={self.current_task_idx} reason=no_candidate_no_frontier_map_complete",
                flush=True,
            )
        if self.stop_by_seem:
            try:
                seem_goal_poses = np.asarray(self.seem_goal_pose)
            except (TypeError, ValueError):
                seem_goal_poses = np.empty((0, 2))
            if (
                seem_goal_poses.ndim != 2
                or seem_goal_poses.shape[0] == 0
                or seem_goal_poses.shape[1] < 2
            ):
                print(
                    f"[SEEM_EMPTY_GOAL] task={self.current_task_idx} fallback=es_candidate_or_exploration",
                    flush=True,
                )
                self._clear_seem_verification_state()
        if self.stop_by_seem:
            min_dist = 1000000.0
            best_goal_idx = -1
            for idx, seem_goal in enumerate(self.seem_goal_pose):
                cur_goal_gps_pose = np.array([seem_goal[0], seem_goal[1]])
                dist = np.linalg.norm(local_pose - cur_goal_gps_pose)
                if dist < min_dist:
                    best_goal_idx = idx
                    min_dist = dist
                    goal_gps_pose = cur_goal_gps_pose
            if best_goal_idx < 0:
                LOGGER.warning(
                    "SEEM requested a stop without a valid goal pose; falling back to exploration"
                )
                self.stop_by_seem = False
            else:
                relative_pose = goal_gps_pose - local_pose
                planner_goal_pose = np.arctan2(relative_pose[1], relative_pose[0])
                relative_pixel_pose = np.around(relative_pose * (100 / 5)).astype(np.int32)
                goal_pixel_pose = cur_self_pixel_pose + relative_pixel_pose
                planner_inputs[0]["found_goal"] = True
                planner_inputs[0]["goal_map"] = np.zeros_like(planner_inputs[0]["goal_map"])
                dilate_radius = 4
                (planner_inputs[0]["goal_map"], dilate_radius, goal_overlap) = (
                    draw_goal_circle_to_explored(
                        planner_inputs[0]["goal_map"],
                        vis_inputs[0]["explored_map"],
                        goal_pixel_pose[1],
                        goal_pixel_pose[0],
                        dilate_radius,
                    )
                )
                if goal_overlap < 20:
                    print(
                        f"[GOAL_MAP_RECOVERY] task={self.current_task_idx} source=seem overlap={goal_overlap}",
                        flush=True,
                    )
                if (
                    np.linalg.norm(relative_pose) < 0.6
                    or np.linalg.norm(relative_pixel_pose) < dilate_radius + 4
                ):
                    planner_inputs[0]["goal_pose"] = math.degrees(planner_goal_pose)
        elif goal_global_pose is not None:
            self.see_around = 0
            goal_gps_pose = global_to_gps2(goal_global_pose, self.rot_matrix_v2, self.tran_v2)
            relative_pose = goal_gps_pose - local_pose
            planner_goal_pose = np.arctan2(relative_pose[1], relative_pose[0])
            relative_pixel_pose = np.around(relative_pose * (100 / 5)).astype(np.int32)
            goal_pixel_pose = cur_self_pixel_pose + relative_pixel_pose
            planner_inputs[0]["found_goal"] = True
            planner_inputs[0]["goal_map"] = np.zeros_like(planner_inputs[0]["goal_map"])
            dilate_radius = 4
            (planner_inputs[0]["goal_map"], dilate_radius, goal_overlap) = (
                draw_goal_circle_to_explored(
                    planner_inputs[0]["goal_map"],
                    vis_inputs[0]["explored_map"],
                    goal_pixel_pose[1],
                    goal_pixel_pose[0],
                    dilate_radius,
                )
            )
            if goal_overlap < 20:
                print(
                    f"[GOAL_MAP_RECOVERY] task={self.current_task_idx} source=global_goal overlap={goal_overlap}",
                    flush=True,
                )
            if (
                np.linalg.norm(relative_pose) < 0.6
                or np.linalg.norm(relative_pixel_pose) < dilate_radius + 4
            ):
                planner_inputs[0]["goal_pose"] = math.degrees(planner_goal_pose)
        else:
            if frontier_global_pose is None and (not planner_inputs[0]["found_goal"]):
                planner_inputs[0]["goal_map"] = planner_inputs[0]["frontier_map"]
                planner_inputs[0]["found_goal"] = False
                planner_inputs[0]["goal_pose"] = None
                self.goal_vis_stage = 0
            if frontier_global_pose is not None:
                planner_inputs[0]["found_goal"] = False
                planner_inputs[0]["goal_pose"] = None
                planner_inputs[0]["goal_map"] = np.zeros_like(planner_inputs[0]["goal_map"])
                frontier_gps_pose = global_to_gps2(
                    frontier_global_pose, self.rot_matrix_v2, self.tran_v2
                )
                relative_pose = frontier_gps_pose - local_pose
                planner_goal_pose = np.arctan2(relative_pose[1], relative_pose[0])
                relative_pixel_pose = np.around(relative_pose * (100 / 5)).astype(np.int32)
                goal_pixel_pose = cur_self_pixel_pose + relative_pixel_pose
                dilate_radius = 6
                (planner_inputs[0]["goal_map"], dilate_radius, goal_overlap) = (
                    draw_goal_circle_to_explored(
                        planner_inputs[0]["goal_map"],
                        vis_inputs[0]["explored_map"],
                        goal_pixel_pose[1],
                        goal_pixel_pose[0],
                        dilate_radius,
                    )
                )
                if goal_overlap < 20:
                    print(
                        f"[GOAL_MAP_RECOVERY] task={self.current_task_idx} source=frontier overlap={goal_overlap}",
                        flush=True,
                    )
                if (
                    np.linalg.norm(relative_pose) < 1.2
                    or np.linalg.norm(relative_pixel_pose) < dilate_radius + 8
                ):
                    if self.see_around <= 12:
                        self.goal_vis_stage = 1
                        action = self.actions.TURN_RIGHT
                        self.see_around += 1
                        (match_score, seem_mask, seem_semantic_image) = self.lagmemo_score(
                            obs, detic_type="interest"
                        )
                        mask_pos = np.empty((0, 2), dtype=np.float32)
                        if np.sum(seem_mask) != 0:
                            seem_mask_mirrored = 1 - seem_mask.astype(np.uint8)
                            seem_mask_mirrored = seem_mask_mirrored.astype(np.uint8)
                            kernel = np.ones((5, 5), np.uint8)
                            seem_mask_mirrored = cv2.dilate(
                                seem_mask_mirrored, kernel, iterations=2
                            )
                            seem_mask = 1 - seem_mask_mirrored
                            seem_mask = seem_mask.astype(np.uint8)
                            hfov_deg = 79
                            depth_image = obs.depth
                            image_width = depth_image.shape[1]
                            mask_pos = self.mask_to_relative_2d_vectorized(
                                seem_mask, depth_image, hfov_deg, image_width, obs.compass[0]
                            )
                            if seem_semantic_image is not None:
                                self.seem_semantic = seem_semantic_image[..., ::-1]
                        if self.two_stage_verification and task_type in [
                            "objectnav",
                            "languagenav",
                        ]:
                            pose_center = (
                                np.median(mask_pos, axis=0)
                                if len(mask_pos)
                                else np.array([np.nan, np.nan])
                            )
                            step = self.sub_task_timesteps[0][self.current_task_idx]
                            (
                                self.verification_history,
                                confirmed,
                                supporting,
                                independent_supporting,
                            ) = update_multiview_verification(
                                self.verification_history,
                                step,
                                self._last_verification_triggered,
                                pose_center,
                                self._last_clip_score,
                                self._last_seem_score,
                                self.verification_window_steps,
                                self.verification_min_confirmations,
                                self.verification_pose_radius,
                                self.verification_min_peak_seem,
                                observer_position=local_pose[:2],
                                observer_yaw=float(obs.compass[0]),
                                min_observer_translation=self.verification_min_observer_translation,
                                min_observer_yaw_deg=self.verification_min_observer_yaw_deg,
                            )
                            verification_event = {
                                "triggered": bool(self._last_verification_triggered),
                                "confirmed": bool(confirmed),
                                "support": len(supporting),
                                "independent_support": len(independent_supporting),
                                "clip_score": float(self._last_clip_score),
                                "seem_score": float(self._last_seem_score),
                                "depth_points": len(mask_pos),
                            }
                            if self.verification_dual_score_gate:
                                peak_seem = max(
                                    (float(item["seem_score"]) for item in supporting), default=0.0
                                )
                                peak_clip = max(
                                    (float(item["clip_score"]) for item in supporting), default=0.0
                                )
                                raw_confirmed = confirmed
                                confirmed = passes_dual_score_verification(
                                    raw_confirmed,
                                    peak_seem,
                                    peak_clip,
                                    self.verification_strong_peak_seem,
                                    self.verification_min_peak_clip,
                                )
                                verification_event.update(
                                    {
                                        "confirmed": bool(confirmed),
                                        "dual_score_rejection": bool(
                                            raw_confirmed and (not confirmed)
                                        ),
                                        "peak_seem": peak_seem,
                                        "peak_clip": peak_clip,
                                    }
                                )
                            if confirmed:
                                self.stop_by_seem = True
                                self.goal_vis_stage = 2
                                self.seem_goal_pose = mask_pos + local_pose[:2]
                                planner_inputs = self.process_relative_poses(
                                    mask_pos, planner_inputs
                                )
                            else:
                                self.stop_by_seem = False
                                self.seem_goal_pose = None
                                self.goal_vis_stage = 1
                            print(
                                f"[VERIFY_DECISION] task={self.current_task_idx} step={step} triggered={int(self._last_verification_triggered)} confirmed={int(confirmed)} support={len(supporting)} independent={len(independent_supporting)} clip={self._last_clip_score:.4f} seem={self._last_seem_score:.4f} depth_points={len(mask_pos)}",
                                flush=True,
                            )
                        elif len(mask_pos):
                            planner_inputs = self.process_relative_poses(mask_pos, planner_inputs)
                            self.seem_goal_pose = mask_pos + local_pose[:2]
                        else:
                            self.seem_goal_pose = np.array([frontier_gps_pose])
                        print(
                            f"seem score: {match_score} clip score: {self._last_clip_score}",
                            flush=True,
                        )
                else:
                    self.goal_vis_stage = 0
                    self.see_around = 0
            elif frontier_global_pose is None:
                planner_inputs[0]["goal_map"] = planner_inputs[0]["frontier_map"]
                planner_inputs[0]["found_goal"] = False
                planner_inputs[0]["goal_pose"] = None
            else:
                self.see_around = 0
                self.goal_vis_stage = 0
        closest_goal_map = None
        dilated_obstacle_map = None
        short_term_goal = None
        could_not_find_path = False
        action = self.actions.TURN_RIGHT
        planner_stop = False
        if planner_inputs[0]["found_goal"]:
            self.episode_panorama_start_steps = 0
        if self.total_timesteps[0] < self.episode_panorama_start_steps:
            action = self.actions.TURN_RIGHT
        elif self.see_around == 0 or self.stop_by_seem:
            metadata = self.last_external_ranking_metadata or {}
            request = NavigationRequest(
                frame=NavigationFrame(obs.rgb, obs.depth, obs.globalpose, obs.gps, obs.compass),
                waypoint_world=frontier_global_pose, task_idx=self.current_task_idx,
                task_step=self.sub_task_timesteps[0][self.current_task_idx],
                scene_id=self.runtime_scene_id or "", episode_id=self.runtime_episode_id or "",
                local_maps=planner_inputs[0], use_dilation_for_stg=self.use_dilation_for_stg,
                version_token=metadata.get("version_token"),
            )
            plan = navigation_backend(self).plan(request)
            (action, closest_goal_map, short_term_goal, dilated_obstacle_map,
             could_not_find_path, planner_stop) = plan.values()
            planner_error = plan.error
        current_position = obs.gps
        if self.prev_position is None:
            self.prev_position = current_position
        initialize_diagnostics = False
        if (
            np.linalg.norm(np.array(current_position) - np.array(self.prev_position)) < 0.001
            and action == self.actions.MOVE_FORWARD
        ):
            self.ctr += 1
            if self.ctr > 20:
                print("The agent is stuck, delete current frontier")
                if self.stuck_num == 0 and closest_goal_map is not None:
                    (goal_x, goal_y) = np.where(closest_goal_map)
                    if goal_x.size and goal_y.size:
                        mean_y = int(np.mean(goal_y))
                        mean_x = int(np.mean(goal_x))
                        closest_goal = np.array([mean_y, mean_x])
                        for frontier in vis_inputs[0].get("frontiers", []):
                            distance = np.linalg.norm(closest_goal - frontier)
                            if distance < 5:
                                sp = planner_inputs[0]["sensor_pose"][3:].astype(np.int32)
                                frontier_pose = np.array([frontier[0] + sp[2], frontier[1] + sp[0]])
                                self.frontier_blacklist.append(frontier_pose)
                if self.stuck_num < 12:
                    if self.stuck_num % 2 == 0:
                        if self.stuck_flag == 0:
                            action = self.actions.TURN_RIGHT
                        else:
                            action = self.actions.TURN_LEFT
                    else:
                        action = self.actions.MOVE_FORWARD
                    self.stuck_num += 1
                else:
                    self.ctr = 0
                    self.stuck_num = 0
        else:
            self.ctr = 0
            if self.stuck_num != 0:
                self.stuck_flag = (self.stuck_flag + 1) % 2
            self.stuck_num = 0
            initialize_diagnostics = True
        self.prev_position = current_position
        self._update_navigation_diagnostics(
            obs, frontier_global_pose, initialize_transform=initialize_diagnostics
        )
        if self.sub_task_timesteps[0][self.current_task_idx] >= self.max_steps:
            print("Reached max number of steps for subgoal, calling STOP")
            action = self.actions.STOP
            stop_reason = "max_steps"
        if stop_by_clip:
            action = self.actions.STOP
            stop_reason = "verified_target"
        if could_not_find_path and (not planner_stop) and (action != self.actions.STOP):
            if frontier_rerank_applied:
                self.frontier_reranker.mark_failed()
            if planner_error:
                print(
                    f"[PLANNER_RECOVERY] task={self.current_task_idx} error={planner_error}; fully_explored unchanged",
                    flush=True,
                )
            else:
                print("Can't find a path. Map fully explored.")
                self.fully_explored[0] = True
                self.force_match_against_memory = True
        if action == self.actions.STOP and stop_reason is None:
            if self.stop_by_seem:
                stop_reason = "verified_target"
            else:
                planner_error = planner_error or "unverified_planner_stop"
                stop_reason = "planner_error"
                action = self.actions.TURN_RIGHT
                print(
                    f"[PLANNER_ERROR] suppressed STOP task={self.current_task_idx} error={planner_error}; recovery=TURN_RIGHT",
                    flush=True,
                )
        if planner_error and stop_reason is None:
            stop_reason = "planner_error"
        trapped_event = self._update_trapped_diagnostic(current_position, action)
        if self.visualize:
            vis_inputs[0]["dilated_obstacle_map"] = dilated_obstacle_map
            collision = {"is_collision": False}
            info = {
                **planner_inputs[0],
                **vis_inputs[0],
                "rgb_frame": obs.rgb,
                "semantic_frame": obs.semantic,
                "closest_goal_map": closest_goal_map,
                "last_collisions": collision,
                "short_term_goal": short_term_goal,
                "goal_vis_type": self.goal_vis_stage,
                "explore_info": (
                    f"match score: {round(match_score, 2)}" if match_score != 0 else None
                ),
            }
            if self.stop_by_seem and self.seem_semantic is not None:
                info["semantic_frame"] = self.seem_semantic
            tasks = obs.task_observations.get("tasks", [])
            current_task = (
                tasks[self.current_task_idx] if 0 <= self.current_task_idx < len(tasks) else {}
            )
            info["last_goal_image"] = current_task.get("image")
            if task_type == "imagenav" and info["last_goal_image"] is None:
                LOGGER.warning("ImageNav task has no goal image: task=%d", self.current_task_idx)
            try:
                info["goal_category"] = (
                    current_task.get("category")
                    or current_task.get("object_category")
                    or current_task.get("description")
                    or "Unavailable"
                )
            except Exception:
                info["goal_category"] = "Unavailable"
            goal_text_desc = {
                x: y
                for (x, y) in obs.task_observations["tasks"][self.current_task_idx].items()
                if x != "image"
            }
            goal_text_desc["action"] = str(action).split(".")[1]
            info["goal_text"] = goal_text_desc
            self._write_navigation_snapshot(info, action, observation=obs)
            info = None
        if action == self.actions.STOP:
            print(f"[STOP_REASON] task={self.current_task_idx} reason={stop_reason}", flush=True)
            self._clear_seem_verification_state()
            self.see_around = 0
            self.goal_disk_data = None
            if len(obs.task_observations["tasks"]) - 1 > self.current_task_idx:
                self.current_task_idx += 1
                self.force_match_against_memory = False
                self.timesteps_before_goal_update[0] = 0
                self.total_timesteps = [0] * self.num_environments
                self.found_goal = torch.zeros(
                    self.num_environments, 1, dtype=bool, device=self.device
                )
                self.reset_sub_episode()
        self.prev_task_type = task_type
        self.last_action_metadata = {
            "stop_reason": stop_reason,
            "planner_error": planner_error,
            "trapped_event_count": len(self.trapped_events),
            "trapped_event": trapped_event,
            "random_exploration_requested": random_exploration_requested,
            "navmesh_override_mode": self.navmesh_override_mode,
            "verification_event": verification_event,
            "candidate_index": int(self.frontier_flag.get(self.current_task_idx, 0)),
            "candidate_evidence": self.candidate_evidence_diagnostics.get(
                self.current_task_idx, {}
            ),
            "frontier_rerank": self.frontier_rerank_diagnostics,
        }
        return (action, info)
