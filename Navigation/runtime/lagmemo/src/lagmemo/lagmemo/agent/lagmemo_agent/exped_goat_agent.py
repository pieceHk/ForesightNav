# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import scipy
import torch
from sklearn.cluster import DBSCAN
from torch.nn import DataParallel

import lagmemo.utils.pose as pu

from lagmemo.agent.imagenav_agent.visualizer import NavVisualizer
from lagmemo.core.abstract_agent import Agent
from lagmemo.core.interfaces import DiscreteNavigationAction, Observations
from lagmemo.mapping.semantic.categorical_2d_semantic_map_state import (
    Categorical2DSemanticMapState,
)
from lagmemo.mapping.semantic.constants import MapConstants as MC
from lagmemo.mapping.semantic.instance_tracking_modules import InstanceMemory
from lagmemo.perception.detection.maskrcnn.coco_categories import coco_categories

from .frontier_agent_module import GoatAgentModule
from .lagmemo_matching import GoatMatching

from scipy.spatial.transform import Rotation as R
import math

# For visualizing exploration issues
debug_frontier_map = False


# for Gaussian goal pose test
def compute_transform_params(global_pose, local_pose):
    # 提取参数并转换为弧度
    x_g, y_g, alpha_g = global_pose
    x_l, y_l, alpha_l = local_pose
    # alpha_g = math.radians(alpha_g_deg)
    # alpha_l = math.radians(alpha_l_deg)
    
    # 计算旋转角度 θ = α_g - α_l
    theta_rad = alpha_g - alpha_l
    
    # 构造方程组矩阵 Ax = b (此处x是平移向量 [tx, ty])
    cos_theta = math.cos(theta_rad)
    sin_theta = math.sin(theta_rad)
    
    A = np.array([
        [cos_theta, -sin_theta],
        [sin_theta, cos_theta]
    ])
    
    global_pose_rot = (A @ np.array([x_g, y_g]).T).T
    
    pose_trans = np.array([x_l, y_l]) - global_pose_rot
    tx, ty = pose_trans[0], pose_trans[1]
    
    return theta_rad, tx, ty

def global_to_gps(global_point, transform_parameters):
    theta_rad, tx, ty = transform_parameters
    x_g, y_g = global_point
    
    cos_theta = math.cos(theta_rad)
    sin_theta = math.sin(theta_rad)
    A = np.array([
        [cos_theta, -sin_theta],
        [sin_theta, cos_theta]
    ])
    
    pose_rot = (A @ np.array([x_g, y_g]).T).T
    
    pose_local = pose_rot + np.array([tx,ty])
    
    pose_local_ret = np.array([-pose_local[1], -pose_local[0]])
    
    return pose_local_ret

def draw_circle(array, x, y, radius):
    """在数组的 (x, y) 处绘制一个半径为 radius 的圆"""
    # 创建网格坐标
    rows, cols = array.shape
    xx, yy = np.mgrid[:rows, :cols]
    
    # 计算每个像素到圆心的距离
    distance = np.sqrt((xx - x)**2 + (yy - y)**2)
    
    # 将圆内像素设为 1
    array[distance <= radius] = 1
    return array


class GoatAgent(Agent):
    """Simple object nav agent based on a 2D semantic map"""

    # Flag for debugging data flow and task configuraiton
    verbose = False

    def __init__(self, config, device_id: int = 0):
        self.arrive_frontier_flag = False
        self.max_steps = config.AGENT.max_steps
        # self.max_steps = [500, 500, 500, 500, 500]
        # self.max_steps = [500, 400, 300, 200, 200, 200, 200, 200, 200, 200, 200]
        # self.max_steps = [400, 300, 200, 200, 200, 200, 200, 200, 200, 200, 200]
        self.num_environments = config.NUM_ENVIRONMENTS
        self.store_all_categories_in_map = getattr(
            config.AGENT, "store_all_categories", False
        )
        if config.AGENT.panorama_start:
            self.panorama_start_steps = int(360 / config.ENVIRONMENT.turn_angle)
        else:
            self.panorama_start_steps = 0

        self.panorama_rotate_steps = int(360 / config.ENVIRONMENT.turn_angle)

        self.goal_matching_vis_dir = f"{config.DUMP_LOCATION}/goal_grounding_vis"
        Path(self.goal_matching_vis_dir).mkdir(parents=True, exist_ok=True)

        self.instance_memory = None
        self.record_instance_ids = getattr(
            config.AGENT.SEMANTIC_MAP, "record_instance_ids", False
        )

        if self.record_instance_ids:
            self.instance_memory = InstanceMemory(
                self.num_environments,
                config.AGENT.SEMANTIC_MAP.du_scale,
                debug_visualize=config.PRINT_IMAGES,
                config=config,
                mask_cropped_instances=False,
                padding_cropped_instances=200,
            )

        ## imagenav stuff
        self.goal_image = None
        self.goal_mask = None
        self.goal_image_keypoints = None

        self.goal_policy_config = config.AGENT.SUPERGLUE

        # self.instance_seg = Detic(config.AGENT.DETIC)
        self.matching = GoatMatching(
            device=0,  # config.simulator_gpu_id
            score_func=self.goal_policy_config.score_function,
            num_sem_categories=config.AGENT.SEMANTIC_MAP.num_sem_categories,
            config=config.AGENT.SUPERGLUE,
            default_vis_dir=f"{config.DUMP_LOCATION}/images/{config.EXP_NAME}",
            print_images=config.PRINT_IMAGES,
            instance_memory=self.instance_memory,
        )

        if self.goal_policy_config.batching:
            self.image_matching_function = self.matching.match_image_batch_to_image
        else:
            self.image_matching_function = self.matching.match_image_to_image

        self._module = GoatAgentModule(
            config, matching=self.matching, instance_memory=self.instance_memory
        )

        if config.NO_GPU:
            self.device = torch.device("cpu")
            self.module = self._module
        else:
            self.device_id = device_id
            self.device = torch.device(f"cuda:{self.device_id}")
            self._module = self._module.to(self.device)
            # Use DataParallel only as a wrapper to move model inputs to GPU
            self.module = DataParallel(self._module, device_ids=[self.device_id])

        self.visualize = config.VISUALIZE or config.PRINT_IMAGES
        self.use_dilation_for_stg = config.AGENT.PLANNER.use_dilation_for_stg
        self.semantic_map = Categorical2DSemanticMapState(
            device=self.device,
            num_environments=self.num_environments,
            num_sem_categories=config.AGENT.SEMANTIC_MAP.num_sem_categories,
            map_resolution=config.AGENT.SEMANTIC_MAP.map_resolution,
            map_size_cm=config.AGENT.SEMANTIC_MAP.map_size_cm,
            global_downscaling=config.AGENT.SEMANTIC_MAP.global_downscaling,
            record_instance_ids=getattr(
                config.AGENT.SEMANTIC_MAP, "record_instance_ids", False
            ),
            max_instances=getattr(config.AGENT.SEMANTIC_MAP, "max_instances", 0),
            evaluate_instance_tracking=getattr(
                config.ENVIRONMENT, "evaluate_instance_tracking", False
            ),
            instance_memory=self.instance_memory,
        )
        agent_radius_cm = config.AGENT.radius * 100.0
        agent_cell_radius = int(
            np.ceil(agent_radius_cm / config.AGENT.SEMANTIC_MAP.map_resolution)
        )
        self.max_num_sub_task_episodes = config.ENVIRONMENT.max_num_sub_task_episodes

        if (
            "planner_type" in config.AGENT.PLANNER
            and config.AGENT.PLANNER.planner_type == "old"
        ):
            print("Using frontier planner")
            from lagmemo.navigation_planner.frontier_planner import (
                DiscretePlanner
            )
            # from lagmemo.navigation_planner.old_discrete_planner import (
            #     DiscretePlanner,
            # )
        else:
            print("Using new planner")
            from lagmemo.navigation_planner.discrete_planner import DiscretePlanner

        self.planner = DiscretePlanner(
            turn_angle=config.ENVIRONMENT.turn_angle,
            collision_threshold=config.AGENT.PLANNER.collision_threshold,
            step_size=config.AGENT.PLANNER.step_size,
            obs_dilation_selem_radius=config.AGENT.PLANNER.obs_dilation_selem_radius,
            goal_dilation_selem_radius=config.AGENT.PLANNER.goal_dilation_selem_radius,
            map_size_cm=config.AGENT.SEMANTIC_MAP.map_size_cm,
            map_resolution=config.AGENT.SEMANTIC_MAP.map_resolution,
            visualize=config.VISUALIZE,
            print_images=config.PRINT_IMAGES,
            dump_location=config.DUMP_LOCATION,
            exp_name=config.EXP_NAME,
            agent_cell_radius=agent_cell_radius,
            min_obs_dilation_selem_radius=config.AGENT.PLANNER.min_obs_dilation_selem_radius,
            map_downsample_factor=config.AGENT.PLANNER.map_downsample_factor,
            map_update_frequency=config.AGENT.PLANNER.map_update_frequency,
            discrete_actions=config.AGENT.PLANNER.discrete_actions,
        )
        self.one_hot_encoding = torch.eye(
            config.AGENT.SEMANTIC_MAP.num_sem_categories, device=self.device
        )

        self.goal_update_steps = self._module.goal_update_steps
        self.sub_task_timesteps = None
        self.total_timesteps = None
        self.timesteps_before_goal_update = None
        self.episode_panorama_start_steps = None
        self.last_poses = None
        self.reject_visited_targets = False
        self.blacklist_target = False

        self.current_task_idx = 0

        self.imagenav_visualizer = NavVisualizer(
            num_sem_categories=config.AGENT.SEMANTIC_MAP.num_sem_categories,
            map_size_cm=config.AGENT.SEMANTIC_MAP.map_size_cm,
            map_resolution=config.AGENT.SEMANTIC_MAP.map_resolution,
            print_images=config.PRINT_IMAGES,
            dump_location=config.DUMP_LOCATION,
            exp_name=config.EXP_NAME,
        )
        # self.imagenav_visualizer = None
        self.found_goal = torch.zeros(
            self.num_environments, 1, dtype=bool, device=self.device
        )
        self.goal_map = torch.zeros(
            self.num_environments,
            1,
            *self.semantic_map.local_map.shape[2:],
            dtype=self.semantic_map.local_map.dtype,
            device=self.device,
        )
        self.goal_pose = None
        self.goal_filtering = config.AGENT.SEMANTIC_MAP.goal_filtering
        self.prev_task_type = None
        
        self.prev_position = None
        self.ctr = 0
        
        self.frontier_blacklist = []
        self.transform_parameters = np.array([0, 0, 0])
        self.see_around = 0
        
        self.gs_poses = {0: [(4.02532354, 0.06128961, -0.41486893), (4.75146657, 0.35748706, -0.56255001), (1.01134514, 0.03149779, -1.12796533), (2.37555182, 0.8156911, 0.20648724), (8.4817268, 0.29505294, -2.83955591)], 1: [(2.37555182, 0.8156911, 0.20648724), (-0.4788183, 0.22664495, -0.09585099), (1.53363464, 0.17891084, 1.36640209), (3.91789758, 0.0137236, -4.41809771), (1.75569129, 0.77342937, 0.70592809)], 2: [(6.13888611, -0.30134331, -5.24979829), (1.75569129, 0.77342937, 0.70592809), (2.37555182, 0.8156911, 0.20648724), (-1.88812369, -0.55209689, 2.84141846), (-1.89233425, -0.10595253, 2.77580646)], 3: [(10.51459641, 0.52345228, -2.45244212), (4.08370501, 0.04427377, -3.31230058), (10.13021166, -0.22507814, 0.36863596), (3.91789758, 0.0137236, -4.41809771), (2.63773931, 0.00353915, -3.30769355)], 4: [(3.91789758, 0.0137236, -4.41809771), (4.08370501, 0.04427377, -3.31230058), (-4.95627013, 0.30799043, 2.39829781), (-5.14862339, 0.34727266, 2.30851321), (7.16717154, -0.39369998, 1.2103122)], 5: [(8.4817268, 0.29505294, -2.83955591), (2.63773931, 0.00353915, -3.30769355), (10.51459641, 0.52345228, -2.45244212), (1.86949732, 0.33284527, -7.14055753), (1.9995978, 0.37727594, -7.17839619)], 6: [(8.14330012, -0.14373919, 4.48113855), (6.24918509, 0.29097367, -0.40364747), (2.10794202, -0.29895391, 3.66285407), (6.1065557, 0.08280667, -0.28367713), (7.32016828, 0.21789991, 0.74037728)], 7: [(-4.84901975, -0.12026228, 4.10336052), (-6.31729328, 0.25626099, 2.74495414), (9.27361365, -0.17385015, 3.93921289), (-4.43429952, 0.42302964, 3.41825165), (2.37555182, 0.8156911, 0.20648724)], 8: [(10.13021166, -0.22507814, 0.36863596), (10.51459641, 0.52345228, -2.45244212), (4.08370501, 0.04427377, -3.31230058), (3.91789758, 0.0137236, -4.41809771), (2.63773931, 0.00353915, -3.30769355)], 9: [(4.08370501, 0.04427377, -3.31230058), (3.91789758, 0.0137236, -4.41809771), (5.24540591, 0.07814172, -3.44474327), (10.73887258, 1.36460341, -1.94639935), (2.63773931, 0.00353915, -3.30769355)], 10: [(1.75569129, 0.77342937, 0.70592809), (3.91789758, 0.0137236, -4.41809771), (4.75146657, 0.35748706, -0.56255001), (1.01134514, 0.03149779, -1.12796533), (2.37555182, 0.8156911, 0.20648724)], 11: [(3.91789758, 0.0137236, -4.41809771), (8.4817268, 0.29505294, -2.83955591), (2.37555182, 0.8156911, 0.20648724), (-1.749722, 0.20563104, 5.25204095), (4.75146657, 0.35748706, -0.56255001)], 12: [(1.01134514, 0.03149779, -1.12796533), (7.85033276, -0.13707534, 5.70276215), (-1.89233425, -0.10595253, 2.77580646), (-1.88812369, -0.55209689, 2.84141846)], 13: [(4.08370501, 0.04427377, -3.31230058), (2.63773931, 0.00353915, -3.30769355), (4.31495526, 0.11044766, -1.59650574), (3.91789758, 0.0137236, -4.41809771), (5.24540591, 0.07814172, -3.44474327)], 14: [(8.4817268, 0.29505294, -2.83955591), (-1.749722, 0.20563104, 5.25204095), (3.91789758, 0.0137236, -4.41809771), (4.75146657, 0.35748706, -0.56255001), (2.63773931, 0.00353915, -3.30769355)], 15: [(6.49367416, 0.14056318, -2.38490601), (7.16717154, -0.39369998, 1.2103122), (-4.36835758, 0.3669075, 5.62629839), (-4.46489139, 0.20965659, 3.23526924), (-4.95627013, 0.30799043, 2.39829781)], 16: [(1.75569129, 0.77342937, 0.70592809), (7.66578462, 0.85489611, -5.21344331), (8.4817268, 0.29505294, -2.83955591), (6.13888611, -0.30134331, -5.24979829), (3.91789758, 0.0137236, -4.41809771)], 17: [(4.08370501, 0.04427377, -3.31230058), (2.63773931, 0.00353915, -3.30769355), (3.91789758, 0.0137236, -4.41809771), (4.31495526, 0.11044766, -1.59650574), (5.24540591, 0.07814172, -3.44474327)], 18: [(6.49367416, 0.14056318, -2.38490601), (7.16717154, -0.39369998, 1.2103122), (-4.36835758, 0.3669075, 5.62629839), (-4.46489139, 0.20965659, 3.23526924), (-4.95627013, 0.30799043, 2.39829781)], 19: [(2.31005286, -0.32574739, -5.38294881), (2.1908668, -0.24144814, -5.6382042), (1.01134514, 0.03149779, -1.12796533), (-3.83252247, -0.286133, 5.53975689), (7.93432032, -0.46474288, -3.96815278)]}
        self.gs_best_pos_idx = {0: 1, 1: 2, 2: 1, 3: 3, 4: 3, 5: 2, 6: 3, 7: 2, 8: 1, 9: 5, 10: 3, 11: 4, 12: 2, 13: 3, 14: 2, 15: 1, 16: 4, 17: 4, 18: 1, 19: 3}
        self.frontier_flag = {i:0 for i in range(len(self.gs_poses))}
        # self.frontier_flag = {i:[0 for j in range(len(self.gs_poses[i]))] for i in range(len(self.gs_poses))}

    # ------------------------------------------------------------------
    # Inference methods to interact with vectorized simulation
    # environments
    # ------------------------------------------------------------------

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
        """Prepare low-level planner inputs from an observation - this is
                the main inference function of the agent that lets it interact with
                vectorized environments.
        s
                This function assumes that the agent has been initialized.

                Args:
                    obs: current frame containing (RGB, depth, segmentation) of shape
                     (num_environments, 3 + 1 + num_sem_categories, frame_height, frame_width)
                    pose_delta: sensor pose delta (dy, dx, dtheta) since last frame
                     of shape (num_environments, 3)
                    object_goal_category: semantic category of small object goals
                    camera_pose: camera extrinsic pose of shape (num_environments, 4, 4)

                Returns:
                    planner_inputs: list of num_environments planner inputs dicts containing
                        obstacle_map: (M, M) binary np.ndarray local obstacle map
                         prediction
                        sensor_pose: (7,) np.ndarray denoting global pose (x, y, o)
                         and local map boundaries planning window (gx1, gx2, gy1, gy2)
                        goal_map: (M, M) binary np.ndarray denoting goal location
                    vis_inputs: list of num_environments visualization info dicts containing
                        explored_map: (M, M) binary np.ndarray local explored map
                         prediction
                        semantic_map: (M, M) np.ndarray containing local semantic map
                         predictions
        """
        dones = torch.tensor([False] * self.num_environments)
        update_global = torch.tensor(
            [
                self.timesteps_before_goal_update[e] == 0
                for e in range(self.num_environments)
            ]
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
            
            # 这个0还是需要调整的, wxl
            camera_pose_yaw = self.semantic_map.get_planner_pose_inputs(0)[2],
        )
        self.semantic_map.local_pose = seq_local_pose[:, -1]
        self.semantic_map.global_pose = seq_global_pose[:, -1]
        self.semantic_map.lmb = seq_lmb[:, -1]
        self.semantic_map.origins = seq_origins[:, -1]

        goal_map = self.goal_map.squeeze(1).cpu().numpy()

        if self.found_goal[0].item():
            goal_map = self._prep_goal_map_input()

        # found_goal = self.found_goal.squeeze(1).cpu()

        for e in range(self.num_environments):
            if frontier_map is not None:
                self.semantic_map.update_frontier_map(
                    e, frontier_map[e][0].cpu().numpy()
                )
            if self.found_goal[e] or self.timesteps_before_goal_update[e] == 0:
                self.semantic_map.update_global_goal_for_env(e, goal_map[e])
                if self.timesteps_before_goal_update[e] == 0:
                    self.timesteps_before_goal_update[e] = self.goal_update_steps
            self.total_timesteps[e] = self.total_timesteps[e] + 1
            try:
                self.sub_task_timesteps[e][self.current_task_idx] += 1
            except:
                import ipdb; ipdb.set_trace()
            self.timesteps_before_goal_update[e] = (
                self.timesteps_before_goal_update[e] - 1
            )


        planner_inputs = [
            {
                "obstacle_map": self.semantic_map.get_obstacle_map(e),
                "goal_map": self.semantic_map.get_goal_map(e), # 其实就是上面的goal_map
                "frontier_map": self.semantic_map.get_frontier_map(e),
                "sensor_pose": self.semantic_map.get_planner_pose_inputs(e),
                "found_goal": self.found_goal[e].item(),
                "goal_pose": self.goal_pose[e] if self.goal_pose is not None else None,
            }
            for e in range(self.num_environments)
        ]
        if self.visualize:
            vis_inputs = [
                {
                    "explored_map": self.semantic_map.get_explored_map(e),
                    "semantic_map": self.semantic_map.get_semantic_map(e),
                    "been_close_map": self.semantic_map.get_been_close_map(e),
                    "timestep": self.total_timesteps[e],
                    "frontiers": frontiers,
                }
                for e in range(self.num_environments)
            ]
            if self.record_instance_ids:
                for e in range(self.num_environments):
                    vis_inputs[e]["instance_map"] = self.semantic_map.get_instance_map(
                        e
                    )
        else:
            vis_inputs = [{} for e in range(self.num_environments)]

        return planner_inputs, vis_inputs

    def reset_vectorized(self):
        """Initialize agent state."""
        self.total_timesteps = [0] * self.num_environments
        self.sub_task_timesteps = [
            [0] * self.max_num_sub_task_episodes
        ] * self.num_environments
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
        self.force_match_against_memory = False

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

    def reset_sub_episode(self) -> None:
        """Reset for a new sub-episode since pre-processing is temporally dependent."""
        self.goal_image = None
        self.goal_image_keypoints = None
        self.goal_mask = None
        self._module.reset_sub_episode()

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
        self.planner.reset()
        self._module.reset()
        self.goal_image = None
        self.goal_image_keypoints = None
        self.goal_mask = None

    # ---------------------------------------------------------------------
    # Inference methods to interact with the robot or a single un-vectorized
    # simulation environment
    # ---------------------------------------------------------------------

    def reset(self, start_position=None, start_rotation=None):
        """Initialize agent state."""
        self.reset_vectorized()
        self.planner.reset()

        self.goal_image = None
        self.goal_mask = None
        self.goal_image_keypoints = None
        self.prev_position = None
        self.ctr = 0
        
        # for gaussian pose test
        # 初始化
        # init_global_pose = [
        #         0,
        #         0.47882,
        #         0,
        #         0.87791,
        #         11.00888,
        #         0.03566,
        #         -5.51595,
        #     ]
        init_local_pose = [0., 0., -2.4021398e-08]   # 朝向30度

        # 全局位姿：qx, qy, qz, qw, x, y, z(注意模拟器中是y轴朝上，即y轴与z轴调换)
        qx, qy, qz, qw = start_rotation
        x_global, _, y_global = start_position
        # qx, qy, qz, qw, x_global, _, y_global = init_global_pose
        quat = np.array([qx, qy, qz, qw])
        theta_global = R.from_quat(quat).as_euler('xyz')[1]  # 提取偏航角，考虑y轴垂直平面
        init_global_pose = [x_global, y_global, theta_global]
        
        # 计算转换参数
        theta_rad, tx, ty = compute_transform_params(init_global_pose, init_local_pose)
        print(f"转换参数:")
        print(f"旋转角度 θ = {math.degrees(theta_rad):.2f}°")
        print(f"平移量 tx = {tx:.2f}, ty = {ty:.2f}")
        
        self.transform_parameters = np.array([theta_rad, tx, ty])

    def score_thresh(self, task_type):
        # If we have fully explored the environment, set the matching threshold to 0.0
        # to go to the highest scoring instance
        if self.fully_explored[0]:
            return 0.0

        if task_type == "languagenav":
            return self.goal_policy_config.score_thresh_lang
        elif task_type == "imagenav":
            return self.goal_policy_config.score_thresh_image
        else:
            return 0.0

    def act(self, obs: Observations) -> Tuple[DiscreteNavigationAction, Dict[str, Any]]:
        """Act end-to-end."""
        current_task = obs.task_observations["tasks"][self.current_task_idx]
        task_type = current_task["type"]
        
        compass = obs.compass

        # t0 = time.time()

        # 1 - Obs preprocessing
        (
            obs_preprocessed,
            pose_delta,
            object_goal_category,
            img_goal,
            camera_pose,
            keypoints,
            matches,
            confidence,
            local_instance_ids,
            all_rgb_keypoints,
            all_matches,
            all_confidences,
            instance_ids,
        ) = self._preprocess_obs(obs, task_type)

        # t1 = time.time()
        # print(f"Obs preprocessing: {t1 - t0:.2f}")

        # if self.total_timesteps[0] >= 80:
        #     import pdb;pdb.set_trace()

        # 2 - Semantic mapping + policy
        # planner_inputs有goal map, goal pose
        planner_inputs, vis_inputs = self.prepare_planner_inputs(
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

        # t2 = time.time()
        # print(f"Mapping and goal selection: {t2 - t1:.2f}")

        # delete frontiers in blacklist
        # 其中vis_inputs的frontiers与global map是一致的，与goal map横纵坐标相反
        try:
            sp = planner_inputs[0]['sensor_pose'][3:].astype(np.int32)
            # neglect frontiers in frontier_blacklist
            for e in range(self.num_environments):
                if self.frontier_blacklist:
                    # 创建一个与 frontier_map 同尺寸的掩码，标记所有需要清除的 frontier
                    mask = np.zeros_like(planner_inputs[e]["frontier_map"], dtype=np.uint8)
                    
                    # 在掩码上标记所有 frontier_blacklist 的位置为 1
                    for frontier in self.frontier_blacklist:
                        x, y = int(frontier[0]-sp[2]), int(frontier[1] - sp[0])
                        mask[y, x] = 1  # 标记当前 frontier 点

                    # 定义一个 11x11 的圆形核（直径 11 = 2*5 + 1）
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
                    
                    # 对掩码进行膨胀，扩展 5 个像素
                    dilated_mask = cv2.dilate(mask, kernel, iterations=1)
                    
                    # 将膨胀后的区域在 frontier_map 中设为 0
                    planner_inputs[e]["frontier_map"][dilated_mask == 1] = 0.0
                    filtered_frontiers = []
                    for frontier in vis_inputs[0]["frontiers"]:
                        # 检查当前 frontier 是否在黑名单附近（距离 < 2）
                        is_near_blacklist = any(
                            np.linalg.norm(np.array(frontier) - np.array(blacklist_point)) < 5
                            for blacklist_point in self.frontier_blacklist
                        )
                        
                        # 如果不在黑名单附近，则保留
                        if not is_near_blacklist:
                            filtered_frontiers.append(frontier)

                    # 更新 frontier_list
                    vis_inputs[0]['frontiers'] = filtered_frontiers
        except:
            # import ipdb; ipdb.set_trace()
            print("Error in filtering frontiers")
            pass

        ''' Gaussian Localization Experiment '''
        ''' START '''
        
        test_gs = 1 # 导航模式，0：不使用Gaussian，1：两阶段导航，2：加载预存储数据导航，3：在线导航
        use_origin_goal = True # 是否在不传入goal、frontier的时候使用原算法的goal map
        frontier_gs_type = 0
        to_store_global = False # 是否存储本次实验得到的2d地图
        store_path = 'global_map_tee.npy' # 存储路径
        if test_gs == 1:
            use_store_global = False # 是否使用存储好的2d地图进行实验
        elif test_gs == 2:
            use_store_global = True # 是否使用存储好的2d地图进行实验
        elif test_gs == 3:
            use_store_global = False # 是否使用存储好的2d地图进行实验
        elif test_gs == 0:
            pass
        else:
            raise NotImplementedError("test_gs must be 0, 1, 2 or 3")

        if test_gs:
        
            if self.current_task_idx == 0 and not use_store_global and test_gs != 3:
                # 探索建图阶段
                if to_store_global:
                    global_obstacle_map = np.array(self.semantic_map.global_map[0, MC.OBSTACLE_MAP, :, :].cpu())
                    sp = planner_inputs[0]['sensor_pose'][3:].astype(np.int32)
                    local_obstacle_map = global_obstacle_map[sp[0]:sp[1], sp[2]:sp[3]]
                    print(np.array_equal(local_obstacle_map, planner_inputs[0]['obstacle_map']))
                    np.save(store_path,np.array(self.semantic_map.global_map.cpu()))
                if np.sum(planner_inputs[0]['frontier_map']) == 0:
                    pass
                planner_inputs[0]['goal_map'] = planner_inputs[0]['frontier_map']
                planner_inputs[0]['found_goal'] = False
                planner_inputs[0]['goal_pose'] = None
                if self.sub_task_timesteps[0][self.current_task_idx] > 700:
                    pass
                
            else:
                # 检验测试，转换得到的gps坐标能否和真实gps坐标对得上
                global_pose = obs.globalpose
                local_pose = obs.gps
                # 全局位姿：qx, qy, qz, qw, x, y, z(或者应该说是x, z, y)
                qx, qy, qz, qw, x_global, _, y_global = global_pose
                test_point = np.array([x_global, y_global])
                converted_point = global_to_gps(test_point, self.transform_parameters)
                print(f"\n当前位置全局坐标 {test_point} —→ GPS坐标【真实GPS坐标】 {converted_point.round(2)}【{local_pose.round(2)}】")
                
                # Observation info
                # obs包含'compass', 'depth', 'globalpose', 'gps', 'rgb', 'semantic', 'task_observations'
                obs
                
                # 例：使用Task info
                self.current_task_idx # 当前的子任务序号
                # current_task:dict = {'category': 'display cabinet', 'image': np.array H*W*3, 'description': str, 'type': 'imagenav' or 'languagenav'} 
                current_task = obs.task_observations["tasks"][self.current_task_idx]
                task_type = current_task["type"]
                
                # TODO: 将计算出的goal pose传入此处，只需2d坐标
                goal_global_pose = None
                # goal_global_pose = self.gs_poses[self.current_task_idx][self.gs_best_pos_idx[self.current_task_idx] - 1]
                # goal_global_pose = np.array([goal_global_pose[0], goal_global_pose[2]])
                # goal_global_pose = np.array([4.06795,-0.44104]) # subtask0:display cabinet_209 [4.06795,0.82117,-0.44104]
                
                # TODO：将interesting frontier pos传入此处，记得上面use_origin_goal的设置
                frontier_global_pose = None
                
                if frontier_gs_type == 2:
                    min_frontier_dist = 1e+5
                    best_frontier_idx = -1
                    for idx,gs_frontier in enumerate(self.gs_poses[self.current_task_idx]):
                        # frontier_global_pose = np.array([gs_frontier[0], gs_frontier[2]])
                        if self.frontier_flag[self.current_task_idx][idx] == 0:
                            cur_frontier_global_pose = np.array([gs_frontier[0], gs_frontier[2]])
                            dist = np.linalg.norm(test_point - cur_frontier_global_pose)
                            if dist < min_frontier_dist:
                                best_frontier_idx = idx
                                min_frontier_dist = dist
                                frontier_global_pose = cur_frontier_global_pose
                    # frontier_global_pose = np.array([gs_frontier[0], gs_frontier[2]])
                    # frontier_global_pose = np.array([4.06795,1.42117]) # subtask0:display cabinet_209 [4.06795,0.82117,-0.44104]
                if frontier_gs_type == 1:
                    if self.frontier_flag[self.current_task_idx] < len(self.gs_poses[self.current_task_idx]):
                        frontier_global_pose = self.gs_poses[self.current_task_idx][self.frontier_flag[self.current_task_idx]]
                        frontier_global_pose = np.array([frontier_global_pose[0], frontier_global_pose[2]])
                    
                # frontier_global_pose = np.array([4.06795,1.42117]) # subtask0:display cabinet_209 [4.06795,0.82117,-0.44104]
                # # 比如在frontier已找到时，可以增加一个类似下面这样的设计
                # if self.arrive_frontier_flag:
                #     goal_global_pose = np.array([4.06795, -0.44104])

                if goal_global_pose is not None:
                    # 也可以直接传 goal_gps_pose
                    goal_gps_pose = global_to_gps(goal_global_pose, self.transform_parameters)
                    relative_pose = goal_gps_pose - local_pose
                    planner_goal_pose = np.arctan2(relative_pose[1], relative_pose[0])
                    relative_pixel_pose = np.around(relative_pose * (100 / 5)).astype(np.int32)
                    goal_pixel_pose = np.array(planner_inputs[0]['goal_map'].shape[:2])//2 + relative_pixel_pose
                    planner_inputs[0]['found_goal'] = True
                    if np.linalg.norm(relative_pose) < 0.6:
                        planner_inputs[0]['goal_pose'] = math.degrees(planner_goal_pose)
                    planner_inputs[0]['goal_map'] = np.zeros_like(planner_inputs[0]['goal_map'])
                    planner_inputs[0]['goal_map'] = draw_circle(planner_inputs[0]['goal_map'], goal_pixel_pose[1], goal_pixel_pose[0], 3)
                else:
                    if ((frontier_global_pose is not None)
                        and ((use_origin_goal and not planner_inputs[0]['found_goal']) or not use_origin_goal)):
                            planner_inputs[0]['found_goal'] = False
                            planner_inputs[0]['goal_pose'] = None
                            if frontier_gs_type == 1:
                                planner_inputs[0]['goal_map'] = np.zeros_like(planner_inputs[0]['goal_map'])
                            # for frontier_pose in frontier_global_pose:
                            frontier_gps_pose = global_to_gps(frontier_global_pose, self.transform_parameters)
                            relative_pose = frontier_gps_pose - local_pose
                            planner_goal_pose = np.arctan2(relative_pose[1], relative_pose[0])
                            relative_pixel_pose = np.around(relative_pose * (100 / 5)).astype(np.int32)
                            goal_pixel_pose = np.array(planner_inputs[0]['goal_map'].shape[:2])//2 + relative_pixel_pose
                            planner_inputs[0]['goal_map'] = draw_circle(planner_inputs[0]['goal_map'], goal_pixel_pose[1], goal_pixel_pose[0], 3)
                            # 如果use_origin_goal为True， 则当原算法检测到目标后就朝着检测到的目标去了，
                            # 没检测的话就根据我们给出的位置增添一个frontier，只有当算法规划不出一条向着这个点去的路径时，才会朝着其他frontier走
                            # 如果use_origin_goal为False，则会朝向我们给出的frontier位置走，直到走到当前
                            # frontier位置并转一圈，触发else条件，然后可以更换frontier或给出goal pose
                            # 所以这个设计只是增加了一个优先级较高的frontier
                            if np.linalg.norm(relative_pose) < 0.8:
                                if self.see_around <=12:
                                    action = DiscreteNavigationAction.TURN_RIGHT
                                    self.see_around += 1
                                else:
                                    self.see_around = 0
                                    # # TODO: 标记当前位置已抵达, 且已看一圈，更换其他想探索的区域或给出目标值
                                    # # 比如类似下面的设计
                                    # frontier_global_pose = None
                                    # self.arrive_frontier_flag = True
                                    if frontier_gs_type == 1:
                                        self.frontier_flag[self.current_task_idx] += 1
                                    elif frontier_gs_type == 2:
                                        self.frontier_flag[self.current_task_idx][best_frontier_idx] = 1
                                    pass
                    elif frontier_global_pose is None and not use_origin_goal:
                        # 如果没有传入frontier pose，也不使用原算法的goal pose，就按frontier explore的方式走
                        planner_inputs[0]['goal_map'] = planner_inputs[0]['frontier_map']
                        planner_inputs[0]['found_goal'] = False
                        planner_inputs[0]['goal_pose'] = None
                    else:
                        self.see_around = 0
                
                if use_store_global:
                    
                    # load global map
                    map_from_globalstore = np.load(store_path)
                    sp = planner_inputs[0]['sensor_pose'][3:].astype(np.int32)
                    local_obs_map_from_global  = np.array(map_from_globalstore[0, MC.OBSTACLE_MAP, sp[0]:sp[1], sp[2]:sp[3]])
                    planner_inputs[0]['obstacle_map'] = local_obs_map_from_global
                    
                    vis_inputs[0]['explored_map'] = np.copy(map_from_globalstore[0, MC.EXPLORED_MAP, sp[0]:sp[1], sp[2]:sp[3]])
                    vis_inputs[0]['been_close_map'] = np.copy(map_from_globalstore[0, MC.BEEN_CLOSE_MAP, sp[0]:sp[1], sp[2]:sp[3]])
                    semantic_map = np.copy(map_from_globalstore[0])
                    semantic_map[
                        MC.NON_SEM_CHANNELS + 379, :, :
                    ] = 1e-5  # Last category is unlabeled
                    semantic_map = semantic_map[
                        MC.NON_SEM_CHANNELS : MC.NON_SEM_CHANNELS + 380, sp[0]:sp[1], sp[2]:sp[3]
                    ].argmax(0)
                    vis_inputs[0]['semantic_map'] = np.copy(semantic_map)
                    vis_inputs[0]['frontiers'] = []

                    
            # vis_inputs = [
            #     {
            #         "explored_map": self.semantic_map.get_explored_map(e),
            #         "semantic_map": self.semantic_map.get_semantic_map(e),
            #         "been_close_map": self.semantic_map.get_been_close_map(e),
            #         "timestep": self.total_timesteps[e],
            #         "frontiers": frontiers,
            #     }
            #     for e in range(self.num_environments)
            # ]
                    
                # planner_inputs[0]['goal_map'][goal_pixel_pose[1], goal_pixel_pose[0]] = 1.0
                # goal_local_pose = 
           
                
        ''' END '''
                
        
        
               
        # planner_inputs = [
        #     {
        #         "obstacle_map": self.semantic_map.get_obstacle_map(e),
        #         "goal_map": self.semantic_map.get_goal_map(e), # 其实就是上面的goal_map
        #         "frontier_map": self.semantic_map.get_frontier_map(e),
        #         "sensor_pose": self.semantic_map.get_planner_pose_inputs(e),
        #         "found_goal": self.found_goal[e].item(),
        #         "goal_pose": self.goal_pose[e] if self.goal_pose is not None else None,
        #     }
        #     for e in range(self.num_environments)
        # ]


        # 3 - Planning
        closest_goal_map = None
        dilated_obstacle_map = None
        short_term_goal = None
        could_not_find_path = False
        if planner_inputs[0]["found_goal"]:
            self.episode_panorama_start_steps = 0
        if self.total_timesteps[0] < self.episode_panorama_start_steps:
            action = DiscreteNavigationAction.TURN_RIGHT
        elif self.see_around == 0:
            # planner_inputs：
                # "obstacle_map"
                # "goal_map"
                # "frontier_map"
                # "sensor_pose"
                # "found_goal"
                # "goal_pose"
            (
                action,
                closest_goal_map, # only for visualize
                short_term_goal, # only for visualize
                dilated_obstacle_map, # only for visualize
                could_not_find_path,
                planner_stop
            ) = self.planner.plan(
                **planner_inputs[0],
                use_dilation_for_stg=self.use_dilation_for_stg,
                timestep=self.sub_task_timesteps[0][self.current_task_idx],
                debug=False
            )

        # t3 = time.time()
        # print(f"Planning: {t3 - t2:.2f}")

        # deal with stuck situation, if the agent is stuck for 20 steps, call STOP
        current_position = obs.gps
        if self.prev_position is None:
            self.prev_position = current_position
        if (np.linalg.norm(np.array(current_position) - np.array(self.prev_position)) < 0.001
            # and action == DiscreteNavigationAction.MOVE_FORWARD
            ):
            self.ctr += 1
            if self.ctr > 20:
                try:
                    print("The agent is stuck, delete current frontier")
                    goal_x, goal_y = np.where(closest_goal_map)
                    mean_y = int(np.mean(goal_y))
                    mean_x = int(np.mean(goal_x))
                    closest_goal = np.array([mean_y, mean_x])
                    for frontier in vis_inputs[0]["frontiers"]:
                        # 这里计算closest goal 和frontier的距离
                        distance = np.linalg.norm(closest_goal - frontier)
                        if distance < 5:
                            sp = planner_inputs[0]['sensor_pose'][3:].astype(np.int32)
                            frontier_pose = np.array([frontier[0]+sp[2], frontier[1]+sp[0]])
                            self.frontier_blacklist.append(frontier_pose)
                    # action = DiscreteNavigationAction.STOP
                    self.ctr = 0
                except:
                    print("Error in deleting current frontier")
                    pass
        else:
            self.ctr = 0
        self.prev_position = current_position  

        if (
            self.sub_task_timesteps[0][self.current_task_idx]
            >= self.max_steps # [self.current_task_idx]
            and not (test_gs == 1 and self.current_task_idx == 0)
        ):
            print("Reached max number of steps for subgoal, calling STOP")
            action = DiscreteNavigationAction.STOP
            
        if (
            self.sub_task_timesteps[0][self.current_task_idx]
            >= self.max_steps + 300 # [self.current_task_idx]
            and (test_gs == 1 and self.current_task_idx == 0)
        ):
            print("Reached max number of steps for subgoal, calling STOP")
            action = DiscreteNavigationAction.STOP
        

        if could_not_find_path and not planner_stop and action != DiscreteNavigationAction.STOP:
            # This doesn't help
            # print("Resetting explored area")
            # self.semantic_map.local_map[0, MC.EXPLORED_MAP] *= 0
            # self.semantic_map.global_map[0, MC.EXPLORED_MAP] *= 0

            # TODO: is this accurate?
            print("Can't find a path. Map fully explored.")
            self.fully_explored[0] = True
            self.force_match_against_memory = True

            # if self.reached_goal_candidate:
            #     # move to next sub-task
            #     # update semantic map
            #     # reset timesteps
            #     pass

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
                "last_td_map": obs.task_observations.get("top_down_map"),
                "short_term_goal": short_term_goal,
            }
            try:
                info['last_goal_image'] = obs.task_observations["tasks"][
                        self.current_task_idx
                    ]["image"]
            except:
                info['last_goal_image'] = None if task_type != "imagenav" else obs.task_observations["tasks"][
                        self.current_task_idx
                    ]["image"]
        
            goal_text_desc = {
                x: y
                for x, y in obs.task_observations["tasks"][
                    self.current_task_idx
                ].items()
                if x != "image"
            }
            goal_text_desc['action'] = str(action).split('.')[1]
            info['goal_text'] = str(goal_text_desc)
            if self.imagenav_visualizer is not None:
                self.imagenav_visualizer.visualize(**info)
                
            info = None

        if action == DiscreteNavigationAction.STOP:
            if len(obs.task_observations["tasks"]) - 1 > self.current_task_idx:
                self.current_task_idx += 1
                self.force_match_against_memory = False
                self.timesteps_before_goal_update[0] = 0
                self.total_timesteps = [0] * self.num_environments
                self.found_goal = torch.zeros(
                    self.num_environments, 1, dtype=bool, device=self.device
                )
                self.reset_sub_episode()
                self.prev_position = None
                self.ctr = 0
        self.prev_task_type = task_type
        # info['compass'] = compass
        return action, info

    def _preprocess_obs(self, obs: Observations, task_type: str):
        """Take a home-robot observation, preprocess it to put it into the correct format for the
        semantic map."""

        rgb = torch.from_numpy(obs.rgb).to(self.device)
        depth = (
            torch.from_numpy(obs.depth).unsqueeze(-1).to(self.device) * 100.0
        )  # m to cm

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

        if not self._module.instance_goal_found:
            if task_type == "imagenav":
                if self.goal_image is None:
                    img_goal = obs.task_observations["tasks"][self.current_task_idx][
                        "image"
                    ]
                    (
                        self.goal_image,
                        self.goal_image_keypoints,
                    ) = self.matching.get_goal_image_keypoints(img_goal)
                    # self.goal_mask, _ = self.instance_seg.get_goal_mask(img_goal)

                (
                    keypoints,
                    matches,
                    confidences,
                    local_instance_ids,
                ) = self.matching.get_matches_against_current_frame(
                    self.image_matching_function,
                    self.total_timesteps[0],
                    image_goal=self.goal_image,
                    goal_image_keypoints=self.goal_image_keypoints,
                    categories=[current_task["semantic_id"]],
                    use_full_image=False,
                )

            elif task_type == "languagenav":
                (
                    keypoints,
                    matches,
                    confidences,
                    local_instance_ids,
                ) = self.matching.get_matches_against_current_frame(
                    self.matching.match_language_to_image,
                    self.total_timesteps[0],
                    language_goal=current_task["description"],
                    categories=[current_task["semantic_id"]],
                    use_full_image=True,
                )
        
        semantic = self.one_hot_encoding[torch.from_numpy(semantic).to(self.device)]

        obs_preprocessed = torch.cat([rgb, depth, semantic], dim=-1)

        if self.record_instance_ids:
            instances = obs.task_observations["instance_map"]
            # first create a mapping to 1, 2, ... num_instances
            instance_ids = np.unique(instances)
            # map instance id to index
            instance_id_to_idx = {
                instance_id: idx for idx, instance_id in enumerate(instance_ids)
            }
            # convert instance ids to indices, use vectorized lookup
            instances = torch.from_numpy(
                np.vectorize(instance_id_to_idx.get)(instances)
            ).to(self.device)
            # create a one-hot encoding
            instances = torch.eye(len(instance_ids), device=self.device)[instances]
            obs_preprocessed = torch.cat([obs_preprocessed, instances], dim=-1)

        obs_preprocessed = obs_preprocessed.unsqueeze(0).permute(0, 3, 1, 2)

        curr_pose = np.array([obs.gps[0], obs.gps[1], obs.compass[0]])
        pose_delta = torch.tensor(
            pu.get_rel_pose_change(curr_pose, self.last_poses[0])
        ).unsqueeze(0)
        self.last_poses[0] = curr_pose

        object_goal_category = torch.tensor(current_goal_semantic_id).unsqueeze(0)

        # NOT USED AT ALL? ->
        camera_pose = obs.camera_pose
        if camera_pose is not None:
            camera_pose = torch.tensor(np.asarray(camera_pose)).unsqueeze(0)

        # Match a goal against every instance in memory the moment we get it
        # or when the map just got fully explored
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
            (all_rgb_keypoints, all_matches, all_confidences, instance_ids) = self._match_against_memory(
                task_type, current_task
            )

        return (
            obs_preprocessed,
            pose_delta,
            object_goal_category,
            self.goal_image,
            camera_pose,
            keypoints,
            matches,
            confidences,
            local_instance_ids,
            all_rgb_keypoints,
            all_matches,
            all_confidences,
            instance_ids,
        )

    def _match_against_memory(self, task_type: str, current_task: Dict):
        print("--------Matching against memory!--------")
        if task_type == "languagenav":
            (
                all_rgb_keypoints,
                all_matches,
                all_confidences,
                instance_ids,
            ) = self.matching.get_matches_against_memory(
                self.matching.match_language_to_image,
                self.total_timesteps[0],
                language_goal=current_task["description"],
                use_full_image=True,
                categories=[current_task["semantic_id"]],
            )
            stats = {
                i: {
                    "mean": float(scores.mean()),
                    "median": float(np.median(scores)),
                    "max": float(scores.max()),
                    "min": float(scores.min()),
                    "all": scores.flatten().tolist(),
                }
                for i, scores in zip(instance_ids, all_confidences)
            }
            with open(
                f"{self.goal_matching_vis_dir}/goal{self.current_task_idx}_language_stats.json",
                "w",
            ) as f:
                json.dump(stats, f, indent=4)

        elif task_type == "imagenav":
            (
                all_rgb_keypoints,
                all_matches,
                all_confidences,
                instance_ids,
            ) = self.matching.get_matches_against_memory(
                self.image_matching_function,
                self.sub_task_timesteps[0][self.current_task_idx],
                image_goal=self.goal_image,
                goal_image_keypoints=self.goal_image_keypoints,
                use_full_image=True,
                categories=[current_task["semantic_id"]],
            )
            stats = {
                i: {
                    "mean": float(scores.sum(axis=1).mean()),
                    "median": float(np.median(scores.sum(axis=1))),
                    "max": float(scores.sum(axis=1).max()),
                    "min": float(scores.sum(axis=1).min()),
                    "all": scores.sum(axis=1).tolist(),
                }
                for i, scores in zip(instance_ids, all_confidences)
            }
            with open(
                f"{self.goal_matching_vis_dir}/goal{self.current_task_idx}_image_stats.json",
                "w",
            ) as f:
                json.dump(stats, f, indent=4)

        return all_rgb_keypoints, all_matches, all_confidences, instance_ids

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

            # cluster goal points
            try:
                c = DBSCAN(eps=4, min_samples=1)
                data = np.array(goal_map[e].nonzero()).T
                c.fit(data)

                # mask all points not in the largest cluster
                mode = scipy.stats.mode(c.labels_, keepdims=False).mode.item()
                mode_mask = (c.labels_ != mode).nonzero()
                x = data[mode_mask]
                goal_map_ = np.copy(goal_map[e])
                goal_map_[x] = 0.0

                # adopt masked map if non-empty
                if goal_map_.sum() > 0:
                    goal_map[e] = goal_map_
            except Exception as e:
                print(e)
                return goal_map

        return goal_map
