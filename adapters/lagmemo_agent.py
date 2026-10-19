"""Assemble LagMemo perception and local-map layers for the PBSP navigator."""

# Portions adapted from LagMemo, Copyright (c) Meta Platforms, Inc. and affiliates.
# Upstream MIT terms and provenance are retained in THIRD_PARTY_NOTICES.md.

from adapters.perception import GoalPerception
from experiments.visualization import NavigationVisualization
from pbsp.navigation.navigator import Navigator

from pathlib import Path
import numpy as np
import torch
from torch.nn import DataParallel
from lagmemo.agent.imagenav_agent.visualizer import NavVisualizer
from lagmemo.core.abstract_agent import Agent
from lagmemo.core.interfaces import DiscreteNavigationAction
from lagmemo.mapping.semantic.categorical_2d_semantic_map_state import Categorical2DSemanticMapState
from lagmemo.mapping.semantic.instance_tracking_modules import InstanceMemory
from lagmemo.agent.lagmemo_agent.frontier_agent_module import GoatAgentModule
from lagmemo.agent.lagmemo_agent.lagmemo_matching import GoatMatching
import mobileclip
import os
from lagmemo.agent.lagmemo_agent.seem_model import SEEMModel
import kornia.feature as KF
from pbsp.navigation.evidence import CandidateEvidenceBank
from pbsp.navigation.ranking import FrontierRerankConfig, FrontierUtilityRanker


class LagMemoAgent(NavigationVisualization, GoalPerception, Navigator, Agent):

    def __init__(self, config, device_id: int = 0, seem_cfg: dict = None, clip_cfg: str = ""):
        Navigator.__init__(self, actions=DiscreteNavigationAction)
        self.arrive_frontier_flag = False
        self.max_steps = config.AGENT.max_steps
        self.num_environments = config.NUM_ENVIRONMENTS
        self.store_all_categories_in_map = getattr(config.AGENT, "store_all_categories", False)
        if config.AGENT.panorama_start:
            self.panorama_start_steps = int(360 / config.ENVIRONMENT.turn_angle)
        else:
            self.panorama_start_steps = 0
        self.panorama_rotate_steps = int(360 / config.ENVIRONMENT.turn_angle)
        self.goal_matching_vis_dir = f"{config.DUMP_LOCATION}/goal_grounding_vis"
        Path(self.goal_matching_vis_dir).mkdir(parents=True, exist_ok=True)
        self.instance_memory = None
        self.record_instance_ids = getattr(config.AGENT.SEMANTIC_MAP, "record_instance_ids", False)
        if self.record_instance_ids:
            self.instance_memory = InstanceMemory(
                self.num_environments,
                config.AGENT.SEMANTIC_MAP.du_scale,
                debug_visualize=config.PRINT_IMAGES,
                config=config,
                mask_cropped_instances=False,
                padding_cropped_instances=200,
            )
        self.goal_image = None
        self.goal_mask = None
        self.goal_image_keypoints = None
        self.goal_policy_config = config.AGENT.SUPERGLUE
        self.matching = GoatMatching(
            device=0,
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
            self.module = DataParallel(self._module, device_ids=[self.device_id])
        self.continuous_print_images = bool(config.PRINT_IMAGES)
        self.episode_final_snapshot = bool(getattr(config, "EPISODE_FINAL_SNAPSHOT", False))
        self.visualize = (
            config.VISUALIZE or self.continuous_print_images or self.episode_final_snapshot
        )
        self.use_dilation_for_stg = config.AGENT.PLANNER.use_dilation_for_stg
        self.semantic_map = Categorical2DSemanticMapState(
            device=self.device,
            num_environments=self.num_environments,
            num_sem_categories=config.AGENT.SEMANTIC_MAP.num_sem_categories,
            map_resolution=config.AGENT.SEMANTIC_MAP.map_resolution,
            map_size_cm=config.AGENT.SEMANTIC_MAP.map_size_cm,
            global_downscaling=config.AGENT.SEMANTIC_MAP.global_downscaling,
            record_instance_ids=getattr(config.AGENT.SEMANTIC_MAP, "record_instance_ids", False),
            max_instances=getattr(config.AGENT.SEMANTIC_MAP, "max_instances", 0),
            evaluate_instance_tracking=getattr(
                config.ENVIRONMENT, "evaluate_instance_tracking", False
            ),
            instance_memory=self.instance_memory,
        )
        agent_radius_cm = config.AGENT.radius * 100.0
        agent_cell_radius = int(np.ceil(agent_radius_cm / config.AGENT.SEMANTIC_MAP.map_resolution))
        self.max_num_sub_task_episodes = config.ENVIRONMENT.max_num_sub_task_episodes
        if "planner_type" in config.AGENT.PLANNER and config.AGENT.PLANNER.planner_type == "old":
            print("Using frontier planner")
            from lagmemo.navigation_planner.frontier_planner import DiscretePlanner
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
            print_images=self.continuous_print_images or self.episode_final_snapshot,
            dump_location=config.DUMP_LOCATION,
            exp_name=config.EXP_NAME,
        )
        self.found_goal = torch.zeros(self.num_environments, 1, dtype=bool, device=self.device)
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
        self.transform_parameters = np.array([0, 0, 0, 0])
        self.td_transform = None
        self.td_goal_pos = None
        self.prev_global_pos = np.zeros(2)
        self.prev_td_pos = np.zeros(2)
        self.see_around = 0
        (self.clip_model, _, self.clip_preprocess) = mobileclip.create_model_and_transforms(
            "mobileclip_s0", pretrained=clip_cfg
        )
        self.clip_tokenizer = mobileclip.get_tokenizer("mobileclip_s0")
        self.seem_model = SEEMModel(**seem_cfg)
        self.stop_by_seem = False
        self.seem_goal_pose = None
        self.seem_semantic = None
        self.stuck_flag = 0
        self.stuck_num = 0
        self.goal_vis_stage = 0
        self.disk = KF.DISK.from_pretrained("depth").to(self.device).eval()
        self.lightglue = KF.LightGlue("disk").to(self.device).eval()
        self.glue_vis_path = f"{config.DUMP_LOCATION}/images/{config.EXP_NAME}/glue_vis"
        self.lightglue_vis_dir = self.glue_vis_path
        self.goal_disk_data = None
        self.exp_first = config.EXP_FIRST
        self.visited_candidate_positions = {}
        self.candidate_match_radius = float(os.environ.get("ES_CANDIDATE_TRACK_RADIUS", "1.0"))
        self.candidate_blacklist_radius = float(
            os.environ.get("ES_CANDIDATE_BLACKLIST_RADIUS", str(self.candidate_match_radius))
        )
        self.two_stage_verification = os.environ.get("HM3D_EXP_TWO_STAGE_VERIFY", "0") == "1"
        self.verification_window_steps = int(os.environ.get("HM3D_EXP_VERIFY_WINDOW_STEPS", "6"))
        self.verification_min_confirmations = int(
            os.environ.get("HM3D_EXP_VERIFY_MIN_CONFIRMATIONS", "2")
        )
        self.verification_pose_radius = float(os.environ.get("HM3D_EXP_VERIFY_POSE_RADIUS", "1.5"))
        self.verification_min_peak_seem = float(
            os.environ.get("HM3D_EXP_VERIFY_MIN_PEAK_SEEM", "1.15")
        )
        self.verification_dual_score_gate = (
            os.environ.get("HM3D_EXP_VERIFY_DUAL_SCORE_GATE", "0") == "1"
        )
        self.verification_strong_peak_seem = float(
            os.environ.get("HM3D_EXP_VERIFY_STRONG_PEAK_SEEM", "1.35")
        )
        self.verification_min_peak_clip = float(
            os.environ.get("HM3D_EXP_VERIFY_MIN_PEAK_CLIP", "0.12")
        )
        self.verification_min_observer_translation = float(
            os.environ.get("HM3D_EXP_VERIFY_MIN_OBSERVER_TRANSLATION", "0.0")
        )
        self.verification_min_observer_yaw_deg = float(
            os.environ.get("HM3D_EXP_VERIFY_MIN_OBSERVER_YAW_DEG", "0.0")
        )
        self.candidate_evidence = CandidateEvidenceBank(
            match_radius_m=float(os.environ.get("ES_CANDIDATE_EVIDENCE_RADIUS", "0.75")),
            promotion_margin=int(os.environ.get("ES_CANDIDATE_PROMOTION_MARGIN", "2")),
            enabled=os.environ.get("ES_CANDIDATE_EVIDENCE_RANKING", "1") == "1",
        )
        self.query_action_rerank_enabled = (
            os.environ.get("ES_QUERY_ACTION_RERANK_ENABLED", "0") == "1"
        )
        self.frontier_reranker = FrontierUtilityRanker(
            enabled=os.environ.get("ES_FRONTIER_RERANK_ENABLED", "0") == "1",
            config=FrontierRerankConfig(
                gain=float(os.environ.get("ES_FRONTIER_W_GAIN", "1.0")),
                clearance=float(os.environ.get("ES_FRONTIER_W_CLEARANCE", "0.25")),
                travel=float(os.environ.get("ES_FRONTIER_W_TRAVEL", "0.35")),
                novelty=float(os.environ.get("ES_FRONTIER_W_NOVELTY", "0.20")),
                revisit=float(os.environ.get("ES_FRONTIER_W_REVISIT", "0.75")),
                failure=float(os.environ.get("ES_FRONTIER_W_FAILURE", "1.0")),
                switch_margin=float(os.environ.get("ES_FRONTIER_SWITCH_MARGIN", "0.15")),
            ),
        )
        self.frontier_rerank_diagnostics = {}
        self.candidate_evidence_diagnostics = {}
        self.last_external_ranking_metadata = None
        self.runtime_scene_id = None
        self.runtime_episode_id = None
        self.trapped_warmup_steps = int(os.environ.get("ES_TRAPPED_WARMUP_STEPS", "30"))
        self.trapped_window_steps = int(os.environ.get("ES_TRAPPED_WINDOW_STEPS", "25"))
        self.trapped_radius = float(os.environ.get("ES_TRAPPED_RADIUS", "0.05"))
        self.random_exploration_warmup_steps = int(
            os.environ.get("ES_RANDOM_EXPLORATION_WARMUP_STEPS", "30")
        )
        self.trapped_position_history = []
        self.trapped_active = False
        self.trapped_events = []
        self.navmesh_override_goal = None
        self.navmesh_override_mode = None
        self.navmesh_override_task = None
