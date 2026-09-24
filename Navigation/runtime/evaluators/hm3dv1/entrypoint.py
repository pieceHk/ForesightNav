#!/usr/bin/env python3
"""Evaluate ES + LagMemo on the official native HM3D-v1 ObjectNav split."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm


_THIS_FILE = Path(__file__).resolve()
_TDONAV_ROOT = Path(os.environ.get("TDONAV_ROOT", _THIS_FILE.parents[4])).resolve()
_EVALUATOR_ROOT = _THIS_FILE.parent
_LAGMEMO_ROOT = Path(os.environ.get("TDONAV_LAGMEMO_ROOT", _TDONAV_ROOT / "Navigation" / "runtime" / "lagmemo")).resolve()
_LAGMEMO_PROJECT = _LAGMEMO_ROOT / "project" / "habitat_lagmemo"
_LAGMEMO_SRC = _LAGMEMO_ROOT / "src" / "lagmemo"
for _path in (_LAGMEMO_PROJECT, _LAGMEMO_SRC, _EVALUATOR_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from habitat.config.default import get_config as get_habitat_config
from habitat.config.default_structured_configs import (
    GlobalPoseSensorConfig,
    TopDownMapMeasurementConfig,
)
from habitat.core.env import Env
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from lagmemo.agent.lagmemo_agent.glue_agent import GLUEAgent
from lagmemo.core.interfaces import DiscreteNavigationAction, Observations
from lagmemo.env.habitat_abstract_env import HabitatEnv
from online_es_provider import OnlineESProvider
from runtime_memory import release_host_memory
from tdonav_navigation.ranking import (
    CandidateActionReranker,
    CandidateRerankConfig,
)
from tdonav_navigation.recovery import NoveltyGoalSelector


OBJECTNAV_SEMANTIC_IDS = {
    "bed": 1,
    "chair": 2,
    "plant": 3,
    "sofa": 4,
    "toilet": 5,
    "tv_monitor": 6,
}


def parse_args() -> argparse.Namespace:
    dataset_root = _TDONAV_ROOT / "Exp" / "assets" / "datasets" / "HM3Dv1_objectnav_val_20scenes"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=dataset_root)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--socket-path", required=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--scenes", default="all")
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--episode-start", type=int, default=0)
    parser.add_argument("--episode-end", type=int, default=None)
    parser.add_argument(
        "--episode-indices",
        default="",
        help="Optional comma-separated global episode indices for focused ablation.",
    )
    parser.add_argument("--resume-metrics", type=Path, default=None)
    parser.add_argument("--enable-timing", action="store_true")
    parser.add_argument(
        "--print-images",
        action="store_true",
        help="Write LagMemo per-step snapshots and FMM planner snapshots.",
    )
    parser.add_argument(
        "--episode-final-snapshot",
        action="store_true",
        help="Write one terminal LagMemo snapshot per episode.",
    )
    parser.add_argument(
        "--experiment-name",
        default="HM3Dv1_objectnav_val_20scenes",
    )
    parser.add_argument(
        "--baseline-config-path",
        type=Path,
        default=_LAGMEMO_ROOT / "project" / "config" / "agent" / "hm3d_eval.yaml",
    )
    parser.add_argument(
        "--seem-ckpt-path",
        type=Path,
        default=_LAGMEMO_ROOT / "src" / "third_party" / "seem" / "checkpoints" / "seem_focall_v0.pt",
    )
    parser.add_argument(
        "--seem-yaml-path",
        type=Path,
        default=_LAGMEMO_ROOT / "src" / "third_party" / "seem" / "configs" / "seem" / "focall_unicl_lang_demo.yaml",
    )
    parser.add_argument(
        "--mobileclip-ckpt-path",
        type=Path,
        default=_LAGMEMO_ROOT / "src" / "third_party" / "ml-mobileclip" / "checkpoints" / "mobileclip_s0.pt",
    )
    return parser.parse_args()


class SingleTargetESProvider(OnlineESProvider):
    """Online ES provider whose one task is the native ObjectNav category."""

    def __init__(self, socket_path: str, top_k: int) -> None:
        super().__init__(
            socket_path=socket_path,
            top_k=top_k,
            connect_timeout=180.0,
            content_root="",
        )
        self._live_mode = True

    def reset_episode(self, scene_id: str, runtime_episode_id: str, category: str) -> None:
        self._scene_id = scene_id
        self._episode_id = runtime_episode_id
        self._tasks = [{"object_category": category}]
        self._goals_cache = {}
        response = self._send_recv(
            {
                "cmd": "reset_episode",
                "scene_id": scene_id,
                "episode_id": runtime_episode_id,
            }
        )
        if not response.get("ok"):
            raise RuntimeError("reset_episode failed: {}".format(response))
        self.n_live_frames = 0
        print(
            "[ObjectNavESProvider] reset {}/{} category={}".format(
                scene_id, runtime_episode_id, category
            ),
            flush=True,
        )


class ObjectNavEnv(HabitatEnv):
    """Translate native ObjectNav observations to the GLUE agent interface."""

    def __init__(self, habitat_env: Env) -> None:
        super().__init__(habitat_env)
        self.current_episode = None

    def reset(self) -> Observations:
        raw_obs = self.habitat_env.reset()
        self.current_episode = self.habitat_env.current_episode
        self._last_obs = self._preprocess_obs(raw_obs)
        return self._last_obs

    def _preprocess_obs(self, habitat_obs: Dict[str, np.ndarray]) -> Observations:
        if self.current_episode is None:
            self.current_episode = self.habitat_env.current_episode
        category = str(self.current_episode.object_category)
        depth = np.asarray(habitat_obs["depth"], dtype=np.float32)[..., -1]
        rgb = np.asarray(habitat_obs["rgb"], dtype=np.uint8)
        semantic = np.zeros(depth.shape, dtype=np.int64)
        task = {
            "category": category,
            "description": "",
            "image": None,
            "semantic_id": OBJECTNAV_SEMANTIC_IDS[category],
            "type": "objectnav",
        }
        raw_top_down_map = self.habitat_env.get_metrics().get("top_down_map")
        if isinstance(raw_top_down_map, dict):
            # The visualizer draws goal overlays in place, so detach the Habitat
            # measure output before exposing it to LagMemo.
            top_down_map = {
                key: value.copy() if isinstance(value, np.ndarray) else value
                for key, value in raw_top_down_map.items()
            }
        else:
            # Keep visualization robust before Habitat has emitted its first map.
            top_down_map = {
                "map": np.zeros((1, 1), dtype=np.uint8),
                "fog_of_war_mask": np.zeros((1, 1), dtype=np.uint8),
                "agent_map_coord": [0, 0],
                "agent_angle": 0.0,
            }
        obs = Observations(
            rgb=rgb,
            depth=depth,
            compass=np.asarray(habitat_obs["compass"], dtype=np.float32),
            gps=self._preprocess_xy(np.asarray(habitat_obs["gps"], dtype=np.float32)),
            task_observations={
                "tasks": [task],
                "top_down_map": top_down_map,
                "instance_map": np.zeros(depth.shape, dtype=np.int64),
                "semantic_frame": np.concatenate(
                    [rgb, np.zeros(depth.shape + (1,), dtype=np.uint8)], axis=2
                ),
            },
            camera_pose=None,
            third_person_image=None,
            globalpose=np.asarray(habitat_obs["globalpose"], dtype=np.float32),
        )
        obs.semantic = semantic
        return obs

    def _preprocess_action(self, action: Any) -> int:
        if isinstance(action, int):
            return action
        return HabitatSimActions[action.name.lower()]

    def _process_info(self, info: Dict[str, Any]) -> None:
        # ObjectNav evaluation runs without visualizer-provided GT semantic maps.
        return None


def _geodesic_distance(pathfinder: Any, start: np.ndarray, end: np.ndarray) -> float:
    try:
        import habitat_sim

        path = habitat_sim.ShortestPath()
        path.requested_start = np.asarray(start, dtype=np.float32)
        path.requested_end = np.asarray(end, dtype=np.float32)
        if pathfinder.find_path(path):
            return float(path.geodesic_distance)
    except Exception:
        pass
    return float(np.linalg.norm(np.asarray(start)[[0, 2]] - np.asarray(end)[[0, 2]]))


def _strict_geodesic_distance(
    pathfinder: Any, start: np.ndarray, end: np.ndarray
) -> float:
    """Return inf when geodesic planning is unavailable so the ranker logs fallback."""
    try:
        import habitat_sim

        path = habitat_sim.ShortestPath()
        path.requested_start = np.asarray(start, dtype=np.float32)
        path.requested_end = np.asarray(end, dtype=np.float32)
        if pathfinder.find_path(path):
            return float(path.geodesic_distance)
    except Exception:
        pass
    return float("inf")


def select_candidate_approach(
    source: np.ndarray,
    pathfinder: Any,
    origin: Optional[np.ndarray],
    ring_radius: float = 0.9,
    ring_samples: int = 12,
    max_snap_distance: float = 0.75,
) -> Optional[np.ndarray]:
    """Select a deterministic reachable observation pose around an ES point."""
    source = np.asarray(source, dtype=np.float32)
    if source.shape != (3,) or not np.isfinite(source).all():
        return None
    origin = None if origin is None else np.asarray(origin, dtype=np.float32)
    ring_candidates: List[np.ndarray] = []
    request_y = float(origin[1]) if origin is not None else float(source[1])
    for index in range(max(1, ring_samples)):
        angle = 2.0 * math.pi * index / max(1, ring_samples)
        requested = np.array(
            [
                source[0] + ring_radius * math.cos(angle),
                request_y,
                source[2] + ring_radius * math.sin(angle),
            ],
            dtype=np.float32,
        )
        try:
            snapped = np.asarray(pathfinder.snap_point(requested), dtype=np.float32)
        except Exception:
            continue
        if snapped.shape != (3,) or not np.isfinite(snapped).all():
            continue
        if np.linalg.norm(snapped[[0, 2]] - requested[[0, 2]]) > max_snap_distance:
            continue
        if origin is not None:
            try:
                if int(pathfinder.get_island(snapped)) != int(pathfinder.get_island(origin)):
                    continue
            except Exception:
                pass
        ring_candidates.append(snapped)

    if ring_candidates:
        if origin is None:
            return ring_candidates[0]
        return min(
            ring_candidates,
            key=lambda point: _geodesic_distance(pathfinder, origin, point),
        )

    try:
        snapped = np.asarray(pathfinder.snap_point(source), dtype=np.float32)
    except Exception:
        return None
    if snapped.shape != (3,) or not np.isfinite(snapped).all():
        return None
    return snapped


def project_goals_to_navigable(
    goals: Dict[int, List[List[float]]],
    pathfinder: Any,
    origin: Optional[np.ndarray] = None,
    approach_ring: bool = False,
    records_by_task: Optional[Dict[int, List[dict]]] = None,
    return_records: bool = False,
):
    max_distance = 1.5
    projected: Dict[int, List[List[float]]] = {}
    projected_records: Dict[int, List[dict]] = {}
    for task_idx, candidates in (goals or {}).items():
        accepted: List[List[float]] = []
        accepted_records: List[dict] = []
        source_records = (records_by_task or {}).get(int(task_idx), [])
        for candidate_index, candidate in enumerate(candidates):
            source = np.asarray(candidate, dtype=np.float32)
            if approach_ring:
                snapped = select_candidate_approach(source, pathfinder, origin)
                if snapped is None:
                    continue
            else:
                try:
                    snapped = np.asarray(pathfinder.snap_point(source), dtype=np.float32)
                except Exception:
                    continue
            if snapped.shape != (3,) or not np.isfinite(snapped).all():
                continue
            source_distance = np.linalg.norm(
                snapped[[0, 2]] - source[[0, 2]]
            )
            if source_distance > max_distance:
                continue
            if any(
                np.linalg.norm(snapped[[0, 2]] - np.asarray(old)[[0, 2]]) < 0.25
                for old in accepted
            ):
                continue
            accepted.append(snapped.tolist())
            if candidate_index < len(source_records):
                record = dict(source_records[candidate_index])
                record["position"] = snapped.tolist()
                accepted_records.append(record)
        projected[int(task_idx)] = accepted
        projected_records[int(task_idx)] = accepted_records
        print(
            "[CANDIDATE_PROJECTION] task={} input={} accepted={} approach_ring={}".format(
                task_idx, len(candidates), len(accepted), int(approach_ring)
            ),
            flush=True,
        )
    if return_records:
        return projected, projected_records
    return projected


def _goal_viewpoints(episode: Any) -> np.ndarray:
    positions: List[np.ndarray] = []
    for goal in getattr(episode, "goals", []) or []:
        for viewpoint in getattr(goal, "view_points", []) or []:
            state = getattr(viewpoint, "agent_state", None)
            point = getattr(state, "position", None)
            if point is not None:
                positions.append(np.asarray(point, dtype=np.float32))
        point = getattr(goal, "position", None)
        if not positions and point is not None:
            positions.append(np.asarray(point, dtype=np.float32))
    valid = [point for point in positions if point.shape == (3,) and np.isfinite(point).all()]
    return np.stack(valid) if valid else np.empty((0, 3), dtype=np.float32)


def update_query_diagnostics(
    diagnostics: Dict[str, Any],
    raw_goals: Dict[int, List[List[float]]],
    projected_goals: Dict[int, List[List[float]]],
    episode: Any,
    reranked_goals: Optional[Dict[int, List[List[float]]]] = None,
) -> None:
    raw = list((raw_goals or {}).get(0, []))
    projected = list((projected_goals or {}).get(0, []))
    diagnostics["refreshes"] += 1
    diagnostics["raw_nonempty_refreshes"] += int(bool(raw))
    diagnostics["projected_nonempty_refreshes"] += int(bool(projected))
    diagnostics["max_raw_candidates"] = max(diagnostics["max_raw_candidates"], len(raw))
    diagnostics["max_projected_candidates"] = max(
        diagnostics["max_projected_candidates"], len(projected)
    )
    viewpoints = _goal_viewpoints(episode)
    if not raw or len(viewpoints) == 0:
        return
    candidates = np.asarray(raw, dtype=np.float32)
    distances = np.linalg.norm(
        candidates[:, None, :][:, :, [0, 2]]
        - viewpoints[None, :, :][:, :, [0, 2]],
        axis=2,
    )
    best = float(distances.min())
    raw_rank_distances = distances.min(axis=1)
    diagnostics["raw_rank1_gt_distance_m"] = min(
        diagnostics.get("raw_rank1_gt_distance_m", float("inf")),
        float(raw_rank_distances[0]),
    )
    diagnostics["raw_rank3_gt_distance_m"] = min(
        diagnostics.get("raw_rank3_gt_distance_m", float("inf")),
        float(raw_rank_distances[:3].min()),
    )
    diagnostics["min_raw_candidate_gt_distance_m"] = min(
        diagnostics["min_raw_candidate_gt_distance_m"], best
    )
    diagnostics["query_hit_1m"] = diagnostics["query_hit_1m"] or best <= 1.0
    diagnostics["query_hit_3m"] = diagnostics["query_hit_3m"] or best <= 3.0
    ranked = list((reranked_goals or {}).get(0, []))
    if ranked:
        ranked_candidates = np.asarray(ranked, dtype=np.float32)
        ranked_distances = np.linalg.norm(
            ranked_candidates[:, None, :][:, :, [0, 2]]
            - viewpoints[None, :, :][:, :, [0, 2]],
            axis=2,
        ).min(axis=1)
        diagnostics["reranked_rank1_gt_distance_m"] = min(
            diagnostics.get("reranked_rank1_gt_distance_m", float("inf")),
            float(ranked_distances[0]),
        )
        diagnostics["reranked_rank3_gt_distance_m"] = min(
            diagnostics.get("reranked_rank3_gt_distance_m", float("inf")),
            float(ranked_distances[:3].min()),
        )


def candidate_reranker_from_env() -> CandidateActionReranker:
    def number(name: str, default: str) -> float:
        return float(os.environ.get(name, default))

    return CandidateActionReranker(
        enabled=os.environ.get("ES_QUERY_ACTION_RERANK_ENABLED", "0") == "1",
        config=CandidateRerankConfig(
            semantic=number("ES_QUERY_W_SEMANTIC", "1.0"),
            support=number("ES_QUERY_W_SUPPORT", "0.35"),
            recurrence=number("ES_QUERY_W_RECURRENCE", "0.30"),
            travel=number("ES_QUERY_W_TRAVEL", "0.35"),
            novelty=number("ES_QUERY_W_NOVELTY", "0.15"),
            revisit=number("ES_QUERY_W_REVISIT", "0.80"),
            failure=number("ES_QUERY_W_FAILURE", "1.0"),
            switch_margin=number("ES_QUERY_SWITCH_MARGIN", "0.15"),
            match_radius_m=number("ES_CANDIDATE_EVIDENCE_RADIUS", "0.75"),
        ),
    )


def rerank_projected_candidates(
    reranker: CandidateActionReranker,
    projected: Dict[int, List[List[float]]],
    projected_records: Dict[int, List[dict]],
    origin: np.ndarray,
    pathfinder: Any,
    agent: Any,
) -> tuple[Dict[int, List[List[float]]], Dict[int, dict]]:
    if not reranker.enabled:
        return projected, {
            int(task_idx): {
                "enabled": 0,
                "candidates": len(candidates),
                "reordered": 0,
            }
            for task_idx, candidates in projected.items()
        }
    output: Dict[int, List[List[float]]] = {}
    diagnostics: Dict[int, dict] = {}
    for raw_task_idx, candidates in projected.items():
        task_idx = int(raw_task_idx)
        records = projected_records.get(task_idx, [])
        if len(records) != len(candidates):
            records = [
                {
                    "position": point,
                    "raw_rank": index,
                    "semantic_score": 0.0,
                    "support_count": 0,
                    "support_score": 0.0,
                    "map_revision": 0,
                }
                for index, point in enumerate(candidates)
            ]
        active = None
        old_points = getattr(agent, "gs_poses", {}).get(task_idx, [])
        active_index = getattr(agent, "frontier_flag", {}).get(task_idx, 0)
        if 0 <= active_index < len(old_points):
            active = old_points[active_index]
        ranked, task_diagnostics = reranker.rank(
            task_idx,
            records,
            origin,
            travel_cost=lambda start, end: _strict_geodesic_distance(
                pathfinder, start, end
            ),
            active_position=active,
            visited_positions=getattr(agent, "visited_candidate_positions", {}).get(
                task_idx, []
            ),
        )
        output[task_idx] = [list(record["position"]) for record in ranked]
        diagnostics[task_idx] = task_diagnostics
        if task_diagnostics.get("reordered"):
            print(
                f"[QUERY_RERANK] task={task_idx} {task_diagnostics}",
                flush=True,
            )
    return output, diagnostics


def sample_navmesh_goal(
    pathfinder: Any,
    origin: np.ndarray,
    radius: float,
    selector: Optional[NoveltyGoalSelector] = None,
) -> Optional[np.ndarray]:
    origin = np.asarray(origin, dtype=np.float32)
    try:
        snapped_origin = np.asarray(pathfinder.snap_point(origin), dtype=np.float32)
        island = int(pathfinder.get_island(snapped_origin))
    except Exception:
        return None
    candidates: List[np.ndarray] = []
    for _ in range(32):
        try:
            point = np.asarray(
                pathfinder.get_random_navigable_point_near(snapped_origin, radius, 100, island),
                dtype=np.float32,
            )
            if int(pathfinder.get_island(point)) != island:
                continue
        except Exception:
            continue
        if abs(float(point[1] - snapped_origin[1])) > 0.35:
            continue
        if np.linalg.norm(point[[0, 2]] - snapped_origin[[0, 2]]) < 1.25:
            continue
        candidates.append(point)
    if not candidates:
        return None
    if selector is not None:
        return selector.select(candidates, snapped_origin)
    return max(
        candidates,
        key=lambda point: np.linalg.norm(
            point[[0, 2]] - snapped_origin[[0, 2]]
        ),
    )


def configure(args: argparse.Namespace) -> DictConfig:
    dataset_root = args.dataset_root.resolve()
    config_path = dataset_root / "configs" / "habitat_hm3d_v1_objectnav.yaml"
    if not config_path.is_file():
        raise FileNotFoundError("dataset config not found: {}".format(config_path))
    if not args.baseline_config_path.is_file():
        raise FileNotFoundError("agent config not found: {}".format(args.baseline_config_path))

    habitat_config = get_habitat_config(
        str(config_path),
        overrides=["habitat.simulator.habitat_sim_v0.gpu_device_id={}".format(args.gpu_id)],
    )
    baseline_config = OmegaConf.load(str(args.baseline_config_path))
    config = DictConfig({**habitat_config, **baseline_config})
    OmegaConf.set_struct(config, False)
    config.EXP_FIRST = False
    config.GROUND_TRUTH_SEMANTICS = 0
    config.NO_GPU = 0
    config.NUM_ENVIRONMENTS = 1
    # Continuous planner images and one terminal snapshot are separate controls.
    config.PRINT_IMAGES = int(args.print_images)
    config.EPISODE_FINAL_SNAPSHOT = int(args.episode_final_snapshot)
    config.VISUALIZE = 0
    config.DUMP_LOCATION = str(args.output_path.resolve())
    config.EXP_NAME = args.experiment_name
    config.ENVIRONMENT.max_num_sub_task_episodes = 1
    config.AGENT.max_steps = 500
    config.habitat.environment.max_episode_steps = 500
    config.habitat.environment.iterator_options.shuffle = False
    config.habitat.dataset.split = "val"
    config.habitat.dataset.data_path = str(
        dataset_root / "data/datasets/objectnav/hm3d/v1/{split}/{split}.json.gz"
    )
    config.habitat.dataset.scenes_dir = str(dataset_root / "data/scene_datasets") + "/"
    config.habitat.task.lab_sensors.globalpose_sensor = GlobalPoseSensorConfig()
    # Native TopDownMap supplies GT geometry only to the renderer.  It is never
    # converted to a semantic map or sent to ES candidate selection.
    top_down_map = TopDownMapMeasurementConfig()
    top_down_map.map_resolution = 1024
    top_down_map.draw_shortest_path = False
    # HM3D-v1 episodes provide goal/view-point geometry without semantic AABBs.
    top_down_map.draw_goal_aabbs = False
    config.habitat.task.measurements.top_down_map = top_down_map
    if args.scenes != "all":
        config.habitat.dataset.content_scenes = [args.scenes]
    return config


def scene_name(scene_id: str) -> str:
    return Path(scene_id).name.split(".")[0]


def load_metrics(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("invalid metrics file: {}".format(path))
    return payload


def write_summary(metrics: Dict[str, Dict[str, Any]], path: Path) -> None:
    metric_names = ("success", "spl", "soft_spl")
    summary: Dict[str, Any] = {"completed_episodes": len(metrics), "overall": {}, "by_category": {}}
    for metric in metric_names:
        values = [float(record["metrics"].get(metric, 0.0)) for record in metrics.values()]
        summary["overall"][metric] = float(np.mean(values)) if values else None
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in metrics.values():
        grouped[str(record["object_category"])].append(record)
    for category, records in sorted(grouped.items()):
        summary["by_category"][category] = {
            "episodes": len(records),
            **{
                metric: float(np.mean([float(record["metrics"].get(metric, 0.0)) for record in records]))
                for metric in metric_names
            },
        }
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def ensure_exploration_vis_inputs(agent: GLUEAgent) -> None:
    """Supply GLUE's ES branch with map state while file visualizations stay off."""
    original = agent.prepare_planner_inputs

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        planner_inputs, vis_inputs = original(*args, **kwargs)
        for env_index, vis_input in enumerate(vis_inputs):
            if "explored_map" not in vis_input:
                explored_map = agent.semantic_map.get_explored_map(env_index)
                vis_input["explored_map"] = explored_map
                vis_input["been_close_map"] = agent.semantic_map.get_been_close_map(env_index)
                vis_input["semantic_map"] = np.zeros_like(explored_map)
                vis_input["frontiers"] = []
        return planner_inputs, vis_inputs

    agent.prepare_planner_inputs = wrapped


def main() -> int:
    args = parse_args()
    args.output_path = args.output_path.resolve()
    args.output_path.mkdir(parents=True, exist_ok=True)
    os.environ["LAGMEMO_SEMANTIC_SOURCE"] = "disabled"

    config = configure(args)
    approach_ring = os.environ.get("HM3D_EXP_APPROACH_RING", "0") == "1"
    candidate_action_rerank_enabled = (
        os.environ.get("ES_QUERY_ACTION_RERANK_ENABLED", "0") == "1"
    )
    protocol = {
        "dataset": "official HM3D ObjectNav-v1 val",
        "task": "native ObjectNav-v1 single target",
        "semantic_input": "disabled; ES candidates plus SEEM verification only",
        "geometry_input": "Habitat RGB-D, GPS, Compass, and GlobalPose",
        "actions": {"forward_m": 0.25, "turn_deg": 30, "max_steps": 500},
        "metrics": ["success", "spl", "soft_spl"],
        "success": {"distance_to": "VIEW_POINTS", "threshold_m": 0.1},
        "planner_images": bool(args.print_images),
        "v2_controls": {
            "approach_ring": approach_ring,
            "candidate_exhaustion_exploration": os.environ.get(
                "HM3D_EXP_CANDIDATE_EXHAUSTION_EXPLORE", "0"
            )
            == "1",
            "verification_es_agreement": os.environ.get(
                "HM3D_EXP_VERIFY_ES_AGREEMENT", "0"
            )
            == "1",
        },
        "improve_controls": {
            "query_action_rerank_enabled": candidate_action_rerank_enabled,
            "candidate_evidence_ranking": os.environ.get(
                "ES_CANDIDATE_EVIDENCE_RANKING", "1"
            )
            == "1",
            "candidate_evidence_radius_m": float(
                os.environ.get("ES_CANDIDATE_EVIDENCE_RADIUS", "0.75")
            ),
            "candidate_promotion_margin": int(
                os.environ.get("ES_CANDIDATE_PROMOTION_MARGIN", "2")
            ),
            "recovery_novelty_weight": float(
                os.environ.get("ES_RECOVERY_NOVELTY_WEIGHT", "0.65")
            ),
            "verification_dual_score_gate": os.environ.get(
                "HM3D_EXP_VERIFY_DUAL_SCORE_GATE", "0"
            )
            == "1",
            "verification_strong_peak_seem": float(
                os.environ.get("HM3D_EXP_VERIFY_STRONG_PEAK_SEEM", "1.35")
            ),
            "verification_min_peak_clip": float(
                os.environ.get("HM3D_EXP_VERIFY_MIN_PEAK_CLIP", "0.12")
            ),
            "verification_min_observer_translation_m": float(
                os.environ.get("HM3D_EXP_VERIFY_MIN_OBSERVER_TRANSLATION", "0.0")
            ),
            "verification_min_observer_yaw_deg": float(
                os.environ.get("HM3D_EXP_VERIFY_MIN_OBSERVER_YAW_DEG", "0.0")
            ),
        },
        "visualization_geometry": "native TopDownMap only; not consumed by ES or semantic mapping",
    }
    (args.output_path / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_path / "runtime_config.yaml").write_text(
        OmegaConf.to_yaml(config), encoding="utf-8"
    )

    provider = SingleTargetESProvider(args.socket_path, args.top_k)
    habitat_env = Env(config)
    total_episodes = len(habitat_env.episodes)
    episode_start = args.episode_start
    episode_end = total_episodes if args.episode_end is None else args.episode_end
    if not (0 <= episode_start < episode_end <= total_episodes):
        raise ValueError(
            "invalid episode range [{}, {}) for {} loaded episodes".format(
                episode_start, episode_end, total_episodes
            )
        )
    if args.episode_indices:
        selected_indices = [
            int(value) for value in args.episode_indices.split(",") if value.strip()
        ]
        if len(selected_indices) != len(set(selected_indices)):
            raise ValueError("episode indices must be unique")
        invalid_indices = [
            value
            for value in selected_indices
            if not episode_start <= value < episode_end
        ]
        if invalid_indices:
            raise ValueError(
                "episode indices outside [{}, {}): {}".format(
                    episode_start, episode_end, invalid_indices
                )
            )
    else:
        selected_indices = list(range(episode_start, episode_end))
    habitat_env.episode_iterator = iter(
        [habitat_env.episodes[index] for index in selected_indices]
    )
    (args.output_path / "episode_range.json").write_text(
        json.dumps(
            {
                "episode_start": episode_start,
                "episode_end": episode_end,
                "selected_episode_count": len(selected_indices),
                "episode_indices": selected_indices if args.episode_indices else None,
                "total_loaded_episodes": total_episodes,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    env = ObjectNavEnv(habitat_env)
    agent = GLUEAgent(
        config=config,
        seem_cfg={"conf_path": str(args.seem_yaml_path), "ckpt_path": str(args.seem_ckpt_path)},
        clip_cfg=str(args.mobileclip_ckpt_path),
    )
    agent._es_test_gs_override = 3
    ensure_exploration_vis_inputs(agent)

    metrics_path = args.output_path / "per_episode_metrics.json"
    summary_path = args.output_path / "cumulative_metrics.json"
    timing_path = args.output_path / "timing_log.csv"
    if args.resume_metrics is not None:
        resume_path = args.resume_metrics.resolve()
        if not resume_path.is_file():
            raise FileNotFoundError("resume metrics not found: {}".format(resume_path))
        if metrics_path.exists():
            raise FileExistsError("refusing to overwrite existing metrics: {}".format(metrics_path))
        metrics = load_metrics(resume_path)
        invalid_resume_keys = [
            key
            for key in metrics
            if not (0 <= int(key.split("_", 1)[0]) < episode_end)
        ]
        if invalid_resume_keys:
            raise ValueError(
                "resume metrics outside dataset/episode_end range: {}".format(
                    invalid_resume_keys[:5]
                )
            )
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (args.output_path / "resume_metadata.json").write_text(
            json.dumps(
                {"resume_metrics": str(resume_path), "resumed_episode_count": len(metrics)},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    else:
        metrics = load_metrics(metrics_path)
    write_summary(metrics, summary_path)
    timing_file = None
    timing_writer = None
    if args.enable_timing:
        timing_file = timing_path.open("a", newline="", encoding="utf-8")
        timing_writer = csv.DictWriter(
            timing_file,
            fieldnames=("episode_key", "step", "dt_total_ms", "dt_push_ms", "dt_goals_ms", "dt_act_ms"),
        )
        if timing_path.stat().st_size == 0:
            timing_writer.writeheader()

    started = time.time()
    completed_this_invocation = 0
    try:
        for episode_index in selected_indices:
            if args.max_episodes and completed_this_invocation >= args.max_episodes:
                break
            env.reset()
            episode = habitat_env.current_episode
            scene = scene_name(episode.scene_id)
            category = str(episode.object_category)
            episode_key = "{:04d}_{}".format(episode_index, scene)
            if episode_key in metrics:
                continue
            runtime_episode_id = "{:04d}".format(episode_index)
            provider.reset_episode(scene, runtime_episode_id, category)
            agent.reset(
                start_position=episode.start_position,
                start_rotation=episode.start_rotation,
                candidate_provider=provider,
            )
            release_host_memory('evaluator episode reset')
            agent.set_runtime_context(scene, runtime_episode_id)
            agent.set_episode_visualization_dir(scene, runtime_episode_id)
            query_diagnostics = {
                "refreshes": 0,
                "raw_nonempty_refreshes": 0,
                "projected_nonempty_refreshes": 0,
                "max_raw_candidates": 0,
                "max_projected_candidates": 0,
                "min_raw_candidate_gt_distance_m": float("inf"),
                "query_hit_1m": False,
                "query_hit_3m": False,
                "gt_used_for_navigation": False,
                "reranked_rank1_gt_distance_m": float("inf"),
                "reranked_rank3_gt_distance_m": float("inf"),
                "raw_rank1_gt_distance_m": float("inf"),
                "raw_rank3_gt_distance_m": float("inf"),
            }
            candidate_reranker = candidate_reranker_from_env()
            raw_goals = agent.gs_poses
            projected_goals, projected_records = project_goals_to_navigable(
                raw_goals,
                habitat_env.sim.pathfinder,
                origin=np.asarray(episode.start_position, dtype=np.float32),
                approach_ring=approach_ring,
                records_by_task=provider.get_all_candidate_records(),
                return_records=True,
            )
            ranked_goals, rerank_diagnostics = rerank_projected_candidates(
                candidate_reranker,
                projected_goals,
                projected_records,
                np.asarray(episode.start_position, dtype=np.float32),
                habitat_env.sim.pathfinder,
                agent,
            )
            update_query_diagnostics(
                query_diagnostics,
                raw_goals,
                projected_goals,
                episode,
                ranked_goals,
            )
            agent.refresh_gs_poses(ranked_goals)

            steps = 0
            stop_reason = None
            verification_counts: Counter = Counter()
            candidate_evidence_counts: Counter = Counter()
            candidate_rerank_counts: Counter = Counter()
            frontier_rerank_counts: Counter = Counter()
            last_frontier_rerank = {}
            last_rerank_diagnostics = rerank_diagnostics.get(0, {})
            for name in (
                "reordered",
                "switches",
                "hysteresis_holds",
                "geodesic_fallbacks",
                "revision_changed",
            ):
                candidate_rerank_counts[name] += int(
                    last_rerank_diagnostics.get(name, 0)
                )
            candidate_rerank_counts["refreshes"] += 1
            exploration_counts: Counter = Counter()
            candidate_evidence = agent.candidate_evidence_diagnostics.get(0, {})
            candidate_evidence_counts["refreshes"] += 1
            candidate_evidence_counts["promotions"] += int(
                candidate_evidence.get("promoted", 0)
            )
            candidate_evidence_counts["max_observations"] = max(
                candidate_evidence_counts["max_observations"],
                int(candidate_evidence.get("max_observations", 0)),
            )
            recovery_selector = NoveltyGoalSelector(
                history_size=int(os.environ.get("ES_RECOVERY_HISTORY_SIZE", "8")),
                novelty_weight=float(
                    os.environ.get("ES_RECOVERY_NOVELTY_WEIGHT", "0.65")
                ),
            )
            max_candidate_index = 0
            pbar = tqdm(total=500, leave=False, desc=episode_key)
            while not env.episode_over:
                steps += 1
                step_started = time.perf_counter()
                obs = env.get_observation()
                push_started = time.perf_counter()
                provider.push_live_frame(obs.rgb, obs.depth, obs.globalpose)
                push_ms = (time.perf_counter() - push_started) * 1000.0

                goals_ms = 0.0
                if steps % 10 == 0:
                    provider._goals_cache = {}
                    goals_started = time.perf_counter()
                    refreshed_raw = provider.get_all_goals()
                    refreshed, refreshed_records = project_goals_to_navigable(
                        refreshed_raw,
                        habitat_env.sim.pathfinder,
                        origin=np.asarray(obs.globalpose[4:7], dtype=np.float32),
                        approach_ring=approach_ring,
                        records_by_task=provider.get_all_candidate_records(),
                        return_records=True,
                    )
                    ranked_refreshed, rerank_diagnostics = rerank_projected_candidates(
                        candidate_reranker,
                        refreshed,
                        refreshed_records,
                        np.asarray(obs.globalpose[4:7], dtype=np.float32),
                        habitat_env.sim.pathfinder,
                        agent,
                    )
                    update_query_diagnostics(
                        query_diagnostics,
                        refreshed_raw,
                        refreshed,
                        episode,
                        ranked_refreshed,
                    )
                    agent.refresh_gs_poses(ranked_refreshed)
                    last_rerank_diagnostics = rerank_diagnostics.get(0, {})
                    for name in (
                        "reordered",
                        "switches",
                        "hysteresis_holds",
                        "geodesic_fallbacks",
                        "revision_changed",
                    ):
                        candidate_rerank_counts[name] += int(
                            last_rerank_diagnostics.get(name, 0)
                        )
                    candidate_rerank_counts["refreshes"] += 1
                    candidate_evidence = agent.candidate_evidence_diagnostics.get(0, {})
                    candidate_evidence_counts["refreshes"] += 1
                    candidate_evidence_counts["promotions"] += int(
                        candidate_evidence.get("promoted", 0)
                    )
                    candidate_evidence_counts["max_observations"] = max(
                        candidate_evidence_counts["max_observations"],
                        int(candidate_evidence.get("max_observations", 0)),
                    )
                    goals_ms = (time.perf_counter() - goals_started) * 1000.0

                act_started = time.perf_counter()
                action, info = agent.act(obs)
                act_ms = (time.perf_counter() - act_started) * 1000.0
                metadata = getattr(agent, "last_action_metadata", {})
                stop_reason = metadata.get("stop_reason")
                if metadata.get("planner_error"):
                    task_idx = int(agent.current_task_idx)
                    points = agent.gs_poses.get(task_idx, [])
                    active_index = agent.frontier_flag.get(task_idx, 0)
                    if 0 <= active_index < len(points):
                        candidate_reranker.mark_failed(
                            task_idx, points[active_index]
                        )
                max_candidate_index = max(
                    max_candidate_index, int(metadata.get("candidate_index", 0))
                )
                verification_event = metadata.get("verification_event")
                last_frontier_rerank = metadata.get("frontier_rerank") or {}
                frontier_rerank_counts["steps"] += int(
                    bool(last_frontier_rerank.get("enabled"))
                )
                frontier_rerank_counts["reordered"] += int(
                    last_frontier_rerank.get("reordered", 0)
                )
                frontier_rerank_counts["hysteresis_holds"] += int(
                    last_frontier_rerank.get("hysteresis_holds", 0)
                )
                frontier_rerank_counts["max_components"] = max(
                    frontier_rerank_counts["max_components"],
                    int(last_frontier_rerank.get("components", 0)),
                )
                if verification_event:
                    verification_counts["events"] += 1
                    verification_counts["triggered"] += int(
                        verification_event.get("triggered", False)
                    )
                    verification_counts["confirmed"] += int(
                        verification_event.get("confirmed", False)
                    )
                    verification_counts["es_disagreement_rejections"] += int(
                        verification_event.get("triggered", False)
                        and not verification_event.get("es_agreement", True)
                    )
                    verification_counts["strong_bypass"] += int(
                        verification_event.get("strong_bypass", False)
                    )
                    verification_counts["dual_score_rejections"] += int(
                        verification_event.get("dual_score_rejection", False)
                    )
                    verification_counts["empty_depth"] += int(
                        verification_event.get("depth_points", 0) == 0
                    )
                    verification_counts["support_total"] += int(
                        verification_event.get("support", 0)
                    )
                    verification_counts["independent_support_total"] += int(
                        verification_event.get("independent_support", 0)
                    )
                    verification_counts["independent_shortfalls"] += int(
                        verification_event.get("triggered", False)
                        and int(verification_event.get("support", 0)) >= 2
                        and int(verification_event.get("independent_support", 0)) < 2
                    )
                if metadata.get("exploration_reason"):
                    exploration_counts[str(metadata["exploration_reason"])] += 1
                if metadata.get("trapped_event"):
                    exploration_counts["trapped"] += 1
                if metadata.get("trapped_event") and action != DiscreteNavigationAction.STOP:
                    goal = sample_navmesh_goal(
                        habitat_env.sim.pathfinder,
                        obs.globalpose[4:],
                        3.0,
                        recovery_selector,
                    )
                    if goal is not None:
                        agent.set_navmesh_override_goal(goal, "escape")
                        action = DiscreteNavigationAction.TURN_RIGHT
                    else:
                        agent.clear_local_navigation_state()
                elif metadata.get("random_exploration_requested") and action != DiscreteNavigationAction.STOP:
                    goal = sample_navmesh_goal(
                        habitat_env.sim.pathfinder,
                        obs.globalpose[4:],
                        8.0,
                        recovery_selector,
                    )
                    if goal is not None:
                        agent.set_navmesh_override_goal(goal, "random")
                        action = DiscreteNavigationAction.TURN_RIGHT
                if action == DiscreteNavigationAction.STOP and stop_reason not in {"verified_target", "max_steps"}:
                    raise RuntimeError(
                        "illegal STOP for {}: {}".format(episode_key, stop_reason)
                    )
                env.apply_action(action, info=info)
                total_ms = (time.perf_counter() - step_started) * 1000.0
                if timing_writer is not None:
                    timing_writer.writerow(
                        {
                            "episode_key": episode_key,
                            "step": steps,
                            "dt_total_ms": round(total_ms, 1),
                            "dt_push_ms": round(push_ms, 1),
                            "dt_goals_ms": round(goals_ms, 1),
                            "dt_act_ms": round(act_ms, 1),
                        }
                    )
                    timing_file.flush()
                pbar.update(1)
            pbar.close()
            flush_status = provider.flush_episode()
            final_artifacts = flush_status.get("final_artifacts", [])
            if not final_artifacts:
                raise RuntimeError(
                    "missing final ES map artifact for {}".format(episode_key)
                )

            all_native_metrics = habitat_env.get_metrics()
            native_metrics = {
                key: float(value)
                for key, value in all_native_metrics.items()
                if isinstance(value, (float, int, np.floating, np.integer))
            }
            top_down_map = all_native_metrics.get("top_down_map")
            exploration_coverage = None
            if isinstance(top_down_map, dict):
                fog = top_down_map.get("fog_of_war_mask")
                if isinstance(fog, np.ndarray) and fog.size:
                    exploration_coverage = float(np.count_nonzero(fog) / fog.size)
            min_gt_distance = query_diagnostics["min_raw_candidate_gt_distance_m"]
            query_diagnostics["min_raw_candidate_gt_distance_m"] = (
                round(min_gt_distance, 4) if np.isfinite(min_gt_distance) else None
            )
            for name in (
                "raw_rank1_gt_distance_m",
                "raw_rank3_gt_distance_m",
                "reranked_rank1_gt_distance_m",
                "reranked_rank3_gt_distance_m",
            ):
                value = query_diagnostics[name]
                query_diagnostics[name] = (
                    round(value, 4) if np.isfinite(value) else None
                )
            metrics[episode_key] = {
                "episode_sequence_index": episode_index,
                "source_episode_id": str(episode.episode_id),
                "scene_id": scene,
                "object_category": category,
                "steps": steps,
                "stop_reason": stop_reason or "episode_limit",
                "metrics": native_metrics,
                "diagnostics": {
                    "query": query_diagnostics,
                    "verification": dict(verification_counts),
                    "candidate_evidence": dict(candidate_evidence_counts),
                    "candidate_rerank": {
                        **dict(candidate_rerank_counts),
                        "last": last_rerank_diagnostics,
                    },
                    "frontier_rerank": {
                        **dict(frontier_rerank_counts),
                        "last": last_frontier_rerank,
                    },
                    "exploration": {
                        **dict(exploration_counts),
                        "fog_coverage_fraction": exploration_coverage,
                        "max_candidate_index": max_candidate_index,
                        "novel_recovery_goals": recovery_selector.selections,
                    },
                },
                "online_es_final": {
                    "artifacts": final_artifacts,
                    "keyframes": flush_status.get("n_keyframes", 0),
                    "raw_frames": flush_status.get("n_raw_frames", 0),
                    "gaussians": flush_status.get("n_gaussians", 0),
                    "flush_ms": round(getattr(provider, "_t_flush_ms", 0.0), 1),
                },
            }
            metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            write_summary(metrics, summary_path)
            completed_this_invocation += 1
            print(
                "[ObjectNav] {} category={} steps={} success={:.3f} spl={:.3f}".format(
                    episode_key,
                    category,
                    steps,
                    native_metrics.get("success", 0.0),
                    native_metrics.get("spl", 0.0),
                ),
                flush=True,
            )
    finally:
        if timing_file is not None:
            timing_file.close()
        provider.close()
        env.close()
        write_summary(metrics, summary_path)
    print("completed={} elapsed_s={:.1f}".format(len(metrics), time.time() - started), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
