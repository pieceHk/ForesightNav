# Portions adapted from LagMemo, Copyright (c) Meta Platforms, Inc. and affiliates.
# Upstream MIT terms and provenance are retained in THIRD_PARTY_NOTICES.md.

from copy import deepcopy
import json
from typing import Mapping

from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np

from .types import Candidate, Navmesh


def project_candidates(
    candidates: Iterable[Candidate],
    navmesh: Navmesh,
    tolerance_m: float = 1.5,
    merge_radius_m: float = 0.25,
) -> list[Candidate]:
    projected: list[Candidate] = []
    for candidate in candidates:
        point = navmesh.project(candidate.world_position)
        if point is None:
            continue
        point = np.asarray(point, dtype=np.float32)
        if np.linalg.norm(point[[0, 2]] - candidate.world_position[[0, 2]]) > tolerance_m:
            continue
        if any(
            np.linalg.norm(point[[0, 2]] - item.world_position[[0, 2]]) < merge_radius_m
            for item in projected
        ):
            continue
        projected.append(Candidate(point, candidate.rank))
    return projected


@dataclass
class ActiveCandidateTracker:
    continuity_radius_m: float = 1.0
    active: Optional[Candidate] = field(default=None, init=False)

    def refresh(self, candidates: list[Candidate]) -> Optional[Candidate]:
        if self.active is not None:
            for candidate in candidates:
                distance = np.linalg.norm(
                    candidate.world_position[[0, 2]] - self.active.world_position[[0, 2]]
                )
                if distance <= self.continuity_radius_m:
                    self.active = candidate
                    return self.active
        self.active = candidates[0] if candidates else None
        return self.active


def reconcile_spatial_candidates(
    old_goals,
    new_goals,
    active_indices,
    visited_positions,
    match_radius=1.0,
    blacklist_radius=None,
    preserve_active=True,
    protected_active_by_task=None,
):
    if blacklist_radius is None:
        blacklist_radius = match_radius
    protected_active_by_task = protected_active_by_task or {}
    reconciled = {}
    next_indices = {}
    next_visited = {
        int(task_idx): [np.asarray(point, dtype=np.float32).tolist() for point in points]
        for (task_idx, points) in visited_positions.items()
    }
    task_ids = set(old_goals) | set(new_goals)
    for task_idx in task_ids:
        task_idx = int(task_idx)
        old_points = [np.asarray(point, dtype=np.float32) for point in old_goals.get(task_idx, [])]
        new_points = [np.asarray(point, dtype=np.float32) for point in new_goals.get(task_idx, [])]
        protected_value = protected_active_by_task.get(task_idx)
        protected_point = (
            None if protected_value is None else np.asarray(protected_value, dtype=np.float32)
        )
        protected_index = None
        if protected_point is not None and new_points:
            protected_distances = [
                np.linalg.norm(point[[0, 2]] - protected_point[[0, 2]]) for point in new_points
            ]
            nearest = int(np.argmin(protected_distances))
            if protected_distances[nearest] <= match_radius:
                protected_index = nearest
        active_index = min(active_indices.get(task_idx, 0), len(old_points))
        history = next_visited.setdefault(task_idx, [])
        for point in old_points[:active_index]:
            if not any(
                (
                    np.linalg.norm(point[[0, 2]] - np.asarray(old)[[0, 2]]) < match_radius
                    for old in history
                )
            ):
                history.append(point.tolist())
        filtered = [
            point
            for (index, point) in enumerate(new_points)
            if index == protected_index
            or not any(
                (
                    np.linalg.norm(point[[0, 2]] - np.asarray(old)[[0, 2]]) < blacklist_radius
                    for old in history
                )
            )
        ]
        active_point = old_points[active_index] if active_index < len(old_points) else None
        if protected_index is not None and filtered:
            distances = [
                np.linalg.norm(point[[0, 2]] - protected_point[[0, 2]]) for point in filtered
            ]
            nearest = int(np.argmin(distances))
            filtered.insert(0, filtered.pop(nearest))
        elif preserve_active and active_point is not None and filtered:
            distances = [np.linalg.norm(point[[0, 2]] - active_point[[0, 2]]) for point in filtered]
            nearest = int(np.argmin(distances))
            if distances[nearest] <= match_radius:
                filtered.insert(0, filtered.pop(nearest))
        reconciled[task_idx] = [point.tolist() for point in filtered]
        next_indices[task_idx] = 0
    return (reconciled, next_indices, next_visited)


class CandidateTracking:
    """Validate a candidate refresh before committing its goals and evidence."""


    @staticmethod
    def _validate_external_ranking_metadata(metadata, new_goals):
        """Validate the transaction metadata before mutating candidate state."""
        if not isinstance(metadata, Mapping):
            raise ValueError("external ranking metadata must be a mapping")
        required = {
            "mode",
            "source",
            "task_idx",
            "version_token",
            "ordered_raw_ranks",
            "dispatch_version_gate",
        }
        missing = sorted(required - set(metadata))
        if missing:
            raise ValueError(f"external ranking metadata is missing required fields: {missing}")
        if metadata["mode"] not in {"idea1_only", "ordered_suffix_mvp"}:
            raise ValueError("external ranking mode must be active")
        if metadata["source"] not in {"idea1", "idea1_fallback", "suffix"}:
            raise ValueError("external ranking source is invalid")
        if (
            metadata["mode"] == "idea1_only"
            and metadata["source"] != "idea1"
            or (metadata["source"] == "suffix" and metadata["mode"] != "ordered_suffix_mvp")
        ):
            raise ValueError("external ranking mode/source combination is invalid")
        task_idx = metadata["task_idx"]
        if isinstance(task_idx, bool) or not isinstance(task_idx, int) or task_idx < 0:
            raise ValueError("external ranking task_idx must be non-negative")
        version_token = metadata["version_token"]
        if not isinstance(version_token, Mapping):
            raise ValueError("external ranking version_token must be a mapping")
        for name in (
            "scene_id",
            "episode_id",
            "map_revision",
            "last_fused_keyframes",
            "query_serial",
        ):
            if name not in version_token:
                raise ValueError(f"external ranking version_token is missing {name}")
        gate = metadata["dispatch_version_gate"]
        if not isinstance(gate, Mapping) or gate.get("would_invalidate") is not False:
            raise ValueError("external ranking requires a valid dispatch version gate")
        parity = (
            ("scene_id", "scene_id"),
            ("episode_id", "episode_id"),
            ("query_serial", "query_serial"),
            ("map_revision", "batch_map_revision"),
            ("last_fused_keyframes", "status_last_fused_keyframes"),
        )
        for token_name, gate_name in parity:
            if gate.get(gate_name) != version_token[token_name]:
                raise ValueError("external ranking version token does not match dispatch gate")
        if gate.get("status_rebuilding") is not False:
            raise ValueError("external ranking dispatch gate must not be rebuilding")
        raw_ranks = metadata["ordered_raw_ranks"]
        if not isinstance(raw_ranks, (list, tuple)) or any(
            (isinstance(rank, bool) or not isinstance(rank, int) or rank < 0 for rank in raw_ranks)
        ):
            raise ValueError("external ranking raw ranks must be non-negative integers")
        if len(raw_ranks) != len(set(raw_ranks)):
            raise ValueError("external ranking raw ranks must be unique")
        task_goals = (new_goals or {}).get(task_idx)
        if task_goals is None or len(task_goals) != len(raw_ranks):
            raise ValueError("external ranking raw ranks must align with current-task goals")
        try:
            return json.loads(json.dumps(metadata, allow_nan=False))
        except (TypeError, ValueError) as exc:
            raise ValueError("external ranking metadata must be strict JSON") from exc

    def refresh_gs_poses(
        self, new_goals, protected_active_by_task=None, *, external_ranking_metadata=None
    ):
        validated_external_ranking = None
        if external_ranking_metadata is not None:
            validated_external_ranking = self._validate_external_ranking_metadata(
                external_ranking_metadata, new_goals
            )
            if validated_external_ranking["task_idx"] != int(self.current_task_idx):
                raise ValueError("external ranking task_idx must equal current_task_idx")
        old_goals = getattr(self, "gs_poses", {}) or {}
        active_goals = {}
        for task_idx, points in old_goals.items():
            active_index = getattr(self, "frontier_flag", {}).get(task_idx, 0)
            if 0 <= active_index < len(points):
                active_goals[int(task_idx)] = points[active_index]
        candidate_evidence_before = deepcopy(self.candidate_evidence)
        diagnostics_before = self.candidate_evidence_diagnostics
        try:
            (ranked_goals, candidate_evidence_diagnostics) = self.candidate_evidence.refresh(
                new_goals or {},
                active_goals,
                preserve_source_order=validated_external_ranking is not None
                or self.query_action_rerank_enabled,
            )
            (reconciled_goals, reconciled_indices, reconciled_visited) = (
                reconcile_spatial_candidates(
                    old_goals,
                    ranked_goals,
                    getattr(self, "frontier_flag", {}),
                    self.visited_candidate_positions,
                    self.candidate_match_radius,
                    self.candidate_blacklist_radius,
                    preserve_active=not (
                        validated_external_ranking is not None or self.query_action_rerank_enabled
                    ),
                    protected_active_by_task=protected_active_by_task,
                )
            )
        except Exception:
            self.candidate_evidence = candidate_evidence_before
            self.candidate_evidence_diagnostics = diagnostics_before
            raise
        self.gs_poses = reconciled_goals
        self.frontier_flag = reconciled_indices
        self.visited_candidate_positions = reconciled_visited
        self.candidate_evidence_diagnostics = candidate_evidence_diagnostics
        self.last_external_ranking_metadata = validated_external_ranking
        counts = {task_idx: len(points) for (task_idx, points) in self.gs_poses.items()}
        print(
            f"[CANDIDATE_TRACK] spatial_refresh counts={counts} evidence={self.candidate_evidence_diagnostics} external_ranking={self.last_external_ranking_metadata}",
            flush=True,
        )

    def _mark_current_candidate_visited(self):
        points = self.gs_poses.get(self.current_task_idx, [])
        index = self.frontier_flag.get(self.current_task_idx, 0)
        if index >= len(points):
            return
        history = self.visited_candidate_positions.setdefault(self.current_task_idx, [])
        point = np.asarray(points[index], dtype=np.float32)
        if not any(
            (
                np.linalg.norm(point[[0, 2]] - np.asarray(old)[[0, 2]])
                < self.candidate_match_radius
                for old in history
            )
        ):
            history.append(point.tolist())
