from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np


def select_independent_evidence(
    observations: Sequence[Mapping[str, object]],
    min_translation_m: float,
    min_yaw_deg: float,
) -> list[Mapping[str, object]]:
    """Keep observations that add camera-baseline or heading diversity."""

    selected: list[Mapping[str, object]] = []
    for observation in observations:
        if not selected:
            selected.append(observation)
            continue
        position = np.asarray(observation["observer_position"], dtype=np.float32)
        yaw = float(observation["observer_yaw"])
        for accepted in selected:
            accepted_position = np.asarray(
                accepted["observer_position"], dtype=np.float32
            )
            accepted_yaw = float(accepted["observer_yaw"])
            translation = (
                np.linalg.norm(position - accepted_position)
                if np.isfinite(position).all()
                and np.isfinite(accepted_position).all()
                else 0.0
            )
            yaw_delta = 0.0
            if np.isfinite(yaw) and np.isfinite(accepted_yaw):
                yaw_delta = abs(
                    (yaw - accepted_yaw + np.pi) % (2 * np.pi) - np.pi
                )
            if translation >= min_translation_m or np.degrees(yaw_delta) >= min_yaw_deg:
                selected.append(observation)
                break
    return selected


def passes_dual_score_verification(
    confirmed: bool,
    peak_seem: float,
    peak_clip: float,
    strong_seem: float,
    minimum_clip: float,
) -> bool:
    """Veto only the intersection where both semantic scores are weak."""

    return bool(
        confirmed
        and (peak_seem >= strong_seem or peak_clip >= minimum_clip)
    )


@dataclass
class _SpatialTrack:
    position: np.ndarray
    observations: int
    last_refresh: int


@dataclass
class CandidateEvidenceBank:
    """Rank recurring ES candidates without using benchmark ground truth."""

    match_radius_m: float = 0.75
    promotion_margin: int = 2
    max_missed_refreshes: int = 3
    enabled: bool = True
    _refresh: int = field(default=0, init=False)
    _tracks: dict[int, list[_SpatialTrack]] = field(default_factory=dict, init=False)

    def reset(self) -> None:
        self._refresh = 0
        self._tracks.clear()

    @staticmethod
    def _distance(first: np.ndarray, second: np.ndarray) -> float:
        return float(np.linalg.norm(first[[0, 2]] - second[[0, 2]]))

    def _update_task(self, task_idx: int, candidates: Sequence[Sequence[float]]) -> list[tuple[np.ndarray, _SpatialTrack, int]]:
        tracks = self._tracks.setdefault(task_idx, [])
        matched_tracks: set[int] = set()
        ranked_items: list[tuple[np.ndarray, _SpatialTrack, int]] = []

        for raw_rank, value in enumerate(candidates):
            point = np.asarray(value, dtype=np.float32)
            eligible = [
                (index, self._distance(point, track.position))
                for index, track in enumerate(tracks)
                if index not in matched_tracks
            ]
            eligible = [item for item in eligible if item[1] <= self.match_radius_m]
            if eligible:
                index, _ = min(eligible, key=lambda item: item[1])
                track = tracks[index]
                track.position = 0.75 * track.position + 0.25 * point
                track.observations += 1
                track.last_refresh = self._refresh
                matched_tracks.add(index)
            else:
                track = _SpatialTrack(point.copy(), 1, self._refresh)
                tracks.append(track)
                matched_tracks.add(len(tracks) - 1)
            ranked_items.append((point, track, raw_rank))

        self._tracks[task_idx] = [
            track
            for track in tracks
            if self._refresh - track.last_refresh <= self.max_missed_refreshes
        ]
        return ranked_items

    def refresh(
        self,
        candidates_by_task: Mapping[int, Sequence[Sequence[float]]],
        active_by_task: Mapping[int, Sequence[float] | None] | None = None,
        preserve_source_order: bool = False,
    ) -> tuple[dict[int, list[list[float]]], dict[int, dict[str, int]]]:
        if not self.enabled:
            output = {
                int(task_idx): [
                    np.asarray(point, dtype=np.float32).tolist()
                    for point in candidates
                ]
                for task_idx, candidates in candidates_by_task.items()
            }
            diagnostics = {
                task_idx: {
                    "candidates": len(candidates),
                    "max_observations": 0,
                    "promoted": 0,
                }
                for task_idx, candidates in output.items()
            }
            return output, diagnostics

        self._refresh += 1
        active_by_task = active_by_task or {}
        output: dict[int, list[list[float]]] = {}
        diagnostics: dict[int, dict[str, int]] = {}

        for raw_task_idx, candidates in candidates_by_task.items():
            task_idx = int(raw_task_idx)
            items = self._update_task(task_idx, candidates)
            if not preserve_source_order:
                items.sort(key=lambda item: (-item[1].observations, item[2]))
            promoted = 0

            active_value = active_by_task.get(task_idx)
            if active_value is not None and items and not preserve_source_order:
                active = np.asarray(active_value, dtype=np.float32)
                active_index = min(
                    range(len(items)),
                    key=lambda index: self._distance(items[index][0], active),
                )
                if self._distance(items[active_index][0], active) <= self.match_radius_m:
                    best_support = items[0][1].observations
                    active_support = items[active_index][1].observations
                    if active_support + self.promotion_margin >= best_support:
                        items.insert(0, items.pop(active_index))
                    else:
                        promoted = int(active_index != 0)

            output[task_idx] = [point.tolist() for point, _, _ in items]
            diagnostics[task_idx] = {
                "candidates": len(items),
                "max_observations": max(
                    (track.observations for _, track, _ in items), default=0
                ),
                "promoted": promoted,
            }
        return output, diagnostics
