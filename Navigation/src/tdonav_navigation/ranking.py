from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np


def _finite_weight(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or abs(value) > 100.0:
        raise ValueError(f"{name} must be finite and within [-100, 100]")
    return value


def _normalize(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if len(array) == 0:
        return array
    span = float(array.max() - array.min())
    if span <= 1e-12:
        return np.zeros_like(array)
    return (array - array.min()) / span


@dataclass(frozen=True)
class CandidateRerankConfig:
    semantic: float = 1.0
    support: float = 0.35
    recurrence: float = 0.30
    travel: float = 0.35
    novelty: float = 0.15
    revisit: float = 0.80
    failure: float = 1.0
    switch_margin: float = 0.15
    match_radius_m: float = 0.75

    def __post_init__(self) -> None:
        for name in (
            "semantic",
            "support",
            "recurrence",
            "travel",
            "novelty",
            "revisit",
            "failure",
            "switch_margin",
            "match_radius_m",
        ):
            _finite_weight(getattr(self, name), name)
        if any(
            getattr(self, name) < 0
            for name in (
                "semantic",
                "support",
                "recurrence",
                "travel",
                "novelty",
                "revisit",
                "failure",
            )
        ):
            raise ValueError("candidate utility weights must be non-negative")
        if self.match_radius_m <= 0 or self.switch_margin < 0:
            raise ValueError("match radius must be positive and margin non-negative")


@dataclass
class _CandidateTrack:
    position: np.ndarray
    observations: int = 1
    failures: int = 0
    last_revision: int = 0


@dataclass
class CandidateActionReranker:
    config: CandidateRerankConfig = field(default_factory=CandidateRerankConfig)
    enabled: bool = False
    _tracks: dict[int, list[_CandidateTrack]] = field(default_factory=dict, init=False)
    _active: dict[int, np.ndarray] = field(default_factory=dict, init=False)
    _revision: dict[int, int] = field(default_factory=dict, init=False)

    def reset(self) -> None:
        self._tracks.clear()
        self._active.clear()
        self._revision.clear()

    @staticmethod
    def _distance(first: np.ndarray, second: np.ndarray) -> float:
        return float(np.linalg.norm(first[[0, 2]] - second[[0, 2]]))

    def _track(
        self, task_idx: int, point: np.ndarray, revision: int
    ) -> _CandidateTrack:
        tracks = self._tracks.setdefault(task_idx, [])
        eligible = [
            (track, self._distance(track.position, point)) for track in tracks
        ]
        eligible = [item for item in eligible if item[1] <= self.config.match_radius_m]
        if eligible:
            track = min(eligible, key=lambda item: item[1])[0]
            track.position = 0.75 * track.position + 0.25 * point
            if track.last_revision != revision:
                track.observations += 1
                track.last_revision = revision
            return track
        track = _CandidateTrack(point.copy(), last_revision=revision)
        tracks.append(track)
        return track

    def mark_failed(self, task_idx: int, position: Sequence[float]) -> None:
        point = np.asarray(position, dtype=np.float64)
        for track in self._tracks.get(int(task_idx), []):
            if self._distance(track.position, point) <= self.config.match_radius_m:
                track.failures += 1
                return

    def rank(
        self,
        task_idx: int,
        records: Sequence[Mapping[str, object]],
        origin: Sequence[float],
        travel_cost: Callable[[np.ndarray, np.ndarray], float] | None = None,
        active_position: Sequence[float] | None = None,
        visited_positions: Iterable[Sequence[float]] = (),
    ) -> tuple[list[dict[str, object]], dict[str, object]]:
        source = [dict(record) for record in records]
        if not self.enabled or not source:
            return source, {
                "enabled": int(self.enabled),
                "candidates": len(source),
                "reordered": 0,
                "switches": 0,
                "hysteresis_holds": 0,
                "geodesic_fallbacks": 0,
            }

        task_idx = int(task_idx)
        origin_array = np.asarray(origin, dtype=np.float64)
        if origin_array.shape != (3,) or not np.isfinite(origin_array).all():
            raise ValueError("origin must be finite xyz")
        visited = [
            point
            for point in (
                np.asarray(value, dtype=np.float64) for value in visited_positions
            )
            if point.shape == (3,) and np.isfinite(point).all()
        ]
        points: list[np.ndarray] = []
        tracks: list[_CandidateTrack] = []
        semantic: list[float] = []
        support: list[float] = []
        travel: list[float] = []
        novelty: list[float] = []
        revisits: list[float] = []
        fallbacks = 0

        revisions = {int(record.get("map_revision", 0)) for record in source}
        revision = max(revisions) if revisions else 0
        revision_changed = int(self._revision.get(task_idx, revision) != revision)
        self._revision[task_idx] = revision

        for record in source:
            point = np.asarray(record["position"], dtype=np.float64)
            if point.shape != (3,) or not np.isfinite(point).all():
                raise ValueError("candidate position must be finite xyz")
            points.append(point)
            record_revision = int(record.get("map_revision", revision))
            tracks.append(self._track(task_idx, point, record_revision))
            semantic_value = float(record.get("semantic_score", 0.0))
            support_value = float(
                record.get("support_score", record.get("support_count", 0.0))
            )
            if not np.isfinite([semantic_value, support_value]).all():
                raise ValueError("candidate metadata scores must be finite")
            semantic.append(semantic_value)
            support.append(support_value)
            value = float("inf")
            if travel_cost is not None:
                try:
                    value = float(travel_cost(origin_array, point))
                except Exception:
                    value = float("inf")
            if not np.isfinite(value):
                value = self._distance(origin_array, point)
                fallbacks += 1
            travel.append(value)
            nearest = min((self._distance(point, old) for old in visited), default=5.0)
            novelty.append(min(nearest, 5.0))
            revisits.append(float(nearest <= self.config.match_radius_m))

        components = {
            "semantic": _normalize(semantic),
            "support": _normalize(support),
            "recurrence": _normalize([track.observations for track in tracks]),
            "travel": _normalize(travel),
            "novelty": _normalize(novelty),
            "revisit": np.asarray(revisits),
            "failure": _normalize([track.failures for track in tracks]),
        }
        cfg = self.config
        utilities = (
            cfg.semantic * components["semantic"]
            + cfg.support * components["support"]
            + cfg.recurrence * components["recurrence"]
            + cfg.novelty * components["novelty"]
            - cfg.travel * components["travel"]
            - cfg.revisit * components["revisit"]
            - cfg.failure * components["failure"]
        )
        order = sorted(
            range(len(source)),
            key=lambda index: (
                -float(utilities[index]),
                int(source[index].get("raw_rank", index)),
                tuple(float(v) for v in points[index]),
            ),
        )

        active = np.asarray(active_position, dtype=np.float64) if active_position is not None else self._active.get(task_idx)
        holds = 0
        switches = 0
        if active is not None:
            active_indices = [
                index for index, point in enumerate(points)
                if self._distance(point, active) <= self.config.match_radius_m
            ]
            if active_indices:
                active_index = min(active_indices, key=lambda index: self._distance(points[index], active))
                if utilities[order[0]] <= utilities[active_index] + cfg.switch_margin:
                    order.remove(active_index)
                    order.insert(0, active_index)
                    holds = 1
                else:
                    switches = int(order[0] != active_index)

        self._active[task_idx] = points[order[0]].copy()
        ranked: list[dict[str, object]] = []
        for index in order:
            record = source[index]
            record["utility"] = float(utilities[index])
            record["utility_components"] = {
                name: float(values[index]) for name, values in components.items()
            }
            ranked.append(record)
        return ranked, {
            "enabled": 1,
            "candidates": len(source),
            "reordered": int(order != list(range(len(source)))),
            "switches": switches,
            "hysteresis_holds": holds,
            "geodesic_fallbacks": fallbacks,
            "revision_changed": revision_changed,
            "raw_order": [int(record.get("raw_rank", index)) for index, record in enumerate(source)],
            "reranked_order": [int(record.get("raw_rank", index)) for index, record in enumerate(ranked)],
            "selected_utility": float(ranked[0]["utility"]),
        }


@dataclass(frozen=True)
class FrontierRerankConfig:
    gain: float = 1.0
    clearance: float = 0.25
    travel: float = 0.35
    novelty: float = 0.20
    revisit: float = 0.75
    failure: float = 1.0
    switch_margin: float = 0.15
    neighborhood: int = 12

    def __post_init__(self) -> None:
        for name in ("gain", "clearance", "travel", "novelty", "revisit", "failure", "switch_margin"):
            _finite_weight(getattr(self, name), name)
        if any(
            getattr(self, name) < 0
            for name in ("gain", "clearance", "travel", "novelty", "revisit", "failure")
        ):
            raise ValueError("frontier utility weights must be non-negative")
        if self.switch_margin < 0 or self.neighborhood < 1:
            raise ValueError("frontier margin must be non-negative and neighborhood positive")


@dataclass
class FrontierUtilityRanker:
    config: FrontierRerankConfig = field(default_factory=FrontierRerankConfig)
    enabled: bool = False
    _active: tuple[int, int] | None = field(default=None, init=False)
    _history: list[tuple[int, int]] = field(default_factory=list, init=False)
    _failures: dict[tuple[int, int], int] = field(default_factory=dict, init=False)

    def reset(self) -> None:
        self._active = None
        self._history.clear()
        self._failures.clear()

    def mark_failed(self) -> None:
        if self._active is not None:
            self._failures[self._active] = self._failures.get(self._active, 0) + 1

    @staticmethod
    def _reachable_free(
        obstacle: np.ndarray,
        explored: np.ndarray,
        agent: tuple[int, int],
    ) -> np.ndarray:
        rows, cols = obstacle.shape
        start = (
            min(max(int(agent[0]), 0), rows - 1),
            min(max(int(agent[1]), 0), cols - 1),
        )
        traversible = np.asarray(explored, dtype=bool) & ~obstacle
        if not traversible[start]:
            free_pixels = np.argwhere(traversible)
            if free_pixels.size == 0:
                return np.zeros_like(obstacle, dtype=bool)
            start_array = np.asarray(start)
            start = tuple(
                free_pixels[
                    np.argmin(np.linalg.norm(free_pixels - start_array, axis=1))
                ]
            )
        try:
            import cv2

            _, labels = cv2.connectedComponents(
                traversible.astype(np.uint8), connectivity=4
            )
            label = labels[start]
            return labels == label if label else np.zeros_like(obstacle, dtype=bool)
        except ImportError:
            pass
        reachable = np.zeros_like(obstacle, dtype=bool)
        reachable[start] = True
        queue = [start]
        while queue:
            row, col = queue.pop()
            for neighbor in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
                nr, nc = neighbor
                if 0 <= nr < rows and 0 <= nc < cols and traversible[nr, nc] and not reachable[nr, nc]:
                    reachable[nr, nc] = True
                    queue.append(neighbor)
        return reachable

    @staticmethod
    def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
        try:
            import cv2

            count, labels = cv2.connectedComponents(
                np.asarray(mask, dtype=np.uint8), connectivity=4
            )
            return [
                [tuple(pixel) for pixel in np.argwhere(labels == label)]
                for label in range(1, count)
            ]
        except ImportError:
            pass
        pending = set(map(tuple, np.argwhere(mask > 0)))
        output: list[list[tuple[int, int]]] = []
        while pending:
            seed = min(pending)
            pending.remove(seed)
            component = [seed]
            queue = [seed]
            while queue:
                row, col = queue.pop()
                for neighbor in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
                    if neighbor in pending:
                        pending.remove(neighbor)
                        queue.append(neighbor)
                        component.append(neighbor)
            output.append(component)
        return output

    def rank(
        self,
        frontier_map: np.ndarray,
        obstacle_map: np.ndarray,
        explored_map: np.ndarray,
        agent_pixel: Sequence[float],
    ) -> tuple[np.ndarray, dict[str, object]]:
        frontier = np.asarray(frontier_map)
        if not self.enabled or not np.any(frontier > 0):
            return frontier.copy(), {"enabled": int(self.enabled), "components": 0, "reordered": 0}
        obstacle = np.asarray(obstacle_map) > 0
        explored = np.asarray(explored_map) > 0
        agent = np.asarray(agent_pixel, dtype=np.float64)
        reachable_free = self._reachable_free(
            obstacle,
            explored,
            (int(round(agent[0])), int(round(agent[1]))),
        )
        reachable_frontier = reachable_free.copy()
        reachable_frontier[1:, :] |= reachable_free[:-1, :]
        reachable_frontier[:-1, :] |= reachable_free[1:, :]
        reachable_frontier[:, 1:] |= reachable_free[:, :-1]
        reachable_frontier[:, :-1] |= reachable_free[:, 1:]
        components = self._components(
            (frontier > 0) & reachable_frontier & ~obstacle
        )
        if not components:
            return frontier.copy(), {"enabled": 1, "components": 0, "reordered": 0, "fallback": 1}

        representatives: list[tuple[int, int]] = []
        gain: list[float] = []
        clearance: list[float] = []
        travel: list[float] = []
        novelty: list[float] = []
        revisit: list[float] = []
        failures: list[float] = []
        radius = max(1, int(self.config.neighborhood))
        rows, cols = frontier.shape
        for component in components:
            center = np.mean(np.asarray(component), axis=0)
            representative = min(component, key=lambda pixel: (np.linalg.norm(np.asarray(pixel) - center), pixel))
            representatives.append(representative)
            row, col = representative
            rs, re = max(0, row - radius), min(rows, row + radius + 1)
            cs, ce = max(0, col - radius), min(cols, col + radius + 1)
            gain.append(float(np.count_nonzero(~explored[rs:re, cs:ce])))
            clearance.append(float(1.0 - np.mean(obstacle[rs:re, cs:ce])))
            travel.append(float(np.linalg.norm(np.asarray(representative) - agent)))
            nearest = min((np.linalg.norm(np.asarray(representative) - old) for old in self._history), default=float(radius))
            novelty.append(float(min(nearest, radius)))
            revisit.append(float(nearest < radius / 2))
            failures.append(float(self._failures.get(representative, 0)))
        values = (
            self.config.gain * _normalize(gain)
            + self.config.clearance * _normalize(clearance)
            + self.config.novelty * _normalize(novelty)
            - self.config.travel * _normalize(travel)
            - self.config.revisit * np.asarray(revisit)
            - self.config.failure * _normalize(failures)
        )
        order = sorted(range(len(components)), key=lambda index: (-float(values[index]), representatives[index]))
        selected = order[0]
        holds = 0
        if self._active in representatives:
            active_index = representatives.index(self._active)
            if values[selected] <= values[active_index] + self.config.switch_margin:
                selected = active_index
                holds = 1
        self._active = representatives[selected]
        self._history.append(self._active)
        self._history = self._history[-16:]
        output = np.zeros_like(frontier)
        output[self._active] = 1
        return output, {
            "enabled": 1,
            "components": len(components),
            "reachable": len(components),
            "reordered": int(selected != 0),
            "hysteresis_holds": holds,
            "selected_pixel": list(self._active),
            "selected_utility": float(values[selected]),
        }
