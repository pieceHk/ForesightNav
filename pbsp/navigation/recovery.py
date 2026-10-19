# Portions adapted from LagMemo, Copyright (c) Meta Platforms, Inc. and affiliates.
# Upstream MIT terms and provenance are retained in THIRD_PARTY_NOTICES.md.

import json

from collections import deque
from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np


@dataclass
class TrappedDetector:
    warmup_steps: int = 30
    window_steps: int = 25
    radius_m: float = 0.05
    _positions: deque = field(init=False)
    _steps: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._positions = deque(maxlen=self.window_steps)

    def reset(self) -> None:
        self._positions.clear()
        self._steps = 0

    def update(self, position: np.ndarray) -> bool:
        self._steps += 1
        self._positions.append(np.asarray(position, dtype=np.float32))
        if self._steps < self.warmup_steps or len(self._positions) < self.window_steps:
            return False
        anchor = self._positions[0]
        max_distance = max(np.linalg.norm(point - anchor) for point in self._positions)
        return max_distance <= self.radius_m


@dataclass
class NoveltyGoalSelector:
    """Select recovery goals that do not repeatedly revisit one local basin."""

    history_size: int = 8
    novelty_weight: float = 0.65
    _history: deque = field(init=False)
    _selections: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._history = deque(maxlen=self.history_size)

    def reset(self) -> None:
        self._history.clear()
        self._selections = 0

    @property
    def selections(self) -> int:
        return self._selections

    def select(
        self,
        candidates: Iterable[np.ndarray],
        origin: np.ndarray,
    ) -> Optional[np.ndarray]:
        points = [np.asarray(point, dtype=np.float32) for point in candidates]
        if not points:
            return None
        origin = np.asarray(origin, dtype=np.float32)

        def score(point: np.ndarray) -> float:
            travel = np.linalg.norm(point[[0, 2]] - origin[[0, 2]])
            if not self._history:
                return float(travel)
            novelty = min(np.linalg.norm(point[[0, 2]] - old[[0, 2]]) for old in self._history)
            return float((1.0 - self.novelty_weight) * travel + self.novelty_weight * novelty)

        selected = max(points, key=score)
        self._history.append(selected.copy())
        self._selections += 1
        return selected


def should_request_random_exploration(
    candidate_available, frontier_map, explored_map, task_step, warmup_steps, fully_explored
):
    no_frontier = not np.any(np.asarray(frontier_map) > 0)
    map_complete = bool(fully_explored) or (
        task_step >= warmup_steps and np.any(np.asarray(explored_map) > 0) and no_frontier
    )
    return not candidate_available and no_frontier and map_complete


class RecoveryState:
    """Track motion failures and reset local state for explicit recovery goals."""


    def _reset_trapped_state(self):
        self.trapped_position_history = []
        self.trapped_active = False

    def _update_trapped_diagnostic(self, position, action):
        step = self.sub_task_timesteps[0][self.current_task_idx]
        point = np.asarray(position, dtype=np.float32).copy()
        self.trapped_position_history.append(point)
        self.trapped_position_history = self.trapped_position_history[-self.trapped_window_steps :]
        if (
            step < self.trapped_warmup_steps
            or len(self.trapped_position_history) < self.trapped_window_steps
            or self.see_around != 0
        ):
            return None
        positions = np.stack(self.trapped_position_history)
        displacement = np.linalg.norm(positions - positions[0], axis=1)
        trapped = float(displacement.max()) <= self.trapped_radius
        if trapped and (not self.trapped_active):
            event = {
                "type": "trapped",
                "scene": self.runtime_scene_id,
                "episode": self.runtime_episode_id,
                "task": self.current_task_idx,
                "step": int(step),
                "window": self.trapped_window_steps,
                "radius_m": self.trapped_radius,
                "position": point.tolist(),
                "action": str(action),
            }
            self.trapped_events.append(event)
            self.trapped_active = True
            print(f"[TRAPPED] {json.dumps(event, ensure_ascii=False)}", flush=True)
            return event
        elif not trapped and float(displacement.max()) > self.trapped_radius * 2:
            self.trapped_active = False
        return None

    def _clear_navmesh_override_goal(self):
        self.navmesh_override_goal = None
        self.navmesh_override_mode = None
        self.navmesh_override_task = None

    def clear_local_navigation_state(self):
        self.ctr = 0
        self.stuck_num = 0
        self._reset_trapped_state()
        if hasattr(self.planner, "reset_subtask_state"):
            self.planner.reset_subtask_state()
        else:
            self.planner.dd = None
            self.planner.reset_collision_state()

    def set_navmesh_override_goal(self, goal, mode):
        point = np.asarray(goal, dtype=np.float32)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError(f"invalid navmesh override goal: {goal}")
        if mode not in {"escape", "random"}:
            raise ValueError(f"invalid navmesh override mode: {mode}")
        self.navmesh_override_goal = point
        self.navmesh_override_mode = mode
        self.navmesh_override_task = self.current_task_idx
        self.see_around = 0
        self.clear_local_navigation_state()
        print(
            f"[NAVMESH_OVERRIDE] task={self.current_task_idx} mode={mode} goal={point.tolist()}",
            flush=True,
        )
