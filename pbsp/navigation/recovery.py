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
            novelty = min(
                np.linalg.norm(point[[0, 2]] - old[[0, 2]])
                for old in self._history
            )
            return float((1.0 - self.novelty_weight) * travel + self.novelty_weight * novelty)

        selected = max(points, key=score)
        self._history.append(selected.copy())
        self._selections += 1
        return selected
