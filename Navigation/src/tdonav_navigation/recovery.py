from collections import deque
from dataclasses import dataclass, field

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
