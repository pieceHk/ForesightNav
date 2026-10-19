from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Protocol

import numpy as np


class Action(str, Enum):
    STOP = "stop"
    TURN_RIGHT = "turn_right"
    MOVE_FORWARD = "move_forward"


@dataclass(frozen=True)
class Candidate:
    world_position: np.ndarray
    rank: int


class Navmesh(Protocol):
    def project(self, point: np.ndarray) -> Optional[np.ndarray]: ...


@dataclass(frozen=True)
class Verification:
    target_evidence: bool
    finite_depth_pose: bool
    mask_pixels: int

    @property
    def confirmed(self) -> bool:
        return self.target_evidence and self.finite_depth_pose and self.mask_pixels > 0


class NavigationObservation(Protocol):
    """RGB-D and local pose supplied by the simulator adapter.

    Depth and GPS use metres; compass uses radians. The adapter supplies the
    seven-element global pose used by the waypoint transform.
    """

    rgb: np.ndarray
    depth: np.ndarray
    semantic: np.ndarray
    gps: np.ndarray
    compass: np.ndarray
    globalpose: np.ndarray
    camera_pose: Any
    task_observations: dict[str, Any]
