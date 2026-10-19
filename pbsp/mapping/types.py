"""Typed boundaries between observation, Mapping, Query, and Navigation."""

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

import numpy as np


@dataclass(frozen=True)
class CameraObservation:
    frame_id: int
    rgb: np.ndarray
    depth_m: np.ndarray
    habitat_pose: np.ndarray
    normalized_intrinsics: np.ndarray


@dataclass(frozen=True)
class WorldCandidate:
    position: np.ndarray
    rank: int


@dataclass(frozen=True)
class MappingSnapshot:
    scene_id: str
    episode_id: str
    accepted_keyframes: int
    raw_frames: int
    active_gaussians: int
    revision: int
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


class MappingBackend(Protocol):
    """Model-specific geometry fusion and semantic-injection port."""

    def reset(self, scene_id: str, episode_id: str) -> None: ...

    def update(self, accepted_frames: Sequence[CameraObservation]) -> MappingSnapshot: ...

    def query(self, category: str, top_k: int) -> Sequence[WorldCandidate]: ...

    def snapshot(self) -> MappingSnapshot: ...

    def close(self) -> None: ...
