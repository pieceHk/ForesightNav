"""RGB-D observation passed to the live mapping service."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CameraObservation:
    """Metric depth, Habitat [qw,qx,qy,qz,x,y,z] pose and normalized intrinsics."""

    frame_id: int
    rgb: np.ndarray
    depth_m: np.ndarray
    habitat_pose: np.ndarray
    normalized_intrinsics: np.ndarray
