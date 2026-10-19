"""Coordinate transforms defined by the Online Mapping formulation."""

import numpy as np


RUB_TO_RDF = np.diag([1.0, -1.0, -1.0])


def habitat_pose_to_c2w(habitat_pose: np.ndarray) -> np.ndarray:
    """Convert `[qw,qx,qy,qz,px,py,pz]` to mapper OpenCV-style c2w."""
    pose = np.asarray(habitat_pose, dtype=np.float64)
    if pose.shape != (7,):
        raise ValueError(f"expected pose shape (7,), got {pose.shape}")
    qw, qx, qy, qz, px, py, pz = pose
    x, y, z, w = qw, qx, qy, qz
    rotation = np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=np.float64,
    )
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = rotation @ RUB_TO_RDF
    c2w[:3, 3] = RUB_TO_RDF @ np.array([px, py, pz], dtype=np.float64)
    return c2w.astype(np.float32)


def mapper_world_to_habitat(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.shape[-1] != 3:
        raise ValueError("world points must end in xyz")
    return (points @ RUB_TO_RDF.T).astype(np.float32)
