"""Coordinate transforms defined by the Online Mapping formulation."""

import numpy as np


RUB_TO_RDF = np.diag([1.0, -1.0, -1.0])


def habitat_pose_to_c2w(habitat_pose: np.ndarray) -> np.ndarray:
    """Convert `[qw,qx,qy,qz,px,py,pz]` to mapper OpenCV-style c2w."""
    pose = np.asarray(habitat_pose, dtype=np.float64)
    if pose.shape != (7,):
        raise ValueError(f"expected pose shape (7,), got {pose.shape}")
    qw, qx, qy, qz, px, py, pz = pose
    x, y, z, w = qx, qy, qz, qw
    rotation = np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=np.float64,
    )
    c2w = np.eye(4, dtype=np.float64)
    # Convert world and optical axes separately; a quaternion permutation only
    # happened to agree for yaw-only poses, not for pitched camera poses.
    c2w[:3, :3] = RUB_TO_RDF @ rotation @ RUB_TO_RDF
    c2w[:3, 3] = RUB_TO_RDF @ np.array([px, py, pz], dtype=np.float64)
    return c2w.astype(np.float32)


def mapper_world_to_habitat(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.shape[-1] != 3:
        raise ValueError("world points must end in xyz")
    return (points @ RUB_TO_RDF.T).astype(np.float32)


def sample_image_grid(shape, device=None):
    """Return normalized xy coordinates and integer ij indices at pixel centers."""
    import torch

    if device is None:
        device = torch.device("cpu")
    indices = [torch.arange(length, device=device) for length in shape]
    stacked_indices = torch.stack(torch.meshgrid(*indices, indexing="ij"), dim=-1)
    coordinates = [(idx + 0.5) / length for (idx, length) in zip(indices, shape)]
    coordinates = reversed(coordinates)
    coordinates = torch.stack(torch.meshgrid(*coordinates, indexing="xy"), dim=-1)
    return (coordinates, stacked_indices)


def rotation_distance(rotations):
    import torch

    R1 = rotations.unsqueeze(2)
    R2 = rotations.unsqueeze(1)
    R_rel = torch.matmul(R1.transpose(-2, -1), R2)
    trace = torch.diagonal(R_rel, dim1=-2, dim2=-1).sum(-1)
    trace = torch.clamp(trace, -1, 3)
    angle = torch.acos((trace - 1) / 2)
    return angle.squeeze(0)


def calculate_distance_matrix(poses):
    import torch

    translations = poses[:, :, :3, 3]
    rotations = poses[:, :, :3, :3]
    translation_dist = torch.cdist(translations, translations).squeeze(0)
    rotation_dist = rotation_distance(rotations)
    combined_dist = translation_dist + rotation_dist
    return combined_dist
