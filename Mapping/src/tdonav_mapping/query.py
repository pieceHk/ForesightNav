"""Model-independent spatial support and greedy NMS for Online Query."""

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class ScoredPoint:
    position: np.ndarray
    score: float


def spatially_supported(
    positions: np.ndarray,
    scores: np.ndarray,
    similarity_threshold: float,
    support_radius_m: float,
    minimum_support: int,
) -> list[ScoredPoint]:
    positions = np.asarray(positions, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3 or scores.shape != (len(positions),):
        raise ValueError("expected positions Nx3 and scores N")
    selected: list[ScoredPoint] = []
    horizontal = positions[:, (0, 2)]
    for index, score in enumerate(scores):
        if score < similarity_threshold:
            continue
        distance = np.linalg.norm(horizontal - horizontal[index], axis=1)
        if int((distance <= support_radius_m).sum()) >= minimum_support:
            selected.append(ScoredPoint(positions[index].astype(np.float32), float(score)))
    return selected


def greedy_nms(points: Iterable[ScoredPoint], radius_m: float, top_k: int) -> list[ScoredPoint]:
    ordered = sorted(points, key=lambda point: point.score, reverse=True)
    kept: list[ScoredPoint] = []
    for point in ordered:
        if all(np.linalg.norm(point.position[[0, 2]] - other.position[[0, 2]]) > radius_m for other in kept):
            kept.append(point)
            if len(kept) == top_k:
                break
    return kept
