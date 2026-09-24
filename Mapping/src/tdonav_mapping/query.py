"""Model-independent spatial support and greedy NMS for Online Query."""

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class ScoredPoint:
    position: np.ndarray
    score: float


def candidate_records(
    positions: Iterable[Iterable[float]],
    semantic_scores: Iterable[float],
    support_counts: Iterable[int],
    support_scores: Iterable[float],
    raw_ranks: Iterable[int],
    map_revision: int,
) -> list[dict[str, object]]:
    """Build the JSON-safe metadata sidecar for a query response."""

    columns = [
        list(positions),
        list(semantic_scores),
        list(support_counts),
        list(support_scores),
        list(raw_ranks),
    ]
    if len({len(column) for column in columns}) != 1:
        raise ValueError("candidate metadata columns must have equal length")
    values = list(zip(*columns))
    records: list[dict[str, object]] = []
    for position, semantic, support, support_score, raw_rank in values:
        point = np.asarray(position, dtype=np.float64)
        numbers = np.asarray([semantic, support_score], dtype=np.float64)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError("candidate position must be a finite xyz vector")
        if not np.isfinite(numbers).all():
            raise ValueError("candidate scores must be finite")
        records.append(
            {
                "position": point.tolist(),
                "raw_rank": int(raw_rank),
                "semantic_score": float(semantic),
                "support_count": int(support),
                "support_score": float(support_score),
                "map_revision": int(map_revision),
            }
        )
    return records


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
