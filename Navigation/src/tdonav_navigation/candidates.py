from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np

from .contracts import Candidate, Navmesh


def project_candidates(
    candidates: Iterable[Candidate],
    navmesh: Navmesh,
    tolerance_m: float = 1.5,
    merge_radius_m: float = 0.25,
) -> list[Candidate]:
    projected: list[Candidate] = []
    for candidate in candidates:
        point = navmesh.project(candidate.world_position)
        if point is None:
            continue
        point = np.asarray(point, dtype=np.float32)
        if np.linalg.norm(point[[0, 2]] - candidate.world_position[[0, 2]]) > tolerance_m:
            continue
        if any(np.linalg.norm(point[[0, 2]] - item.world_position[[0, 2]]) < merge_radius_m for item in projected):
            continue
        projected.append(Candidate(point, candidate.rank))
    return projected


@dataclass
class ActiveCandidateTracker:
    continuity_radius_m: float = 1.0
    active: Optional[Candidate] = field(default=None, init=False)

    def refresh(self, candidates: list[Candidate]) -> Optional[Candidate]:
        if self.active is not None:
            for candidate in candidates:
                distance = np.linalg.norm(
                    candidate.world_position[[0, 2]] - self.active.world_position[[0, 2]]
                )
                if distance <= self.continuity_radius_m:
                    self.active = candidate
                    return self.active
        self.active = candidates[0] if candidates else None
        return self.active
