"""Portable semantic index: metric world positions and sparse feature coefficients."""

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class FeatureSpace:
    name: str
    dimension: int

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Feature space needs a model/checkpoint identifier")
        if type(self.dimension) is not int or self.dimension < 1:
            raise ValueError("Feature dimension must be positive")


@dataclass(frozen=True)
class SemanticIndex:
    """Positions use Habitat world xyz in metres, y up; arrays live on CPU.

    Codebook rows are normalized embeddings. Each point has up to three codebook
    indices and normalized weights. The producing map owns snapshot isolation.
    """

    positions: Any
    codebook: Any
    indices: Any
    weights: Any
    feature_space: FeatureSpace

    def __post_init__(self):
        points = self.positions.shape[0]
        if self.positions.shape != (points, 3):
            raise ValueError("Query positions must be Nx3 world coordinates")
        if (
            len(self.codebook.shape) != 2
            or self.codebook.shape[1] != self.feature_space.dimension
        ):
            raise ValueError("Codebook dimension does not match its feature space")
        if self.indices.shape != (points, 3) or self.weights.shape != (points, 3, 1):
            raise ValueError(
                "Query coefficients must align with positions and three feature slots"
            )

    def data(self):
        return self.codebook, self.positions, self.indices, self.weights


class TextEncoder(Protocol):
    feature_space: FeatureSpace

    def encode(self, prompts, *, device): ...


class QueryBackend(Protocol):
    """A mapper passes an opaque, stable snapshot to its configured reader."""

    top_k: int

    def query(
        self, state, category, top_k, *, return_records=False, map_revision=0
    ): ...

    def query_many(
        self, state, categories, top_k, *, return_records=False, map_revision=0
    ): ...

    def close(self): ...
