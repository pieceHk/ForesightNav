"""Explicit auxiliary checkpoints for the default model adapters."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MappingModelResources:
    fastsam: Path
    maskadapter: Path
    efficientnet: Path


@dataclass(frozen=True)
class MatchingResources:
    superpoint: Path
    superglue: Path
    clip: Path
