"""Model image and depth bounds shared by validation and inference."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ModelInputSpec:
    image_height: int
    image_width: int
    near: float
    far: float

    def __post_init__(self):
        for value in (self.image_height, self.image_width):
            if type(value) is not int or value < 2 or value % 2:
                raise ValueError(
                    "Model image dimensions must be positive even integers"
                )
        if (
            type(self.near) not in (int, float)
            or type(self.far) not in (int, float)
            or not math.isfinite(self.near)
            or not math.isfinite(self.far)
            or not 0 < self.near < self.far
        ):
            raise ValueError("Model bounds require finite 0 < near < far")
