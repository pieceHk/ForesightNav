"""Deterministic keyframe admission independent of the fusion backend."""

from dataclasses import dataclass
from typing import Optional

import numpy as np


def _area_downsample(image: np.ndarray, height: int, width: int) -> np.ndarray:
    src_h, src_w = image.shape[:2]
    ys = np.linspace(0, src_h, height + 1, dtype=int)
    xs = np.linspace(0, src_w, width + 1, dtype=int)
    out = np.empty((height, width, 3), dtype=np.float32)
    for row in range(height):
        for col in range(width):
            patch = image[ys[row] : ys[row + 1], xs[col] : xs[col + 1]]
            out[row, col] = patch.mean(axis=(0, 1))
    return out


def frame_change_ratio(
    image: np.ndarray,
    reference: np.ndarray,
    sample_size: tuple[int, int] = (48, 64),
    pixel_delta: float = 0.15,
) -> float:
    """Fraction of sampled pixels with mean normalized RGB delta above threshold."""
    if image.shape != reference.shape or image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("RGB frames must have identical HxWx3 shapes")
    current = image.astype(np.float32)
    previous = reference.astype(np.float32)
    if current.max(initial=0) > 1.0 or previous.max(initial=0) > 1.0:
        current /= 255.0
        previous /= 255.0
    height, width = sample_size
    current = _area_downsample(current, height, width)
    previous = _area_downsample(previous, height, width)
    changed = np.abs(current - previous).mean(axis=-1) >= pixel_delta
    return float(changed.mean())


@dataclass
class KeyframeAdmission:
    threshold: float = 0.30
    pixel_delta: float = 0.15
    sample_size: tuple[int, int] = (48, 64)
    _last_rgb: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._last_rgb = None

    def accept(self, rgb: np.ndarray) -> tuple[bool, float]:
        if self._last_rgb is None:
            self._last_rgb = np.array(rgb, copy=True)
            return True, 1.0
        ratio = frame_change_ratio(rgb, self._last_rgb, self.sample_size, self.pixel_delta)
        accepted = ratio >= self.threshold
        if accepted:
            self._last_rgb = np.array(rgb, copy=True)
        return accepted, ratio
