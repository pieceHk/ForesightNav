"""Compile frontier kernels before navigation creates large GPU map tensors."""

from functools import lru_cache
import gc


@lru_cache(maxsize=1)
def warm_up_frontier_kernels():
    """Keep first-use Numba compiler frames out of the GPU mapping loop."""
    import numpy as np
    from frontier_exploration.frontier_detection import (
        contour_to_frontiers,
        get_frontier_midpoint,
    )

    # OpenCV contours and Bresenham interpolation use int32. Fortran contours
    # and their slices also exercise F and arbitrary-stride specializations.
    contour = np.array([[[1, 1]], [[1, 5]], [[5, 5]], [[5, 1]]], dtype=np.int32)
    mask = np.full((8, 8), 255, dtype=np.uint8)
    for points in (contour, np.asfortranarray(contour), np.asfortranarray(contour)[::2]):
        contour_to_frontiers(points, mask)
        get_frontier_midpoint(points)

    # Numba compilation can leave cyclic Python frames alive. Collect them
    # once, before any navigation map tensors can be retained through callers.
    gc.collect()
