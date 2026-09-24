import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from tdonav_mapping.coordinates import habitat_pose_to_c2w
from tdonav_mapping.keyframes import frame_change_ratio


REFERENCE = os.environ.get("THREEDONAV_REFERENCE_MAPPING")


def _load_reference():
    if not REFERENCE:
        pytest.skip("THREEDONAV_REFERENCE_MAPPING is not configured")
    source = Path(REFERENCE)
    sys.path.insert(0, str(source.parent))
    spec = importlib.util.spec_from_file_location("tdonav_reference_mapping", source)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_coordinate_and_keyframe_math_matches_reference():
    reference = _load_reference()
    pose = np.array([0.5, -0.5, 0.5, -0.5, 1.25, -0.75, 2.5], dtype=np.float32)
    assert np.allclose(
        habitat_pose_to_c2w(pose),
        reference.globalpose_to_c2w(pose),
        atol=1e-6,
    )

    rng = np.random.default_rng(37)
    previous = rng.integers(0, 256, size=(384, 512, 3), dtype=np.uint8)
    current = np.array(previous, copy=True)
    current[:, :256] = 255 - current[:, :256]

    import torch

    previous_tensor = torch.from_numpy(previous).permute(2, 0, 1).float() / 255.0
    current_tensor = torch.from_numpy(current).permute(2, 0, 1).float() / 255.0
    expected = reference.frame_change_ratio(current_tensor, previous_tensor)
    actual = frame_change_ratio(current, previous)
    assert actual == pytest.approx(expected, abs=1e-7)
