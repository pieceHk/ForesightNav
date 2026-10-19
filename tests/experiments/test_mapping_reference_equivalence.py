import importlib.util
import ast
import os
import sys
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pbsp.mapping.geometry import habitat_pose_to_c2w
from pbsp.mapping.keyframes import frame_change_ratio


REFERENCE = os.environ.get("THREEDONAV_REFERENCE_MAPPING")


def _load_reference():
    if not REFERENCE:
        # Read only the two pure helpers from the frozen repository revision;
        # importing that entire server would require its GPU model packages.
        import torch
        import torch.nn.functional as F

        root = Path(__file__).resolve().parents[2]
        tree = ast.parse(subprocess.check_output([
            "git", "show", "422cb8e:Mapping/runtime/server/online_es_mapper.py",
        ], cwd=root, text=True))
        constants = {"_RUB2RDF", "_FRAME_PIXEL_DELTA", "_FRAME_COMPARE_H", "_FRAME_COMPARE_W"}
        functions = {"globalpose_to_c2w", "frame_change_ratio"}
        body = [node for node in tree.body if (
            isinstance(node, ast.FunctionDef) and node.name in functions
        ) or (
            isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id in constants for target in node.targets)
        )]
        namespace = {"np": np, "torch": torch, "F": F, "os": SimpleNamespace(environ={})}
        exec(compile(ast.Module(body=body, type_ignores=[]), "<frozen-mapping-helpers>", "exec"), namespace)
        return SimpleNamespace(**{name: namespace[name] for name in functions})
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
