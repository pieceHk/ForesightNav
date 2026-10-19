"""Check the extracted encoder helpers against the frozen neural implementation."""

import ast
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
import torch

from pbsp.mapping.encoder import OnlineEncoder
from pbsp.mapping.geometry import calculate_distance_matrix, sample_image_grid


ROOT = Path(__file__).resolve().parents[2]
SOURCE = "Mapping/runtime/embodied_splat/src/model/encoder/encoder_test_unet3d_online.py"


def _source(path):
    return ast.parse(subprocess.check_output(["git", "show", "422cb8e:" + path], cwd=ROOT))


def _without_annotations(function):
    function.returns = None
    for argument in function.args.args:
        argument.annotation = None
    return function


def _frozen_helpers():
    tree = _source(SOURCE)
    encoder = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name.startswith("EmbodiedSplatEncoder")
    )
    matching = next(
        node
        for node in encoder.body
        if isinstance(node, ast.FunctionDef) and node.name == "compute_matching_feats"
    )
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"rotation_distance", "calculate_distance_matrix"}
    ]
    utils = _source("Mapping/runtime/embodied_splat/sr_utils/generic_utils.py")
    functions += [
        node
        for node in utils.body
        if isinstance(node, ast.FunctionDef) and node.name in {"tensor_B_to_bM", "tensor_bM_to_B"}
    ]
    grid = next(
        node
        for node in _source("Mapping/runtime/embodied_splat/src/geometry/projection.py").body
        if isinstance(node, ast.FunctionDef) and node.name == "sample_image_grid"
    )
    functions.append(_without_annotations(grid))
    functions.append(matching)
    namespace = {"torch": torch}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[])),
            "<frozen-encoder-helpers>",
            "exec",
        ),
        namespace,
    )
    return namespace


class BatchSensitiveMatching:
    def __init__(self):
        self.calls = []

    def __call__(self, frames):
        self.calls.append(tuple(frames.shape))
        return frames[:, :1] * 2 + frames[:, :1].mean(0, keepdim=True)


def _formatter(frames, *, apply_func):
    batches, views = frames.shape[:2]
    result = apply_func(frames.flatten(0, 1))
    return result.view(batches, views, *result.shape[1:])


@pytest.mark.parametrize("unbatched", (False, True))
def test_matching_encoder_batch_order_and_outputs_match_frozen(unbatched):
    original = _frozen_helpers()["compute_matching_feats"]
    current = torch.arange(2 * 3 * 4 * 6, dtype=torch.float32).reshape(2, 3, 4, 6) / 10
    sources = torch.arange(2 * 3 * 3 * 4 * 6, dtype=torch.float32).reshape(2, 3, 3, 4, 6) / 100
    actual = SimpleNamespace(matching_net=BatchSensitiveMatching(), tensor_formatter=_formatter)
    expected = SimpleNamespace(matching_net=BatchSensitiveMatching(), tensor_formatter=_formatter)
    output = OnlineEncoder.compute_matching_feats(actual, current, sources, unbatched)
    reference = original(expected, current, sources, unbatched)
    for tensor, baseline in zip(output, reference):
        torch.testing.assert_close(tensor, baseline, rtol=0, atol=0)
    assert actual.matching_net.calls == expected.matching_net.calls
    assert len(actual.matching_net.calls) == (8 if unbatched else 1)


def test_view_distances_and_pixel_centers_match_frozen():
    original = _frozen_helpers()
    poses = torch.eye(4).repeat(1, 6, 1, 1)
    poses[0, 2, 0, 3] = 0.5
    poses[0, 3, :2, :2] = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
    torch.testing.assert_close(
        calculate_distance_matrix(poses),
        original["calculate_distance_matrix"](poses),
        rtol=0,
        atol=0,
    )
    for shape in ((3, 5), (4, 6)):
        for actual, expected in zip(sample_image_grid(shape), original["sample_image_grid"](shape)):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("name", ("extract_fastsam_clip_feats", "extract_sam_clip_feats"))
def test_semantic_feature_extraction_preserves_executable_ast(name):
    previous = _source(SOURCE)
    old_class = next(
        node
        for node in previous.body
        if isinstance(node, ast.ClassDef) and node.name.startswith("EmbodiedSplatEncoder")
    )
    new_class = next(
        node
        for node in ast.parse((ROOT / "pbsp/mapping/features.py").read_text()).body
        if isinstance(node, ast.ClassDef)
    )
    old = next(
        node for node in old_class.body if isinstance(node, ast.FunctionDef) and node.name == name
    )
    new = next(
        node for node in new_class.body if isinstance(node, ast.FunctionDef) and node.name == name
    )
    assert ast.dump(old, include_attributes=False) == ast.dump(new, include_attributes=False)


@pytest.mark.parametrize(
    "override",
    (
        {"is_testing": False},
        {"online_incremental": False},
        {"export_ply": True},
        {"test_bev": True},
    ),
)
def test_online_entry_rejects_unsupported_offline_modes(override):
    model = OnlineEncoder(SimpleNamespace())
    options = dict(is_testing=True, online_incremental=True, export_ply=False, test_bev=False)
    options.update(override)
    with pytest.raises(ValueError, match="OnlineEncoder requires"):
        model([], global_step=0, **options)


def test_profile_observer_is_optional_and_does_not_enter_parameter_state():
    records = []
    model = OnlineEncoder(SimpleNamespace(), profile_sink=records.append)
    model._record_profile({"phase": "warmup_construct", "ms": 1.0})
    assert records == [{"phase": "warmup_construct", "ms": 1.0}]
    assert not model.state_dict()
    OnlineEncoder(SimpleNamespace())._record_profile({"phase": "fuse_frame", "ms": 2.0})
