"""Compare actual Gaussian storage and fusion against the frozen network source."""

import ast
import copy
from pathlib import Path
import subprocess
from types import SimpleNamespace

from einops import rearrange, reduce, repeat
import numpy as np
import pytest
import torch
from torch import nn

from pbsp.mapping.fusion import GaussianFusion, positional_encoding
from pbsp.mapping.gaussians import EfficientGaussians


ROOT = Path(__file__).resolve().parents[2]
SOURCE = 'Mapping/runtime/embodied_splat/src/model/encoder/encoder_test_unet3d_online.py'


def _frozen():
    source = subprocess.check_output(['git', 'show', f'422cb8e:{SOURCE}'], cwd=ROOT)
    tree = ast.parse(source)
    encoder = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                   and node.name.startswith('EmbodiedSplatEncoder'))
    methods = [node for node in encoder.body if isinstance(node, ast.FunctionDef)
               and node.name in {'get_transformation_matrix', 'fuse_gaussians', 'refine_gaussians'}]
    nodes = [node for node in tree.body if (
        isinstance(node, ast.ClassDef) and node.name == 'EfficientGaussians'
        or isinstance(node, ast.FunctionDef) and node.name == 'positional_encoding')]
    nodes.append(ast.ClassDef(name='FrozenFusion', bases=[], keywords=[], body=methods, decorator_list=[]))
    namespace = dict(torch=torch, nn=nn, np=np, rearrange=rearrange, reduce=reduce, repeat=repeat)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), '<frozen-fusion>', 'exec'), namespace)
    return namespace


def _assert_state_equal(actual, expected):
    assert type(actual) is type(expected) or hasattr(actual, '__dict__') and hasattr(expected, '__dict__')
    if torch.is_tensor(actual):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            _assert_state_equal(actual[key], expected[key])
    elif isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected)
        for value, reference in zip(actual, expected):
            _assert_state_equal(value, reference)
    elif hasattr(actual, '__dict__'):
        _assert_state_equal(vars(actual), vars(expected))
    else:
        assert actual == expected


def _storage_input(n, offset, domain, export):
    values = torch.arange(n * 3, dtype=torch.float32).view(1, n, 3) + offset
    return dict(
        means=values, covariances=torch.eye(3).repeat(1, n, 1, 1),
        harmonics=torch.arange(n * 27, dtype=torch.float32).reshape(1, n, 3, 9),
        opacities=torch.ones((1, n)), features=torch.full((1, n, 4), float(offset)),
        clip_features_3d=torch.full((1, n, 4), .25) if domain != '2d' else None,
        coords=values, densities=torch.full((1, n, 1, 1), .5),
        weights=torch.ones((1, n, 1, 1)), extrinsics=torch.eye(4).repeat(1, n, 1, 1),
        depths=torch.ones((1, n)),
        codebooks_idx=torch.arange(n, dtype=torch.float32).view(1, n) if domain != '3d' else None,
        codebooks_weights=torch.full((1, n), .75) if domain != '3d' else None,
        scales=torch.ones((1, n, 3)) if export else None,
        rotations=torch.ones((1, n, 4)) if export else None,
    )


@pytest.mark.parametrize('domain', ('2d', '3d', 'ensemble'))
@pytest.mark.parametrize('export', (False, True))
def test_gaussian_storage_growth_replacement_and_codebooks_match_frozen(domain, export):
    original = _frozen()['EfficientGaussians']
    stores = [cls(initial_capacity=2, growth_factor=2., feat_dim=4,
                  device='cpu', domain=domain, export_ply=export)
              for cls in (EfficientGaussians, original)]
    for n, offset in ((3, 0), (4, 10)):
        for store in stores:
            store.append(**_storage_input(n, offset, domain, export))
        _assert_state_equal(stores[0], stores[1])
    # Exercise repeated replacement and the six-slot semantic-codebook ordering.
    mask = torch.tensor([True, False, False, False, False, False, True])
    for offset in range(7):
        for store in stores:
            store.append(**_storage_input(2, offset + 20, domain, export), mask=mask, fuse=True)
        _assert_state_equal(stores[0], stores[1])


class GeometryAdapter:
    def forward(self, extrinsics, intrinsics, rays, depths, opacities, features, shape, *, coords, **kwargs):
        dimensions = coords.shape[:-1]
        return SimpleNamespace(
            means=coords, covariances=torch.eye(3).expand(*dimensions, 3, 3),
            harmonics=coords.unsqueeze(-1).expand(*dimensions, 3, 9), opacities=opacities,
            scales=torch.ones_like(coords), rotations=torch.ones((*dimensions, 4)),
        )


def _owner(cls, domain):
    owner = cls()
    owner.device = 'cpu'
    owner.cfg = SimpleNamespace(openvocab_domain=domain, unet_3d=domain != '2d',
                                num_surfaces=1, load_depth=True, pcd_aug=False,
                                voxel_size=.05, vis_refine=False)
    owner.gaussian_adapter = GeometryAdapter()
    owner.to_gaussians = lambda features: torch.cat([features, features], dim=-1)
    owner.gru = lambda local, previous, *weights: .4 * local + .6 * previous
    owner.semantic_gru = lambda local, previous, *weights: .25 * local + .75 * previous
    owner.backbone_3d = lambda coords, features, *state: features.unsqueeze(0)
    return owner


def _fusion_inputs(domain):
    views, pixels, channels = 3, 4, 4
    features = torch.arange(views * pixels * channels, dtype=torch.float32).reshape(1, views, pixels, channels) / 50
    depth = torch.full((views, 1, 2, 2), 2.)
    depth[-1, :, :, 1] = 1.
    grid = torch.tensor([[0., 0.], [.5, 0.], [0., .5], [.5, .5]])
    coords = torch.cat([grid[None].expand(views, -1, -1) * depth.view(views, pixels, 1),
                        depth.view(views, pixels, 1)], dim=-1).reshape(1, views, pixels, 1, 1, 3)
    return dict(
        gaussians=features, clip_gaussians_3d=features * .5 if domain != '2d' else None,
        codebook_caches=torch.arange(views * pixels, dtype=torch.float32).reshape(1, views, pixels), coords=coords,
        densities=torch.full((1, views, pixels, 1, 1), .5),
        weight_emb=torch.ones((1, views, pixels, 1, 1)), depths=depth,
        extrinsics=torch.eye(4).repeat(1, views, 1, 1),
        intrinsics=torch.eye(3).repeat(1, views, 1, 1), image_shape=(2, 2),
    )


@pytest.mark.parametrize('domain', ('2d', 'ensemble'))
@pytest.mark.parametrize('use_gru', (False, True))
def test_fusion_and_occlusion_refinement_match_frozen(domain, use_gru):
    original = _frozen()
    native = _owner(GaussianFusion, domain)
    reference = _owner(original['FrozenFusion'], domain)
    states = [cls(initial_capacity=2, feat_dim=4, device='cpu', domain=domain)
              for cls in (EfficientGaussians, original['EfficientGaussians'])]
    inputs = _fusion_inputs(domain)
    for owner, state in zip((native, reference), states):
        owner.fuse_gaussians(**copy.deepcopy(inputs), globals=state, use_gru=use_gru, fore_fusion=True)
    assert int(states[0].valid.sum()) > 4
    _assert_state_equal(states[0], states[1])
    for owner, state in zip((native, reference), states):
        owner.refine_gaussians(
            state, torch.full((1, 1, 2, 2), 5.), torch.ones((1, 1, 4, 1, 1)),
            torch.eye(4).reshape(1, 1, 4, 4), torch.eye(3).reshape(1, 1, 3, 3), (2, 2),
        )
    _assert_state_equal(states[0], states[1])


def test_moved_storage_and_encoding_definitions_preserve_executable_ast():
    original = _frozen()
    source = subprocess.check_output(['git', 'show', f'422cb8e:{SOURCE}'], cwd=ROOT)
    previous = ast.parse(source)
    current = ast.parse((ROOT / 'pbsp/mapping/gaussians.py').read_text())
    old_class = next(node for node in previous.body if isinstance(node, ast.ClassDef) and node.name == 'EfficientGaussians')
    new_class = next(node for node in current.body if isinstance(node, ast.ClassDef))
    assert ast.dump(old_class, include_attributes=False) == ast.dump(new_class, include_attributes=False)
    value = torch.tensor([[[[.5, 1.]]]])
    torch.testing.assert_close(positional_encoding(value, 6), original['positional_encoding'](value, 6), rtol=0, atol=0)
