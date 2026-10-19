"""Exercise semantic codebook updates with no geometry model or Gaussian tensors."""

import gc
from types import SimpleNamespace
import weakref

import pytest
import torch

from adapters.embodied_splat_semantics import EmbodiedSplatSemanticInjector


class Resources:
    def __init__(self, domain):
        self.cfg = SimpleNamespace(
            sam_model_type="fastsam", clip_model="lseg", openvocab_domain=domain
        )
        self.device = "cpu"


@pytest.mark.parametrize("domain", ("2d", "3d", "ensemble"))
def test_injector_only_needs_semantic_state_and_keeps_codebook_ids(domain, tmp_path):
    masks = [
        torch.tensor([[True, False], [True, False]]),
        torch.tensor([[False, True], [False, True]]),
    ]
    cache = tmp_path / "features.pt"
    features = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    torch.save(
        {"fastsam": [[{"segmentation": mask} for mask in masks]], "clip": [features]},
        cache,
    )
    resources = Resources(domain)
    injector = EmbodiedSplatSemanticInjector(resources)
    state = SimpleNamespace(clip_features_2d={"feat": [], "img_stamp": [0]})
    for offset in (0, 2):
        result = injector.inject(
            torch.zeros((1, 3, 2, 2)), [(str(cache),)], (2, 2), state
        )
        assert result.state is state
        if domain in {"2d", "ensemble"}:
            torch.testing.assert_close(
                result.pixel_indices,
                torch.tensor(
                    [[[offset, offset + 1], [offset, offset + 1]]], dtype=torch.float32
                ),
                rtol=0,
                atol=0,
            )
            assert state.clip_features_2d["img_stamp"][-1] == offset + 2
            torch.testing.assert_close(
                state.clip_features_2d["feat"][-1], features, rtol=0, atol=0
            )
        else:
            assert result.pixel_indices is None
        if domain in {"3d", "ensemble"}:
            expected = (
                torch.nn.functional.normalize(features, dim=1)
                .T[:, None, :]
                .expand(2, 2, 2)
                .unsqueeze(0)
            )
            torch.testing.assert_close(result.dense_features, expected, rtol=0, atol=0)
        else:
            assert result.dense_features is None


def test_semantic_adapter_does_not_keep_model_resources_alive():
    resources = Resources("2d")
    reference = weakref.ref(resources)
    injector = EmbodiedSplatSemanticInjector(resources)
    del resources
    gc.collect()
    assert reference() is None
    with pytest.raises(RuntimeError, match="released"):
        injector.inject(None, None, None, None)
