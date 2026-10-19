"""Use retrieval without a geometry encoder, Gaussian store or navigation policy."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from pbsp.query.service import SemanticQuery
from pbsp.query.types import FeatureSpace, SemanticIndex
from runtime.config import load_experiment


ROOT = Path(__file__).resolve().parents[2]
SPACE = FeatureSpace("test-clip", 2)


class TextEncoder:
    def __init__(self, space=SPACE, reverse=False):
        self.feature_space = space
        self.reverse = reverse
        self.calls = []

    def encode(self, prompts, *, device):
        self.calls.append(tuple(prompts))
        vectors = {"a chair in a photo.": [1.0, 0.0], "a bed in a photo.": [0.0, 1.0]}
        values = [
            vectors[prompt][::-1] if self.reverse else vectors[prompt]
            for prompt in prompts
        ]
        return torch.tensor(values, device=device)


def index():
    indices = torch.zeros((6, 3), dtype=torch.long)
    indices[3:, 0] = 1
    weights = torch.zeros((6, 3, 1))
    weights[:, 0] = 1
    return SemanticIndex(
        np.array(
            [
                [0, 0, 0],
                [0.1, 0, 0.1],
                [-0.1, 0, -0.1],
                [2, 0, -3],
                [2.1, 0, -3.1],
                [1.9, 0, -2.9],
            ],
            dtype=np.float32,
        ),
        torch.eye(2),
        indices,
        weights,
        SPACE,
    )


def query(text):
    return SemanticQuery(
        settings=load_experiment(
            ROOT / "configs/experiments/hm3d_4mon.yaml"
        ).method.query,
        text_encoder=text,
        read_index=lambda state: state,
        device="cpu",
    )


def test_portable_index_deduplicates_queries_and_preserves_world_positions_and_revision():
    text = TextEncoder()
    positions, records = query(text).query_many(
        index(), ["chair", "bed", "chair"], map_revision=23, return_records=True
    )
    assert text.calls == [("a chair in a photo.",), ("a bed in a photo.",)]
    assert positions == {"chair": [[0.0, 0.0, 0.0]], "bed": [[2.0, 0.0, -3.0]]}
    for category in positions:
        assert records[category][0]["position"] == positions[category][0]
        assert records[category][0]["map_revision"] == 23
        assert records[category][0]["support_count"] == 3


def test_text_backend_can_be_replaced_without_changing_map_state():
    state = index()
    original = state.codebook.clone()
    assert query(TextEncoder(reverse=True)).query(state, "chair") == [[2.0, 0.0, -3.0]]
    torch.testing.assert_close(state.codebook, original, rtol=0, atol=0)


def test_feature_space_mismatch_fails_before_text_inference():
    text = TextEncoder(FeatureSpace("different-checkpoint", 2))
    with pytest.raises(ValueError, match="feature spaces"):
        query(text).query(index(), "chair")
    assert text.calls == []


def test_index_rejects_incompatible_dimensions():
    with pytest.raises(ValueError, match="dimension"):
        replace(index(), feature_space=FeatureSpace("test-clip", 3))
    with pytest.raises(ValueError, match="align"):
        replace(index(), weights=torch.zeros((5, 3, 1)))


def test_empty_index_and_close_do_not_call_text_models():
    text = TextEncoder()
    service = query(text)
    assert service.query_many(None, ["chair"], return_records=True) == (
        {"chair": []},
        {"chair": []},
    )
    assert text.calls == []
    service.close()
    assert service.text_encoder is None and service.read_index is None
    with pytest.raises(RuntimeError, match="closed"):
        service.query(index(), "chair")
