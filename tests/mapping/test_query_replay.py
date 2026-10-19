"""Replay outputs recorded from the frozen spatial query implementation."""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pbsp.query.engine import select_spatial_candidates
from adapters.embodied_splat_query import legacy_query_categories as query_categories, legacy_query_category as query_category
from runtime.config import load_experiment


ROOT = Path(__file__).resolve().parents[2]
BASELINE = json.loads((ROOT / "tests/fixtures/mapping_query_baseline.json").read_text())
SETTINGS = load_experiment(ROOT / "configs/experiments/hm3d_4mon.yaml").method.query


@pytest.mark.parametrize("case", BASELINE["cases"], ids=lambda case: case["name"])
def test_spatial_selection_replays_frozen_outputs(case):
    selected, diagnostics = select_spatial_candidates(
        np.asarray(case["positions"], dtype=np.float64).reshape(-1, 3),
        torch.tensor(case["scores"], dtype=torch.float32), case["top_k"],
        settings=replace(SETTINGS, **case.get("settings_overrides", {})),
    )
    assert selected == case["expected_indices"]
    assert set(diagnostics) == set(case["expected_diagnostics"])
    for key, expected in case["expected_diagnostics"].items():
        actual = diagnostics[key]
        if isinstance(expected, (float, list)):
            assert actual == pytest.approx(expected, rel=1e-6, abs=1e-7), key
        else:
            assert actual == expected, key


def test_query_settings_ignore_inherited_environment(monkeypatch):
    case = next(case for case in BASELINE["cases"] if case["name"] == "support_reranking")
    positions = np.asarray(case["positions"], dtype=np.float64)
    scores = torch.tensor(case["scores"], dtype=torch.float32)
    expected = select_spatial_candidates(positions, scores, case["top_k"], settings=SETTINGS)
    monkeypatch.setenv("ES_QUERY_MIN_SIMILARITY", "999")
    monkeypatch.setenv("ES_QUERY_SUPPORT_RERANK_WEIGHT", "999")
    assert select_spatial_candidates(positions, scores, case["top_k"], settings=SETTINGS) == expected


class TextClassifier:
    def __init__(self):
        self.calls = []

    def get_text_classifier(self, prompts, device):
        self.calls.append(tuple(prompts))
        vectors = {"a chair in a photo.": [1.0, 0.0], "a bed in a photo.": [0.0, 1.0]}
        return torch.tensor([vectors[prompt] for prompt in prompts], device=device)


def _map():
    indices = torch.full((1, 6, 3), -1, dtype=torch.long)
    indices[0, :, 0] = torch.tensor([0, 0, 0, 1, 1, 1])
    weights = torch.zeros((1, 6, 3))
    weights[0, :, 0] = 1.0
    return SimpleNamespace(
        valid=torch.ones((1, 6), dtype=torch.bool),
        means=torch.tensor([[[0.0,0.0,0.0], [.1,0.0,.1], [-.1,0.0,-.1],
                             [2.0,0.0,0.0], [2.1,0.0,.1], [1.9,0.0,-.1]]]),
        clip_features_2d={"feat": [torch.tensor([[1.0,0.0], [0.0,1.0]])], "idx": indices, "weight": weights},
    )


def test_batch_query_deduplicates_text_and_preserves_versioned_records():
    classifier = TextClassifier()
    encoder = SimpleNamespace(clip=classifier)
    state = _map()
    positions, records = query_categories(
        encoder, state, ["chair", "bed", "chair"], 10, "cpu",
        settings=SETTINGS, return_records=True, map_revision=13,
    )
    assert classifier.calls == [("a chair in a photo.",), ("a bed in a photo.",)]
    assert set(positions) == set(records) == {"chair", "bed"}
    for category in positions:
        assert len(positions[category]) == 1
        assert records[category][0]["position"] == positions[category][0]
        assert records[category][0]["map_revision"] == 13
        assert records[category][0]["support_count"] == 3
        single = query_category(encoder, state, category, 10, "cpu", settings=SETTINGS,
                                return_records=True, map_revision=13)
        assert single == (positions[category], records[category])


def test_empty_map_keeps_the_response_schema():
    state = _map()
    state.valid.zero_()
    assert query_categories(None, state, ["chair", "bed"], 10, "cpu", settings=SETTINGS,
                            return_records=True) == ({"chair": [], "bed": []}, {"chair": [], "bed": []})
