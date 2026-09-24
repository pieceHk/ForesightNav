import numpy as np
import sys
from pathlib import Path

import torch

from tdonav_mapping.coordinates import habitat_pose_to_c2w, mapper_world_to_habitat
from tdonav_mapping.keyframes import KeyframeAdmission, frame_change_ratio
from tdonav_mapping.query import (
    ScoredPoint,
    candidate_records,
    greedy_nms,
    spatially_supported,
)

SERVER_ROOT = Path(__file__).resolve().parents[1] / "runtime" / "server"
sys.path.insert(0, str(SERVER_ROOT))
from online_es_mapper import _select_spatial_candidates


def test_coordinate_round_trip_translation():
    pose = np.array([0.0, 0.0, 0.0, 1.0, 1.5, 2.0, -3.0], dtype=np.float32)
    c2w = habitat_pose_to_c2w(pose)
    assert np.allclose(c2w[:3, 3], [1.5, -2.0, 3.0])
    assert np.allclose(mapper_world_to_habitat(c2w[:3, 3]), pose[4:])


def test_keyframe_admission_uses_last_accepted_frame():
    admission = KeyframeAdmission()
    black = np.zeros((96, 128, 3), dtype=np.uint8)
    near_black = np.full_like(black, 10)
    white = np.full_like(black, 255)
    assert admission.accept(black)[0]
    assert not admission.accept(near_black)[0]
    assert admission.accept(white)[0]
    assert frame_change_ratio(white, black) == 1.0


def test_query_support_then_nms():
    positions = np.array([[0, 0, 0], [0.1, 0, 0.1], [2, 0, 2], [2.1, 0, 2.1]])
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    supported = spatially_supported(positions, scores, 0.18, 0.35, 2)
    kept = greedy_nms(supported, radius_m=1.0, top_k=10)
    assert [round(point.score, 1) for point in kept] == [0.9, 0.7]


def test_candidate_records_are_json_safe_and_stable():
    records = candidate_records(
        [[1.0, 0.0, 2.0]], [0.42], [7], [0.43], [3], map_revision=20
    )
    assert records == [
        {
            "position": [1.0, 0.0, 2.0],
            "raw_rank": 3,
            "semantic_score": 0.42,
            "support_count": 7,
            "support_score": 0.43,
            "map_revision": 20,
        }
    ]


def test_candidate_records_reject_nonfinite_values():
    with np.testing.assert_raises(ValueError):
        candidate_records([[np.nan, 0.0, 0.0]], [1.0], [1], [1.0], [0], 1)


def test_empty_spatial_selection_keeps_candidate_record_schema(monkeypatch):
    monkeypatch.setenv("ES_QUERY_MIN_SIMILARITY", "0.18")
    selected, diagnostics = _select_spatial_candidates(
        np.zeros((2, 3), dtype=np.float32),
        torch.tensor([0.01, 0.02]),
        top_k=10,
    )

    assert selected == []
    assert diagnostics["selected"] == 0
    assert diagnostics["selected_supports"] == []
    assert diagnostics["selected_support_scores"] == []
    assert diagnostics["selected_scores"] == []
    assert diagnostics["selected_combined_scores"] == []
    assert diagnostics["selected_raw_ranks"] == []
