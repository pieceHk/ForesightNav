import numpy as np

from tdonav_mapping.coordinates import habitat_pose_to_c2w, mapper_world_to_habitat
from tdonav_mapping.keyframes import KeyframeAdmission, frame_change_ratio
from tdonav_mapping.query import ScoredPoint, greedy_nms, spatially_supported


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
