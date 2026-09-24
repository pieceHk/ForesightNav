import numpy as np

from tdonav_navigation.contracts import Action, Candidate, Verification
from tdonav_navigation.candidates import ActiveCandidateTracker, project_candidates
from tdonav_navigation.policy import GoalHierarchy, StopBoundary
from tdonav_navigation.recovery import TrappedDetector


class FlatNavmesh:
    def project(self, point):
        point = np.array(point, dtype=np.float32)
        point[1] = 0
        return point


def test_candidate_projection_and_continuity():
    candidates = [Candidate(np.array([1.0, 0.2, 1.0]), 0), Candidate(np.array([3.0, 0.2, 3.0]), 1)]
    projected = project_candidates(candidates, FlatNavmesh())
    tracker = ActiveCandidateTracker()
    assert tracker.refresh(projected).rank == 0
    refreshed = [Candidate(np.array([1.2, 0.0, 1.1]), 2), projected[1]]
    assert tracker.refresh(refreshed).rank == 2


def test_stop_and_goal_boundaries():
    failed = Verification(True, False, 100)
    confirmed = Verification(True, True, 100)
    assert StopBoundary.resolve(Action.STOP, failed, False) == Action.TURN_RIGHT
    assert StopBoundary.resolve(Action.MOVE_FORWARD, confirmed, False) == Action.STOP
    assert GoalHierarchy.choose(False, False, False) == GoalHierarchy.RECOVERY


def test_trapped_detector():
    detector = TrappedDetector(warmup_steps=3, window_steps=3, radius_m=0.05)
    assert not detector.update(np.zeros(3))
    assert not detector.update(np.array([0.01, 0, 0]))
    assert detector.update(np.array([0.02, 0, 0]))
