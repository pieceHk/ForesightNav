"""Visual confirmation history and task-boundary reset for the navigation agent."""

# Portions adapted from LagMemo, Copyright (c) Meta Platforms, Inc. and affiliates.
# Upstream MIT terms and provenance are retained in THIRD_PARTY_NOTICES.md.

import numpy as np
from .evidence import select_independent_evidence


def update_multiview_verification(
    history,
    step,
    triggered,
    pose_center,
    clip_score,
    seem_score,
    window_steps=6,
    min_confirmations=2,
    pose_radius=1.5,
    min_peak_seem=1.15,
    observer_position=None,
    observer_yaw=None,
    min_observer_translation=0.0,
    min_observer_yaw_deg=0.0,
):
    history = [item for item in history if step - item["step"] <= window_steps]
    center = np.asarray(pose_center, dtype=np.float32)
    added = triggered and center.shape == (2,) and np.isfinite(center).all()
    if added:
        history.append(
            {
                "step": int(step),
                "pose": center.tolist(),
                "clip_score": float(clip_score),
                "seem_score": float(seem_score),
                "observer_position": np.asarray(
                    observer_position if observer_position is not None else [np.nan, np.nan],
                    dtype=np.float32,
                ).tolist(),
                "observer_yaw": float(observer_yaw) if observer_yaw is not None else float("nan"),
            }
        )
    confirmed = False
    supporting = []
    independent = []
    if added and history:
        latest = np.asarray(history[-1]["pose"], dtype=np.float32)
        supporting = [
            item
            for item in history
            if np.linalg.norm(np.asarray(item["pose"], dtype=np.float32) - latest) <= pose_radius
        ]
        independent = select_independent_evidence(
            supporting, min_observer_translation, min_observer_yaw_deg
        )
        confirmed = (
            len(independent) >= min_confirmations
            and max((item["seem_score"] for item in independent)) >= min_peak_seem
        )
    return (history, confirmed, supporting, independent)


class VerificationState:
    """Clear visual confirmation state at task and episode boundaries."""


    def _clear_seem_verification_state(self):
        """Clear verification state at episode and subtask boundaries."""
        self.stop_by_seem = False
        self.seem_goal_pose = None
        self.seem_semantic = None
        self.goal_vis_stage = 0
        self.verification_history = []
        self._last_verification_triggered = False
        self._last_clip_score = 0.0
        self._last_seem_score = 0.0

    def _clear_candidate_verification_evidence(self):
        self.verification_history = []
        self._last_verification_triggered = False
        self._last_clip_score = 0.0
        self._last_seem_score = 0.0
        self.seem_goal_pose = None
        self.seem_semantic = None
