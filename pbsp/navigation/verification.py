"""Visual confirmation history and task-boundary reset for the navigation agent."""

# Portions adapted from LagMemo, Copyright (c) Meta Platforms, Inc. and affiliates.
# Upstream MIT terms and provenance are retained in THIRD_PARTY_NOTICES.md.

import numpy as np
from .types import NavigationObservation
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


class GoalVerifier(VerificationState):

    def lagmemo_score(self, obs: NavigationObservation, detic_type: str = "interest"):
        """Calculate the lagmemo score for the current observation and set flags accordingly."""
        import cv2

        img_thresh = 200
        txt_clip_thresh = 0.3
        txt_seem_thresh = 1.35
        if detic_type == "interest":
            txt_clip_thresh = 0.23
            txt_seem_thresh = 1.1
        elif detic_type == "common":
            txt_clip_thresh = 0.23
            txt_seem_thresh = 1.1
        else:
            print(f"Unknown detic type: {detic_type}. Using default thresholds.")
        match_score = 0
        self._last_verification_triggered = False
        self._last_clip_score = 0.0
        self._last_seem_score = 0.0
        task_type = obs.task_observations["tasks"][self.current_task_idx]["type"]
        image_input = obs.rgb
        img_ref = np.asarray(obs.task_observations["tasks"][self.current_task_idx]["image"])
        if task_type == "imagenav":
            if self.goal_disk_data is None:
                self.goal_disk_data = self.disk_preprocess(img_ref)
            (match_score, _) = self.lightglue_score(obs.rgb)
            if match_score > img_thresh:
                self.stop_by_seem = True
                self.goal_vis_stage = 2
                reference = obs.task_observations["tasks"][self.current_task_idx]["category"]
                (_, goal_mask, image_output) = self.seem_model.img2txt_seem_seg(
                    image_input, reference
                )
                if np.sum(goal_mask) != 0:
                    goal_mask_mirrored = 1 - goal_mask.astype(np.uint8)
                    goal_mask_mirrored = goal_mask_mirrored.astype(np.uint8)
                    kernel = np.ones((5, 5), np.uint8)
                    goal_mask_mirrored = cv2.dilate(goal_mask_mirrored, kernel, iterations=2)
                    goal_mask = 1 - goal_mask_mirrored
                    goal_mask = goal_mask.astype(np.uint8)
                else:
                    goal_mask = None
                (_, seem_mask, seem_semantic_image) = self.seem_model.img2img_seem_seg(
                    image_input, img_ref, goal_mask
                )
            else:
                seem_mask = np.zeros(2)
                seem_semantic_image = None
        elif task_type in ["objectnav", "languagenav"]:
            if task_type == "objectnav":
                reference = obs.task_observations["tasks"][self.current_task_idx]["category"]
            elif task_type == "languagenav":
                reference = obs.task_observations["tasks"][self.current_task_idx]["description"]
            clip_score = self.mobileclip_score(obs.rgb, reference, "text")
            (seem_score, seem_mask, seem_semantic_image) = self.seem_model.img2txt_seem_seg(
                image_input, reference
            )
            match_score = seem_score
            triggered = clip_score > txt_clip_thresh or seem_score > txt_seem_thresh
            self._last_verification_triggered = bool(triggered)
            self._last_clip_score = float(clip_score)
            self._last_seem_score = float(seem_score)
            if triggered and (not self.two_stage_verification):
                self.stop_by_seem = True
                self.goal_vis_stage = 2
            if seem_mask is None:
                seem_mask = np.zeros(2)
                seem_semantic_image = None
        return (match_score, seem_mask, seem_semantic_image)
