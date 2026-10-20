"""Image/text matching backends used by the navigation kernel."""

# Portions adapted from LagMemo, Copyright (c) Meta Platforms, Inc. and affiliates.
# Upstream MIT terms and provenance are retained in THIRD_PARTY_NOTICES.md.

import numpy as np
import torch
import cv2
import kornia as K
from PIL import Image


class GoalPerception:

    def disk_preprocess(self, goal_img):
        img_bgr = cv2.cvtColor(goal_img, cv2.COLOR_RGB2BGR)
        img_tensor = (
            K.image_to_tensor(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), keepdim=False)
            .float()
            .to(self.device)
            / 255.0
        )
        with torch.inference_mode():
            features = self.disk(img_tensor)[0]
        total_keypoints = len(features.keypoints)
        goal_data = {
            "image_bgr": img_bgr,
            "keypoints": features.keypoints,
            "descriptors": features.descriptors,
            "total_keypoints": total_keypoints,
        }
        self.goal_disk_data = goal_data
        return goal_data

    def lightglue_score(self, img: np.array):
        img_obs_bgr = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2BGR) if img.ndim == 3 else None
        if img_obs_bgr is None:
            raise ValueError("Observation image must be a 3-channel RGB image.")
        img_obs_tensor = (
            K.image_to_tensor(cv2.cvtColor(img_obs_bgr, cv2.COLOR_BGR2RGB), keepdim=False)
            .float()
            .to(self.device)
            / 255.0
        )
        with torch.inference_mode():
            features_obs = self.disk(img_obs_tensor)[0]
        (kps_obs, desc_obs) = (features_obs.keypoints, features_obs.descriptors)
        (h1, w1) = self.goal_disk_data["image_bgr"].shape[:2]
        (h2, w2) = img_obs_bgr.shape[:2]
        data_to_match = {
            "image0": {
                "keypoints": self.goal_disk_data["keypoints"][None],
                "descriptors": self.goal_disk_data["descriptors"][None],
                "image_size": torch.tensor([[w1, h1]]).to(self.device),
            },
            "image1": {
                "keypoints": kps_obs[None],
                "descriptors": desc_obs[None],
                "image_size": torch.tensor([[w2, h2]]).to(self.device),
            },
        }
        with torch.inference_mode():
            out = self.lightglue(data_to_match)
        idxs = out["matches"][0]
        mkpts0 = self.goal_disk_data["keypoints"][idxs[:, 0]].cpu().numpy()
        mkpts1 = kps_obs[idxs[:, 1]].cpu().numpy()
        match_results = {"image_obs_bgr": img_obs_bgr, "mkpts0": mkpts0, "mkpts1": mkpts1}
        self.save_glue_visualization(self.goal_disk_data["image_bgr"], img_obs_bgr, mkpts0, mkpts1)
        return (
            len(mkpts0),
            (
                len(mkpts0) / self.goal_disk_data["total_keypoints"] * 100
                if self.goal_disk_data["total_keypoints"] > 0
                else 0
            ),
        )

    @torch.no_grad()
    @torch.cuda.amp.autocast()
    def mobileclip_score(self, image, text, score_type="text"):
        if score_type == "text":
            text = self.clip_tokenizer([text])
            image = self.clip_preprocess(
                Image.fromarray(image.astype(np.uint8), mode="RGB")
            ).unsqueeze(0)
            with torch.no_grad(), torch.cuda.amp.autocast():
                image_features = self.clip_model.encode_image(image)
                text_features = self.clip_model.encode_text(text)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
            probs = image_features @ text_features.T
        elif score_type == "image":
            image1 = self.clip_preprocess(
                Image.fromarray(image.astype(np.uint8), mode="RGB")
            ).unsqueeze(0)
            image2 = self.clip_preprocess(
                Image.fromarray(text.astype(np.uint8), mode="RGB")
            ).unsqueeze(0)
            with torch.no_grad(), torch.cuda.amp.autocast():
                image1_features = self.clip_model.encode_image(image1)
                image2_features = self.clip_model.encode_image(image2)
                image1_features /= image1_features.norm(dim=-1, keepdim=True)
                image2_features /= image2_features.norm(dim=-1, keepdim=True)
            probs = image1_features @ image2_features.T
        else:
            raise ValueError("mobile clip score_type must be 'text' or 'image'.")
        return probs[0, 0].item()
