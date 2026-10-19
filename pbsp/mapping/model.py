"""Frame preparation, checkpoint loading and incremental model inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ModelInputSpec:
    image_height: int
    image_width: int
    near: float
    far: float

    def __post_init__(self):
        for value in (self.image_height, self.image_width):
            if type(value) is not int or value < 2 or value % 2:
                raise ValueError("Model image dimensions must be positive even integers")
        if (type(self.near) not in (int, float) or type(self.far) not in (int, float)
                or not math.isfinite(self.near) or not math.isfinite(self.far)
                or not 0 < self.near < self.far):
            raise ValueError("Model bounds require finite 0 < near < far")


def preprocess_frame(rgb, depth, c2w, intrinsic, *, input_spec: ModelInputSpec, device="cpu"):
    """Resize RGB-D to the model resolution, preserving the supplied camera matrices."""
    shape = (input_spec.image_height, input_spec.image_width)
    image = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    image = F.interpolate(image.unsqueeze(0), shape, mode="bilinear", align_corners=False).squeeze(0)
    depth_tensor = torch.from_numpy(depth).unsqueeze(0).float()
    depth_tensor = F.interpolate(depth_tensor.unsqueeze(0), shape, mode="bilinear", align_corners=False).squeeze(0)
    return {
        "image": image.to(device), "depth": depth_tensor.to(device),
        "extrinsic": torch.from_numpy(c2w).to(device),
        "intrinsic": torch.from_numpy(np.asarray(intrinsic).copy()).to(device),
    }


class OnlineMappingModel:
    """Own the encoder/decoder and preserve the inference call used by full PBSP."""

    def __init__(self, encoder, decoder, input_spec: ModelInputSpec, work_directory: Path):
        self.encoder = encoder
        self.decoder = decoder
        self.input_spec = input_spec
        self.work_directory = Path(work_directory)

    def load_checkpoint(self, checkpoint: Path) -> dict:
        checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = checkpoint.get("state_dict", checkpoint)
        results = {}
        for name, module in (("encoder", self.encoder), ("decoder", self.decoder)):
            prefix = name + "."
            parameters = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
            missing, unexpected = module.load_state_dict(parameters, strict=False)
            results[name] = {"missing_keys": list(missing), "unexpected_keys": list(unexpected)}
        return results

    def build_context(self, frames, *, device):
        n = len(frames)
        spec = self.input_spec
        values = {key: torch.stack([frame[key] for frame in frames]).unsqueeze(0).to(device)
                  for key in ("image", "depth", "extrinsic", "intrinsic")}
        depths = values["depth"]
        half_depth = F.interpolate(
            depths.view(n, 1, spec.image_height, spec.image_width),
            size=(spec.image_height // 2, spec.image_width // 2),
            mode="bilinear", align_corners=False,
        ).view(1, n, 1, spec.image_height // 2, spec.image_width // 2).to(device)
        return {
            "image": values["image"], "depth": depths, "depth_s0": half_depth,
            "extrinsics": values["extrinsic"], "intrinsics": values["intrinsic"],
            "near": torch.full((1, n), spec.near, dtype=torch.float32).to(device),
            "far": torch.full((1, n), spec.far, dtype=torch.float32).to(device),
            "cache": [None] * n,
        }

    def update(self, context, *, scene_key, resume_state):
        with torch.no_grad():
            return self.encoder(
                [context], global_step=0, is_testing=True, export_ply=False,
                decoder=self.decoder, source=None, dense=False, scene=scene_key,
                target_indices=None, output_path=self.work_directory, target=None,
                test_bev=False, path=self.work_directory, resume_state=resume_state,
                online_incremental=True, return_state=True,
            )

    def close(self):
        self.encoder = None
        self.decoder = None
