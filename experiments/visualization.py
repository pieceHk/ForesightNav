"""Map diagnostics written by the experiment recorder."""

from pathlib import Path
import os

import numpy as np
import torch


def write_fusion_debug_render(render):
    """Record the optional reconstruction-error diagnostic used by the network."""
    import os
    from PIL import Image

    count = len(os.listdir("./debug"))
    pixels = (render.squeeze().permute(1, 2, 0) * 255).byte()
    Image.fromarray(pixels.detach().cpu().numpy(), "RGB").save(f"debug/render_{count}.png")


def render_bev_cv2(
    globals_state,
    out_path,
    n_keyframes=None,
    n_raw_frames=None,
    size=420,
    margin=22,
    max_points=200000,
):
    """Draw the observed Gaussian map in the x-z plane.

    Blue points have CLIP features; grey points have none. Frame counters match
    the navigation recorder so that both views can be inspected together.
    """
    import cv2

    valid = globals_state.valid[0].bool()
    if not valid.any():
        return
    valid_indices = torch.nonzero(valid, as_tuple=False).flatten()
    n_total = int(valid_indices.numel())
    if max_points > 0 and n_total > max_points:
        sample = torch.linspace(
            0, n_total - 1, steps=max_points, device=valid_indices.device
        ).long()
        valid_indices = valid_indices[sample]
    means_src = globals_state.means
    idx_src = globals_state.clip_features_2d["idx"]
    means = means_src[0, valid_indices.to(means_src.device), :].detach().cpu().numpy()
    idx = idx_src[0, valid_indices.to(idx_src.device), :3]
    has = (idx != -1).any(dim=-1).detach().cpu().numpy()  # (N,) bool
    xz = means[:, [0, 2]]
    lo = xz.min(0)
    span = float((xz.max(0) - lo).max()) + 1e-6
    cv = np.full((size, size, 3), 255, np.uint8)
    px = ((xz[:, 0] - lo[0]) / span * (size - 2 * margin) + margin).astype(np.int32)
    py = ((xz[:, 1] - lo[1]) / span * (size - 2 * margin) + margin).astype(np.int32)
    py = size - 1 - py
    px = np.clip(px, 0, size - 1)
    py = np.clip(py, 0, size - 1)
    g = ~has
    cv[py[g], px[g]] = (205, 205, 205)  # grey: no-CLIP gaussian
    cv[py[has], px[has]] = (205, 130, 50)  # BGR: CLIP gaussian (blue-ish)
    n_rendered = int(valid_indices.numel())
    cov = 100.0 * has.sum() / max(n_rendered, 1)
    cv2.rectangle(cv, (0, 0), (size, 26), (255, 255, 255), -1)
    label = f"ES online map  N={n_total} sampled={n_rendered} CLIP={cov:.0f}%"
    if n_keyframes is not None and n_raw_frames is not None:
        label = f"kf={n_keyframes} raw_frame={n_raw_frames}  " + label
    cv2.putText(cv, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.imwrite(out_path, cv)


def render_empty_bev_cv2(out_path, n_keyframes, n_raw_frames, size=720):
    """Write a diagnostic artifact when an episode ends before ES warmup."""
    import cv2

    cv = np.full((size, size, 3), 255, np.uint8)
    label = f"ES online map unavailable: kf={n_keyframes} raw_frame={n_raw_frames}"
    cv2.putText(cv, label, (18, size // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 180), 1)
    cv2.imwrite(out_path, cv)


class MappingArtifacts:
    """Write optional map snapshots and one final BEV per episode."""

    def __init__(self, *, final_directory=None, snapshot_directory=None, max_points=200000):
        self.final_directory = Path(final_directory) if final_directory else None
        self.snapshot_directory = Path(snapshot_directory) if snapshot_directory else None
        self.max_points = max_points

    def on_rebuild(self, *, scene_id, episode_id, globals_state, n_keyframes, n_raw_frames):
        if self.snapshot_directory is None:
            return
        folder = self.snapshot_directory / f"{scene_id}_{episode_id}"
        folder.mkdir(parents=True, exist_ok=True)
        output = folder / f"bev_{n_raw_frames:05d}_kf{n_keyframes:03d}.png"
        render_bev_cv2(
            globals_state,
            str(output),
            n_keyframes=n_keyframes,
            n_raw_frames=n_raw_frames,
            max_points=self.max_points,
        )

    def on_finish(self, *, scene_id, episode_id, globals_state, n_keyframes, n_raw_frames):
        if self.final_directory is None:
            return []
        folder = self.final_directory / scene_id / episode_id / "es_final_bev"
        folder.mkdir(parents=True, exist_ok=True)
        output = folder / "final_bev.png"
        if globals_state is None:
            render_empty_bev_cv2(str(output), n_keyframes, n_raw_frames)
        else:
            render_bev_cv2(
                globals_state,
                str(output),
                n_keyframes=n_keyframes,
                n_raw_frames=n_raw_frames,
                size=720,
                max_points=self.max_points,
            )
        return [str(output)] if output.is_file() else []


# Portions adapted from LagMemo, Copyright (c) Meta Platforms, Inc. and affiliates.
# Upstream MIT terms and provenance are retained in THIRD_PARTY_NOTICES.md.


class NavigationVisualization:

    def set_episode_visualization_dir(self, scene_id, episode_id):
        if (
            self.continuous_print_images or self.episode_final_snapshot
        ) and self.imagenav_visualizer is not None:
            relative_dir = os.path.join(str(scene_id), str(episode_id), "snapshot")
            self.imagenav_visualizer.set_vis_dir(relative_dir)
            shared_vis_dir = self.imagenav_visualizer.vis_dir
            if self.planner is not None:
                self.planner.vis_dir = shared_vis_dir
            if self.matching is not None:
                self.matching.vis_dir = shared_vis_dir
            self.lightglue_vis_dir = shared_vis_dir

    def save_glue_visualization(self, goal_img_bgr, obs_img_bgr, mkpts0, mkpts1):
        """将匹配结果可视化并保存到文件。"""
        import cv2
        import matplotlib.pyplot as plt

        (h1, w1) = goal_img_bgr.shape[:2]
        (h2, w2) = obs_img_bgr.shape[:2]
        vis = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
        vis[:h1, :w1] = goal_img_bgr
        vis[:h2, w1 : w1 + w2] = obs_img_bgr
        mkpts0_int = np.round(mkpts0).astype(int)
        mkpts1_int = np.round(mkpts1).astype(int)
        for p1, p2 in zip(mkpts0_int, mkpts1_int):
            pt2_shifted = (p2[0] + w1, p2[1])
            cv2.line(vis, tuple(p1), tuple(pt2_shifted), (0, 255, 0), 1)
        fig = plt.figure(figsize=(16, 8))
        plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        plt.title(f"Matches: {len(mkpts0)} points")
        plt.axis("off")
        plt.tight_layout()
        save_path = f"{self.lightglue_vis_dir}/"
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        plt.savefig(
            save_path + str(self.sub_task_timesteps[0][self.current_task_idx]) + ".png",
            dpi=150,
            bbox_inches="tight",
            pad_inches=0.1,
        )
        plt.close(fig)

    def set_lightglue_vis_dir(self, episode_id: str):
        self.lightglue_vis_dir = f"{self.glue_vis_path}/{episode_id}/"
        if not os.path.exists(self.lightglue_vis_dir):
            os.makedirs(self.lightglue_vis_dir, exist_ok=True)

    def _write_navigation_snapshot(self, info, action):
        should_write_snapshot = self.continuous_print_images or (
            self.episode_final_snapshot and action == self.actions.STOP
        )
        if should_write_snapshot and self.imagenav_visualizer is not None:
            self.imagenav_visualizer.visualize(**info)
            if self.episode_final_snapshot and (not self.continuous_print_images):
                snapshot_paths = [
                    os.path.join(self.imagenav_visualizer.vis_dir, name)
                    for name in os.listdir(self.imagenav_visualizer.vis_dir)
                    if name.startswith("snapshot_") and name.endswith(".png")
                ]
                if snapshot_paths:
                    os.replace(
                        max(snapshot_paths, key=os.path.getmtime),
                        os.path.join(self.imagenav_visualizer.vis_dir, "final.png"),
                    )
