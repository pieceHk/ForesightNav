"""Online RGB-D encoding and incremental map updates derived from EmbodiedSplat.

Network construction, checkpoints and resource locations belong to the adapter.
Upstream attribution and license scope are recorded in THIRD_PARTY_NOTICES.md.
"""

from typing import Optional
import time

import numpy as np
import torch
from torch import Tensor, nn
from einops import rearrange

from pbsp.semantics.types import SemanticInjector
from .fusion import GaussianFusion
from .gaussians import EfficientGaussians
from .geometry import calculate_distance_matrix, sample_image_grid


def tensor_B_to_bM(tensor_BS, batch_size, num_views):
    """Unpacks a flattened tensor of tupled elements (BS) into bMS. Tuple size
    is M."""
    tensor_bMS = tensor_BS.view([batch_size, num_views] + list(tensor_BS.shape[1:]))
    return tensor_bMS


def tensor_bM_to_B(tensor_bMS):
    """Packs an inflated tensor of tupled elements (bMS) into BS. Tuple size
    is M."""
    num_views = tensor_bMS.shape[1]
    num_batches = tensor_bMS.shape[0]
    tensor_BS = tensor_bMS.view([num_views * num_batches] + list(tensor_bMS.shape[2:]))
    return tensor_BS


class OnlineEncoder(GaussianFusion, nn.Module):
    """Incremental inference with network layers supplied by the model factory."""

    def __init__(self, cfg, *, profile_sink=None, semantic_injector: Optional[SemanticInjector] = None):
        super().__init__()
        self.cfg = cfg
        self._profile_sink = profile_sink
        self.semantic_injector = semantic_injector

    def _record_profile(self, record):
        if self._profile_sink is not None:
            self._profile_sink(record)

    def map_pdf_to_opacity(self, pdf: Tensor, global_step: int) -> Tensor:
        cfg = self.cfg.opacity_mapping
        x = cfg.initial + min(global_step / cfg.warm_up, 1) * (cfg.final - cfg.initial)
        exponent = 2**x
        return 0.5 * (1 - (1 - pdf) ** exponent + pdf ** (1 / exponent))

    def compute_matching_feats(
        self, cur_image, src_image, unbatched_matching_encoder_forward=False
    ):
        """Encode reference B3HW and source BM3HW images into matching features.

        The unbatched path avoids the batch-dependent numerical variation of
        the reference matching encoder during evaluation. Results have shapes
        BChw and BMChw.
        """
        if unbatched_matching_encoder_forward:
            all_frames_bm3hw = torch.cat([cur_image.unsqueeze(1), src_image], dim=1)
            (batch_size, num_views) = all_frames_bm3hw.shape[:2]
            all_frames_B3hw = tensor_bM_to_B(all_frames_bm3hw)
            matching_feats = [self.matching_net(f) for f in all_frames_B3hw.split(1, dim=0)]
            matching_feats = torch.cat(matching_feats, dim=0)
            matching_feats = tensor_B_to_bM(
                matching_feats, batch_size=batch_size, num_views=num_views
            )
        else:
            matching_feats = self.tensor_formatter(
                torch.cat([cur_image.unsqueeze(1), src_image], dim=1), apply_func=self.matching_net
            )
        matching_cur_feats = matching_feats[:, 0]
        matching_src_feats = matching_feats[:, 1:].contiguous()
        return (matching_cur_feats, matching_src_feats)

    def warmup_construct(
        self, context, target, decoder, is_testing=False, export_ply=False, test_bev=False
    ):
        """Encode the first view group and fuse its geometric and semantic state."""
        _t_wu = time.perf_counter()
        device = context["image"].device
        (b, n_views, _, h, w) = context["image"].shape
        results = {}
        num_context_views = self.cfg.num_views
        warmup_n_views = self.cfg.num_views
        context["image_shape"] = (h, w)
        self.cfg.gaussians_per_pixel = 1
        context_intrinsics = context["intrinsics"].clone()
        context_intrinsics[:, :, 0] *= w // 4
        context_intrinsics[:, :, 1] *= h // 4
        globals = EfficientGaussians(
            initial_capacity=n_views * 10000,
            growth_factor=1.5,
            device=context["image"].device,
            testing=is_testing,
            export_ply=export_ply,
            domain=self.cfg.openvocab_domain,
        )
        cur_indices = torch.arange(warmup_n_views, device=context["image"].device)
        cur_intrinsics = context_intrinsics.gather(
            dim=1, index=cur_indices.view(1, -1, 1, 1).repeat(b, 1, 3, 3)
        )
        cur_extrinsics = context["extrinsics"].gather(
            dim=1, index=cur_indices.view(1, -1, 1, 1).repeat(b, 1, 4, 4)
        )
        cur_image = (
            context["image"]
            .gather(dim=1, index=cur_indices.view(1, -1, 1, 1, 1).repeat(b, 1, 3, h, w))
            .view(-1, 3, h, w)
        )
        cur_cache = [context["cache"][i] for i in cur_indices]
        cur_feats = self.backbone(cur_image)
        resized = 0
        s = -1 if not self.cfg.low_res else 0
        full_indices = torch.arange(warmup_n_views, device=context["image"].device)[None].repeat(
            warmup_n_views, 1
        )
        src_indices = (
            full_indices[~(full_indices == cur_indices[:, None])]
            .view(1, warmup_n_views, warmup_n_views - 1)
            .repeat(b, 1, 1)
        )
        bs = self.cfg.max_batch
        depth_outputs_all = {f"depth_pred_s{s}_b1hw": [], "densities": []}
        for batch in range(
            min(np.ceil(warmup_n_views / bs).astype(int), self.cfg.max_batch_length)
        ):
            semantic = self.semantic_injector.inject(
                cur_image[batch * bs : (batch + 1) * bs],
                cur_cache[batch * bs : (batch + 1) * bs], (h, w), globals,
            )
            clip_img_feats, codebook_caches, globals = semantic.values()
            del semantic
            if self.cfg.openvocab_domain in ["ensemble", "3d"]:
                clip_gaussian_feats = torch.nn.functional.interpolate(
                    clip_img_feats, size=(h // 2, w // 2), mode="nearest"
                )
            else:
                clip_gaussian_feats = None
            if self.cfg.openvocab_domain in ["ensemble", "2d"]:
                cur_codebook_caches = torch.nn.functional.interpolate(
                    codebook_caches.unsqueeze(1), size=(h // 2, w // 2), mode="nearest"
                ).squeeze(1)
            else:
                cur_codebook_caches = None
            warmup_n_views_now = min(bs, warmup_n_views - batch * bs)
            matching_cur_feats = cur_feats[1][batch * bs : (batch + 1) * bs]
            dim = matching_cur_feats.shape[-3]
            matching_src_feats = (
                rearrange(cur_feats[1], "(b v) c h w -> b v c h w", b=b, v=warmup_n_views)[:, None]
                .repeat(1, warmup_n_views_now, 1, 1, 1, 1)
                .gather(
                    dim=2,
                    index=src_indices[:, batch * bs : (batch + 1) * bs, :, None, None, None].repeat(
                        1, 1, 1, dim, h // 4, w // 4
                    ),
                )
                .view(-1, warmup_n_views - 1, dim, h // 4, w // 4)
            )
            src_extrinsics = (
                context["extrinsics"][:, None]
                .repeat(1, warmup_n_views_now, 1, 1, 1)
                .gather(
                    dim=2,
                    index=src_indices[:, batch * bs : (batch + 1) * bs, :, None, None].repeat(
                        1, 1, 1, 4, 4
                    ),
                )
            )
            src_intrinsics = (
                context_intrinsics[:, None]
                .repeat(1, warmup_n_views_now, 1, 1, 1)
                .gather(
                    dim=2,
                    index=src_indices[:, batch * bs : (batch + 1) * bs, :, None, None].repeat(
                        1, 1, 1, 3, 3
                    ),
                )
            )
            src_cam_t_world = src_extrinsics.inverse()
            cur_cam_t_world = cur_extrinsics[:, batch * bs : (batch + 1) * bs].inverse()
            src_cam_T_cur_cam = src_cam_t_world @ cur_extrinsics[
                :, batch * bs : (batch + 1) * bs
            ].unsqueeze(2)
            cur_cam_T_src_cam = cur_cam_t_world.unsqueeze(2) @ src_extrinsics
            src_cam_T_cur_cam_ = rearrange(src_cam_T_cur_cam, "b v n x y -> (b v) n x y")
            cur_cam_T_src_cam_ = rearrange(cur_cam_T_src_cam, "b v n x y -> (b v) n x y")
            src_intrinsics_ = rearrange(src_intrinsics, "b v n x y -> (b v) n x y")
            cur_intrinsics_ = rearrange(
                cur_intrinsics[:, batch * bs : (batch + 1) * bs], "b v x y -> (b v) x y"
            )
            src_K = torch.eye(4, device=context["image"].device)[None, None].repeat(
                src_intrinsics_.shape[0], src_intrinsics_.shape[1], 1, 1
            )
            src_K[:, :, :3, :3] = src_intrinsics_
            cur_inverse = torch.eye(4, device=context["image"].device)[None].repeat(
                cur_intrinsics_.shape[0], 1, 1
            )
            cur_inverse[:, :3, :3] = cur_intrinsics_.inverse()
            results["cur_invK"] = cur_inverse
            results["src_K"] = src_K
            results["cur_wtc"] = cur_extrinsics[:, batch * bs : (batch + 1) * bs]
            results["cur_ctw"] = cur_cam_t_world
            near = context["near"][:1, 0].type_as(src_K).view(1, 1, 1, 1)
            far = context["far"][:1, 0].type_as(src_K).view(1, 1, 1, 1)
            cost_volume = self.cost_volume(
                cur_feats=matching_cur_feats,
                src_feats=matching_src_feats,
                src_extrinsics=src_cam_T_cur_cam_,
                src_poses=cur_cam_T_src_cam_,
                src_Ks=src_K,
                cur_invK=cur_inverse,
                min_depth=near,
                max_depth=far,
            )
            cost_volume_features = self.cv_encoder(
                cost_volume, [x[batch * bs : (batch + 1) * bs] for x in cur_feats[1:]]
            )
            depth_outputs = self.depth_decoder(
                [cur_feats[0][batch * bs : (batch + 1) * bs]] + cost_volume_features,
                imgs=cur_image[batch * bs : (batch + 1) * bs],
            )
            to_skip = cur_image[None][:, batch * bs : (batch + 1) * bs]
            to_skip = rearrange(to_skip, "b v c h w -> (b v) c h w")
            skip = self.high_resolution_skip[s + 1](to_skip)
            margin = 0
            (xy_ray, _) = sample_image_grid(
                (h // (1 + self.cfg.low_res), w // (1 + self.cfg.low_res)), device
            )
            gaussians_feats = rearrange(
                depth_outputs[f"output_pred_s{s}_b1hw"][:, 1:],
                "(b v) c h w -> b v h w c",
                b=b,
                v=warmup_n_views_now,
            )
            gaussians_feats = gaussians_feats + rearrange(
                skip, "(b v) c h w -> b v h w c", b=b, v=warmup_n_views_now
            )
            if self.cfg.openvocab_domain in ["ensemble", "3d"]:
                if self.cfg.semantic_early_fusion:
                    clip_gaussians_feats_3d = (
                        gaussians_feats
                        + self.semantic_proj(clip_gaussian_feats).permute(0, 2, 3, 1)[None]
                    )
                else:
                    clip_gaussians_feats_3d = gaussians_feats.clone()
                clip_gaussians_feats_3d = rearrange(
                    clip_gaussians_feats_3d, "b v h w c -> b v (h w) c"
                )
            else:
                clip_gaussians_feats_3d = None
            densities = nn.Sigmoid()(
                rearrange(
                    depth_outputs[f"output_pred_s{s}_b1hw"][:, :1],
                    "(b v) c h w -> b v (c h w) () ()",
                    b=b,
                    v=warmup_n_views_now,
                )
            )
            depth_outputs["densities"] = rearrange(densities, "b v x y z -> (b v) () x y z")
            if self.cfg.use_gt_depth:
                depths_raw = context[f"depth_s0"][0, batch * bs : batch * bs + warmup_n_views_now]
                depths = depth_outputs[f"depth_pred_s{s}_b1hw"]
                weights = depth_outputs[f"depth_weights"]
                mask = (depths_raw > 0.001) * (depths_raw < 10)
                depths[mask] = depths_raw[mask].float()
                depth_outputs[f"depth_pred_s{s}_b1hw"] = depths
                weights[mask] = 1.0
                depth_outputs["depth_weights"] = weights
            depths = rearrange(
                depth_outputs[f"depth_pred_s{s}_b1hw"], "(b v) c h w -> b v (c h w) () ()", b=b
            )
            weights = rearrange(
                depth_outputs[f"depth_weights"], "(b v) c h w -> b v (c h w) () ()", b=b
            )
            for key in depth_outputs:
                if key not in depth_outputs_all:
                    continue
                if is_testing:
                    depth_outputs_all[key].append(depth_outputs[key])
                else:
                    depth_outputs_all[key].append(depth_outputs[key])
            gaussians_feats = rearrange(gaussians_feats, "b v h w c -> b v (h w) c")
            xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")
            offset_xy = torch.zeros_like(
                rearrange(
                    gaussians_feats[..., :2], "... (srf c) -> ... srf c", srf=self.cfg.num_surfaces
                ),
                device=gaussians_feats.device,
            )
            xy_ray = xy_ray + offset_xy
            coords = self.gaussian_adapter.forward(
                rearrange(
                    context["extrinsics"][:, batch * bs : batch * bs + warmup_n_views_now],
                    "b v i j -> b v () () () i j",
                ),
                rearrange(
                    context["intrinsics"][:, batch * bs : batch * bs + warmup_n_views_now],
                    "b v i j -> b v () () () i j",
                ),
                rearrange(xy_ray, "b v r srf xy -> b v r srf () xy"),
                depths,
                densities,
                gaussians_feats,
                (
                    h // (1 + self.cfg.low_res) - margin * 2,
                    w // (1 + self.cfg.low_res) - margin * 2,
                ),
                load_depth=self.cfg.load_depth,
                fusion=True,
            )
            num_raw_gaussians = gaussians_feats.shape[2] * gaussians_feats.shape[1]
            B = gaussians_feats.shape[0]
            for bb in range(B):
                cur_gs = gaussians_feats[bb : bb + 1]
                cur_clip_gs_3d = (
                    clip_gaussians_feats_3d[bb : bb + 1]
                    if clip_gaussians_feats_3d is not None
                    else None
                )
                cur_coords = coords[bb : bb + 1]
                cur_densities = densities[bb : bb + 1]
                cur_weights = weights[bb : bb + 1]
                cur_depth = rearrange(
                    depth_outputs[f"depth_pred_s{s}_b1hw"], "(b v) c h w -> b v c h w", b=B
                )[bb]
                if self.cfg.openvocab_domain in ["ensemble", "2d"]:
                    cur_codebook_caches = rearrange(cur_codebook_caches, "b h w ->() b (h w)")
                globals = self.fuse_gaussians(
                    cur_gs,
                    cur_clip_gs_3d,
                    cur_codebook_caches,
                    cur_coords,
                    cur_densities,
                    cur_weights,
                    cur_depth,
                    context["extrinsics"][
                        bb : bb + 1, batch * bs : batch * bs + warmup_n_views_now
                    ],
                    context["intrinsics"][
                        bb : bb + 1, batch * bs : batch * bs + warmup_n_views_now
                    ],
                    (
                        h // (1 + self.cfg.low_res) - margin * 2,
                        w // (1 + self.cfg.low_res) - margin * 2,
                    ),
                    vis=self.cfg.vis,
                    globals=globals,
                    remove=self.cfg.remove,
                    use_em=self.cfg.use_em,
                    decoder=decoder,
                    near=context["near"][bb : bb + 1, batch * bs : batch * bs + warmup_n_views_now],
                    far=context["far"][bb : bb + 1, batch * bs : batch * bs + warmup_n_views_now],
                    img=context["image"][bb : bb + 1, batch * bs : batch * bs + warmup_n_views_now],
                    fusion=self.cfg.fusion,
                    fore_fusion=self.cfg.fore_fusion,
                    concat_depth=self.cfg.concat_depth,
                    use_gru=self.cfg.use_gru,
                    export_ply=export_ply,
                )
        depth_preds_all = torch.cat(depth_outputs_all[f"depth_pred_s{s}_b1hw"], dim=0)
        densities_preds_all = torch.cat(depth_outputs_all[f"densities"], dim=0)
        if self.cfg.refine_gs:
            for _ in range(self.cfg.refine_times):
                for i in range(len(depth_preds_all)):
                    cur_depth = depth_preds_all[i : i + 1]
                    cur_weights = densities_preds_all[i : i + 1]
                    if self.cfg.refine_uniform:
                        cur_weights = torch.ones_like(cur_weights, device=cur_weights.device)
                        globals.densities = torch.ones_like(
                            globals.densities, device=globals.densities.device
                        )
                    globals = self.refine_gaussians(
                        globals,
                        cur_depth,
                        cur_weights,
                        context["extrinsics"][:, i : i + 1],
                        context["intrinsics"][:, i : i + 1],
                        (
                            h // (1 + self.cfg.low_res) - margin * 2,
                            w // (1 + self.cfg.low_res) - margin * 2,
                        ),
                        ws=self.cfg.refine_ws,
                        depth_thres=self.cfg.refine_thres,
                        soft_thres=self.cfg.refine_soft_thres,
                        refine_pp=self.cfg.refine_pp,
                        export_ply=export_ply or self.cfg.ft,
                        num=i,
                    )
        warmup_cache = {
            "warmup_indices": cur_indices,
            "warmup_intrinsics": cur_intrinsics,
            "warmup_extrinsics": cur_extrinsics,
            "warmup_img_feats": cur_feats,
        }
        self._record_profile(
            {"phase": "warmup_construct", "ms": round((time.perf_counter() - _t_wu) * 1000, 2)}
        )
        return (globals, depth_outputs_all, warmup_cache)

    def forward(
        self,
        contexts,
        global_step: int,
        deterministic: bool = False,
        visualization_dump: Optional[dict] = None,
        is_testing: bool = False,
        export_ply: bool = False,
        dataset_name: str = "scannet",
        decoder=None,
        source=None,
        dense: int = 0,
        scene=None,
        target_indices=None,
        output_path=None,
        target=None,
        test_bev=False,
        path=None,
        resume_state=None,
        online_incremental=False,
        return_state=False,
    ) -> dict:
        """Restore the episode state, encode unseen frames and return the next state."""
        if not is_testing or not online_incremental or export_ply or test_bev:
            raise ValueError(
                "OnlineEncoder requires testing, incremental updates and experiment-owned visualization"
            )
        if self.cfg.est_depth != "cost" or self.cfg.vis or self.cfg.vis_gs or self.cfg.ft:
            raise ValueError("OnlineEncoder requires the full PBSP inference configuration")
        device = contexts[0]["image"].device
        (b, n_views, _, h, w) = contexts[0]["image"].shape
        results = {}
        num_context_views = self.cfg.num_views
        warmup_n_views = self.cfg.num_views
        if resume_state is None:
            (globals, depth_outputs_all, warmup_cache) = self.warmup_construct(
                contexts[0],
                target,
                decoder,
                is_testing=is_testing,
                export_ply=export_ply,
                test_bev=test_bev,
            )
        else:
            globals = resume_state["globals"]
            depth_outputs_all = resume_state["depth_outputs_all"]
            warmup_cache = {
                "warmup_indices": resume_state["cur_indices"],
                "warmup_intrinsics": resume_state["cur_intrinsics"],
                "warmup_extrinsics": resume_state["cur_extrinsics"],
                "warmup_img_feats": resume_state["cur_feats"],
            }
        length = len(contexts)
        our_gaussians = []
        gaussians = []
        coords = []
        results = {}
        if dataset_name == "scannet" or not is_testing or self.cfg.num_views > 2:
            self.backbone.train()
        else:
            print("freezing backbone...")
        num = 0
        context = contexts[num]
        context["image_shape"] = (h, w)
        self.cfg.gaussians_per_pixel = 1
        context_intrinsics = context["intrinsics"].clone()
        context_intrinsics[:, :, 0] *= w // 4
        context_intrinsics[:, :, 1] *= h // 4
        all_indices = torch.arange(n_views, device=context["image"].device)[warmup_n_views:]
        all_intrinsics = context_intrinsics.gather(
            dim=1, index=all_indices.view(1, -1, 1, 1).repeat(b, 1, 3, 3)
        )
        all_extrinsics = context["extrinsics"].gather(
            dim=1, index=all_indices.view(1, -1, 1, 1).repeat(b, 1, 4, 4)
        )
        all_image = context["image"][:, warmup_n_views:].reshape(-1, 3, h, w)
        all_cache = [context["cache"][i] for i in all_indices]
        cur_indices = warmup_cache["warmup_indices"]
        cur_intrinsics = warmup_cache["warmup_intrinsics"]
        cur_extrinsics = warmup_cache["warmup_extrinsics"]
        cur_feats = warmup_cache["warmup_img_feats"]
        # Historical feature values stay on the host, at original precision.
        # Only the views selected by the unchanged distance ranking go to GPU.
        # Per-frame chunks also avoid copying the complete history each step.
        cur_feats = [list(feat.detach().cpu().split(1, dim=0))
                     if torch.is_tensor(feat) else feat for feat in cur_feats]
        depth_outputs_all = {key: [v.detach().cpu() for v in values]
                             for key, values in depth_outputs_all.items()}
        _start_idx = len(cur_indices) - warmup_n_views
        for idx in range(_start_idx, n_views - warmup_n_views):
            cur_indices = torch.cat([cur_indices, all_indices[idx : idx + 1]], dim=0)
            cur_intrinsics = torch.cat([cur_intrinsics, all_intrinsics[:, idx : idx + 1]], dim=1)
            cur_extrinsics = torch.cat([cur_extrinsics, all_extrinsics[:, idx : idx + 1]], dim=1)
            cur_image = all_image[idx : idx + 1]
            cur_cache = [all_cache[idx]]
            new_feats = self.backbone(cur_image)
            semantic = self.semantic_injector.inject(cur_image, cur_cache, (h, w), globals)
            new_clip_feats, codebook_caches, globals = semantic.values()
            del semantic
            if self.cfg.openvocab_domain in ["ensemble", "3d"]:
                clip_gaussians_feats = torch.nn.functional.interpolate(
                    new_clip_feats, size=(h // 2, w // 2), mode="nearest"
                )
            else:
                clip_gaussians_feats = None
            if self.cfg.openvocab_domain in ["ensemble", "2d"]:
                cur_codebook_caches = torch.nn.functional.interpolate(
                    codebook_caches.unsqueeze(1), size=(h // 2, w // 2), mode="nearest"
                ).squeeze(1)
                cur_codebook_caches = rearrange(cur_codebook_caches, "b h w -> () b (h w)")
            else:
                cur_codebook_caches = None
            for history, new_feat in zip(cur_feats, new_feats):
                history.extend(new_feat.detach().cpu().split(1, dim=0))
            resized = 0
            s = -1 if not self.cfg.low_res else 0
            cur_n_views = len(cur_indices)
            full_indices = torch.arange(cur_n_views, device=context["image"].device)[None]
            slide_mask = torch.zeros((1, cur_n_views), dtype=torch.bool, device=full_indices.device)
            if self.cfg.recon_mode == "incremental":
                dist_matrix = calculate_distance_matrix(cur_extrinsics)[-1:]
                dist_matrix = dist_matrix.clone()
                dist_matrix[0, -1] = float("inf")
                _k = min(num_context_views - 1, dist_matrix.shape[1] - 1)
                (_, indices) = torch.topk(dist_matrix, _k, largest=False, sorted=False, dim=1)
            elif self.cfg.recon_mode == "online":
                indices = torch.arange(idx + 1, idx + self.cfg.num_views + 1)[None].cuda()
            slide_mask.scatter_(1, indices, True)
            slide_mask[0, -1] = False
            src_indices = full_indices[slide_mask][None, None].repeat(b, 1, 1)
            matching_cur_feats = new_feats[1]
            dim = matching_cur_feats.shape[-3]
            assert b == 1, 'Online mapping owns one episode at a time'
            selected = src_indices.reshape(-1).tolist()
            matching_src_feats = torch.cat([cur_feats[1][i] for i in selected], dim=0).to(device)
            matching_src_feats = matching_src_feats.view(
                -1, num_context_views - 1, dim, h // 4, w // 4)
            src_extrinsics = cur_extrinsics[:, None].gather(
                dim=2, index=src_indices[:, :, :, None, None].repeat(1, 1, 1, 4, 4)
            )
            src_intrinsics = cur_intrinsics[:, None].gather(
                dim=2, index=src_indices[:, :, :, None, None].repeat(1, 1, 1, 3, 3)
            )
            src_cam_t_world = src_extrinsics.inverse()
            cur_cam_t_world = cur_extrinsics[:, -1:].inverse()
            src_cam_T_cur_cam = src_cam_t_world @ cur_extrinsics[:, -1:].unsqueeze(2)
            cur_cam_T_src_cam = cur_cam_t_world.unsqueeze(2) @ src_extrinsics
            src_cam_T_cur_cam_ = rearrange(src_cam_T_cur_cam, "b v n x y -> (b v) n x y")
            cur_cam_T_src_cam_ = rearrange(cur_cam_T_src_cam, "b v n x y -> (b v) n x y")
            src_intrinsics_ = rearrange(src_intrinsics, "b v n x y -> (b v) n x y")
            cur_intrinsics_ = rearrange(cur_intrinsics[:, -1:], "b v x y -> (b v) x y")
            src_K = torch.eye(4, device=context["image"].device)[None, None].repeat(
                src_intrinsics_.shape[0], src_intrinsics_.shape[1], 1, 1
            )
            src_K[:, :, :3, :3] = src_intrinsics_
            cur_inverse = torch.eye(4, device=context["image"].device)[None].repeat(
                cur_intrinsics_.shape[0], 1, 1
            )
            cur_inverse[:, :3, :3] = cur_intrinsics_.inverse()
            results["cur_invK"] = cur_inverse
            results["src_K"] = src_K
            results["cur_wtc"] = cur_extrinsics[:, -1:]
            results["cur_ctw"] = cur_cam_t_world
            near = context["near"][:1, cur_indices[-1]].type_as(src_K).view(1, 1, 1, 1)
            far = context["far"][:1, cur_indices[-1]].type_as(src_K).view(1, 1, 1, 1)
            cost_volume = self.cost_volume(
                cur_feats=matching_cur_feats,
                src_feats=matching_src_feats,
                src_extrinsics=src_cam_T_cur_cam_,
                src_poses=cur_cam_T_src_cam_,
                src_Ks=src_K,
                cur_invK=cur_inverse,
                min_depth=near,
                max_depth=far,
            )
            cost_volume_features = self.cv_encoder(cost_volume, new_feats[1:])
            depth_outputs = self.depth_decoder(
                [new_feats[0]] + cost_volume_features, imgs=cur_image
            )
            to_skip = cur_image
            skip = self.high_resolution_skip[s + 1](to_skip)
            margin = 0
            (xy_ray, _) = sample_image_grid(
                (h // (1 + self.cfg.low_res), w // (1 + self.cfg.low_res)), device
            )
            if self.cfg.depth_pad:
                border = 8 if not self.cfg.low_res else 4
                for name in [f"depth_pred_s{s}_b1hw", f"depth_weights"]:
                    depth_outputs[name][:, :, :border, :] = depth_outputs[name][:, :, border, None]
                    depth_outputs[name][:, :, -border:, :] = depth_outputs[name][
                        :, :, -border - 1, None
                    ]
                    depth_outputs[name][:, :, :, :border] = depth_outputs[name][
                        :, :, :, border, None
                    ]
                    depth_outputs[name][:, :, :, -border:] = depth_outputs[name][
                        :, :, :, -border - 1, None
                    ]
            gaussians_feats = rearrange(
                depth_outputs[f"output_pred_s{s}_b1hw"][:, 1:], "b c h w -> b () h w c"
            )
            gaussians_feats = gaussians_feats + rearrange(skip, "b c h w -> b () h w c")
            if self.cfg.openvocab_domain in ["ensemble", "3d"]:
                if self.cfg.semantic_early_fusion:
                    clip_gaussians_feats_3d = (
                        gaussians_feats
                        + self.semantic_proj(clip_gaussians_feats).permute(0, 2, 3, 1)[None]
                    )
                else:
                    clip_gaussians_feats_3d = gaussians_feats.clone()
                clip_gaussians_feats_3d = rearrange(
                    clip_gaussians_feats_3d, "b v h w c -> b v (h w) c"
                )
            else:
                clip_gaussians_feats_3d = None
            densities = nn.Sigmoid()(
                rearrange(
                    depth_outputs[f"output_pred_s{s}_b1hw"][:, :1], "b c h w -> b () (c h w) () ()"
                )
            )
            depth_outputs["densities"] = rearrange(densities, "b v x y z -> (b v) () x y z")
            if self.cfg.use_gt_depth:
                depths_raw = context[f"depth_s0"][0, cur_indices[-1] : cur_indices[-1] + 1]
                depths = depth_outputs[f"depth_pred_s{s}_b1hw"]
                weights = depth_outputs[f"depth_weights"]
                mask = (depths_raw > 0.001) * (depths_raw < 10)
                depths[mask] = depths_raw[mask].float()
                depth_outputs[f"depth_pred_s{s}_b1hw"] = depths
                weights[mask] = 1.0
                depth_outputs["depth_weights"] = weights
            depths = rearrange(
                depth_outputs[f"depth_pred_s{s}_b1hw"], "(b v) c h w -> b v (c h w) () ()", b=b
            )
            weights = rearrange(
                depth_outputs[f"depth_weights"], "(b v) c h w -> b v (c h w) () ()", b=b
            )
            for key in depth_outputs:
                if key not in depth_outputs_all:
                    continue
                if is_testing:
                    depth_outputs_all[key].append(depth_outputs[key].detach().cpu())
                else:
                    depth_outputs_all[key].append(depth_outputs[key].detach().cpu())
            gaussians_feats = rearrange(gaussians_feats, "b v h w c -> b v (h w) c")
            xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")
            offset_xy = torch.zeros_like(
                rearrange(
                    gaussians_feats[..., :2], "... (srf c) -> ... srf c", srf=self.cfg.num_surfaces
                ),
                device=gaussians_feats.device,
            )
            xy_ray = xy_ray + offset_xy
            coords = self.gaussian_adapter.forward(
                rearrange(
                    context["extrinsics"][:, cur_indices[-1] : cur_indices[-1] + 1],
                    "b v i j -> b v () () () i j",
                ),
                rearrange(
                    context["intrinsics"][:, cur_indices[-1] : cur_indices[-1] + 1],
                    "b v i j -> b v () () () i j",
                ),
                rearrange(xy_ray, "b v r srf xy -> b v r srf () xy"),
                depths,
                densities,
                gaussians_feats,
                (
                    h // (1 + self.cfg.low_res) - margin * 2,
                    w // (1 + self.cfg.low_res) - margin * 2,
                ),
                load_depth=self.cfg.load_depth,
                fusion=True,
            )
            num_raw_gaussians = gaussians_feats.shape[2] * gaussians_feats.shape[1]
            B = gaussians_feats.shape[0]
            for bb in range(B):
                cur_gs = gaussians_feats[bb : bb + 1]
                cur_clip_gs_3d = (
                    clip_gaussians_feats_3d[bb : bb + 1]
                    if clip_gaussians_feats_3d is not None
                    else None
                )
                cur_coords = coords[bb : bb + 1]
                cur_densities = densities[bb : bb + 1]
                cur_weights = weights[bb : bb + 1]
                cur_depth = rearrange(
                    depth_outputs[f"depth_pred_s{s}_b1hw"], "(b v) c h w -> b v c h w", b=B
                )[bb]
                _t_fuse = time.perf_counter()
                (globals, new_gaussian_mask) = self.fuse_gaussians(
                    cur_gs,
                    cur_clip_gs_3d,
                    cur_codebook_caches,
                    cur_coords,
                    cur_densities,
                    cur_weights,
                    cur_depth,
                    context["extrinsics"][bb : bb + 1, cur_indices[-1] : cur_indices[-1] + 1],
                    context["intrinsics"][bb : bb + 1, cur_indices[-1] : cur_indices[-1] + 1],
                    (
                        h // (1 + self.cfg.low_res) - margin * 2,
                        w // (1 + self.cfg.low_res) - margin * 2,
                    ),
                    vis=self.cfg.vis,
                    globals=globals,
                    remove=self.cfg.remove,
                    use_em=self.cfg.use_em,
                    decoder=decoder,
                    near=context["near"][bb : bb + 1, cur_indices[-1] : cur_indices[-1] + 1],
                    far=context["far"][bb : bb + 1, cur_indices[-1] : cur_indices[-1] + 1],
                    img=cur_image[bb : bb + 1][:, None],
                    fusion=self.cfg.fusion,
                    fore_fusion=self.cfg.fore_fusion,
                    concat_depth=self.cfg.concat_depth,
                    use_gru=self.cfg.use_gru,
                    export_ply=export_ply or self.cfg.ft,
                    output_new_idx=True,
                )
                self._record_profile(
                    {"phase": "fuse_frame", "ms": round((time.perf_counter() - _t_fuse) * 1000, 2)}
                )
            # Historical depth outputs are returned for provenance; the online
            # path does not refine them and need not concatenate them per frame.
        results["globals"] = globals
        if return_state:
            results["online_state"] = {
                "globals": globals,
                "depth_outputs_all": depth_outputs_all,
                "cur_indices": cur_indices,
                "cur_intrinsics": cur_intrinsics,
                "cur_extrinsics": cur_extrinsics,
                "cur_feats": cur_feats,
            }
        return results
