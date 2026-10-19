"""Project, associate and fuse Gaussian geometry and semantic features.

Derived from EmbodiedSplat. See THIRD_PARTY_NOTICES.md for attribution and scope.
The encoder supplies its trained Gaussian adapter and recurrent fusion layers.
"""

import numpy as np
import torch
from torch import nn
from einops import rearrange, reduce, repeat


def positional_encoding(positions, freqs, ori=False):
    '''encode positions with positional encoding
        positions: :math:`(...,D)`
        freqs: int
    Return:
        pts: :math:`(..., 2DF)`
    '''
    freq_bands = (2**torch.arange(freqs).float()).to(positions.device)  # (F,)
    ori_c = positions.shape[-1]
    pts = (positions[..., None] * freq_bands).reshape(positions.shape[:-1] +
                                                      (freqs * positions.shape[-1], ))  # (..., DF)
    if ori:
        pts = torch.cat([positions, torch.sin(pts), torch.cos(pts)], dim=-1).reshape(pts.shape[:-1]+(pts.shape[-1]*2+ori_c,))
    else:
        pts = torch.stack([torch.sin(pts), torch.cos(pts)], dim=-1).reshape(pts.shape[:-1]+(pts.shape[-1]*2,))
    return pts


class GaussianFusion:
    """Fusion operations sharing the encoder weights without adding parameter prefixes."""

    def get_transformation_matrix(self):
        import collections
        from scipy.linalg import expm, norm

        def M(axis, theta):
            return expm(np.cross(np.eye(3), axis / norm(axis) * theta))

        voxelization_matrix, rotation_matrix = np.eye(4), np.eye(4)

        rot_mat = np.eye(3)
        if self.cfg.pcd_aug and self.ROTATION_AUGMENTATION_BOUND is not None:
            if isinstance(self.ROTATION_AUGMENTATION_BOUND, collections.abc.Iterable):
                rot_mats = []
                for axis_ind, rot_bound in enumerate(self.ROTATION_AUGMENTATION_BOUND):
                    theta = 0
                    axis = np.zeros(3)
                    axis[axis_ind] = 1
                    if rot_bound is not None:
                        theta = np.random.uniform(*rot_bound)
                    rot_mats.append(M(axis, theta))

                np.random.shuffle(rot_mats)
                rot_mat = rot_mats[0] @ rot_mats[1] @ rot_mats[2]
            else:
                raise ValueError()
        rotation_matrix[:3, :3] = rot_mat

        scale = 1 / self.cfg.voxel_size
        if self.cfg.pcd_aug and self.SCALE_AUGMENTATION_BOUND is not None:
            scale *= np.random.uniform(*self.SCALE_AUGMENTATION_BOUND)
        np.fill_diagonal(voxelization_matrix[:3, :3], scale)

        return torch.from_numpy(voxelization_matrix).to(self.device).float(), torch.from_numpy(rotation_matrix).to(self.device).float()

    def refine_gaussians(
        self,
        globals,
        depths,
        weights,
        extrinsics,
        intrinsics,
        image_shape,
        depth_thres=0.3,
        ws=True,
        soft_thres=False,
        refine_pp=False,
        export_ply=False,
        num=0,
        new_gaussian_mask=None
    ):
        depths = rearrange(depths, "v c h w -> v (c h w)")
        h, w = image_shape

        # global_gaussians = globals['gs']
        global_coords = globals.coords[:, globals.valid[0]]
        global_weights = globals.densities[:, globals.valid[0]]

        i = 0
        extrinsic = extrinsics[0,i]
        intrinsic = intrinsics[0,i].clone()
        intrinsic[:1,:] *= w
        intrinsic[1:2,:] *= h
        focal_length = (intrinsic[0, 0], intrinsic[1, 1])
        principal_point = (intrinsic[0, 2], intrinsic[1, 2])
        principal_point_mat = torch.tensor([principal_point[0], principal_point[1]]).to(intrinsic.device)
        principal_point_mat = principal_point_mat.reshape(1, 2)
        focal_length_mat = torch.tensor([focal_length[0], focal_length[1]]).to(intrinsic.device)
        focal_length_mat = focal_length_mat.reshape(1, 2)
        means1 = torch.cat([global_coords[0], torch.ones_like(global_coords[..., :1][0])], dim=-1).permute(1,0) # [4, 196608]
        post_xy_coords = torch.matmul(extrinsic.inverse(), means1)[:3]
        curr_depths = post_xy_coords[2:3, :]
        post_xy_coords = (post_xy_coords / curr_depths)[:2].permute(1,0)
        curr_depths = curr_depths.squeeze()
        post_xy_coords = post_xy_coords * focal_length_mat.reshape(1,2) + principal_point_mat # [196608, 2]
        pixel_coords = post_xy_coords.round().long()[:,[1,0]]
        valid = (pixel_coords[:, 0] >= 0) & (pixel_coords[:, 0] < h) & (pixel_coords[:, 1] >= 0) & (pixel_coords[:, 1] < w) & (curr_depths > 0)
        proj_map = - torch.ones((h*w), device=depths.device, dtype=curr_depths.dtype)
        depth_map = torch.ones((h*w), device=depths.device, dtype=curr_depths.dtype) * 10000

        pixel_indices = (pixel_coords[:, 1] + pixel_coords[:, 0]*w)[valid]
        depth_map.scatter_reduce_(0, pixel_indices, curr_depths[valid], reduce='amin')

        if soft_thres:
            remove_mask = depths[i] - depth_map > 0
        else:
            remove_mask = depths[i] - depth_map > depth_thres

        proj_map = torch.where(depth_map[pixel_indices] == curr_depths[valid])[0]
        fusion_indices = torch.where(remove_mask[pixel_indices])[0]
        fusion_indices_ = fusion_indices[torch.isin(fusion_indices, proj_map)]
        corr_indices = proj_map[torch.isin(proj_map, fusion_indices)]
        valid_indices = torch.zeros(valid.sum(), device=valid.device, dtype=torch.bool)
        valid_indices.scatter_(0, corr_indices, True)
        mask = torch.zeros_like(valid, device=valid.device, dtype=torch.bool)
        mask[valid] = valid_indices

        if refine_pp:
            curr_depths = curr_depths[valid]
            N = len(curr_depths)
            h, w = 384, 512  # Example dimensions for depth_map
            # gs_feat = torch.randn(N, 3, device=curr_depths.device)
            gs_weights = global_weights[0, valid]
            gs_coords = global_coords[0, valid]

            expanded_depths = depths[i][pixel_indices.squeeze()]

            # Compute the differences and apply the condition
            diff = curr_depths - expanded_depths
            condition_mask = torch.abs(diff) < 0.1

            # Filter indices using the condition mask
            valid_indices = torch.nonzero(condition_mask).squeeze()

            # We need to adjust the indices map to only include the valid indices
            filtered_pixel_indices = pixel_indices[valid_indices].squeeze()
            filtered_gaussian_indices = valid_indices

            # Re-create the sparse tensor with the new filtered indices
            sparse_indices = torch.stack([filtered_pixel_indices, filtered_gaussian_indices], dim=0)
            values = torch.ones_like(filtered_gaussian_indices, dtype=torch.float32)  # We can use ones as dummy values

            if sparse_indices.dim() == 1:
                sparse_indices = sparse_indices.unsqueeze(1)

            # Build the new sparse tensor
            new_indices_map = torch.sparse_coo_tensor(sparse_indices, values, size=(h * w, N))

            # Ensure that the sparse tensor is coalesced
            new_indices_map = new_indices_map.coalesce()

            # Retrieve all non-zero indices (which are the valid Gaussians for each pixel)
            gaussian_indices = new_indices_map.indices()[1]
            new_pixel_indices = new_indices_map.indices()[0]

            # Gather Gaussian features using the valid indices
            gaussian_weights = gs_weights[gaussian_indices]

            # Compute the sum of features for each pixel via a scatter operation
            weight_sums = torch.zeros((h * w, 1, 1), device=curr_depths.device)
            weight_sums.index_add_(0, new_pixel_indices, gaussian_weights)

            curr_weights = weight_sums.unsqueeze(0)

            # Find the global gs that are near the closest ones
            expanded_depth_map = depth_map[pixel_indices.squeeze()]
            diff = curr_depths - expanded_depth_map
            condition_mask = torch.abs(diff) < 0.1

            # Filter indices using the condition mask
            valid_indices = torch.nonzero(condition_mask).squeeze()

            # We need to adjust the indices map to only include the valid indices
            filtered_pixel_indices = pixel_indices[valid_indices].squeeze()
            filtered_gaussian_indices = valid_indices

            # Re-create the sparse tensor with the new filtered indices
            sparse_indices = torch.stack([filtered_pixel_indices, filtered_gaussian_indices], dim=0)
            values = torch.ones_like(filtered_gaussian_indices, dtype=torch.float32)  # We can use ones as dummy values

            if sparse_indices.dim() == 1:
                sparse_indices = sparse_indices.unsqueeze(1)

            # Build the new sparse tensor
            new_indices_map = torch.sparse_coo_tensor(sparse_indices, values, size=(h * w, N))

            # Ensure that the sparse tensor is coalesced
            new_indices_map = new_indices_map.coalesce()

            # Retrieve all non-zero indices (which are the valid Gaussians for each pixel)
            gaussian_indices = new_indices_map.indices()[1]
            new_pixel_indices = new_indices_map.indices()[0]

            # Gather Gaussian features using the valid indices
            gaussian_weights = gs_weights[gaussian_indices]

            # Compute the sum of features for each pixel via a scatter operation
            weight_sums = torch.zeros((h * w, 1, 1), device=curr_depths.device)
            weight_sums.index_add_(0, new_pixel_indices, gaussian_weights)

            curr_global_weights = global_weights[:, mask]
            to_add = weight_sums[pixel_indices][fusion_indices_]
            # curr_global_weights = curr_global_weights + to_add
            curr_global_weights = to_add
            multiplier = curr_global_weights[:, 0, 0] / (curr_global_weights[:, 0, 0]+ curr_weights[:, pixel_indices][:, fusion_indices_, 0, 0]+ \
                                                         weights[:, i, pixel_indices][:,fusion_indices_, 0, 0])
        else:
            multiplier = global_weights[:, mask, 0, 0] / (global_weights[:, mask, 0, 0]+ weights[:, i, pixel_indices][:,fusion_indices_, 0, 0])

        if new_gaussian_mask is not None:
            filtered_mask = mask & new_gaussian_mask
            multiplier_mask = filtered_mask[mask]
            mask = filtered_mask
        else:
            multiplier_mask = None

        if mask.sum() > 0 and (not self.cfg.vis_refine):
            if ws:
                if multiplier_mask is not None:
                    multiplier = multiplier[:, multiplier_mask]

                globals.append(
                    means=globals.means[:, globals.valid[0]][:, mask],
                    covariances=globals.covariances[:, globals.valid[0]][:, mask],
                    harmonics=globals.harmonics[:, globals.valid[0]][:, mask],
                    opacities=globals.opacities[:, globals.valid[0]][:, mask] * multiplier,
                    features=None,
                    clip_features_3d=None,
                    densities=globals.densities[:, globals.valid[0]][:, mask],
                    weights=None,
                    coords=globals.coords[:, globals.valid[0]][:, mask],
                    extrinsics=None,
                    depths=None,
                    codebooks_idx=None,
                    codebooks_weights=None,
                    mask=mask,
                    scales=globals.scales[:, globals.valid[0]][:, mask] if export_ply else None,
                    rotations=globals.rotations[:, globals.valid[0]][:, mask] if export_ply else None,
                    features_invalid=torch.ones_like(globals.clip_features_2d["invalid"][:, globals.valid[0]][:, mask], device=self.device) * -1
                )
            else:
                globals.append(
                    means=globals.means[:, globals.valid[0]][:, mask],
                    covariances=globals.covariances[:, globals.valid[0]][:, mask],
                    harmonics=globals.harmonics[:, globals.valid[0]][:, mask],
                    opacities=globals.opacities[:, globals.valid[0]][:, mask] * 0,
                    features=None,
                    clip_features_3d=None,
                    densities=globals.densities[:, globals.valid[0]][:, mask],
                    weights=None,
                    coords=globals.coords[:, globals.valid[0]][:, mask],
                    extrinsics=None,
                    depths=None,
                    codebooks_idx=None,
                    codebooks_weights=None,
                    mask=mask,
                    scales=globals.scales[:, globals.valid[0]][:, mask] if export_ply else None,
                    rotations=globals.rotations[:, globals.valid[0]][:, mask] if export_ply else None,
                    features_invalid=torch.ones(globals.clip_features_2d["invalid"][:, globals.valid[0]][:, mask], device=self.device) * -1
                )

        return globals

    def fuse_gaussians(
        self,
        gaussians,
        clip_gaussians_3d,
        codebook_caches,
        coords,
        densities,
        weight_emb,
        depths,
        extrinsics,
        intrinsics,
        image_shape,
        depth_thres=0.1,
        limit=100,
        vis=False,
        globals=None,
        remove=False,
        use_em=False,
        decoder=None,
        near=0.5,
        far=5.0,
        img=None,
        fusion=True,
        fore_fusion=False,
        concat_depth=False,
        use_gru=True,
        depth_fore=15.0,
        export_ply=False,
        output_new_idx=False
    ):

        length = min(gaussians.shape[1], limit)
        depths = rearrange(depths, "v c h w -> v (c h w)")
        initial = globals.valid.sum() == 0
        h, w = image_shape

        if initial:
            global_gaussians_feat = gaussians[:,0]

            if self.cfg.openvocab_domain in ["ensemble", "3d"]:
                global_clip_gaussians_feat_3d = clip_gaussians_3d[:, 0]
            else:
                global_clip_gaussians_feat_3d = None

            if self.cfg.openvocab_domain in ["ensemble", "2d"]:
                global_codebook_caches = codebook_caches[:, 0]
                global_codebook_weights = torch.ones_like(global_codebook_caches).to(self.device)
                invalid = global_codebook_caches == -1
                global_codebook_weights[:, invalid[0]] = 0.
            else:
                global_codebook_caches = None
                global_codebook_weights = None

            global_densities = densities[:, 0]
            global_weight_emb = weight_emb[:, 0]
            global_coords = coords[:,0,:,0,0]

            if self.cfg.unet_3d and self.cfg.openvocab_domain in ["ensemble", "3d"]:
                self.M_v, self.M_r = self.get_transformation_matrix()
                global_clip_gaussians_feat_3d = self.backbone_3d(
                    global_coords[0],
                    global_clip_gaussians_feat_3d[0],
                    self.M_v,
                    self.M_r
                )

            global_extrinsics = extrinsics[:,0][:,None].repeat(1,global_gaussians_feat.shape[1],1,1)
            global_depths = depths[None, 0]

            global_gaussians = rearrange(
                self.to_gaussians(global_gaussians_feat),
                "... (srf c) -> ... srf c",
                srf=self.cfg.num_surfaces,
            )

            global_gaussians = self.gaussian_adapter.forward(
                rearrange(global_extrinsics, "b r i j -> b () r () () i j"),
                repeat(intrinsics[:,0], "b i j -> b () N () () i j", N=global_gaussians.shape[1]),
                None,
                rearrange(global_depths, "b r -> b () r () ()"),
                nn.Sigmoid()(rearrange(global_gaussians[..., :1], "b r srf c -> b () r srf c")),
                rearrange(global_gaussians[..., 2:], "b r srf c -> b () r srf () c"),
                (h, w),
                load_depth=self.cfg.load_depth,
                fusion=False,
                coords=rearrange(global_coords, "b r c -> b () r () () c"),
            )

            globals.append(
                means=rearrange(
                    global_gaussians.means,
                    "b v r srf spp xyz -> b (v r srf spp) xyz",
                ),
                covariances=rearrange(
                    global_gaussians.covariances,
                    "b v r srf spp i j -> b (v r srf spp) i j",
                ),
                harmonics=rearrange(
                    global_gaussians.harmonics,
                    "b v r srf spp c d_sh -> b (v r srf spp) c d_sh",
                ),
                opacities=rearrange(
                    global_gaussians.opacities,
                    "b v r srf spp -> b (v r srf spp)",
                ),
                features=global_gaussians_feat,
                clip_features_3d=global_clip_gaussians_feat_3d,
                densities=global_densities,
                weights=global_weight_emb,
                coords=global_coords,
                extrinsics=global_extrinsics,
                depths=global_depths,
                codebooks_idx=global_codebook_caches,
                codebooks_weights=global_codebook_weights,
                mask=None,
                scales=rearrange(
                    global_gaussians.scales,
                    "b v r srf spp xyz -> b (v r srf spp) xyz",
                ) if export_ply else None,
                rotations=rearrange(
                    global_gaussians.rotations,
                    "b v r srf spp xyz -> b (v r srf spp) xyz",
                ) if export_ply else None,
            )
        else:
            pass

        removed = []

        for i in range(initial, length):
            # start = time.time()
            extrinsic = extrinsics[0,i]
            intrinsic = intrinsics[0,i].clone()
            intrinsic[:1,:] *= w
            intrinsic[1:2,:] *= h
            focal_length = (intrinsic[0, 0], intrinsic[1, 1])
            principal_point = (intrinsic[0, 2], intrinsic[1, 2])
            principal_point_mat = torch.tensor([principal_point[0], principal_point[1]]).to(intrinsic.device)
            principal_point_mat = principal_point_mat.reshape(1, 2)
            focal_length_mat = torch.tensor([focal_length[0], focal_length[1]]).to(intrinsic.device)
            focal_length_mat = focal_length_mat.reshape(1, 2)
            global_coords = globals.coords[:, globals.valid[0]]
            means1 = torch.cat([global_coords[0], torch.ones_like(global_coords[..., :1][0])], dim=-1).permute(1,0) # [4, 196608]
            post_xy_coords = torch.matmul(extrinsic.inverse(), means1)[:3]
            curr_depths = post_xy_coords[2:3, :]
            post_xy_coords = (post_xy_coords / curr_depths)[:2].permute(1,0)
            curr_depths = curr_depths.squeeze()
            post_xy_coords = post_xy_coords * focal_length_mat.reshape(1,2) + principal_point_mat # [196608, 2]
            pixel_coords = post_xy_coords.round().long()[:,[1,0]]
            valid = (pixel_coords[:, 0] >= 0) & (pixel_coords[:, 0] < h) & (pixel_coords[:, 1] >= 0) & (pixel_coords[:, 1] < w) & (curr_depths > 0)
            proj_map = - torch.ones((h*w), device=coords.device, dtype=curr_depths.dtype)
            depth_map = torch.ones((h*w), device=coords.device, dtype=curr_depths.dtype) * 10000

            pixel_indices = (pixel_coords[:, 1] + pixel_coords[:, 0]*w)[valid]
            depth_map.scatter_reduce_(0, pixel_indices, curr_depths[valid], reduce='amin')

            if not fore_fusion:
                fusion_mask = torch.abs(depth_map - depths[i]) < torch.clamp_min(depths[i] * 0.05, depth_thres)
            else:
                fusion_mask = (depths[i] - depth_map > - torch.clamp_min(depths[i] * 0.05, depth_thres)) \
                                & (depths[i] - depth_map < torch.clamp_min(depths[i] * 0.05, depth_fore))

            concat_mask = torch.ones_like(depths[i], device=depths[i].device, dtype=torch.bool)
            if fusion:
                concat_mask = ~fusion_mask

            if remove:
                remove_mask = depths[i] - depth_map > torch.clamp_min(depths[i] * 0.05, depth_thres)
                concat_mask = concat_mask & (~remove_mask)

            if use_em:
                render = decoder.forward(
                    global_gaussians,
                    extrinsics[:,i:i+1],
                    intrinsics[:,i:i+1],
                    near[:,i:i+1],
                    far[:,i:i+1],
                    (h, w),
                    depth_mode='depth',
                ).color

                gt = img[:,i]
                error_map = reduce((gt - render) ** 2, "b v c h w -> b v h w", "mean")

                error_mask = (error_map > 10**(-2.5)).view(-1)
                concat_mask = concat_mask & error_mask

                debug_sink = getattr(self, 'fusion_debug_sink', None)
                if debug_sink is not None:
                    debug_sink(render)

            proj_map = torch.where(depth_map[pixel_indices] == curr_depths[valid])[0]
            fusion_indices = torch.where(fusion_mask[pixel_indices])[0]
            fusion_indices_ = fusion_indices[torch.isin(fusion_indices, proj_map)]
            corr_indices = proj_map[torch.isin(proj_map, fusion_indices)]
            valid_indices = torch.zeros(valid.sum(), device=valid.device, dtype=torch.bool)
            valid_indices.scatter_(0, corr_indices, True)
            mask = torch.zeros_like(valid, device=valid.device, dtype=torch.bool)
            mask[valid] = valid_indices

            if self.cfg.unet_3d and self.cfg.openvocab_domain in ["ensemble", "3d"]:
                clip_gaussians_3d_unet = self.backbone_3d(coords[0, i, :, 0, 0], clip_gaussians_3d[0, i], self.M_v, self.M_r, globals)
            else:
                clip_gaussians_3d_unet = None

            if mask.sum() > 0 and fusion:
                input_weights_emb = positional_encoding(torch.cat([globals.densities[:, globals.valid[0]][:, mask], weight_emb[:, i, pixel_indices][:,fusion_indices_]], dim=-1), 6)
                hidden_weights_emb = positional_encoding(torch.cat([densities[:, i, pixel_indices][:,fusion_indices_], globals.weights[:, globals.valid[0]][:, mask]], dim=-1), 6)
                local_latent = gaussians[:, i, pixel_indices][:,fusion_indices_].unsqueeze(2)
                global_latent = globals.features[:, globals.valid[0]][:, mask].unsqueeze(2)
                local_depth = depths[i][pixel_indices][fusion_indices_]
                global_depth = depth_map[pixel_indices][fusion_indices_]

                if self.cfg.openvocab_domain in ["ensemble", "3d"]:
                    local_clip_latent_3d = clip_gaussians_3d_unet[:, pixel_indices][:,fusion_indices_].unsqueeze(2)
                    global_clip_latent_3d = globals.clip_features_3d[:, globals.valid[0]][:, mask].unsqueeze(2)
                else:
                    local_clip_latent_3d = None
                    global_clip_latent_3d = None

                weights_0 = globals.densities[:, globals.valid[0]][:, mask].repeat(1, 1, 1, 2)
                weights_1 = densities[:, i, pixel_indices][:,fusion_indices_].repeat(1, 1, 1, 2)

                if self.cfg.openvocab_domain in ["ensemble", "2d"]:
                    local_codebook_caches = codebook_caches[:, i, pixel_indices][:,fusion_indices_]
                    local_codebook_weights = (weights_1[..., 1] / (weights_0[..., 1] + weights_1[..., 1]))[..., 0]
                    invalid = local_codebook_caches == -1
                    local_codebook_weights[:, invalid[0]] = 0.
                else:
                    local_codebook_caches = None
                    local_codebook_weights = None

                if use_gru:
                    if not concat_depth:
                        fusion_feat = self.gru(
                            local_latent,
                            global_latent,
                            input_weights_emb,
                            hidden_weights_emb
                        ).squeeze(2)

                        if self.cfg.openvocab_domain in ["ensemble", "3d"]:
                            clip_fusion_feat_3d = self.semantic_gru(
                                local_clip_latent_3d,
                                global_clip_latent_3d,
                                input_weights_emb,
                                hidden_weights_emb
                            ).squeeze(2)
                        else:
                            clip_fusion_feat_3d = None
                    else:
                        fusion_feat = self.gru(
                            torch.cat([local_latent, local_depth.view(1,-1,1,1)], dim=-1),
                            torch.cat([global_latent, global_depth.view(1,-1,1,1)], dim=-1),
                            input_weights_emb,
                            hidden_weights_emb
                        ).squeeze(2)

                        if self.cfg.openvocab_domain in ["ensemble", "3d"]:
                            clip_fusion_feat_3d = self.semantic_gru(
                                torch.cat([local_clip_latent_3d, local_depth.view(1,-1,1,1)], dim=-1),
                                torch.cat([global_clip_latent_3d, global_depth.view(1,-1,1,1)], dim=-1),
                                input_weights_emb,
                                hidden_weights_emb
                            ).squeeze(2)
                        else:
                            clip_fusion_feat_3d = None
                else:
                    fusion_feat = (local_latent * weights_1[...,:1] + global_latent * weights_0[...,:1]) / (weights_0[...,:1] + weights_1[...,:1])
                    fusion_feat = fusion_feat.squeeze(2)

                    if self.cfg.openvocab_domain in ["ensemble", "3d"]:
                        clip_fusion_feat_3d = (local_clip_latent_3d * weights_1[...,:1] + global_clip_latent_3d * weights_0[...,:1]) \
                                            / (weights_0[...,:1] + weights_1[...,:1])
                        clip_fusion_feat_3d = clip_fusion_feat_3d.squeeze(2)
                    else:
                        clip_fusion_feat_3d = None

                update_coords = (
                    global_coords[:, mask] * weights_0[...,1] + coords[:, i, pixel_indices] \
                    [:,fusion_indices_,0,0] * weights_1[...,1]) / (weights_0[...,1] + weights_1[...,1]
                )
                update_extrinsics = (
                    globals.extrinsics[:, globals.valid[0]][:, mask]*weights_0[...,:1] + extrinsics[:, i, None] \
                    * weights_1[...,:1]) / (weights_0[...,:1]+weights_1[...,:1]
                )
                update_depths = (
                    globals.depths[:, globals.valid[0]][:, mask] *weights_0[...,0,0] + \
                    depths[None, i, pixel_indices][:,fusion_indices_] \
                    * weights_1[...,0,0]) / (weights_0[...,0,0] + weights_1[...,0,0]
                )

                update_gaussians = rearrange(
                    self.to_gaussians(fusion_feat),
                    "... (srf c) -> ... srf c",
                    srf=self.cfg.num_surfaces,
                )

                update_gaussians = self.gaussian_adapter.forward(
                    rearrange(update_extrinsics, "b r i j -> b () r () () i j"),
                    repeat(intrinsics[:,0], "b i j -> b () N () () i j", N=update_gaussians.shape[1]),
                    None,
                    rearrange(update_depths, "b r -> b () r () ()"),
                    nn.Sigmoid()(rearrange(update_gaussians[..., :1], "b r srf c -> b () r srf c")),
                    rearrange(update_gaussians[..., 2:], "b r srf c -> b () r srf () c"),
                    (h, w),
                    load_depth=self.cfg.load_depth,
                    fusion=False,
                    coords=rearrange(update_coords, "b r c -> b () r () () c"),
                )

            new_extrinsics = extrinsics[:,i,None].repeat(1,(concat_mask).sum(),1,1)
            new_gaussians_feat = gaussians[:,i][:,concat_mask]
            if self.cfg.openvocab_domain in ["ensemble", "3d"]:
                new_clip_gaussians_feat_3d = clip_gaussians_3d_unet[:,concat_mask]
            else:
                new_clip_gaussians_feat_3d = None
            new_densities = densities[:,i][:,concat_mask]
            new_weight_emb = weight_emb[:,i][:,concat_mask]
            new_coords = coords[:,i][:,concat_mask,0,0]
            new_depths = depths[None,i][:,concat_mask]
            new_gaussians = rearrange(
                self.to_gaussians(gaussians[:,i][:,concat_mask]),
                "... (srf c) -> ... srf c",
                srf=self.cfg.num_surfaces,
            )

            if self.cfg.openvocab_domain in ["ensemble", "2d"]:
                new_codebook_caches = codebook_caches[:,i][:,concat_mask]
                new_codebook_weights = torch.ones_like(new_codebook_caches).to(self.device)
                invalid = new_codebook_caches == -1
                new_codebook_weights[:, invalid[0]] = 0.
            else:
                new_codebook_caches = None
                new_codebook_weights = None

            new_gaussians = self.gaussian_adapter.forward(
                rearrange(new_extrinsics, "b r i j -> b () r () () i j"),
                repeat(intrinsics[:,0], "b i j -> b () N () () i j", N=new_gaussians.shape[1]),
                None,
                rearrange(new_depths, "b r -> b () r () ()"),
                nn.Sigmoid()(rearrange(new_gaussians[..., :1], "b r srf c -> b () r srf c")),
                rearrange(new_gaussians[..., 2:], "b r srf c -> b () r srf () c"),
                (h, w),
                load_depth=self.cfg.load_depth,
                fusion=False,
                coords=rearrange(new_coords, "b r c -> b () r () () c"),
            )

            if mask.sum() > 0 and fusion:
                globals.append(
                    means=rearrange(
                        update_gaussians.means,
                        "b v r srf spp xyz -> b (v r srf spp) xyz",
                    ),
                    covariances=rearrange(
                        update_gaussians.covariances,
                        "b v r srf spp i j -> b (v r srf spp) i j",
                    ),
                    harmonics=rearrange(
                        update_gaussians.harmonics,
                        "b v r srf spp c d_sh -> b (v r srf spp) c d_sh",
                    ),
                    opacities=rearrange(
                        update_gaussians.opacities,
                        "b v r srf spp -> b (v r srf spp)",
                    ),
                    features=fusion_feat,
                    clip_features_3d=clip_fusion_feat_3d,
                    densities=globals.densities[:, globals.valid[0]][:, mask] + densities[:, i, pixel_indices][:,fusion_indices_],
                    weights=globals.weights[:, globals.valid[0]][:, mask] + weight_emb[:, i, pixel_indices][:,fusion_indices_],
                    coords=update_coords,
                    extrinsics=update_extrinsics,
                    depths=update_depths,
                    codebooks_idx=local_codebook_caches,
                    codebooks_weights=local_codebook_weights,
                    mask=mask,
                    scales=globals.scales[:, globals.valid[0]][:, mask] if export_ply else None,
                    rotations=globals.rotations[:, globals.valid[0]][:, mask] if export_ply else None,
                    fuse=True
                )

            globals.append(
                means=rearrange(
                    new_gaussians.means,
                    "b v r srf spp xyz -> b (v r srf spp) xyz",
                ),
                covariances=rearrange(
                    new_gaussians.covariances,
                    "b v r srf spp i j -> b (v r srf spp) i j",
                ),
                harmonics=rearrange(
                    new_gaussians.harmonics,
                    "b v r srf spp c d_sh -> b (v r srf spp) c d_sh",
                ),
                opacities=rearrange(
                    new_gaussians.opacities,
                    "b v r srf spp -> b (v r srf spp)",
                ),
                features=new_gaussians_feat,
                clip_features_3d=new_clip_gaussians_feat_3d,
                densities=new_densities,
                weights=new_weight_emb,
                coords=new_coords,
                extrinsics=new_extrinsics,
                depths=new_depths,
                codebooks_idx=new_codebook_caches,
                codebooks_weights=new_codebook_weights,
                mask=None,
                scales=rearrange(
                    new_gaussians.scales,
                    "b v r srf spp xyz -> b (v r srf spp) xyz",
                ) if export_ply else None,
                rotations=rearrange(
                    new_gaussians.rotations,
                    "b v r srf spp xyz -> b (v r srf spp) xyz",
                ) if export_ply else None,
            )

            if output_new_idx:
                new_mask = torch.ones(concat_mask.sum()).bool().to(mask.device)
                new_mask = torch.cat([mask, new_mask], dim=0)

        if output_new_idx:
            return globals, new_mask

        return globals
