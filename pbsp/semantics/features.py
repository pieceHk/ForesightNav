"""Region embedding and semantic-codebook construction derived from EmbodiedSplat.

Upstream attribution and license scope are recorded in THIRD_PARTY_NOTICES.md.
"""

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange


class SemanticFeatures:
    """Pool region embeddings and append stable semantic-codebook indices."""

    def extract_fastsam_clip_feats(self, images, caches, image_shape, globals):
        (h, w) = image_shape
        with torch.no_grad():
            if self.cfg.clip_model in ["lseg", "llava_ov"]:
                everything_results = [torch.load(cache[0], weights_only=False) for cache in caches]
            else:
                inputs = (images * 255.0).clone()
                everything_results = self.fastsam(
                    inputs,
                    device=self.device,
                    retina_masks=True,
                    imgsz=(w, h),
                    conf=self.cfg.fastsam_conf,
                    iou=self.cfg.fastsam_iou,
                )
                prompt = self.fastsam_prompt(inputs, everything_results, device=self.device)
            if self.cfg.clip_model == "maskclip":
                clip_feats = self.maskclip.encode_image(self.clip_preprocess(images))[1]
                clip_feats = rearrange(clip_feats, "b (h w) d -> b d h w", h=24, w=24)
                clip_feats = torch.nn.functional.interpolate(
                    clip_feats, size=(h, w), mode="bilinear"
                )
            elif self.cfg.clip_model == "maskadapter":
                clip_image = (images * 255.0 - self.PIXEL_MEAN) / self.PIXEL_STD
                x = self.clip.clip_model.visual.trunk.stem(clip_image.float())
                for i in range(4):
                    x = self.clip.clip_model.visual.trunk.stages[i](x)
                clip_feature = self.clip.clip_model.visual.trunk.norm_pre(x).contiguous()
                clip_feature_dense = self.clip.clip_model.visual.trunk.head.norm(clip_feature)
                clip_feature_dense = self.clip.clip_model.visual.trunk.head.drop(
                    clip_feature_dense.permute(0, 2, 3, 1)
                )
                clip_feature_dense = self.clip.clip_model.visual.head(clip_feature_dense).permute(
                    0, 3, 1, 2
                )
            codebook_caches = []
            clip_img_feats = []
            for idx, everything_result in enumerate(everything_results):
                if self.cfg.clip_model in ["lseg", "llava_ov"]:
                    format_results = everything_result["fastsam"][0]
                else:
                    (format_results, masks_torch) = prompt._format_results(
                        everything_result, 0, sort=True
                    )
                if self.cfg.clip_model == "maskclip":
                    clip_feats_flat = (
                        rearrange(clip_feats[idx], "d h w -> (h w) d").to(self.device).float()
                    )
                    masks_flat = rearrange(masks_torch, "n h w -> n (h w)").to(self.device).float()
                    pooled = masks_flat @ clip_feats_flat
                    denom = masks_flat.sum(dim=-1, keepdim=True).clamp(min=1e-06)
                    mask_feats = pooled / denom
                elif self.cfg.clip_model == "maskadapter":
                    masks_torch = masks_torch.to(clip_feature_dense.device).float()
                    semantic_activation_maps = self.maskadapter(
                        clip_feature_dense[idx : idx + 1], masks_torch[None]
                    )
                    maps_for_pooling = F.interpolate(
                        semantic_activation_maps,
                        size=clip_feature.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                    (B, C) = (
                        clip_feature[idx : idx + 1].size(0),
                        clip_feature[idx : idx + 1].size(1),
                    )
                    N = maps_for_pooling.size(1)
                    num_instances = N // 16
                    maps_for_pooling = F.softmax(
                        F.logsigmoid(maps_for_pooling).view(B, N, -1), dim=-1
                    )
                    pooled_clip_feature = torch.bmm(
                        maps_for_pooling,
                        clip_feature[idx : idx + 1].view(B, C, -1).permute(0, 2, 1),
                    )
                    (batch, num_query, channel) = pooled_clip_feature.shape
                    pooled_clip_feature = pooled_clip_feature.reshape(
                        batch * num_query, channel, 1, 1
                    )
                    x = self.clip.clip_model.visual.trunk.head(pooled_clip_feature)
                    x = self.clip.clip_model.visual.head(x)
                    mask_feats = x.reshape(B, num_instances, 16, -1).mean(dim=-2).contiguous()[0]
                elif self.cfg.clip_model == "openseg":
                    image = images[idx]
                    clip_feats = self.extract_openseg_img_feature(
                        image, self.openseg, self.openseg_text_emb, img_size=[h, w]
                    )
                    clip_feats_flat = (
                        rearrange(clip_feats, "h w d -> (h w) d").to(self.device).float()
                    )
                    masks_flat = rearrange(masks_torch, "n h w -> n (h w)").to(self.device).float()
                    pooled = masks_flat @ clip_feats_flat
                    denom = masks_flat.sum(dim=-1, keepdim=True).clamp(min=1e-06)
                    mask_feats = pooled / denom
                elif self.cfg.clip_model in ["lseg", "llava_ov"]:
                    mask_feats = everything_result["clip"][0].to(self.device)
                if self.cfg.openvocab_domain in ["ensemble", "2d"]:
                    globals.clip_features_2d["feat"].append(mask_feats)
                    globals.clip_features_2d["img_stamp"].append(
                        globals.clip_features_2d["img_stamp"][-1] + mask_feats.shape[0]
                    )
                    codebook_cache = torch.ones((h, w)).to(self.device) * -1
                if self.cfg.openvocab_domain in ["ensemble", "3d"]:
                    clip_img_feat = torch.zeros((h, w, mask_feats.shape[-1])).to(self.device)
                for jdx, (feat, mask) in enumerate(zip(mask_feats, format_results)):
                    if self.cfg.openvocab_domain in ["ensemble", "3d"]:
                        clip_img_feat[mask["segmentation"], :] = feat.float()
                    if self.cfg.openvocab_domain in ["ensemble", "2d"]:
                        codebook_cache[mask["segmentation"]] = (
                            jdx + globals.clip_features_2d["img_stamp"][-2]
                        )
                if self.cfg.openvocab_domain in ["ensemble", "3d"]:
                    clip_img_feats.append(clip_img_feat)
                if self.cfg.openvocab_domain in ["ensemble", "2d"]:
                    codebook_caches.append(codebook_cache)
            if self.cfg.openvocab_domain in ["ensemble", "3d"]:
                clip_img_feats = torch.nn.functional.normalize(
                    torch.stack(clip_img_feats), dim=-1, eps=1e-05
                )
                clip_img_feats = clip_img_feats.permute(0, 3, 1, 2)
            if self.cfg.openvocab_domain in ["ensemble", "2d"]:
                codebook_caches = torch.stack(codebook_caches, dim=0)
        if self.cfg.openvocab_domain == "ensemble":
            return (clip_img_feats, codebook_caches, globals)
        elif self.cfg.openvocab_domain == "2d":
            return (None, codebook_caches, globals)
        elif self.cfg.openvocab_domain == "3d":
            return (clip_img_feats, None, globals)

    def extract_sam_clip_feats(self, images, caches, image_shape, globals):
        (h, w) = image_shape
        with torch.no_grad():
            inputs = (images * 255.0).clone().permute(0, 2, 3, 1).cpu().numpy()
            masks_collate = []
            feats_collate = []
            for input in inputs:
                (seg_images, seg_maps) = self.sam.sam_encoder(input)
                tiles = seg_images["l"]
                tiles = tiles.to("cuda")
                tiles = tiles.reshape(-1, 3, 224, 224)
                clip_embed = self.clip.encode_image(tiles)
                clip_embed = clip_embed.reshape(clip_embed.shape[0] // 4, 4, 512)
                clip_embed = clip_embed.mean(dim=1)
                feats_collate.append(clip_embed)
                masks = []
                for mask_idx in np.unique(seg_maps["l"]):
                    if mask_idx == -1:
                        continue
                    masks.append(seg_maps["l"] == mask_idx)
                masks = torch.from_numpy(np.stack(masks)).to(self.device)
                masks_collate.append(masks)
            codebook_caches = []
            clip_img_feats = []
            for mask, feats in zip(masks_collate, feats_collate):
                if self.cfg.openvocab_domain in ["ensemble", "2d"]:
                    globals.clip_features_2d["feat"].append(feats)
                    globals.clip_features_2d["img_stamp"].append(
                        globals.clip_features_2d["img_stamp"][-1] + feats.shape[0]
                    )
                    codebook_cache = torch.ones((h, w)).to(self.device) * -1
                if self.cfg.openvocab_domain in ["ensemble", "3d"]:
                    clip_img_feat = torch.zeros((h, w, feats.shape[-1])).to(self.device)
                for jdx, (feat_, mask_) in enumerate(zip(feats, mask)):
                    if self.cfg.openvocab_domain in ["ensemble", "3d"]:
                        clip_img_feat[mask_, :] = feat_.float()
                    if self.cfg.openvocab_domain in ["ensemble", "2d"]:
                        codebook_cache[mask_] = jdx + globals.clip_features_2d["img_stamp"][-2]
                if self.cfg.openvocab_domain in ["ensemble", "3d"]:
                    clip_img_feats.append(clip_img_feat)
                if self.cfg.openvocab_domain in ["ensemble", "2d"]:
                    codebook_caches.append(codebook_cache)
            if self.cfg.openvocab_domain in ["ensemble", "3d"]:
                clip_img_feats = torch.nn.functional.normalize(
                    torch.stack(clip_img_feats), dim=-1, eps=1e-05
                )
                clip_img_feats = clip_img_feats.permute(0, 3, 1, 2)
            if self.cfg.openvocab_domain in ["ensemble", "2d"]:
                codebook_caches = torch.stack(codebook_caches, dim=0)
        if self.cfg.openvocab_domain == "ensemble":
            return (clip_img_feats, codebook_caches, globals)
        elif self.cfg.openvocab_domain == "2d":
            return (None, codebook_caches, globals)
        elif self.cfg.openvocab_domain == "3d":
            return (clip_img_feats, None, globals)
