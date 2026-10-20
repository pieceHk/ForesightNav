"""Construct EmbodiedSplat layers and provide optional sparse/OpenSeg backends.

The encoder arithmetic is implemented in pbsp.mapping. The layer definitions
remain in the pinned EmbodiedSplat dependency. See THIRD_PARTY_NOTICES.md.
"""

from pbsp.mapping.encoder import OnlineEncoder


import torch
from torch import nn
from collections import OrderedDict
import numpy as np
from pathlib import Path

from src.dataset.shims.patch_shim import apply_patch_shim
from src.dataset.types import BatchedExample, DataShim
from src.model.encoder.backbone import Backbone, get_backbone, BackboneMultiview
from src.model.encoder.common.gaussian_adapter import GaussianAdapter
from src.model.encoder.costvolume.depth_predictor_multiview import DepthPredictorMultiView
from src.model.encoder.epipolar.depth_predictor_monocular import DepthPredictorMonocular
from src.model.encoder.epipolar.epipolar_transformer import EpipolarTransformer
from src.global_cfg import get_cfg
import os


from src.model.encoder.common.keypoint_scorer import ContextScorer
from src.model.encoder.adapter.cascade_gaussian_adapter import CascadeGaussianAdapter
from src.model.encoder.adapter.gaussian_refiner import IterativeGaussianRefiner

from modules.networks import CVEncoder, DepthDecoderPP, ResnetMatchingEncoder
from modules.cost_volume import AVGFeatureVolumeManager
import timm
from modules.layers import TensorFormatter
import torchvision.transforms as tf

from src.model.encoder.attention.transformer import GRU2D_naive_Wweights


from src.model.encoder.encoder_config import EncoderEpipolarCfg

import math


class EmbodiedSplatEncoder(OnlineEncoder):
    backbone: Backbone
    backbone_projection: nn.Sequential
    epipolar_transformer: EpipolarTransformer | None
    depth_predictor: DepthPredictorMonocular
    to_gaussians: nn.Sequential
    gaussian_adapter: GaussianAdapter
    high_resolution_skip: nn.Sequential

    def __init__(self, cfg: EncoderEpipolarCfg, depth_range=[0.5, 15.0], *, semantic_injector=None, model_resources=None) -> None:
        super().__init__(cfg, semantic_injector=semantic_injector)
        activation_func = nn.ReLU()
        self.to_tensor = tf.ToTensor()

        self.depth_range = depth_range

        self.gaussian_adapter = GaussianAdapter(cfg.gaussian_adapter)

        if cfg.backbone.name == "dino":
            self.backbone = get_backbone(cfg.backbone, 3 + cfg.load_depth)
            self.backbone_projection = nn.Sequential(
                activation_func,
                nn.Linear(self.backbone.d_out, cfg.d_feature),
            )
            if cfg.use_epipolar_transformer:
                self.epipolar_transformer = EpipolarTransformer(
                    cfg.epipolar_transformer,
                    cfg.d_feature,
                    n_views=cfg.max_batch,
                )
            else:
                self.epipolar_transformer = None
            if cfg.est_depth == "est":
                self.depth_predictor = DepthPredictorMonocular(
                    cfg.d_feature,
                    cfg.num_monocular_samples,
                    cfg.num_surfaces,
                    cfg.use_transmittance,
                )
            else:
                self.opacity_mlp = nn.Sequential(
                    activation_func,
                    nn.Linear(cfg.d_feature, 1),
                    nn.Sigmoid(),
                )
                self.depth_refine = nn.Sequential(
                    activation_func,
                    nn.Linear(
                        cfg.d_feature,
                        cfg.d_feature,
                    ),
                    activation_func,
                    nn.Linear(
                        cfg.d_feature,
                        1,
                    ),
                )
            if cfg.predict_opacity:
                self.to_opacity = nn.Sequential(
                    activation_func,
                    nn.Linear(cfg.d_feature, 1),
                    nn.Sigmoid(),
                )
            self.to_gaussians = nn.Sequential(
                activation_func,
                nn.Linear(
                    cfg.d_feature,
                    cfg.num_surfaces * (2 + self.gaussian_adapter.d_in),
                ),
            )

            self.high_resolution_skip = nn.Sequential(
                nn.Conv2d(3 + self.cfg.load_depth, cfg.d_feature, 7, 1, 3),
                activation_func,
            )
        elif cfg.backbone.name == "cost_volume":
            self.backbone = BackboneMultiview(
                feature_channels=cfg.d_feature,
                downscale_factor=cfg.downscale_factor,
                no_cross_attn=cfg.wo_backbone_cross_attn,
                use_epipolar_trans=cfg.use_epipolar_trans,
                limit=cfg.backbone_limit,
            )
            ckpt_path = cfg.unimatch_weights_path
            if get_cfg().mode == "train":
                if cfg.unimatch_weights_path is None:
                    print("==> Init multi-view transformer backbone from scratch")
                else:
                    print("==> Load multi-view transformer backbone checkpoint: %s" % ckpt_path)
                    unimatch_pretrained_model = torch.load(ckpt_path)["model"]
                    updated_state_dict = OrderedDict(
                        {
                            k: v
                            for k, v in unimatch_pretrained_model.items()
                            if k in self.backbone.state_dict()
                        }
                    )
                    is_strict_loading = not cfg.wo_backbone_cross_attn
                    self.backbone.load_state_dict(updated_state_dict, strict=is_strict_loading)

            # gaussians convertor
            self.gaussian_adapter = GaussianAdapter(cfg.gaussian_adapter)

            # cost volume based depth predictor
            self.depth_predictor = DepthPredictorMultiView(
                feature_channels=cfg.d_feature,
                upscale_factor=cfg.downscale_factor,
                num_depth_candidates=cfg.num_depth_candidates,
                costvolume_unet_feat_dim=cfg.costvolume_unet_feat_dim,
                costvolume_unet_channel_mult=tuple(cfg.costvolume_unet_channel_mult),
                costvolume_unet_attn_res=tuple(cfg.costvolume_unet_attn_res),
                gaussian_raw_channels=cfg.num_surfaces * (self.gaussian_adapter.d_in + 2),
                gaussians_per_pixel=cfg.gaussians_per_pixel,
                # num_views=get_cfg().dataset.view_sampler.num_context_views,
                num_views=min(get_cfg().dataset.view_sampler.num_context_views, cfg.max_batch),
                depth_unet_feat_dim=cfg.depth_unet_feat_dim,
                depth_unet_attn_res=cfg.depth_unet_attn_res,
                depth_unet_channel_mult=cfg.depth_unet_channel_mult,
                wo_depth_refine=cfg.wo_depth_refine,
                wo_cost_volume=cfg.wo_cost_volume,
                wo_cost_volume_refine=cfg.wo_cost_volume_refine,
            )

            if self.cfg.adaptive_gaussian:
                self.keypoint_scorer = ContextScorer(
                    channels=cfg.score_channels,
                    num_layers=cfg.num_layers,
                    max_num_view=cfg.max_num_view,
                )

                self.cascade_gaussian_adapter = CascadeGaussianAdapter(
                    stages=cfg.cga_stages,
                    opacity_thres=cfg.opacity_thres,
                    split_count=cfg.split_count,
                    scaling_factor=cfg.scaling_factor,
                    opacity_factor=cfg.opacity_factor,
                    num_groups=cfg.cga_num_groups,
                    max_num_view=cfg.max_num_view,
                    num_levels=cfg.cga_num_levels,
                    attn_drop=cfg.attn_drop,
                    num_learnable_pts=cfg.num_learnable_pts,
                    fix_scale=cfg.fix_scale,
                    num_anchors=cfg.cga_num_anchors,
                    score_embed=cfg.score_embed,
                )

                self.iterative_refiner = IterativeGaussianRefiner(
                    stages=cfg.igr_stages,
                    num_groups=cfg.igr_num_groups,
                    num_levels=cfg.igr_num_levels,
                    attn_drop=cfg.attn_drop,
                    max_num_view=cfg.max_num_view,
                    num_learnable_pts=cfg.num_learnable_pts,
                    fix_scale=cfg.fix_scale,
                    num_anchors=cfg.igr_num_anchors,
                    embed_dim=cfg.d_feature,
                )
        else:
            effv2_weight = Path(model_resources.efficientnet) if model_resources else Path(
                os.environ.get(
                    "ES_TIMM_EFFV2_WEIGHTS",
                    "pretrained/tf_efficientnetv2_s_21ft1k-d7dafa41.pth",
                )
            )
            pretrained_kwargs = {}
            if effv2_weight.exists():
                pretrained_kwargs["pretrained_cfg_overlay"] = {"file": str(effv2_weight)}
            self.backbone = timm.create_model(
                "tf_efficientnetv2_s_in21ft1k",
                pretrained=effv2_weight.exists(),
                features_only=True,
                **pretrained_kwargs,
            )

            self.backbone.num_ch_enc = self.backbone.feature_info.channels()

            for name, module in self.backbone.named_modules():
                if isinstance(module, torch.nn.BatchNorm2d):
                    module.momentum = 0.01  # A lower value than the typical 0.1

            if cfg.use_epipolar_transformer:
                self.epipolar_transformer = EpipolarTransformer(
                    cfg.epipolar_transformer,
                    self.backbone.feature_info.channels()[0],
                    n_views=cfg.max_batch,
                )
            else:
                self.epipolar_transformer = None

            self.high_resolution_skip = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(3 + self.cfg.load_depth, 64, 7, 1, 3),
                        activation_func,
                    ),
                    nn.Sequential(
                        nn.Conv2d(3 + self.cfg.load_depth, 64, 6, 2, 2),
                        activation_func,
                    ),
                    nn.Sequential(
                        nn.Conv2d(3 + self.cfg.load_depth, 64, 8, 4, 2),
                        activation_func,
                    ),
                    nn.Sequential(
                        nn.Conv2d(3 + self.cfg.load_depth, 64, 16, 8, 4),
                        activation_func,
                    ),
                    nn.Sequential(
                        nn.Conv2d(3 + self.cfg.load_depth, 64, 32, 16, 8),
                        activation_func,
                    ),
                ]
            )

            self.to_gaussians = nn.Sequential(
                activation_func,
                nn.Linear(
                    64,
                    cfg.num_surfaces * (2 + self.gaussian_adapter.d_in),
                ),
            )

        self.gausisans_ch = cfg.num_surfaces * (2 + self.gaussian_adapter.d_in)

        self.load_depth = cfg.load_depth
        self.est_depth = cfg.est_depth

        if self.cfg.est_depth == "cost":
            if not self.cfg.wo_cost_volume:
                if not self.cfg.wo_matchnet:
                    self.matching_net = ResnetMatchingEncoder(
                        self.cfg.matchnet_type,
                        self.cfg.matchnet_dim,
                    )

                self.cost_volume = AVGFeatureVolumeManager(
                    matching_height=self.cfg.image_H // 4,
                    matching_width=self.cfg.image_W // 4,
                    num_depth_bins=self.cfg.num_depth_candidates,
                    matching_dim_size=self.cfg.matchnet_dim if (not self.cfg.wo_matchnet) else 48,
                    num_source_views=self.cfg.num_views - 1,
                    log_plane=self.cfg.log_cv,
                )

                if not self.cfg.wo_msd:
                    self.cv_encoder = CVEncoder(
                        num_ch_cv=self.cfg.num_depth_candidates,
                        num_ch_enc=self.backbone.num_ch_enc[1:],
                        num_ch_outs=[64, 128, 256, 384],
                    )
                    dec_num_input_ch = self.backbone.num_ch_enc[:1] + self.cv_encoder.num_ch_enc
                else:
                    dec_num_input_ch = (
                        self.backbone.num_ch_enc[:1]
                        + [self.backbone.num_ch_enc[1] + self.cfg.num_depth_candidates]
                        + self.backbone.num_ch_enc[2:]
                    )
            else:
                dec_num_input_ch = self.backbone.num_ch_enc

            self.depth_decoder = DepthDecoderPP(
                dec_num_input_ch,
                num_output_channels=1 + 64,
                n_levels=self.cfg.n_levels,
                use_planes=self.cfg.use_planes,
                near=depth_range[0],
                far=depth_range[1],
                num_samples=self.cfg.num_depth_candidates,
                log_plane=self.cfg.log_plane,
                wo_msd=self.cfg.wo_msd,
                refine=self.cfg.depth_refine,
                num_context_views=self.cfg.num_views,
                low_res=self.cfg.low_res,
            )
            self.max_depth = 2 + 2 * (not self.cfg.wo_msd)
            self.tensor_formatter = TensorFormatter()

            if self.cfg.fusion:
                self.weight_embedding = nn.Sequential(
                    nn.Linear(2, 12),
                    activation_func,
                    nn.Linear(12, 12),
                )
                self.gru = GRU2D_naive_Wweights(concat_depth=self.cfg.concat_depth)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if semantic_injector is None and not self.cfg.load_cache:
            if self.cfg.sam_model_type == "sam":
                from hiersam import Tools

                self.sam = Tools(
                    points_per_side=32,
                    pred_iou_thresh=0.7,
                    box_nms_thresh=0.7,
                    stability_score_thresh=0.85,
                    crop_n_layers=1,
                    crop_n_points_downscale_factor=1,
                    min_mask_region_area=100,
                    load_sam="pretrained/sam_vit_h_4b8939.pth",
                    load_tracker="pretrained/sam2.1_hiera_base_plus.pt",
                )
            elif self.cfg.sam_model_type == "fastsam":
                from fastsam import FastSAM, FastSAMPrompt

                self.fastsam = FastSAM(str(model_resources.fastsam) if model_resources else self.cfg.fastsam_model_path)
                self.fastsam_prompt = FastSAMPrompt

            if self.cfg.clip_model == "clip":
                from open_clip_network import OpenCLIPNetwork, OpenCLIPNetworkConfig

                self.clip = OpenCLIPNetwork(OpenCLIPNetworkConfig)
                self.clip_process = tf.Compose(
                    [
                        tf.Normalize(
                            mean=[0.48145466, 0.4578275, 0.40821073],
                            std=[0.26862954, 0.26130258, 0.27577711],
                        ),
                    ]
                )
            elif self.cfg.clip_model == "openseg":
                from pbsp.adapters.openseg import tensorflow_runtime

                tff2, tff = tensorflow_runtime()
                saved_model_path = self.cfg.clip_model_path
                self.openseg = tff2.saved_model.load(
                    saved_model_path,
                    tags=[tff.saved_model.tag_constants.SERVING],
                )
                self.openseg_text_emb = tff.zeros([1, 1, 768])
            elif self.cfg.clip_model == "maskadapter":
                from maskadapter.mask_adapter_head import MASKAdapterHead as MaskAdapter
                from maskadapter.clip import CLIP

                self.maskadapter = MaskAdapter(checkpoint=model_resources.maskadapter) if model_resources else MaskAdapter()
                self.clip = CLIP(
                    model_name=self.cfg.clip_model_type, pretrained=self.cfg.clip_model_path
                )

                self.PIXEL_MEAN = (
                    torch.tensor([122.7709383, 116.7460125, 104.09373615])
                    .view(1, -1, 1, 1)
                    .to(self.device)
                )
                self.PIXEL_STD = (
                    torch.tensor([68.5005327, 66.6321579, 70.32316305])
                    .view(1, -1, 1, 1)
                    .to(self.device)
                )

        self.clip_dim = self.cfg.clip_dim

        if self.cfg.openvocab_domain in ["ensemble", "3d"]:
            if self.cfg.semantic_early_fusion:
                self.semantic_proj = nn.Sequential(
                    nn.Conv2d(in_channels=self.clip_dim, out_channels=64, kernel_size=1),
                    activation_func,
                )

            self.semantic_gru = GRU2D_naive_Wweights(concat_depth=self.cfg.concat_depth)

            self.to_semantic = nn.Sequential(
                activation_func,
                nn.Linear(64, self.clip_dim),
            )

            if self.cfg.unet_3d:
                from src.model.encoder.backbone.mink_unet import mink_unet

                self.unet3d = mink_unet(
                    in_channels=self.cfg.unet3d_in_channels,
                    out_channels=self.cfg.unet3d_out_channels,
                    D=self.cfg.unet3d_D,
                    arch=self.cfg.unet3d_arch,
                )

                if self.cfg.pcd_aug:
                    self.SCALE_AUGMENTATION_BOUND = (0.9, 1.1)
                    self.ROTATION_AUGMENTATION_BOUND = (
                        (-np.pi / 64, np.pi / 64),
                        (-np.pi / 64, np.pi / 64),
                        (-np.pi, np.pi),
                    )

                    self.M_v, self.M_r = None, None

            if self.cfg.memory:
                from src.model.encoder.backbone.multilevel_memory import MultilevelMemory

                self.memory = MultilevelMemory(
                    in_channels=self.cfg.memory_in_channels,
                    queue=self.cfg.memory_queue,
                    vmp_layer=self.cfg.memory_vmp_layer,
                )
            else:
                self.memory = None

            if self.cfg.non_object_embedding:
                self.non_object_embedding = nn.Parameter(torch.empty(1, self.clip_dim))
                torch.nn.init.kaiming_uniform_(self.non_object_embedding, a=math.sqrt(5))
                self.non_object_embedding.requires_grad = True

        if semantic_injector is None:
            from pbsp.adapters.embodied_splat_semantics import EmbodiedSplatSemanticInjector

            self.semantic_injector = EmbodiedSplatSemanticInjector(self)

    def backbone_3d(self, coords, feats, voxelization_matrix, rotation_matrix, globals=None):
        import MinkowskiEngine as ME

        if globals is not None:
            global_coords = globals.means[:, globals.valid[0]][0]
            global_feats = globals.clip_features_3d[:, globals.valid[0]][0]

            global_homo_coords = torch.cat(
                [global_coords, torch.ones(global_coords.shape[0], 1).to(self.device)], dim=-1
            )

            rigid_transformation = rotation_matrix @ voxelization_matrix
            global_voxel_coords = torch.floor(global_homo_coords @ rigid_transformation.T[:, :3])

            voxelization_dict = {
                "return_index": True,
                "return_inverse": True,
                "coordinates": global_voxel_coords.contiguous(),
                "features": global_feats,
            }

            _, _, unique_map, _ = ME.utils.sparse_quantize(**voxelization_dict)

            input_dict = {
                "coords": [global_voxel_coords[unique_map].int()],
                "feats": [global_feats[unique_map]],
            }
            coordinates, features = ME.utils.sparse_collate(**input_dict)
            accumulated_feats = ME.SparseTensor(
                coordinates=coordinates,
                features=features,
                device=self.device,
            )
        else:
            accumulated_feats = None

        homo_coords = torch.cat([coords, torch.ones(coords.shape[0], 1).to(self.device)], dim=-1)

        rigid_transformation = rotation_matrix @ voxelization_matrix
        voxel_coords = torch.floor(homo_coords @ rigid_transformation.T[:, :3])

        voxelization_dict = {
            "return_index": True,
            "return_inverse": True,
            "coordinates": voxel_coords.contiguous(),
            "features": feats,
        }

        _, _, unique_map, inverse_map = ME.utils.sparse_quantize(**voxelization_dict)

        input_dict = {"coords": [voxel_coords[unique_map].int()], "feats": [feats[unique_map]]}
        coordinates, features = ME.utils.sparse_collate(**input_dict)
        sinput = ME.SparseTensor(
            coordinates=coordinates,
            features=features,
            device=self.device,
        )

        global_clip_gaussians_feat_3d = self.unet3d(sinput, self.memory, accumulated_feats)
        global_clip_gaussians_feat_3d = global_clip_gaussians_feat_3d.decomposed_features[0][None]

        return global_clip_gaussians_feat_3d[:, inverse_map, :]

    def extract_openseg_img_feature(
        self, img, openseg_model, text_emb, img_size=None, regional_pool=True
    ):
        """Extract per-pixel OpenSeg features."""
        from pbsp.adapters.openseg import tensorflow_runtime

        _, tff = tensorflow_runtime()

        def array_to_bytes(array):
            assert array.ndim == 3 and array.shape[2] == 3, "Expected shape (H, W, 3)"
            assert array.dtype in [np.float32, np.float64], "Array must be float"

            uint8_img = tff.image.convert_image_dtype(array, dtype=tff.uint8, saturate=True)
            png_bytes = tff.io.encode_png(uint8_img).numpy()
            return png_bytes

        np_image_string = array_to_bytes(img.permute(1, 2, 0).cpu().numpy())

        results = openseg_model.signatures["serving_default"](
            inp_image_bytes=tff.convert_to_tensor(np_image_string), inp_text_emb=text_emb
        )
        img_info = results["image_info"]
        crop_sz = [int(img_info[0, 0] * img_info[2, 0]), int(img_info[0, 1] * img_info[2, 1])]
        if regional_pool:
            image_embedding_feat = results["ppixel_ave_feat"][:, : crop_sz[0], : crop_sz[1]]
        else:
            image_embedding_feat = results["image_embedding_feat"][:, : crop_sz[0], : crop_sz[1]]
        if img_size is not None:
            feat_2d = tff.cast(
                tff.image.resize_nearest_neighbor(
                    image_embedding_feat, img_size, align_corners=True
                )[0],
                dtype=tff.float16,
            ).numpy()
        else:
            feat_2d = tff.cast(image_embedding_feat[[0]], dtype=tff.float16).numpy()

        feat_2d = torch.from_numpy(feat_2d)

        return feat_2d

    def get_data_shim(self) -> DataShim:
        def data_shim(batch: BatchedExample) -> BatchedExample:
            batch = apply_patch_shim(
                batch,
                patch_size=self.cfg.epipolar_transformer.self_attention.patch_size
                * self.cfg.epipolar_transformer.downscale,
            )

            return batch

        return data_shim

    @property
    def sampler(self):
        # hack to make the visualizer work
        return self.epipolar_transformer.epipolar_sampler
