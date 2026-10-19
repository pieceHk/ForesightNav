"""Translate frozen-launcher environment controls while its worker is migrated."""

import os
from pathlib import Path

from pbsp.settings import MappingSettings
from .config import read_yaml


ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENT_FIELDS = {
    "keyframe_threshold": "ES_FRAME_CHANGE_RATIO",
    "pixel_delta": "ES_FRAME_PIXEL_DELTA",
    "warmup_keyframes": "ES_WARMUP_KEYFRAMES",
    "rebuild_interval": "ES_REBUILD_INTERVAL",
    "query_similarity": "ES_QUERY_MIN_SIMILARITY",
    "query_support_radius_m": "ES_QUERY_SUPPORT_RADIUS",
    "query_minimum_support": "ES_QUERY_MIN_SUPPORT",
    "query_nms_radius_m": "ES_QUERY_NMS_RADIUS",
    "query_pool_size": "ES_QUERY_POOL_SIZE",
    "query_support_rerank_weight": "ES_QUERY_SUPPORT_RERANK_WEIGHT",
    "raw_frame_buffer": "ES_RAW_FRAME_BUFFER",
    "frame_compare_height": "ES_FRAME_COMPARE_H",
    "frame_compare_width": "ES_FRAME_COMPARE_W",
    "max_gaussians": "ES_MAX_GAUSSIANS",
    "max_keyframes": "ES_MAX_KEYFRAMES",
    "offload_fastsam_after_rebuild": "ES_OFFLOAD_FASTSAM_AFTER_REBUILD",
    "clip_coverage_diagnostic": "ES_ENABLE_CLIP_COVERAGE_DIAGNOSTIC",
}


def mapping_settings_from_environment() -> MappingSettings:
    values = read_yaml(ROOT / "configs/methods/full_pbsp.yaml")["mapping"]
    for name, environment_name in ENVIRONMENT_FIELDS.items():
        if environment_name in os.environ:
            raw = os.environ[environment_name]
            values[name] = raw != "0" if type(values[name]) is bool else type(values[name])(raw)
    return MappingSettings(**values)


def model_specification():
    from pbsp.mapping.model import ModelInputSpec

    return ModelInputSpec(**read_yaml(ROOT / "configs/models/embodied_splat.yaml")["input"])


def benchmark_intrinsic():
    import math
    import numpy as np

    camera = read_yaml(ROOT / "configs/models/embodied_splat.yaml")["camera"]
    focal = camera["width"] / (2.0 * math.tan(math.radians(camera["hfov_degrees"] / 2.0)))
    return np.array([[focal / camera["width"], 0.0, 0.5],
                     [0.0, focal / camera["height"], 0.5], [0.0, 0.0, 1.0]], dtype=np.float32)


def create_mapper(*, device="cuda", top_k=10, warmup_n=None, rebuild_interval=None,
                  ckpt_path=None, cfg_path=None, local_pos_path=None, bev_dir=None):
    from dataclasses import replace
    from adapters.embodied_splat import load_model
    from experiments.visualization import MappingArtifacts, write_fusion_debug_render
    from pbsp.mapping.mapper import OnlineMapper
    from Mapping.runtime.server.runtime_memory import release_host_memory

    settings = mapping_settings_from_environment()
    overrides = {"top_k": top_k}
    if warmup_n is not None:
        overrides["warmup_keyframes"] = warmup_n
    if rebuild_interval is not None:
        overrides["rebuild_interval"] = rebuild_interval
    settings = replace(settings, **overrides)
    engine = Path(os.environ.get("ES_ROOT", ROOT / "Mapping/runtime/embodied_splat"))
    checkpoint = ckpt_path or os.environ.get("ES_CKPT", engine / "pretrained/freesplatpp_scannet_gtdepth.ckpt")
    work = Path(os.environ.get("ES_SNAP_LIVE", ROOT / "Exp/runs/mapping_tmp"))
    # cfg_path was accepted by the frozen wrapper; its loader selected this preset.
    del cfg_path, local_pos_path
    model = load_model(engine_root=engine, checkpoint=checkpoint, work_directory=work,
                       input_spec=model_specification(), device=device)
    model.encoder.fusion_debug_sink = write_fusion_debug_render
    artifacts = MappingArtifacts(
        final_directory=os.environ.get("ES_FINAL_BEV_DIR"), snapshot_directory=bev_dir,
        max_points=int(os.environ.get("ES_FINAL_BEV_MAX_POINTS", "200000")),
    )
    return OnlineMapper(settings=settings, model=model, device=device, artifacts=artifacts,
                        host_memory_hook=release_host_memory, camera_intrinsic=benchmark_intrinsic())
