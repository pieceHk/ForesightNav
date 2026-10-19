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
}


def mapping_settings_from_environment() -> MappingSettings:
    values = read_yaml(ROOT / "configs/methods/full_pbsp.yaml")["mapping"]
    for name, environment_name in ENVIRONMENT_FIELDS.items():
        if environment_name in os.environ:
            values[name] = type(values[name])(os.environ[environment_name])
    return MappingSettings(**values)
