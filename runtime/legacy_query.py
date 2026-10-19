"""Translate retained launch controls into independent retrieval settings."""

import os
from pathlib import Path

from pbsp.settings import QuerySettings
from runtime.config import read_yaml


ENVIRONMENT_FIELDS = {
    "query_similarity": "ES_QUERY_MIN_SIMILARITY",
    "query_support_radius_m": "ES_QUERY_SUPPORT_RADIUS",
    "query_minimum_support": "ES_QUERY_MIN_SUPPORT",
    "query_nms_radius_m": "ES_QUERY_NMS_RADIUS",
    "query_pool_size": "ES_QUERY_POOL_SIZE",
    "query_support_rerank_weight": "ES_QUERY_SUPPORT_RERANK_WEIGHT",
}


def query_settings_from_environment():
    values = read_yaml(
        Path(__file__).resolve().parents[1] / "configs/methods/full_pbsp.yaml"
    )["query"]
    for field, variable in ENVIRONMENT_FIELDS.items():
        if variable in os.environ:
            values[field] = type(values[field])(os.environ[variable])
    return QuerySettings(**values)
