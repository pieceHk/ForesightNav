"""Compatibility exports; semantic retrieval lives in pbsp.query."""

from pbsp.query.engine import (
    ScoredPoint, candidate_records, greedy_nms, spatially_supported,
    select_spatial_candidates, query_category, query_categories,
)
