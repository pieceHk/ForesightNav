"""Frozen launcher compatibility for the canonical PBSP online mapper."""

from pbsp.mapping.geometry import habitat_pose_to_c2w as globalpose_to_c2w
from pbsp.mapping.geometry import mapper_world_to_habitat as es_world_to_habitat
from pbsp.mapping.keyframes import tensor_frame_change_ratio
from pbsp.mapping.model import preprocess_frame as _preprocess_frame
from pbsp.query.engine import select_spatial_candidates
from adapters.embodied_splat_query import legacy_query_category as query_category, legacy_query_categories as query_categories
from runtime.legacy_query import query_settings_from_environment
from runtime.legacy_mapping import (
    benchmark_intrinsic, create_mapper, mapping_settings_from_environment, model_specification,
)


def OnlineESMapper(*args, **kwargs):
    if args:
        raise TypeError("OnlineESMapper requires explicit keyword arguments")
    return create_mapper(**kwargs)


def preprocess_frame(rgb, depth, c2w, device):
    return _preprocess_frame(rgb, depth, c2w, benchmark_intrinsic(),
                             input_spec=model_specification(), device=device)


def frame_change_ratio(image, reference):
    return tensor_frame_change_ratio(image, reference, settings=mapping_settings_from_environment())


def _select_spatial_candidates(ok_means, scores, top_k):
    return select_spatial_candidates(ok_means, scores, top_k, settings=query_settings_from_environment())


def clip_query_globals(encoder, globals_state, category, top_k, device,
                       return_records=False, map_revision=0):
    return query_category(encoder, globals_state, category, top_k, device,
                          settings=query_settings_from_environment(),
                          return_records=return_records, map_revision=map_revision)


def clip_query_batch_globals(encoder, globals_state, categories, top_k, device,
                             return_records=False, map_revision=0):
    return query_categories(encoder, globals_state, categories, top_k, device,
                            settings=query_settings_from_environment(),
                            return_records=return_records, map_revision=map_revision)
