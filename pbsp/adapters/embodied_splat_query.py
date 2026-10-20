"""Convert EmbodiedSplat storage to a portable query index and text encoder."""

from functools import partial

import numpy as np
import torch
import torch.nn.functional as F

from pbsp.mapping.geometry import mapper_world_to_habitat
from pbsp.query.service import SemanticQuery
from pbsp.query.types import FeatureSpace, SemanticIndex


class ClipTextEncoder:
    def __init__(self, classifier, feature_space):
        self.classifier = classifier
        self.feature_space = feature_space

    def encode(self, prompts, *, device):
        return self.classifier.get_text_classifier(prompts, device=device)


def read_gaussian_index(globals_state, *, feature_space=None):
    valid = globals_state.valid[0].bool()
    if not valid.any():
        return None

    feat_list = globals_state.clip_features_2d["feat"]
    idx_src = globals_state.clip_features_2d["idx"]
    wts_src = globals_state.clip_features_2d["weight"]
    means_src = globals_state.means
    idx_t = idx_src[:, valid.to(idx_src.device), :3].clone().cpu()
    wts_t = wts_src[:, valid.to(wts_src.device), :3].clone().cpu()
    means = means_src[:, valid.to(means_src.device), :][0].cpu().numpy()
    feat_dict = F.normalize(torch.cat(feat_list, dim=0).cpu().float(), dim=-1)

    bad = (idx_t != -1).sum(dim=-1)[0] == 0
    idx_t[idx_t == -1] = 0
    wts_t = wts_t / wts_t.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    ok = (~bad).cpu().numpy()
    if not np.any(ok):
        return None
    feature_space = feature_space or FeatureSpace(
        "legacy-clip", int(feat_dict.shape[-1])
    )
    return SemanticIndex(
        positions=mapper_world_to_habitat(means[ok]),
        codebook=feat_dict,
        indices=idx_t[0][ok].long(),
        weights=wts_t[0][ok].unsqueeze(-1),
        feature_space=feature_space,
    )


def create_query_backend(encoder, *, settings, device, feature_space=None):
    feature_space = feature_space or FeatureSpace(
        "embodiedsplat:" + str(encoder.cfg.clip_model), int(encoder.clip_dim)
    )
    return SemanticQuery(
        settings=settings,
        text_encoder=ClipTextEncoder(encoder.clip, feature_space),
        read_index=partial(read_gaussian_index, feature_space=feature_space),
        device=device,
    )
