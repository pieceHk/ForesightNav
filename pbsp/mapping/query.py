"""Semantic scoring and spatial candidate retrieval from fused Gaussian maps."""

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

from pbsp.settings import MappingSettings
from .geometry import mapper_world_to_habitat as es_world_to_habitat


@dataclass(frozen=True)
class ScoredPoint:
    position: np.ndarray
    score: float


def candidate_records(
    positions: Iterable[Iterable[float]],
    semantic_scores: Iterable[float],
    support_counts: Iterable[int],
    support_scores: Iterable[float],
    raw_ranks: Iterable[int],
    map_revision: int,
) -> list[dict[str, object]]:
    """Build the JSON-safe metadata sidecar for a query response."""

    columns = [
        list(positions),
        list(semantic_scores),
        list(support_counts),
        list(support_scores),
        list(raw_ranks),
    ]
    if len({len(column) for column in columns}) != 1:
        raise ValueError("candidate metadata columns must have equal length")
    values = list(zip(*columns))
    records: list[dict[str, object]] = []
    for position, semantic, support, support_score, raw_rank in values:
        point = np.asarray(position, dtype=np.float64)
        numbers = np.asarray([semantic, support_score], dtype=np.float64)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError("candidate position must be a finite xyz vector")
        if not np.isfinite(numbers).all():
            raise ValueError("candidate scores must be finite")
        records.append(
            {
                "position": point.tolist(),
                "raw_rank": int(raw_rank),
                "semantic_score": float(semantic),
                "support_count": int(support),
                "support_score": float(support_score),
                "map_revision": int(map_revision),
            }
        )
    return records


def spatially_supported(
    positions: np.ndarray,
    scores: np.ndarray,
    similarity_threshold: float,
    support_radius_m: float,
    minimum_support: int,
) -> list[ScoredPoint]:
    positions = np.asarray(positions, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3 or scores.shape != (len(positions),):
        raise ValueError("expected positions Nx3 and scores N")
    selected: list[ScoredPoint] = []
    horizontal = positions[:, (0, 2)]
    for index, score in enumerate(scores):
        if score < similarity_threshold:
            continue
        distance = np.linalg.norm(horizontal - horizontal[index], axis=1)
        if int((distance <= support_radius_m).sum()) >= minimum_support:
            selected.append(ScoredPoint(positions[index].astype(np.float32), float(score)))
    return selected


def greedy_nms(points: Iterable[ScoredPoint], radius_m: float, top_k: int) -> list[ScoredPoint]:
    ordered = sorted(points, key=lambda point: point.score, reverse=True)
    kept: list[ScoredPoint] = []
    for point in ordered:
        if all(np.linalg.norm(point.position[[0, 2]] - other.position[[0, 2]]) > radius_m for other in kept):
            kept.append(point)
            if len(kept) == top_k:
                break
    return kept


def _weighted_codebook_scores(feat_dict, ok_idx, ok_wts, text_feature):
    raw = feat_dict @ text_feature.T
    selected = raw[ok_idx]
    return (selected * ok_wts).sum(dim=1).squeeze(-1)


def _rerank_selected_by_support(selected, supports, semantic_scores, weight):
    if weight <= 0 or len(selected) <= 1:
        return selected, supports, semantic_scores, list(semantic_scores)
    log_support = np.log1p(np.asarray(supports, dtype=np.float32))
    span = float(log_support.max() - log_support.min())
    normalized = (
        (log_support - log_support.min()) / span
        if span > 0 else np.zeros_like(log_support)
    )
    combined = np.asarray(semantic_scores, dtype=np.float32) + weight * normalized
    order = np.argsort(-combined, kind='stable')
    return (
        [selected[int(i)] for i in order],
        [supports[int(i)] for i in order],
        [semantic_scores[int(i)] for i in order],
        [float(combined[int(i)]) for i in order],
    )


def select_spatial_candidates(ok_means, scores, top_k, *, settings: MappingSettings):
    min_similarity = settings.query_similarity
    min_support = settings.query_minimum_support
    support_radius = settings.query_support_radius_m
    nms_radius = settings.query_nms_radius_m
    pool_size = settings.query_pool_size
    support_rerank_weight = settings.query_support_rerank_weight

    eligible = torch.nonzero(scores >= min_similarity, as_tuple=False).flatten()
    if eligible.numel() == 0:
        return [], {
            'eligible': 0,
            'pool': 0,
            'selected': 0,
            'selected_supports': [],
            'selected_support_scores': [],
            'selected_scores': [],
            'selected_combined_scores': [],
            'selected_raw_ranks': [],
            'top_score': float(scores.max()) if scores.numel() else None,
        }

    eligible_scores = scores[eligible]
    order = torch.argsort(eligible_scores, descending=True)
    ordered_indices = eligible[order].cpu().numpy()
    raw_rank_by_index = {
        int(index): raw_rank for raw_rank, index in enumerate(ordered_indices)
    }
    selected = []
    selected_positions = []
    selected_supports = []
    selected_scores = []
    support_radius_sq = support_radius ** 2

    def run_nms(pool_indices):
        if len(pool_indices) == 0:
            return 0
        pool_positions = ok_means[pool_indices]
        pool_xz = pool_positions[:, [0, 2]]
        added = 0
        for index, position in zip(pool_indices, pool_positions):
            delta = pool_xz - position[[0, 2]]
            support = int(np.count_nonzero(
                np.sum(delta * delta, axis=1) <= support_radius_sq
            ))
            if support < min_support:
                continue
            if any(
                np.linalg.norm(position[[0, 2]] - other[[0, 2]]) < nms_radius
                for other in selected_positions
            ):
                continue
            selected.append(int(index))
            selected_positions.append(position)
            selected_supports.append(support)
            selected_scores.append(float(scores[int(index)]))
            added += 1
            if len(selected) >= top_k:
                break
        return added

    first_pool = ordered_indices[:pool_size]
    first_selected = run_nms(first_pool)

    second_pool = np.empty((0,), dtype=ordered_indices.dtype)
    second_selected = 0
    if len(selected) < top_k and len(ordered_indices) > len(first_pool):
        remaining = ordered_indices[len(first_pool):]
        remaining_positions = ok_means[remaining]
        if selected_positions:
            selected_xz = np.asarray(selected_positions)[:, [0, 2]]
            delta = (
                remaining_positions[:, None, [0, 2]]
                - selected_xz[None, :, :]
            )
            far_from_selected = np.all(
                np.linalg.norm(delta, axis=-1) >= nms_radius,
                axis=1,
            )
            remaining = remaining[far_from_selected]
        second_pool = remaining[:pool_size]
        second_selected = run_nms(second_pool)

    total_pool = len(first_pool) + len(second_pool)
    (
        selected,
        selected_supports,
        selected_scores,
        selected_combined_scores,
    ) = _rerank_selected_by_support(
        selected,
        selected_supports,
        selected_scores,
        support_rerank_weight,
    )
    log_support = np.log1p(np.asarray(selected_supports, dtype=np.float32))
    support_span = float(log_support.max() - log_support.min()) if len(log_support) else 0.0
    selected_support_scores = (
        ((log_support - log_support.min()) / support_span).tolist()
        if support_span > 0
        else [0.0] * len(selected_supports)
    )
    return selected, {
        'eligible': int(eligible.numel()),
        'pool': int(total_pool),
        'first_pool': int(len(first_pool)),
        'first_selected': int(first_selected),
        'second_pool': int(len(second_pool)),
        'second_selected': int(second_selected),
        'selected': len(selected),
        'selected_supports': selected_supports,
        'selected_support_scores': selected_support_scores,
        'selected_scores': selected_scores,
        'selected_combined_scores': selected_combined_scores,
        'selected_raw_ranks': [raw_rank_by_index[index] for index in selected],
        'top_score': (
            float(scores[int(ordered_indices[0])])
            if len(ordered_indices) else None
        ),
        'min_similarity': min_similarity,
        'min_support': min_support,
        'support_radius': support_radius,
        'nms_radius': nms_radius,
        'support_rerank_weight': support_rerank_weight,
    }


def _prepare_query_data(globals_state):
    valid = globals_state.valid[0].bool()
    if not valid.any():
        return None

    feat_list = globals_state.clip_features_2d['feat']
    idx_src = globals_state.clip_features_2d['idx']
    wts_src = globals_state.clip_features_2d['weight']
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
    return (
        feat_dict,
        means[ok],
        idx_t[0][ok].long(),
        wts_t[0][ok].unsqueeze(-1),
    )


def query_category(
    encoder, globals_state, category: str, top_k: int, device: str, *,
    settings: MappingSettings, return_records: bool = False, map_revision: int = 0,
):
    query_data = _prepare_query_data(globals_state)
    if query_data is None:
        return ([], []) if return_records else []
    feat_dict, ok_means, ok_idx, ok_wts = query_data

    prompt = f'a {category} in a photo.'
    with torch.no_grad():
        text_feature = encoder.clip.get_text_classifier([prompt], device=device)
        text_feature = F.normalize(text_feature.float(), dim=-1).cpu()
    scores = _weighted_codebook_scores(
        feat_dict, ok_idx, ok_wts, text_feature
    )
    selected, diagnostics = select_spatial_candidates(
        ok_means, scores, top_k, settings=settings
    )
    print(f'[ES_QUERY] category={category!r} {diagnostics}', flush=True)
    if not selected:
        return ([], []) if return_records else []
    positions = es_world_to_habitat(ok_means[np.asarray(selected)]).tolist()
    if not return_records:
        return positions
    records = candidate_records(
        positions,
        diagnostics['selected_scores'],
        diagnostics['selected_supports'],
        diagnostics['selected_support_scores'],
        diagnostics['selected_raw_ranks'],
        map_revision,
    )
    return positions, records


def query_categories(
    encoder, globals_state, categories: list, top_k: int, device: str, *,
    settings: MappingSettings, return_records: bool = False, map_revision: int = 0,
):
    """Query distinct categories using one prepared map feature dictionary.

    Return positions by category and optionally matching versioned records.
    """
    query_data = _prepare_query_data(globals_state)
    if query_data is None:
        empty = {cat: [] for cat in categories}
        return (empty, dict(empty)) if return_records else empty
    feat_dict, ok_means, ok_idx, ok_wts = query_data

    results = {}
    records_by_category = {}
    seen = {}
    for category in categories:
        if category in seen:
            results[category] = seen[category]
            continue

        prompt = f'a {category} in a photo.'
        with torch.no_grad():
            text_feature = encoder.clip.get_text_classifier([prompt], device=device)
            text_feature = F.normalize(text_feature.float(), dim=-1).cpu()
        scores = _weighted_codebook_scores(
            feat_dict, ok_idx, ok_wts, text_feature
        )
        selected, diagnostics = select_spatial_candidates(
            ok_means, scores, top_k, settings=settings
        )
        print(f'[ES_QUERY] category={category!r} {diagnostics}', flush=True)
        result = (
            es_world_to_habitat(ok_means[np.asarray(selected)]).tolist()
            if selected else []
        )

        results[category] = result
        seen[category] = result
        if return_records:
            records_by_category[category] = candidate_records(
                result,
                diagnostics['selected_scores'],
                diagnostics['selected_supports'],
                diagnostics['selected_support_scores'],
                diagnostics['selected_raw_ranks'],
                map_revision,
            )

    return (results, records_by_category) if return_records else results
