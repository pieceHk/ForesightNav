#!/usr/bin/env python3
"""
online_es_mapper.py  — Phase 5 Online Incremental ES Mapper

Architecture
============
- Keeps accepted keyframes on CPU and only moves a transient context to CUDA.
- Admits a frame when at least 30% of sampled RGB pixels differ from the
  previously accepted keyframe.
- Background rebuilds incrementally fuse newly accepted keyframes.
- Published globals and persistent online state live on CPU between rebuilds.

Coordinate conventions
======================
globalpose (Habitat): [qw, qx, qy, qz, px, py, pz]
ES extrinsic (c2w):   built via snapshot_mapper.py convention:
  x,y,z,w = qw,qx,qy,qz  (w-last quaternion formula → correct c2w)
  R = rotation_matrix(x,y,z,w);  c2w[:3,:3] = R @ _RUB2RDF
  c2w[:3,3] = _RUB2RDF @ [px,py,pz]

Intrinsic (normalized ES format):
  K_norm[0,0] = fx/W,  K_norm[1,1] = fy/H
  K_norm[0,2] = cx/W,  K_norm[1,2] = cy/H
  for HFOV=79°, H=480, W=640:
    fx = fy = W / (2*tan(HFOV/2 * pi/180)) ≈ 388.19
    K_norm ≈ [[0.6066, 0, 0.5], [0, 0.8087, 0.5], [0, 0, 1]]
"""

import sys, os, time, threading, math, collections
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from runtime_memory import release_host_memory
from pbsp.mapping.query import candidate_records

# ── ES paths ──────────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_INTEG_ROOT = _SCRIPT_DIR.parent
_REPO_ROOT = _INTEG_ROOT.parent
ES_ROOT   = os.environ.get('ES_ROOT', str(_REPO_ROOT / 'EmbodiedSplat'))
ES_CKPT   = os.environ.get('ES_CKPT', f'{ES_ROOT}/pretrained/freesplatpp_scannet_gtdepth.ckpt')
ES_CFG    = os.environ.get('ES_CFG', f'{ES_ROOT}/config/experiment/scannet/embodiedsplat_fast_test.yaml')
SNAP_LIVE = os.environ.get(
    'ES_SNAP_LIVE',
    str(_INTEG_ROOT / 'tmp' / 'es_p5_live'),
)  # kept for encoder API compatibility; live-only emits no tensor checkpoints
_FINAL_BEV_DIR = os.environ.get('ES_FINAL_BEV_DIR', '').strip()
_FINAL_BEV_MAX_POINTS = int(os.environ.get('ES_FINAL_BEV_MAX_POINTS', '200000'))

for p in [ES_ROOT, f'{ES_ROOT}/src/third_party', f'{ES_ROOT}/src']:
    if p not in sys.path:
        sys.path.insert(0, p)

# ── Camera constants (Habitat GOAT benchmark) ─────────────────────────────────
_H        = 480
_W        = 640
_HFOV_DEG = 79.0
_fx = _W  / (2.0 * math.tan(math.radians(_HFOV_DEG / 2.0)))
_fy = _fx
_K_NORM = np.array([
    [_fx / _W, 0.0,       0.5],
    [0.0,      _fy / _H,  0.5],
    [0.0,      0.0,       1.0],
], dtype=np.float32)

# Habitat camera → OpenCV camera coordinate convention
_RUB2RDF = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)

# A non-positive value disables the legacy single-GPU safety ceiling. Mapping
# then continues for the full episode and remains bounded by keyframe admission.
_MAX_GAUSSIANS = int(os.environ.get('ES_MAX_GAUSSIANS', '0'))
_MAX_KEYFRAMES = int(os.environ.get('ES_MAX_KEYFRAMES', '0'))
_FRAME_CHANGE_RATIO = float(os.environ.get('ES_FRAME_CHANGE_RATIO', '0.30'))
_FRAME_PIXEL_DELTA = float(os.environ.get('ES_FRAME_PIXEL_DELTA', '0.15'))
_FRAME_COMPARE_H = int(os.environ.get('ES_FRAME_COMPARE_H', '48'))
_FRAME_COMPARE_W = int(os.environ.get('ES_FRAME_COMPARE_W', '64'))
_RAW_FRAME_BUFFER = int(os.environ.get('ES_RAW_FRAME_BUFFER', '2'))
_OFFLOAD_FASTSAM_AFTER_REBUILD = os.environ.get('ES_OFFLOAD_FASTSAM_AFTER_REBUILD', '0') != '0'
_ENABLE_CLIP_COVERAGE_DIAGNOSTIC = (
    os.environ.get('ES_ENABLE_CLIP_COVERAGE_DIAGNOSTIC', '0') != '0'
)

# ES near/far (matches scannet config)
_NEAR = 0.5
_FAR  = 15.0

# ES encoder expected input resolution (from embodiedsplat_fast_test.yaml)
_ES_H = 384
_ES_W = 512


# ── Coordinate helpers ────────────────────────────────────────────────────────

def globalpose_to_c2w(globalpose: np.ndarray) -> np.ndarray:
    """Convert Habitat obs.globalpose → ES camera-to-world 4×4 (float32).

    Matches snapshot_mapper.py convention exactly so live frames and
    offline snapshots share the same coordinate system.

    globalpose layout: [qw, qx, qy, qz, px, py, pz]
    """
    qw, qx, qy, qz = globalpose[:4]
    px, py, pz = globalpose[4:7]

    # snapshot_mapper unpacks local_pos.txt as x,y,z,w = [qw,qx,qy,qz]
    # → x=qw, y=qx, z=qy, w=qz fed into standard w-last rotation formula
    x, y, z, w = qw, qx, qy, qz

    R = np.array([
        [1-2*y*y-2*z*z,  2*x*y-2*z*w,   2*x*z+2*y*w],
        [2*x*y+2*z*w,    1-2*x*x-2*z*z,  2*y*z-2*x*w],
        [2*x*z-2*y*w,    2*y*z+2*x*w,   1-2*x*x-2*y*y],
    ], dtype=np.float64)

    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = R @ _RUB2RDF
    c2w[:3,  3] = _RUB2RDF @ np.array([px, py, pz])
    return c2w.astype(np.float32)


def es_world_to_habitat(points: np.ndarray) -> np.ndarray:
    """Convert absolute online ES-world points back to Habitat world.

    Live c2w translations are constructed as ``RUB2RDF @ p_habitat``.  The
    conversion matrix is self-inverse, so query points need exactly one inverse
    axis conversion and must not use an offline first-frame anchor.
    """
    points = np.asarray(points, dtype=np.float64)
    return points @ _RUB2RDF.T


# ── Frame preprocessing ───────────────────────────────────────────────────────

def preprocess_frame(rgb: np.ndarray, depth: np.ndarray,
                     c2w: np.ndarray, device: str) -> dict:
    """Convert numpy Habitat obs to ES-format tensors on `device`.

    rgb   : (H, W, 3) uint8
    depth : (H, W)    float32, metres
    c2w   : (4, 4)    float32
    """
    # Image: resize to ES encoder resolution (384×512), normalize to [0,1]
    img = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0  # (3,H,W)
    img = F.interpolate(img.unsqueeze(0), (_ES_H, _ES_W), mode='bilinear',
                        align_corners=False).squeeze(0)

    # Depth: resize to ES encoder resolution
    d = torch.from_numpy(depth).unsqueeze(0).float()  # (1,H,W)
    d = F.interpolate(d.unsqueeze(0), (_ES_H, _ES_W), mode='bilinear',
                      align_corners=False).squeeze(0)

    extr = torch.from_numpy(c2w)                            # (4,4)
    intr = torch.from_numpy(_K_NORM.copy())                  # (3,3)

    return {
        'image':     img.to(device),
        'depth':     d.to(device),
        'extrinsic': extr.to(device),
        'intrinsic': intr.to(device),
    }


def frame_change_ratio(image: torch.Tensor, reference: torch.Tensor) -> float:
    """Fraction of sampled pixels whose mean RGB delta is significant."""
    pair = torch.stack((image, reference))
    sampled = F.interpolate(
        pair,
        size=(_FRAME_COMPARE_H, _FRAME_COMPARE_W),
        mode='area',
    )
    changed = (sampled[0] - sampled[1]).abs().mean(dim=0) >= _FRAME_PIXEL_DELTA
    return float(changed.float().mean().item())



def _release_cuda_cache(device: str, reason: str = ''):
    """Return cached CUDA blocks after dropping Python references.

    This does not free live tensors, but it prevents transient rebuild/context
    allocations from staying reserved in PyTorch's caching allocator after a
    rebuild or episode reset.
    """
    if isinstance(device, str) and device.startswith('cuda') and torch.cuda.is_available():
        try:
            import gc
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            alloc = torch.cuda.memory_allocated() / (1024 ** 2)
            reserved = torch.cuda.memory_reserved() / (1024 ** 2)
            suffix = f' ({reason})' if reason else ''
            print(f'[OnlineESMapper] CUDA cache released{suffix}: '
                  f'allocated={alloc:.0f}MiB reserved={reserved:.0f}MiB',
                  flush=True)
        except Exception as exc:
            print(f'[OnlineESMapper] CUDA cache release warn: {exc}', flush=True)


def _move_tree_to_device(obj, device: str):
    """Move all tensors in a nested ES state object to `device` in place."""
    if obj is None:
        return None
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, torch.nn.Module):
        return obj
    if isinstance(obj, dict):
        for key, value in list(obj.items()):
            obj[key] = _move_tree_to_device(value, device)
        return obj
    if isinstance(obj, list):
        for idx, value in enumerate(obj):
            obj[idx] = _move_tree_to_device(value, device)
        return obj
    if isinstance(obj, tuple):
        return tuple(_move_tree_to_device(value, device) for value in obj)
    if hasattr(obj, '__dict__'):
        for name, value in list(vars(obj).items()):
            setattr(obj, name, _move_tree_to_device(value, device))
        if isinstance(getattr(obj, 'device', None), str):
            obj.device = device
        return obj
    return obj


def _log_cuda_memory(device: str, label: str):
    if isinstance(device, str) and device.startswith('cuda') and torch.cuda.is_available():
        try:
            alloc = torch.cuda.memory_allocated() / (1024 ** 2)
            reserved = torch.cuda.memory_reserved() / (1024 ** 2)
            print(f'[OnlineESMapper] CUDA memory {label}: '
                  f'allocated={alloc:.0f}MiB reserved={reserved:.0f}MiB',
                  flush=True)
        except Exception:
            pass


def _release_encoder_transient_caches(encoder):
    """Drop encoder-side forward caches that are safe to rebuild next pass."""
    try:
        released = []
        cost_volume = getattr(encoder, 'cost_volume', None)
        if cost_volume is not None and getattr(cost_volume, 'depth_planes_bdhw', None) is not None:
            cost_volume.depth_planes_bdhw = None
            released.append('cost_volume.depth_planes_bdhw')

        fastsam = getattr(encoder, 'fastsam', None)
        predictor = getattr(fastsam, 'predictor', None)
        if predictor is not None:
            for attr in ('results', 'batch', 'dataset', 'plotted_img'):
                if hasattr(predictor, attr) and getattr(predictor, attr) is not None:
                    setattr(predictor, attr, None)
                    released.append(f'fastsam.predictor.{attr}')
            if _OFFLOAD_FASTSAM_AFTER_REBUILD:
                predictor_model = getattr(predictor, 'model', None)
                if isinstance(predictor_model, torch.nn.Module):
                    predictor_model.to('cpu')
                    released.append('fastsam.predictor.model->cpu')
                fastsam.predictor = None
                released.append('fastsam.predictor')

        if _OFFLOAD_FASTSAM_AFTER_REBUILD and fastsam is not None:
            fastsam_model = getattr(fastsam, 'model', None)
            if isinstance(fastsam_model, torch.nn.Module):
                fastsam_model.to('cpu')
                released.append('fastsam.model->cpu')

        if released:
            print('[OnlineESMapper] Released encoder transient caches: '
                  + ', '.join(released), flush=True)
    except Exception as exc:
        print(f'[OnlineESMapper] Encoder cache release warn: {exc}', flush=True)


# ── ES model loader ───────────────────────────────────────────────────────────

def load_es_model(ckpt_path: str, cfg_path: str, device: str):
    """Load ES encoder + decoder via Hydra compose (no load_typed_root_config needed).

    Uses Hydra to compose the full config from config/main.yaml +
    experiment/scannet/embodiedsplat_fast_test overrides, then constructs
    encoder/decoder directly from OmegaConf sub-configs.

    Returns (encoder, decoder) both in eval mode on `device`.
    """
    import os as _os
    _os.makedirs(SNAP_LIVE, exist_ok=True)
    _os.chdir(ES_ROOT)

    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    GlobalHydra.instance().clear()

    with initialize_config_dir(
        config_dir=f'{ES_ROOT}/config', version_base=None
    ):
        cfg_dict = compose(
            config_name='main',
            overrides=[
                '+experiment=scannet/embodiedsplat_fast_test',
                'mode=test',
                f'model.encoder.output_path={SNAP_LIVE}',
                'model.encoder.use_gt_depth=true',
                'wandb.mode=disabled',
            ]
        )

    from src.global_cfg import set_cfg
    set_cfg(cfg_dict)

    from src.config import load_typed_config
    from src.model.encoder.encoder_config import EncoderEpipolarCfg
    from src.model.encoder import get_encoder
    from src.model.decoder import get_decoder

    # Build typed encoder cfg (fills all dataclass defaults not in YAML)
    enc_cfg = load_typed_config(cfg_dict.model.encoder, EncoderEpipolarCfg)
    enc_cfg.output_path = SNAP_LIVE

    encoder, _vis = get_encoder(
        enc_cfg,
        depth_range=[float(cfg_dict.dataset.near), float(cfg_dict.dataset.far)]
    )
    encoder = encoder.to(device).eval()

    # Decoder uses OmegaConf cfg directly (only needs background_color from dataset)
    decoder = get_decoder(cfg_dict.model.decoder, cfg_dict.dataset)
    decoder = decoder.to(device).eval()

    print(f'[OnlineESMapper] Loading checkpoint {ckpt_path} ...', flush=True)
    ckpt  = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = ckpt.get('state_dict', ckpt)

    enc_state = {k[len('encoder.'):]: v for k, v in state.items()
                 if k.startswith('encoder.')}
    dec_state = {k[len('decoder.'):]: v for k, v in state.items()
                 if k.startswith('decoder.')}

    miss_e, _ = encoder.load_state_dict(enc_state, strict=False)
    miss_d, _ = decoder.load_state_dict(dec_state, strict=False)
    if miss_e:
        print(f'[OnlineESMapper] encoder missing keys: {miss_e[:3]}', flush=True)

    print(f'[OnlineESMapper] Model loaded — '
          f'enc={sum(p.numel() for p in encoder.parameters()):,} params', flush=True)

    # Keep encoder output_path for API compatibility; live-only emits no tensor checkpoints
    try:
        encoder.cfg.output_path = SNAP_LIVE
    except Exception:
        pass

    return encoder, decoder



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


def _select_spatial_candidates(ok_means, scores, top_k):
    min_similarity = float(os.environ.get('ES_QUERY_MIN_SIMILARITY', '0.18'))
    min_support = int(os.environ.get('ES_QUERY_MIN_SUPPORT', '3'))
    support_radius = float(os.environ.get('ES_QUERY_SUPPORT_RADIUS', '0.35'))
    nms_radius = float(os.environ.get('ES_QUERY_NMS_RADIUS', '1.0'))
    pool_size = int(os.environ.get(
        'ES_QUERY_POOL_SIZE', str(max(top_k * 200, 2000))
    ))
    support_rerank_weight = float(os.environ.get(
        'ES_QUERY_SUPPORT_RERANK_WEIGHT', '0.0'
    ))

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


def clip_query_globals(encoder, globals_state, category: str,
                       top_k: int, device: str, return_records: bool = False,
                       map_revision: int = 0):
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
    selected, diagnostics = _select_spatial_candidates(
        ok_means, scores, top_k
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




def clip_query_batch_globals(encoder, globals_state, categories: list,
                              top_k: int, device: str,
                              return_records: bool = False,
                              map_revision: int = 0) -> dict:
    """Batch CLIP query: load feat_dict ONCE, query N categories.

    8× faster than calling clip_query_globals per-category.
    Returns dict: {category: [[x,y,z], ...]}
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
        selected, diagnostics = _select_spatial_candidates(
            ok_means, scores, top_k
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

# ── P6_VIS: ES map BEV renderer (agent-built map, for Oracle-vs-ES comparison) ──
def render_bev_cv2(globals_state, out_path, n_keyframes=None, n_raw_frames=None,
                   size=420, margin=22, max_points=_FINAL_BEV_MAX_POINTS):
    """Fast cv2 BEV (x-z) of the ES online Gaussian map.
    CLIP-covered gaussians = blue, others = grey. Shows what the agent actually
    built (unexplored = empty, no contour) — contrast with the oracle top-down.

    P6_FIX13: n_keyframes/n_raw_frames (if given) are burned into the label so
    a BEV snapshot can be cross-referenced against LagMemo's own per-step
    visualizations (images/.../planner_snapshot_<raw_step>.png uses the same
    raw nav-step counter as n_raw_frames here).
    """
    import cv2
    valid = globals_state.valid[0].bool()
    if not valid.any():
        return
    valid_indices = torch.nonzero(valid, as_tuple=False).flatten()
    n_total = int(valid_indices.numel())
    if max_points > 0 and n_total > max_points:
        sample = torch.linspace(
            0, n_total - 1, steps=max_points, device=valid_indices.device
        ).long()
        valid_indices = valid_indices[sample]
    means_src = globals_state.means
    idx_src = globals_state.clip_features_2d['idx']
    means = means_src[0, valid_indices.to(means_src.device), :].detach().cpu().numpy()
    idx = idx_src[0, valid_indices.to(idx_src.device), :3]
    has = (idx != -1).any(dim=-1).detach().cpu().numpy()                     # (N,) bool
    xz = means[:, [0, 2]]
    lo = xz.min(0); span = float((xz.max(0) - lo).max()) + 1e-6
    cv = np.full((size, size, 3), 255, np.uint8)
    px = ((xz[:, 0] - lo[0]) / span * (size - 2 * margin) + margin).astype(np.int32)
    py = ((xz[:, 1] - lo[1]) / span * (size - 2 * margin) + margin).astype(np.int32)
    py = size - 1 - py
    px = np.clip(px, 0, size - 1); py = np.clip(py, 0, size - 1)
    g = ~has
    cv[py[g], px[g]] = (205, 205, 205)          # grey: no-CLIP gaussian
    cv[py[has], px[has]] = (205, 130, 50)        # BGR: CLIP gaussian (blue-ish)
    n_rendered = int(valid_indices.numel())
    cov = 100.0 * has.sum() / max(n_rendered, 1)
    cv2.rectangle(cv, (0, 0), (size, 26), (255, 255, 255), -1)
    label = f'ES online map  N={n_total} sampled={n_rendered} CLIP={cov:.0f}%'
    if n_keyframes is not None and n_raw_frames is not None:
        label = f'kf={n_keyframes} raw_frame={n_raw_frames}  ' + label
    cv2.putText(cv, label, (6, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.imwrite(out_path, cv)


def render_empty_bev_cv2(out_path, n_keyframes, n_raw_frames, size=720):
    """Write a diagnostic artifact when an episode ends before ES warmup."""
    import cv2
    cv = np.full((size, size, 3), 255, np.uint8)
    label = f"ES online map unavailable: kf={n_keyframes} raw_frame={n_raw_frames}"
    cv2.putText(cv, label, (18, size // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 180), 1)
    cv2.imwrite(out_path, cv)


# ── Main class ────────────────────────────────────────────────────────────────

class OnlineESMapper:
    """
    Phase 5 online ES mapper.

    Lifecycle per episode
    ---------------------
    reset_episode() → N × add_live_frame() → N × query()

    Thread safety
    -------------
    add_live_frame() and query() are called from the IPC server threads.
    _rebuild_worker() runs in a daemon thread.
    Globals hot-swap is atomic under self._globals_lock.
    """

    def __init__(self, device: str = 'cuda',
                 top_k: int = 10,
                 warmup_n: int = None,
                 rebuild_interval: int = None,
                 ckpt_path: str = ES_CKPT,
                 cfg_path: str = ES_CFG,
                 local_pos_path: str = None,
                 bev_dir: str = None):

        self._bev_dir        = bev_dir   # P6_VIS: dir to save ES map BEV per rebuild
        self.device          = device
        self.top_k           = top_k
        self.warmup_n = (
            int(os.environ.get('ES_WARMUP_KEYFRAMES', '10'))
            if warmup_n is None else warmup_n
        )
        self.rebuild_interval = (
            int(os.environ.get('ES_REBUILD_INTERVAL', '10'))
            if rebuild_interval is None else rebuild_interval
        )
        if local_pos_path:
            print('[OnlineESMapper] Ignoring local_pos_path in live absolute-world mode.',
                  flush=True)

        # Rejected raw frames are retained only briefly. Accepted keyframes remain
        # on CPU because the incremental encoder requires stable frame indexing.
        self._frames: list = []
        self._n_total_frames: int = 0
        self._buf_lock = threading.Lock()

        self._keyframes: list = []
        self._admission_counts = collections.Counter()
        self._last_frame_change = 1.0

        # ── P6_INCREMENTAL: persistent encoder online state (resume across rebuilds)
        self._online_state = None

        # ── globals (hot-swappable) ─────────────────────────────────────────
        self._globals     = None
        self._globals_lock = threading.Lock()
        self._model_lock = threading.RLock()
        self.n_frames     = 0
        self.n_gaussians  = 0

        # ── rebuild state ───────────────────────────────────────────────────
        self._rebuild_thread    = None
        self._rebuild_lock      = threading.Lock()
        self._rebuild_active    = False
        self._rebuild_pending   = False
        self._resetting         = False
        self._last_rebuild_n    = 0
        self._last_fused_n      = 0

        # ── scene / episode info ────────────────────────────────────────────
        self._scene_id   = ''
        self._episode_id = ''
        self._episode_artifacts_exported = False

        # ── load model ──────────────────────────────────────────────────────
        print('[OnlineESMapper] Loading ES model ...', flush=True)
        self._encoder, self._decoder = load_es_model(ckpt_path, cfg_path, device)
        print('[OnlineESMapper] Ready.', flush=True)

    # ── episode lifecycle ─────────────────────────────────────────────────────

    def _export_final_map(self):
        if not _FINAL_BEV_DIR or not self._scene_id:
            return []
        if self._episode_artifacts_exported:
            return []
        episode_name = f'{self._scene_id}_{self._episode_id}'
        outputs = []
        with self._globals_lock:
            globals_state = self._globals
            if _FINAL_BEV_DIR:
                output_dir = os.path.join(
                    _FINAL_BEV_DIR, self._scene_id, self._episode_id, "es_final_bev"
                )
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, 'final_bev.png')
                try:
                    if globals_state is None:
                        render_empty_bev_cv2(
                            output_path, len(self._keyframes), self._n_total_frames
                        )
                    else:
                        render_bev_cv2(
                            globals_state, output_path,
                            n_keyframes=len(self._keyframes),
                            n_raw_frames=self._n_total_frames,
                            size=720,
                        )
                    if os.path.isfile(output_path):
                        outputs.append(output_path)
                except Exception as exc:
                    print(
                        f'[OnlineESMapper] Final BEV export failed for {episode_name}: {exc}',
                        flush=True,
                    )
        self._episode_artifacts_exported = bool(outputs)
        if outputs:
            print(
                f'[OnlineESMapper] Final episode artifacts exported: {outputs}',
                flush=True,
            )
        return outputs


    def flush_episode(self):
        """Synchronously fuse and optionally export the completed episode map."""
        for _ in range(3):
            with self._buf_lock:
                n_kf = len(self._keyframes)
            if n_kf < self.warmup_n or self._last_fused_n >= n_kf:
                break
            self._maybe_start_rebuild(force=True)
            with self._rebuild_lock:
                rebuild_thread = self._rebuild_thread
            if rebuild_thread is None:
                break
            rebuild_thread.join()
        if self._last_fused_n < n_kf:
            print(f'[OnlineESMapper] Flush incomplete: fused={self._last_fused_n}, '
                  f'accepted={n_kf}', flush=True)
        artifacts = self._export_final_map()
        status = self.get_status()
        status['final_artifacts'] = artifacts
        return status

    def reset_episode(self, scene_id: str, episode_id: str):
        if self._scene_id:
            self.flush_episode()
        with self._rebuild_lock:
            self._resetting = True
            self._rebuild_pending = False

        if self._rebuild_thread and self._rebuild_thread.is_alive():
            self._rebuild_thread.join()

        with self._model_lock:
            with self._buf_lock:
                old_frames = self._frames
                old_keyframes = self._keyframes
                self._frames = []
                self._keyframes = []
                self._admission_counts.clear()
                self._last_frame_change = 1.0
                self._n_total_frames = 0
                self._last_rebuild_n = 0
                self._last_fused_n = 0
                self._episode_artifacts_exported = False
            with self._globals_lock:
                old_globals = self._globals
                self._globals      = None
                self.n_gaussians   = 0
                self.n_frames      = 0

            old_online_state = self._online_state
            self._online_state = None   # P6_INCREMENTAL: fresh accumulation per episode
            del old_frames, old_keyframes, old_globals, old_online_state
        _release_cuda_cache(self.device, 'episode reset')
        release_host_memory('ES mapper episode reset')
        self._scene_id   = scene_id
        self._episode_id = episode_id
        with self._rebuild_lock:
            self._rebuild_active = False
            self._rebuild_pending = False
            self._resetting = False

        print(f'[OnlineESMapper] Episode reset: {scene_id}/{episode_id}', flush=True)

    # ── live frame ingestion ─────────────────────────────────────────────────

    def _consider_keyframe(self, frame: dict):
        """Compare the current RGB frame only with the last accepted keyframe."""
        if _MAX_GAUSSIANS > 0 and self.n_gaussians >= _MAX_GAUSSIANS:
            return False, 'gaussian_limit', 0.0
        if _MAX_KEYFRAMES > 0 and len(self._keyframes) >= _MAX_KEYFRAMES:
            return False, 'keyframe_limit', 0.0
        if not self._keyframes:
            return True, 'first_frame', 1.0
        change = frame_change_ratio(
            frame['image'],
            self._keyframes[-1]['image'],
        )
        if change >= _FRAME_CHANGE_RATIO:
            return True, 'frame_change', change
        return False, 'frame_similarity', change

    def _maybe_start_rebuild(self, force: bool = False):
        with self._buf_lock:
            n_kf = len(self._keyframes)
        if n_kf < self.warmup_n:
            return

        with self._rebuild_lock:
            due = (
                force
                or self._last_rebuild_n == 0
                or n_kf - self._last_rebuild_n >= self.rebuild_interval
            )
            if not due or self._resetting:
                return
            if self._rebuild_active:
                self._rebuild_pending = True
                return
            self._rebuild_active = True
            self._rebuild_pending = False
            self._last_rebuild_n = n_kf
            self._rebuild_thread = threading.Thread(
                target=self._rebuild_worker,
                daemon=True,
                name=f'es_rebuild_kf{n_kf}',
            )
            self._rebuild_thread.start()

    def add_live_frame(self, rgb: np.ndarray, depth: np.ndarray,
                       extrinsic: np.ndarray, intrinsic: np.ndarray):
        """
        Ingest one Habitat navigation frame.

        rgb       : (H, W, 3) uint8
        depth     : (H, W)    float32, metres
        extrinsic : (4, 4)    c2w float32 (from globalpose_to_c2w)
        intrinsic : (3, 3)    normalized (not used — we use _K_NORM)
        """
        # Keep live/keyframe buffers on CPU; _build_context() performs the
        # short-lived CUDA transfer for rebuild. This prevents raw frame and
        # keyframe history from permanently occupying GPU6 memory.
        frame = preprocess_frame(rgb, depth, extrinsic, 'cpu')
        with self._buf_lock:
            self._frames.append(frame)
            self._n_total_frames += 1
            n = self._n_total_frames
            if _RAW_FRAME_BUFFER <= 0:
                self._frames.clear()
            elif len(self._frames) > _RAW_FRAME_BUFFER:
                del self._frames[:-_RAW_FRAME_BUFFER]
        with self._globals_lock:
            self.n_frames = n

        with self._buf_lock:
            admitted, reason, change = self._consider_keyframe(frame)
            self._last_frame_change = change
            self._admission_counts[reason] += 1
            if admitted:
                self._keyframes.append(frame)
            n_kf = len(self._keyframes)

        if admitted:
            if n_kf == 1 or n_kf % 25 == 0:
                print(f'[OnlineESMapper] Keyframes: {n_kf}/{n} raw '
                      f'(change={change:.3f}, threshold={_FRAME_CHANGE_RATIO:.2f})',
                      flush=True)
            self._maybe_start_rebuild()

    # ── rebuild worker (background) ──────────────────────────────────────────

    def _rebuild_worker(self):
        t0 = time.perf_counter()
        context = None
        results = None
        new_globals = None
        next_online_state = None
        old_globals = None
        old_online_state = None
        resume_state = None
        frames_snap = None
        # Capture episode identity at start — used to detect episode resets
        # that happened while this rebuild was running (race condition guard).
        _ep_scene = self._scene_id
        _ep_id    = self._episode_id
        self._model_lock.acquire()
        try:
            with self._buf_lock:
                # P6_FIX10: persistent keyframe set, never truncated/reset.
                # Append-only ordering keeps resume_state's index alignment
                # valid across rebuilds -- pure incremental, ~200-300ms/new kf
                # regardless of total accumulated keyframe count (verified up
                # to 308 keyframes via run_keyframe_verification.sh).
                frames_snap = list(self._keyframes)
                # P6_FIX13: raw nav-step count paired with this exact keyframe
                # snapshot, for BEV filename/label (cross-ref vs LagMemo's own
                # per-step visualizations, which use the same step counter).
                n_total_frames_snap = self._n_total_frames

            n = len(frames_snap)
            print(f'[OnlineESMapper] Rebuild start: {n} keyframes / '
                  f'raw_frame={n_total_frames_snap} '
                  f'(n_gaussians={self.n_gaussians})...', flush=True)

            context = self._build_context(frames_snap)

            # Episode identifier for ES save path (must end with '/')
            scene_ep = f'{self._scene_id}_{self._episode_id}/'

            # Keep the persistent state on CPU between rebuilds. The ES encoder
            # mutates it in place, so it only lives on GPU during this forward.
            resume_state = _move_tree_to_device(self._online_state, self.device)
            _log_cuda_memory(self.device, 'before encoder forward')

            with torch.no_grad():
                # P6_INCREMENTAL: resume from persistent state → only NEW frames are
                # fused (bounded ~1s/frame), instead of from-scratch O(N) rebuild.
                results = self._encoder(
                    [context],              # full accumulated context (index alignment)
                    global_step=0,
                    is_testing=True,
                    export_ply=False,
                    decoder=self._decoder,
                    source=None,
                    dense=False,
                    scene=scene_ep,
                    target_indices=None,
                    output_path=Path(SNAP_LIVE),
                    target=None,
                    test_bev=False,
                    path=Path(SNAP_LIVE),
                    resume_state=resume_state,
                    online_incremental=True,
                    return_state=True,
                )
            _log_cuda_memory(self.device, 'after encoder forward')

            next_online_state = _move_tree_to_device(results.get('online_state'), 'cpu')
            new_globals = _move_tree_to_device(results.get('globals'), 'cpu')
            if new_globals is not None:
                n_valid = int(new_globals.valid[0].sum().item())
                # Episode guard: discard results if reset_episode() fired during rebuild
                if self._scene_id != _ep_scene or self._episode_id != _ep_id:
                    print(f'[OnlineESMapper] Rebuild discarded: episode changed '
                          f'({_ep_scene}/{_ep_id} -> {self._scene_id}/{self._episode_id})',
                          flush=True)
                else:
                    # Persist CPU-resident online state for the next incremental
                    # rebuild, and drop stale references so CUDA tensors can die.
                    old_online_state = self._online_state
                    self._online_state = next_online_state
                    if old_online_state is next_online_state:
                        old_online_state = None
                    with self._globals_lock:
                        old_globals = self._globals
                        self._globals    = new_globals
                        self.n_gaussians = n_valid
                    self._last_fused_n = n
                    if old_globals is new_globals:
                        old_globals = None
                    dt = (time.perf_counter() - t0) * 1000
                    print(f'[OnlineESMapper] Rebuild done: {n} keyframes, '
                          f'{n_valid} Gaussians, {dt:.0f} ms', flush=True)
                if _ENABLE_CLIP_COVERAGE_DIAGNOSTIC:
                    try:
                        _idx = new_globals.clip_features_2d['idx'][
                            :, new_globals.valid[0], :3
                        ]
                        _has_clip = int((_idx != -1).any(dim=-1).sum())
                        print(f'[OnlineESMapper] CLIP coverage: {_has_clip}/{n_valid} '
                              f'({100*_has_clip/max(n_valid,1):.1f}%)', flush=True)
                    except Exception:
                        pass
                # P6_VIS: save ES map BEV (agent-built) for Oracle-vs-ES comparison
                if self._bev_dir:
                    try:
                        _bd = os.path.join(self._bev_dir, f'{self._scene_id}_{self._episode_id}')
                        os.makedirs(_bd, exist_ok=True)
                        # P6_FIX13: filename led by the raw nav-step number (same
                        # counter as images/.../planner_snapshot_<N>.png) so BEV
                        # snapshots can be cross-referenced directly; keyframe
                        # count kept as a secondary qualifier, zero-padded so
                        # alphabetical sort still matches chronological order.
                        _bev_name = f'bev_{n_total_frames_snap:05d}_kf{n:03d}.png'
                        render_bev_cv2(new_globals, os.path.join(_bd, _bev_name),
                                      n_keyframes=n, n_raw_frames=n_total_frames_snap)
                    except Exception as _be:
                        print(f'[OnlineESMapper] BEV save warn: {_be}', flush=True)
            else:
                print(f'[OnlineESMapper] Rebuild returned no globals', flush=True)

        except Exception as e:
            import traceback
            print(f'[OnlineESMapper] Rebuild ERROR: {e}', flush=True)
            traceback.print_exc()
        finally:
            # Drop transient CUDA-heavy rebuild objects before marking the worker
            # idle. Current self._globals/self._online_state are CPU-resident by
            # design; context/results/resume_state should not pin GPU memory.
            _release_encoder_transient_caches(self._encoder)
            del context, results, new_globals, next_online_state, old_globals, old_online_state, resume_state, frames_snap
            _release_cuda_cache(self.device, 'rebuild cleanup')
            self._model_lock.release()
            with self._rebuild_lock:
                self._rebuild_active = False
                restart = self._rebuild_pending and not self._resetting
                self._rebuild_pending = False
            if restart:
                self._maybe_start_rebuild()

    # ── context builder ───────────────────────────────────────────────────────

    def _build_context(self, frames: list) -> dict:
        """Stack CPU-resident frames into a transient ES context on self.device."""
        N = len(frames)
        dev = self.device

        images     = torch.stack([f['image']     for f in frames]).unsqueeze(0).to(dev)  # (1,N,3,H,W)
        depths     = torch.stack([f['depth']     for f in frames]).unsqueeze(0).to(dev)  # (1,N,1,H,W)
        extrinsics = torch.stack([f['extrinsic'] for f in frames]).unsqueeze(0).to(dev)  # (1,N,4,4)
        intrinsics = torch.stack([f['intrinsic'] for f in frames]).unsqueeze(0).to(dev)  # (1,N,3,3)

        near = torch.full((1, N), _NEAR, dtype=torch.float32).to(dev)
        far  = torch.full((1, N), _FAR,  dtype=torch.float32).to(dev)

        # depth_s0: half-resolution depth for use_gt_depth mode (low_res=True, s=0)
        depth_s0 = F.interpolate(
            depths.view(N, 1, _ES_H, _ES_W), size=(_ES_H // 2, _ES_W // 2),
            mode='bilinear', align_corners=False
        ).view(1, N, 1, _ES_H // 2, _ES_W // 2).to(dev)

        return {
            'image':      images,
            'depth':      depths,
            'depth_s0':   depth_s0,      # required for use_gt_depth=True
            'extrinsics': extrinsics,
            'intrinsics': intrinsics,
            'near':       near,
            'far':        far,
            'cache':      [None] * N,    # not used for fastsam+maskadapter path
        }

    # ── query ─────────────────────────────────────────────────────────────────

    def query(self, category: str, top_k: int = None) -> list:
        """Return top-k Habitat-world positions for `category`."""
        with self._model_lock:
            with self._globals_lock:
                g = self._globals
            if g is None:
                return []
            top_k = top_k or self.top_k
            return clip_query_globals(
                self._encoder, g, category, top_k, self.device
            )

    def query_with_records(self, category: str, top_k: int = None):
        """Return legacy positions plus a backward-compatible metadata sidecar."""
        with self._model_lock:
            with self._globals_lock:
                g = self._globals
            if g is None:
                return [], []
            top_k = top_k or self.top_k
            return clip_query_globals(
                self._encoder,
                g,
                category,
                top_k,
                self.device,
                return_records=True,
                map_revision=self._last_fused_n,
            )

    # ── status ────────────────────────────────────────────────────────────────

    def query_batch(self, categories: list, top_k: int = None) -> dict:
        """Batch query all categories in one feat_dict pass."""
        with self._model_lock:
            with self._globals_lock:
                g = self._globals
            if g is None:
                return {cat: [] for cat in categories}
            top_k = top_k or self.top_k
            return clip_query_batch_globals(
                self._encoder, g, categories, top_k, self.device
            )

    def query_batch_with_records(self, categories: list, top_k: int = None):
        with self._model_lock:
            with self._globals_lock:
                g = self._globals
            if g is None:
                empty = {cat: [] for cat in categories}
                return empty, dict(empty)
            top_k = top_k or self.top_k
            return clip_query_batch_globals(
                self._encoder,
                g,
                categories,
                top_k,
                self.device,
                return_records=True,
                map_revision=self._last_fused_n,
            )

    def get_status(self) -> dict:
        with self._globals_lock:
            n_g = self.n_gaussians
            n_f = self.n_frames
        with self._rebuild_lock:
            rebuilding = self._rebuild_active
            pending = self._rebuild_pending
        with self._buf_lock:
            n_kf = len(self._keyframes)
            counts = dict(self._admission_counts)
            last_change = self._last_frame_change
        return {
            'n_frames':    n_f,
            'n_raw_frames': self._n_total_frames,
            'n_keyframes': n_kf,
            'last_fused_keyframes': self._last_fused_n,
            'n_gaussians': n_g,
            'rebuilding':  rebuilding,
            'rebuild_pending': pending,
            'last_frame_change': last_change,
            'admission_counts': counts,
            'scene_id':    self._scene_id,
            'episode_id':  self._episode_id,
        }
