#!/usr/bin/env python3
"""
eval_episode_glue_es.py  — Phase 4 end-to-end nav with ES online mapper.

Adds --enable_timing flag: records per-step latency breakdown and
reports online-navigation compliance summary at the end.

Timing columns saved to results_dir/timing_log.csv:
  scene_id, episode_id, subtask_index, frame_index, snap,
  dt_total_ms, dt_push_ms, dt_goals_ms, dt_act_ms, is_upgrade_step

Online standard thresholds (printed at end):
  ≥10 FPS → 100 ms/step   (real-time robot, e.g. 10 Hz)
  ≥ 5 FPS → 200 ms/step   (high-performance)
  ≥ 1 FPS → 1000 ms/step  (quasi-real-time)
  ≥ 0.2 FPS → 5000 ms/step (acceptable simulation overhead)
"""

import warnings
warnings.filterwarnings("ignore")
import os, sys, csv

os.environ['MAGNUM_LOG'] = 'quiet'
os.environ['HABITAT_SIM_LOG'] = 'quiet'

import argparse
import hashlib
import json
import time
import types
from dataclasses import replace
from functools import partial
from pathlib import Path
from pprint import pprint

import numpy as np
from tqdm import tqdm

_THIS_FILE = Path(__file__).resolve()
_TDONAV_ROOT = Path(os.environ.get("TDONAV_ROOT", _THIS_FILE.parents[4])).resolve()
_LAGMEMO_ROOT = Path(os.environ.get("TDONAV_LAGMEMO_ROOT", _TDONAV_ROOT / "Navigation" / "runtime" / "lagmemo")).resolve()
_LAGMEMO_SRC = _LAGMEMO_ROOT / 'src' / 'lagmemo'
if str(_LAGMEMO_SRC) not in sys.path:
    sys.path.insert(0, str(_LAGMEMO_SRC))


def _install_detic_inference_import_shims():
    """Shim absent training-only imports in the pruned Detic vendor tree."""

    detic_root = (
        _LAGMEMO_SRC
        / 'lagmemo/perception/detection/detic/Detic'
    )
    groups = (
        (
            detic_root / 'third_party/CenterNet2/centernet/data',
            'centernet.data',
            {
                'datasets.objects365': {'categories_v1': ()},
                'datasets.coco': {'_PREDEFINED_SPLITS_COCO': {}},
                'datasets.nuimages': {},
            },
        ),
        (
            detic_root / 'detic/data',
            (
                'lagmemo.perception.detection.detic.Detic.detic.data'
            ),
            {
                'datasets.lvis_v1': {},
                'datasets.imagenet': {},
                'datasets.cc': {},
                'datasets.objects365': {},
                'datasets.oid': {},
                'datasets.coco_zeroshot': {},
            },
        ),
    )
    installed = []
    for source_dir, package_root, child_specs in groups:
        if source_dir.is_dir():
            continue
        package_names = {package_root}
        for relative_name in child_specs:
            pieces = relative_name.split('.')
            for length in range(1, len(pieces)):
                package_names.add(
                    package_root + '.' + '.'.join(pieces[:length])
                )
        for name in sorted(package_names, key=lambda item: item.count('.')):
            if name not in sys.modules:
                module = types.ModuleType(name)
                module.__path__ = []
                module.__scsp_inference_stub__ = True
                sys.modules[name] = module
                installed.append(name)
            if '.' in name:
                parent_name, attribute = name.rsplit('.', 1)
                parent = sys.modules.get(parent_name)
                if parent is not None:
                    setattr(parent, attribute, sys.modules[name])
        for relative_name, attributes in child_specs.items():
            name = package_root + '.' + relative_name
            if name not in sys.modules:
                module = types.ModuleType(name)
                module.__scsp_inference_stub__ = True
                sys.modules[name] = module
                installed.append(name)
            for attribute, value in attributes.items():
                setattr(sys.modules[name], attribute, value)
            parent_name, attribute = name.rsplit('.', 1)
            setattr(sys.modules[parent_name], attribute, sys.modules[name])
    try:
        __import__('timm.models.convnext')
    except ModuleNotFoundError as exc:
        if exc.name != 'timm.models.convnext':
            raise
        name = (
            'lagmemo.perception.detection.detic.Detic.detic.'
            'modeling.backbone.timm'
        )
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__scsp_inference_stub__ = True
            sys.modules[name] = module
            installed.append(name)
    return tuple(installed)


_EARLY_DETIC_REGISTRATION_STUBS = ()
if os.environ.get('ES_SCSP_RUNTIME_MODE', 'off') != 'off':
    _EARLY_DETIC_REGISTRATION_STUBS = (
        _install_detic_inference_import_shims()
    )


from config_utils import get_config
from habitat.core.env import Env

from lagmemo.agent.lagmemo_agent.glue_agent import GLUEAgent
from lagmemo.core.interfaces import DiscreteNavigationAction
from lagmemo.env.habitat_lagmemo_env import HabitatGoatEnv

from runtime.mapping_client import MappingClient
from experiments.dataset import load_runtime_tasks
from pbsp.navigation.cooldown import (
    CategorySwitchCooldown,
    cooldown_for_category,
    parse_category_cooldowns,
)
from experiments.metrics import (
    DISTANCE_TO_GOAL_KEY,
    DISTANCE_TO_GOAL_STATUS_KEY,
    SHORTEST_PATH_UNREACHABLE,
    SPL_KEY,
    SUCCESS_KEY,
    build_subtask_telemetry,
    summarize_episode,
    summarize_records,
    write_json_strict,
)
from pbsp.navigation.ranking import (
    CandidateActionReranker,
    CandidateRerankConfig,
)
from pbsp.types import (
    BatchValidationStatus,
    CandidateBatch,
    CandidateRecord,
    FallbackReason,
    GeodesicDistance,
    GeodesicMatrix,
    GeodesicStatus,
    SCSPContractError,
    validate_candidate_batch,
)
from experiments.telemetry import (
    SHADOW_SCHEMA_VERSION,
    build_scsp_shadow_decision,
    dumps_json_strict,
    to_jsonable,
)
from pbsp.navigation.suffix import (
    DEFAULT_CANDIDATE_CAP,
    DEFAULT_SUFFIX_WEIGHT,
    Idea1Candidate,
    SuffixPlanningResult,
    plan_ordered_suffix,
)


# ── Timing helpers ────────────────────────────────────────────────────────────


def _finite_metric_summary(subtasks, metric):
    from experiments.recorder import finite_metric_summary

    return finite_metric_summary(subtasks, metric)



def _project_goals_to_navigable(goals, pathfinder, *, return_indices=False):
    from adapters.geometry import project_goals_to_navigable
    return project_goals_to_navigable(
        goals, pathfinder, return_indices=return_indices,
        max_distance=float(os.environ.get('ES_NAV_PROJECTION_MAX_DISTANCE', '1.5')),
    )


def _task_category(task):
    for name in ('category', 'object_category'):
        value = task.get(name)
        if value is not None and str(value).strip():
            return str(value)
    raise ValueError(f'MON task is missing its target category: {task}')


def _record_scsp_provider_event(
    provider,
    *,
    phase,
    current_task,
    scene_id,
    episode_id,
    ordered_categories=None,
    output_path=None,
    query_error=None
):
    from experiments.recorder import record_provider_event

    return record_provider_event(
        provider,
        phase=phase,
        current_task=current_task,
        scene_id=scene_id,
        episode_id=episode_id,
        ordered_categories=ordered_categories,
        output_path=output_path,
        query_error=query_error,
    )


def _scsp_runtime_mode():
    mode = os.environ.get('ES_SCSP_RUNTIME_MODE', 'off')
    if mode not in {'off', 'shadow', 'idea1_only', 'ordered_suffix_mvp'}:
        raise ValueError(
            'ES_SCSP_RUNTIME_MODE must be off, shadow, idea1_only, or '
            'ordered_suffix_mvp'
        )
    profile_mode = os.environ.get('ES_SCSP_PROFILE_MODE', 'exact_baseline')
    compatible = {
        'exact_baseline': {'off'},
        'idea1_only': {'shadow', 'idea1_only'},
        'ordered_suffix_mvp': {
            'shadow', 'idea1_only', 'ordered_suffix_mvp'
        },
    }
    if profile_mode not in compatible or mode not in compatible[profile_mode]:
        raise ValueError(
            f'SCSP runtime/profile mode mismatch: {mode!r}/{profile_mode!r}'
        )
    return mode


def _enable_scsp_no_image_planning_state(
    agent, *, scsp_enabled, print_images
):
    """Keep planner-only maps available without enabling image writes."""

    if not scsp_enabled or bool(print_images):
        return False
    if bool(getattr(agent, 'continuous_print_images', False)) or bool(
        getattr(agent, 'episode_final_snapshot', False)
    ):
        raise RuntimeError(
            'SCSP no-image planning requires every image write gate disabled'
        )
    # GLUEAgent historically uses ``visualize`` both to materialize planner
    # inputs (explored map/frontiers) and to prepare optional render payloads.
    # Its writers remain guarded by the two flags checked above, while its
    # planner retained config.VISUALIZE=False during construction.
    agent.visualize = True
    return True


def _load_scsp_frozen_identities(
    dataset_manifest_sha256, ordered_subtasks=3
):
    """Load identity-only frozen metadata; never read policy outcomes."""

    path = _TDONAV_ROOT / 'Exp/experiments/scsp_idea2_frozen_pairs_v1.json'
    payload = json.loads(path.read_text(encoding='utf-8'))
    if payload.get('dataset', {}).get(
        'dataset_manifest_sha256'
    ) != dataset_manifest_sha256:
        return {}
    if ordered_subtasks != 3:
        raise ValueError('SCSP frozen identities require three subtasks')
    identities = {}
    for row in payload.get('episodes', ()):
        identity = row.get('identity', {})
        key = (
            str(identity.get('scene_id', '')),
            str(identity.get('runtime_episode_id', '')),
        )
        if not all(key) or key in identities:
            raise ValueError('SCSP frozen manifest has an invalid identity key')
        source_ids = tuple(
            str(value) for value in identity.get('source_episode_ids', ())
        )
        categories = tuple(
            str(value) for value in identity.get('ordered_categories', ())
        )
        if len(source_ids) != 3 or len(categories) != 3:
            raise ValueError('SCSP frozen identity must contain three subtasks')
        identities[key] = {
            'scene_id': key[0],
            'runtime_episode_id': key[1],
            'source_episode_ids': source_ids,
            'ordered_categories': categories,
        }
    if len(identities) != 48:
        raise ValueError('SCSP frozen manifest must contain 48 identities')
    return identities


def _load_scsp_identity_sidecar(
    path, dataset_manifest_sha256, ordered_subtasks=3
):
    """Load optional identity-only smoke metadata without unrelated fields."""

    if not path:
        return {}, None
    source = Path(path).expanduser().resolve()
    raw = source.read_bytes()
    def reject_constant(value):
        raise ValueError(f'non-strict JSON constant: {value}')

    payload = json.loads(
        raw.decode('utf-8'), parse_constant=reject_constant
    )
    if not isinstance(payload, dict) or set(payload) != {
        'schema_version',
        'dataset_manifest_sha256',
        'episodes',
    }:
        raise ValueError('SCSP identity sidecar contains non-identity fields')
    if payload.get('schema_version') != 'scsp_identity_sidecar_v1':
        raise ValueError('SCSP identity sidecar schema mismatch')
    if (
        not isinstance(payload.get('dataset_manifest_sha256'), str)
        or payload['dataset_manifest_sha256'] != dataset_manifest_sha256
    ):
        raise ValueError('SCSP identity sidecar dataset hash mismatch')
    identities = {}
    rows = payload.get('episodes')
    if not isinstance(rows, list):
        raise ValueError('SCSP identity sidecar episodes must be a list')
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            'scene_id',
            'runtime_episode_id',
            'source_episode_ids',
            'ordered_categories',
        }:
            raise ValueError('SCSP identity sidecar contains non-identity fields')
        if any(
            not isinstance(row[name], str) or not row[name].strip()
            for name in ('scene_id', 'runtime_episode_id')
        ):
            raise ValueError('SCSP identity sidecar IDs must be non-empty strings')
        if not isinstance(row['source_episode_ids'], list) or not isinstance(
            row['ordered_categories'], list
        ):
            raise ValueError('SCSP identity sidecar identity values must be lists')
        if any(
            not isinstance(value, str) or not value.strip()
            for name in ('source_episode_ids', 'ordered_categories')
            for value in row[name]
        ):
            raise ValueError(
                'SCSP identity sidecar identity values must be non-empty strings'
            )
        key = (row['scene_id'], row['runtime_episode_id'])
        source_ids = tuple(row['source_episode_ids'])
        categories = tuple(row['ordered_categories'])
        if (
            not all(key)
            or key in identities
            or len(source_ids) != ordered_subtasks
            or len(categories) != ordered_subtasks
            or not all(source_ids)
            or not all(categories)
        ):
            raise ValueError('SCSP identity sidecar has an invalid episode')
        identities[key] = {
            'scene_id': key[0],
            'runtime_episode_id': key[1],
            'source_episode_ids': source_ids,
            'ordered_categories': categories,
        }
    if not identities:
        raise ValueError('SCSP identity sidecar must contain at least one episode')
    return identities, hashlib.sha256(raw).hexdigest()


def _candidate_reranker_from_env(enabled):
    def number(name, default):
        return float(os.environ.get(name, default))

    return CandidateActionReranker(
        enabled=bool(enabled),
        config=CandidateRerankConfig(
            semantic=number('ES_QUERY_W_SEMANTIC', '1.0'),
            support=number('ES_QUERY_W_SUPPORT', '0.35'),
            recurrence=number('ES_QUERY_W_RECURRENCE', '0.30'),
            travel=number('ES_QUERY_W_TRAVEL', '0.35'),
            novelty=number('ES_QUERY_W_NOVELTY', '0.15'),
            revisit=number('ES_QUERY_W_REVISIT', '0.80'),
            failure=number('ES_QUERY_W_FAILURE', '1.0'),
            switch_margin=number('ES_QUERY_SWITCH_MARGIN', '0.15'),
            match_radius_m=number('ES_CANDIDATE_EVIDENCE_RADIUS', '0.75'),
        ),
    )


def _strict_geodesic_distance(pathfinder, start, end):
    from adapters.geometry import strict_geodesic_distance
    return strict_geodesic_distance(pathfinder, start, end)


def _projected_candidate_batch(candidate_batch, projected, projected_indices):
    from pbsp.navigation.planning import bind_projected_candidates
    return bind_projected_candidates(candidate_batch, projected, projected_indices)


def _subtask_has_acted(agent, task_idx):
    from pbsp.navigation.planning import subtask_has_acted
    return subtask_has_acted(agent, task_idx)


def _rank_scsp_idea1_layers(reranker, candidate_batch, origin, pathfinder, agent):
    from pbsp.navigation.planning import rank_candidate_layers
    return rank_candidate_layers(reranker, candidate_batch, origin, lambda start, end: _strict_geodesic_distance(pathfinder, start, end), agent)


def _build_scsp_geodesic_matrices(candidate_batch, current_task, idea1_candidates_by_task, pathfinder, distance_cache, candidate_cap):
    from pbsp.navigation.planning import build_geodesic_matrices
    return build_geodesic_matrices(candidate_batch, current_task, idea1_candidates_by_task, lambda start, end: _strict_geodesic_distance(pathfinder, start, end), distance_cache, candidate_cap)


def _scsp_shadow_fallback_event(
    *,
    phase,
    arm,
    current_task,
    reason,
    version_gate,
    candidate_batch,
    origin,
    started_at,
    error=None,
    failure_stage="shadow_compute",
    provenance=None,
    candidate_cap=None,
    suffix_weight=None,
    idea1_config=None,
    episode_identity=None
):
    from experiments.recorder import planning_fallback_event

    return planning_fallback_event(
        phase=phase,
        arm=arm,
        current_task=current_task,
        reason=reason,
        version_gate=version_gate,
        candidate_batch=candidate_batch,
        origin=origin,
        started_at=started_at,
        error=error,
        failure_stage=failure_stage,
        provenance=provenance,
        candidate_cap=candidate_cap,
        suffix_weight=suffix_weight,
        idea1_config=idea1_config,
        episode_identity=episode_identity,
    )


def _compute_scsp_shadow_decision(**kwargs):
    from runtime.legacy_experiments import planning_bridge_from_environment

    return planning_bridge_from_environment().compute_decision(**kwargs)


def _record_scsp_shadow_decision(**kwargs):
    from runtime.legacy_experiments import planning_bridge_from_environment

    return planning_bridge_from_environment().record_decision(**kwargs)


def _fallback_value(value):
    from pbsp.navigation.planning import fallback_value
    return fallback_value(value)


def _scsp_dispatch_gate(provider, decision, *, scene_id, episode_id, ordered_categories):
    from pbsp.navigation.planning import dispatch_version_gate

    return dispatch_version_gate(
        provider,
        decision,
        scene_id=scene_id,
        episode_id=episode_id,
        ordered_categories=ordered_categories,
    )


def _prepare_scsp_refresh(**kwargs):
    from runtime.legacy_experiments import planning_bridge_from_environment

    return planning_bridge_from_environment().prepare_refresh(**kwargs)


def _scsp_raw_ranks_for_positions(candidate_batch, task_idx, positions):
    from pbsp.navigation.planning import raw_ranks_for_positions
    return raw_ranks_for_positions(candidate_batch, task_idx, positions)


def _scsp_raw_rank_for_position(candidate_batch, task_idx, position):
    from pbsp.navigation.planning import raw_rank_for_position
    return raw_rank_for_position(candidate_batch, task_idx, position)


def _apply_scsp_runtime_order(projected_goals, decision, runtime_mode):
    from pbsp.navigation.planning import apply_runtime_order
    return apply_runtime_order(projected_goals, decision, runtime_mode)


def _finalize_scsp_dispatch_event(
    decision,
    dispatch_metadata,
    *,
    cooldown_event,
    refresh_applied,
    realized_active_position,
    realized_raw_rank,
    refresh_error=None
):
    from experiments.recorder import finalize_dispatch_event

    return finalize_dispatch_event(
        decision,
        dispatch_metadata,
        cooldown_event=cooldown_event,
        refresh_applied=refresh_applied,
        realized_active_position=realized_active_position,
        realized_raw_rank=realized_raw_rank,
        refresh_error=refresh_error,
    )


def _reset_scsp_task_state(reranker, distance_cache):
    from pbsp.navigation.planning import reset_task_planning
    return reset_task_planning(reranker, distance_cache)


def _active_candidate(agent, task_idx):
    from pbsp.navigation.planning import active_candidate
    return active_candidate(agent, task_idx)


def _sample_navmesh_goal(pathfinder, origin, mode):
    from adapters.geometry import sample_navmesh_goal

    max_vertical_delta = float(os.environ.get('ES_NAVMESH_MAX_VERTICAL_DELTA', '0.35'))
    sample_count = int(os.environ.get('ES_NAVMESH_SAMPLE_COUNT', '32'))
    parameters = {}
    if mode == 'escape':
        parameters['escape_radius_m'] = float(os.environ.get('ES_NAVMESH_ESCAPE_RADIUS', '3.0'))
        parameters['escape_min_distance_m'] = float(os.environ.get('ES_NAVMESH_ESCAPE_MIN_DISTANCE', '1.25'))
    elif mode == 'random':
        parameters['random_min_distance_m'] = float(os.environ.get('ES_NAVMESH_RANDOM_MIN_DISTANCE', '2.0'))
    return sample_navmesh_goal(
        pathfinder, origin, mode,
        max_vertical_delta=max_vertical_delta,
        sample_count=sample_count,
        **parameters,
    )


def _write_timing_rows(rows, results_dir):
    from experiments.recorder import write_timing_rows

    return write_timing_rows(rows, results_dir)


def _print_timing_summary(rows, results_dir, save_csv=True):
    from experiments.recorder import print_timing_summary

    return print_timing_summary(rows, results_dir, save_csv)


if __name__ == '__main__':
    from runtime.navigation_worker import main
    main()
