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

from online_es_provider import OnlineESProvider
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


def _compute_scsp_shadow_decision(
    *,
    phase,
    arm,
    provider_event,
    candidate_batch,
    projected_goals,
    projected_indices,
    current_task,
    origin,
    pathfinder,
    agent,
    reranker,
    distance_cache,
    candidate_cap,
    suffix_weight,
    idea1_config,
    provenance,
    episode_identity
):
    """Compute one suffix decision while leaving legacy candidates untouched."""
    started_at = time.perf_counter()
    version_gate = dict(provider_event.get("version_gate", {}))
    if candidate_batch is None:
        return _scsp_shadow_fallback_event(
            phase=phase,
            arm=arm,
            current_task=current_task,
            reason=FallbackReason.INVALID_CANDIDATE_BATCH.value,
            version_gate=version_gate,
            candidate_batch=None,
            origin=origin,
            started_at=started_at,
            failure_stage="candidate_batch",
            provenance=provenance,
            candidate_cap=candidate_cap,
            suffix_weight=suffix_weight,
            idea1_config=idea1_config,
            episode_identity=episode_identity,
        )
    if version_gate.get("would_invalidate", True):
        return _scsp_shadow_fallback_event(
            phase=phase,
            arm=arm,
            current_task=current_task,
            reason=version_gate.get("reason", FallbackReason.PROVIDER_STATUS_UNAVAILABLE.value),
            version_gate=version_gate,
            candidate_batch=candidate_batch,
            origin=origin,
            started_at=started_at,
            failure_stage="version_gate",
            provenance=provenance,
            candidate_cap=candidate_cap,
            suffix_weight=suffix_weight,
            idea1_config=idea1_config,
            episode_identity=episode_identity,
        )
    try:
        from pbsp.navigation.planning import compute_candidate_plan

        calculation = compute_candidate_plan(
            candidate_batch=candidate_batch,
            projected_goals=projected_goals,
            projected_indices=projected_indices,
            current_task=current_task,
            origin=origin,
            geodesic=lambda start, end: _strict_geodesic_distance(pathfinder, start, end),
            agent=agent,
            reranker=reranker,
            distance_cache=distance_cache,
            candidate_cap=candidate_cap,
            suffix_weight=suffix_weight,
            clock=time.perf_counter,
            started_at=started_at,
        )
        event = build_scsp_shadow_decision(
            phase=phase,
            arm=arm,
            candidate_batch=calculation.candidate_batch,
            suffix_plan=calculation.suffix_plan,
            legacy_order=calculation.legacy_order,
            idea1_order=calculation.idea1_order,
            suffix_order=tuple(calculation.suffix_order),
            idea1_candidates=calculation.idea1_candidates,
            rerank_diagnostics=calculation.rerank_diagnostics,
            version_gate=version_gate,
            origin_position=[float(value) for value in origin],
            timing_ms=calculation.timing_ms,
            distance_cache=calculation.distance_cache,
            candidate_cap=int(candidate_cap),
            suffix_weight=float(suffix_weight),
            idea1_config=idea1_config,
            provenance=provenance,
            episode_identity=episode_identity,
        )
        event["status"] = "planned"
        event["provider_candidate_batch"] = candidate_batch
        event["identity_sidecar_sha256"] = provenance.get("identity_sidecar_sha256")
        return event
    except Exception as exc:
        reason = (
            FallbackReason.INVALID_CANDIDATE_BATCH.value
            if isinstance(exc, SCSPContractError)
            else FallbackReason.PLANNER_ERROR.value
        )
        return _scsp_shadow_fallback_event(
            phase=phase,
            arm=arm,
            current_task=current_task,
            reason=reason,
            version_gate=version_gate,
            candidate_batch=candidate_batch,
            origin=origin,
            started_at=started_at,
            error=exc,
            failure_stage="projection_rerank_or_planning",
            provenance=provenance,
            candidate_cap=candidate_cap,
            suffix_weight=suffix_weight,
            idea1_config=idea1_config,
            episode_identity=episode_identity,
        )


def _record_scsp_shadow_decision(
    *, output_path=None, decision_event=None, **kwargs
):
    """Persist best-effort SCSP telemetry after the dispatch transaction."""

    compute_kwargs = dict(kwargs)
    try:
        if decision_event is not None:
            event = dict(decision_event)
        else:
            provider = compute_kwargs.pop('provider', None)
            if provider is not None and 'candidate_batch' not in compute_kwargs:
                compute_kwargs['candidate_batch'] = provider.get_candidate_batch()
            event = _compute_scsp_shadow_decision(**compute_kwargs)
    except Exception as exc:
        try:
            event = _scsp_shadow_fallback_event(
                phase=kwargs.get('phase', 'unknown'),
                arm=kwargs.get('arm', 'unknown'),
                current_task=kwargs.get('current_task', 0),
                reason=FallbackReason.PLANNER_ERROR.value,
                version_gate=(
                    kwargs.get('provider_event', {}).get('version_gate', {})
                ),
                candidate_batch=compute_kwargs.get('candidate_batch'),
                origin=kwargs.get('origin', (0.0, 0.0, 0.0)),
                started_at=time.perf_counter(),
                error=exc,
                failure_stage='unhandled_shadow_compute',
                provenance=kwargs.get('provenance'),
                candidate_cap=kwargs.get('candidate_cap'),
                suffix_weight=kwargs.get('suffix_weight'),
                idea1_config=kwargs.get('idea1_config'),
                episode_identity=kwargs.get('episode_identity'),
            )
        except Exception as fallback_exc:
            fallback_mode = str(kwargs.get('runtime_mode', 'shadow'))
            event = {
                'schema_version': SHADOW_SCHEMA_VERSION,
                'phase': str(kwargs.get('phase', 'unknown')),
                'arm': str(kwargs.get('arm', 'unknown')),
                'mode': fallback_mode,
                'shadow_only': fallback_mode == 'shadow',
                'status': 'fallback',
                'dispatch_order_source': 'not_applied',
                'current_task': 0,
                'fallback_reason': FallbackReason.PLANNER_ERROR.value,
                'error': {
                    'type': type(exc).__name__,
                    'message': str(exc),
                },
                'fallback_construction_error': {
                    'type': type(fallback_exc).__name__,
                    'message': str(fallback_exc),
                },
            }
    from experiments.recorder import record_planning_decision

    return record_planning_decision(
        event, output_path=output_path,
        current_task_hint=kwargs.get('current_task', 0),
        serialize=dumps_json_strict,
    )


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


def _prepare_scsp_refresh(
    *,
    phase,
    runtime_mode,
    arm,
    provider,
    provider_event,
    raw_goals,
    current_task,
    origin,
    pathfinder,
    agent,
    reranker,
    distance_cache,
    candidate_cap,
    suffix_weight,
    idea1_config,
    provenance,
    episode_identity,
    scene_id,
    episode_id,
    ordered_categories,
    provider_events_path=None,
    preserve_existing_on_missing=False,
):
    from pbsp.navigation.planning import prepare_candidate_refresh

    def plan_once(attempt_goals, attempt_provider_event):
        (projected, projected_indices) = _project_goals_to_navigable(
            attempt_goals, pathfinder, return_indices=True
        )
        decision = _compute_scsp_shadow_decision(
            phase=phase,
            arm=arm,
            provider_event=attempt_provider_event,
            candidate_batch=provider.get_candidate_batch(),
            projected_goals=projected,
            projected_indices=projected_indices,
            current_task=current_task,
            origin=origin,
            pathfinder=pathfinder,
            agent=agent,
            reranker=reranker,
            distance_cache=distance_cache,
            candidate_cap=candidate_cap,
            suffix_weight=0.0 if runtime_mode == "idea1_only" else suffix_weight,
            idea1_config=idea1_config,
            provenance=provenance,
            episode_identity=episode_identity,
        )
        gate = _scsp_dispatch_gate(
            provider,
            decision,
            scene_id=scene_id,
            episode_id=episode_id,
            ordered_categories=ordered_categories,
        )
        return (projected, decision, gate)

    def make_fallback(exc):
        return _scsp_shadow_fallback_event(
            phase=phase,
            arm=arm,
            current_task=current_task,
            reason=FallbackReason.PLANNER_ERROR.value,
            version_gate=provider_event.get("version_gate", {}),
            candidate_batch=None,
            origin=origin,
            started_at=time.perf_counter(),
            error=exc,
            failure_stage="unhandled_plan_attempt",
            provenance=provenance,
            candidate_cap=candidate_cap,
            suffix_weight=suffix_weight,
            idea1_config=idea1_config,
            episode_identity=episode_identity,
        )

    def read_retry_event():
        return _record_scsp_provider_event(
            provider,
            phase=f"{phase}_synchronous_requery",
            current_task=current_task,
            scene_id=scene_id,
            episode_id=episode_id,
            ordered_categories=ordered_categories,
            output_path=provider_events_path,
        )

    result = prepare_candidate_refresh(
        runtime_mode=runtime_mode,
        provider=provider,
        provider_event=provider_event,
        raw_goals=raw_goals,
        plan_once=plan_once,
        project_fallback=lambda goals: _project_goals_to_navigable(goals, pathfinder),
        make_fallback=make_fallback,
        read_dispatch_gate=lambda decision: _scsp_dispatch_gate(
            provider,
            decision,
            scene_id=scene_id,
            episode_id=episode_id,
            ordered_categories=ordered_categories,
        ),
        read_retry_event=read_retry_event,
        preserve_existing_on_missing=preserve_existing_on_missing,
    )
    return (result.projected_goals, result.decision, result.retry_provider_events)


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
    print('Start time:', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--habitat_config_path', type=str,
                        default='project/config/habitat/lagmemo_hm3d.yaml')
    parser.add_argument('--baseline_config_path', type=str,
                        default='project/config/agent/hm3d_eval.yaml')
    parser.add_argument('--scene_idx', type=int, default=0)
    parser.add_argument('--scenes', type=str, default='5cdEh9F2hJL')
    parser.add_argument('--episode_start', type=int, default=0)
    parser.add_argument('--episode_end', type=int, default=None)
    parser.add_argument('--seem_ckpt_path', type=str,
                        default='./src/third_party/seem/checkpoints/seem_focall_v0.pt')
    parser.add_argument('--seem_yaml_path', type=str,
                        default='./src/third_party/seem/configs/seem/focall_unicl_lang_demo.yaml')
    parser.add_argument('--mobileclip_ckpt_path', type=str,
                        default='./src/third_party/ml-mobileclip/checkpoints/mobileclip_s0.pt')
    parser.add_argument('--output_path', type=str,
                        default=str(_TDONAV_ROOT / 'Exp' / 'runs' / 'p4_demo'))
    parser.add_argument('--input_data', type=str, default='3_episode_data')
    parser.add_argument(
        '--dataset_root',
        type=str,
        default=None,
        help='Absolute GOAT dataset root containing val_seen.json.gz and content/.',
    )
    parser.add_argument('--socket_path', type=str, default='/tmp/es_mapper.sock')
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument(
        '--max_steps_per_subtask',
        type=int,
        default=None,
        help='Override config.AGENT.max_steps for each ordered subtask.',
    )
    parser.add_argument('--live_mode', action='store_true',
                        help='Phase 5: stream live obs to ES server')
    parser.add_argument('--enable_timing', action='store_true',
                        help='Record per-step timing and print compliance report')
    parser.set_defaults(print_images=True)
    parser.add_argument('--print-images', dest='print_images', action='store_true',
                        help='Write the shared 3x2 snapshot at each planner keyframe')
    parser.add_argument('--no-print-images', dest='print_images', action='store_false',
                        help='Disable per-keyframe snapshots for throughput-only runs')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    print(json.dumps(vars(args), indent=4))
    print('-' * 80)

    dataset_root = (
        Path(args.dataset_root).expanduser().resolve()
        if args.dataset_root
        else _LAGMEMO_ROOT / 'data' / 'datasets' / 'goat' / 'hm3d' / args.input_data
    )
    dataset_index = dataset_root / 'val_seen.json.gz'
    dataset_content = dataset_root / 'content'
    if not dataset_index.is_file() or not dataset_content.is_dir():
        raise FileNotFoundError(
            f'dataset_root must contain val_seen.json.gz and content/: {dataset_root}'
        )
    print(f'[ES eval] dataset_root={dataset_root}', flush=True)

    timing_rows = []   # filled when --enable_timing

    # ── Provider ────────────────────────────────────────────────────────
    print('[P4] Connecting to ES Mapper Server...', flush=True)
    candidate_provider = OnlineESProvider(
        socket_path=args.socket_path,
        top_k=args.top_k,
        connect_timeout=120.0,
        content_root=str(dataset_content),
    )
    if args.live_mode:
        candidate_provider._live_mode = True
        print('[Phase 5] Live mode ON: streaming frames to ES server', flush=True)
    print('[P4] ES Mapper Server connected.', flush=True)

    # ── LagMemo setup ───────────────────────────────────────────────────
    seem_cfg = {
        'conf_path': args.seem_yaml_path,
        'ckpt_path': args.seem_ckpt_path,
    }

    config = get_config(args.habitat_config_path, args.baseline_config_path)
    scsp_runtime_mode = _scsp_runtime_mode()
    scsp_arm = os.environ.get('ES_SCSP_ARM', 'legacy')
    scsp_profile_mode = os.environ.get(
        'ES_SCSP_PROFILE_MODE', 'exact_baseline'
    )
    expected_scsp_arm = {
        'exact_baseline': 'legacy',
        'idea1_only': 'idea1_only',
        'ordered_suffix_mvp': 'ordered_suffix_mvp',
    }[scsp_profile_mode]
    if scsp_arm != expected_scsp_arm:
        raise ValueError(
            'SCSP arm/profile mismatch: '
            f'{scsp_arm!r}/{scsp_profile_mode!r}'
        )
    if scsp_runtime_mode != 'off':
        if os.environ.get('LAGMEMO_SEMANTIC_SOURCE') != 'detic':
            raise ValueError('SCSP runtime requires Detic semantics')
        if os.environ.get('ES_GROUND_TRUTH_SEMANTICS') != '0':
            raise ValueError(
                'SCSP runtime requires GROUND_TRUTH_SEMANTICS=0'
            )
        config.GROUND_TRUTH_SEMANTICS = 0
        print(
            '[SCSP_RUNTIME] inference-only Detic registration stubs='
            f'{list(_EARLY_DETIC_REGISTRATION_STUBS)}',
            flush=True,
        )
    if args.max_steps_per_subtask is not None:
        if args.max_steps_per_subtask < 1:
            raise ValueError('--max_steps_per_subtask must be positive')
        config.AGENT.max_steps = args.max_steps_per_subtask
    print(
        f'[ES eval] max_steps_per_subtask={config.AGENT.max_steps}',
        flush=True,
    )
    # ES online candidates are indexed directly by Habitat subtask id.
    # LagMemo's EXP_FIRST mutates Habitat episodes by inserting a duplicate
    # first subtask, which shifts candidate/query indices and corrupts SR.
    config.EXP_FIRST = False
    try:
        config.habitat.environment.iterator_options.shuffle = False
    except Exception:
        pass
    print('[ES eval] EXP_FIRST=False; episode iterator shuffle=False', flush=True)
    if args.output_path:
        config.DUMP_LOCATION = args.output_path
    config['habitat']['dataset']['data_path'] = str(dataset_index)
    all_scenes = sorted([
        x.name.removesuffix('.json.gz')
        for x in dataset_content.glob('*.json.gz')
    ])
    config.habitat.dataset.content_scenes = \
        all_scenes if args.scenes == 'all' else [args.scenes]
    config.NUM_ENVIRONMENTS = 1
    config.PRINT_IMAGES = int(args.print_images)
    config.EXP_NAME = f'{config.EXP_NAME}_{args.scene_idx}'

    habitat_env = Env(config)
    total_episodes = len(habitat_env.episodes)
    episode_start = args.episode_start
    episode_end = total_episodes if args.episode_end is None else args.episode_end
    if not 0 <= episode_start < episode_end <= total_episodes:
        raise ValueError(
            f'invalid episode range [{episode_start}, {episode_end}) '
            f'for {total_episodes} episodes'
        )
    habitat_env.episode_iterator = iter(
        habitat_env.episodes[episode_start:episode_end]
    )
    print(
        f'[ES eval] episode_range=[{episode_start}, {episode_end})',
        flush=True,
    )
    env = HabitatGoatEnv(habitat_env, config=config)
    evaluator = env.evaluator
    agent = GLUEAgent(config=config, seem_cfg=seem_cfg,
                      clip_cfg=args.mobileclip_ckpt_path)
    agent.top_down_map_reader = evaluator.top_down_map
    category_cooldown = CategorySwitchCooldown(
        parse_category_cooldowns(
            os.environ.get('ES_QUERY_SWITCH_COOLDOWN_BY_CATEGORY', '{}')
        ),
        match_radius_m=float(
            os.environ.get('ES_CANDIDATE_TRACK_RADIUS', '1.0')
        ),
    )
    if category_cooldown.enabled and not agent.query_action_rerank_enabled:
        raise ValueError(
            'category cooldown requires ES_QUERY_ACTION_RERANK_ENABLED=1'
    )
    scsp_enabled = scsp_runtime_mode != 'off'
    if _enable_scsp_no_image_planning_state(
        agent,
        scsp_enabled=scsp_enabled,
        print_images=args.print_images,
    ):
        print(
            '[SCSP_RUNTIME] planner maps enabled with image writes disabled',
            flush=True,
        )
    if scsp_enabled:
        scsp_ordered_subtasks = int(
            os.environ.get('ES_SCSP_ORDERED_SUBTASKS', '0')
        )
        if scsp_ordered_subtasks < 1:
            raise ValueError(
                'SCSP requires ES_SCSP_ORDERED_SUBTASKS to be positive'
            )
        scsp_candidate_cap = int(
            os.environ.get(
                'ES_SCSP_CANDIDATE_CAP', str(DEFAULT_CANDIDATE_CAP)
            )
        )
        scsp_suffix_weight = float(
            os.environ.get(
                'ES_SCSP_SUFFIX_WEIGHT', str(DEFAULT_SUFFIX_WEIGHT)
            )
        )
        if (
            os.environ.get('ES_SCSP_STRICT_GEODESIC', '1') != '1'
            or os.environ.get('ES_SCSP_EUCLIDEAN_FALLBACK', '0') != '0'
        ):
            raise ValueError(
                'SCSP requires strict geodesics without fallback'
            )
        scsp_reranker = _candidate_reranker_from_env(True)
        scsp_idea1_config = to_jsonable(scsp_reranker.config)
        scsp_provenance = {
            'profile_sha256': os.environ.get(
                'ES_SCSP_PROFILE_SHA256', ''
            ),
            'code_sha': os.environ.get('ES_SCSP_CODE_SHA', ''),
        }
        dataset_manifest_path = dataset_root / 'dataset_manifest.json'
        if not dataset_manifest_path.is_file():
            raise FileNotFoundError(
                f'SCSP dataset manifest is missing: {dataset_manifest_path}'
            )
        observed_dataset_sha256 = hashlib.sha256(
            dataset_manifest_path.read_bytes()
        ).hexdigest()
        expected_dataset_sha256 = os.environ.get(
            'ES_SCSP_EXPECTED_DATASET_MANIFEST_SHA256', ''
        )
        if observed_dataset_sha256 != expected_dataset_sha256:
            raise ValueError('SCSP dataset manifest hash mismatch')
        scsp_provenance['dataset_manifest_sha256'] = (
            observed_dataset_sha256
        )
        scsp_episode_identities = _load_scsp_frozen_identities(
            observed_dataset_sha256, scsp_ordered_subtasks
        )
        extra_identities, sidecar_sha256 = _load_scsp_identity_sidecar(
            os.environ.get('ES_SCSP_IDENTITY_SIDECAR', ''),
            observed_dataset_sha256,
            scsp_ordered_subtasks,
        )
        overlap = set(scsp_episode_identities) & set(extra_identities)
        if overlap:
            raise ValueError(
                f'SCSP identity sidecar overlaps frozen identities: {overlap}'
            )
        scsp_episode_identities.update(extra_identities)
        scsp_provenance['identity_sidecar_sha256'] = sidecar_sha256
    else:
        scsp_candidate_cap = DEFAULT_CANDIDATE_CAP
        scsp_suffix_weight = DEFAULT_SUFFIX_WEIGHT
        scsp_reranker = CandidateActionReranker(enabled=False)
        scsp_idea1_config = {}
        scsp_provenance = {}
        scsp_episode_identities = {}
    scsp_distance_cache = {}
    print(
        '[SCSP_RUNTIME] '
        f'mode={scsp_runtime_mode} arm={scsp_arm} '
        f'candidate_cap={scsp_candidate_cap} '
        f'suffix_weight={scsp_suffix_weight}',
        flush=True,
    )

    results_dir = os.path.join(config.DUMP_LOCATION,
                               f'results_{args.scenes}', config.EXP_NAME)
    os.makedirs(results_dir, exist_ok=True)
    existing_outputs = [
        name for name in (
            'per_episode_metrics.json',
            'timing_log.csv',
            'scsp_provider_events.jsonl',
            'scsp_decisions.jsonl',
        )
        if os.path.exists(os.path.join(results_dir, name))
    ]
    if existing_outputs:
        raise RuntimeError(
            'MON standard-metric runs require a fresh results directory; '
            f'in-place resume is unsupported: {existing_outputs}'
        )
    scsp_provider_events_path = Path(results_dir) / 'scsp_provider_events.jsonl'
    scsp_decisions_path = Path(results_dir) / 'scsp_decisions.jsonl'

    metrics = {}
    es_status_log = []

    # ── Episode loop ─────────────────────────────────────────────────────
    for ep_i in range(episode_start, episode_end):
        env.reset()
        scene_id = env.habitat_env.current_episode.scene_id.split('/')[-1].split('.')[0]
        episode  = env.habitat_env.current_episode
        episode_id = episode.episode_id
        scsp_episode_identity = scsp_episode_identities.get(
            (str(scene_id), str(episode_id))
        )
        if scsp_enabled and scsp_episode_identity is None:
            raise RuntimeError(
                'SCSP active/shadow execution requires a complete '
                'identity-only episode record'
            )

        candidate_provider.reset_episode(scene_id, episode_id)
        candidate_provider.invalidate_cache()

        agent.reset(start_position=episode.start_position,
                    start_rotation=episode.start_rotation,
                    candidate_provider=candidate_provider)
        agent.set_runtime_context(scene_id, episode_id)
        category_cooldown.reset()
        scsp_reranker.reset()
        scsp_distance_cache.clear()
        category_cooldown_events = []
        scsp_decisions = []
        initial_provider_event = _record_scsp_provider_event(
            candidate_provider,
            phase='initial_query',
            current_task=0,
            scene_id=scene_id,
            episode_id=episode_id,
            output_path=scsp_provider_events_path,
        )
        scsp_provider_events = [initial_provider_event]
        initial_raw_goals = agent.gs_poses
        initial_refresh_pending = scsp_enabled
        if not scsp_enabled:
            initial_goals = _project_goals_to_navigable(
                agent.gs_poses, env.habitat_env.sim.pathfinder
            )
            agent.refresh_gs_poses(initial_goals)
        # ES: disable exploration-first at task_idx=0 (legacy LagMemo behavior)
        # test_gs=3 → no exploration phase, use gs_poses for all tasks directly
        agent._es_test_gs_override = 3

        t = 0
        scsp_last_task_idx = 0
        scsp_subtask_reset_count = 0
        scsp_boundary_replan_pending = False

        if os.path.exists(os.path.join(results_dir, 'per_episode_metrics.json')):
            with open(os.path.join(results_dir, 'per_episode_metrics.json')) as fp:
                metrics = json.load(fp)
        if f'{scene_id}_{episode_id}' in metrics:
            continue

        agent.planner.set_vis_dir(scene_id,
            f'{episode_id}_{env.habitat_env.task.current_task_idx}')
        agent.imagenav_visualizer.set_vis_dir(
            f'{scene_id}_{episode_id}_{env.habitat_env.task.current_task_idx}')
        agent.matching.set_vis_dir(
            f'{scene_id}_{episode_id}_{env.habitat_env.task.current_task_idx}')
        env.visualizer.set_vis_dir(scene_id,
            f'{episode_id}_{env.habitat_env.task.current_task_idx}')
        agent.set_lightglue_vis_dir(
            f'{scene_id}_{episode_id}_{env.habitat_env.task.current_task_idx}')

        all_subtask_metrics = []
        all_stop_reasons = []
        episode_frame_times = []
        subtask_frame_times = {}
        pbar = tqdm(total=config.AGENT.max_steps)

        n_subtasks = None   # P6_FALLBACK: set at t==1; used by episode-boundary guard
        while not env.episode_over:
            current_task_idx = env.habitat_env.task.current_task_idx
            # P6_FALLBACK (PROGRESS §4.3): habitat advances current_task_idx past the
            # last sub-task before flipping episode_over, so agent.act() gets an
            # out-of-range idx and KeyErrors (frontier_flag/gs_poses keyed 0..N-1),
            # which previously killed the WHOLE run at every episode boundary.
            # End the episode cleanly instead.
            if n_subtasks is not None and current_task_idx >= n_subtasks:
                print(f'[P6 boundary] current_task_idx={current_task_idx} >= '
                      f'n_subtasks={n_subtasks}; ending episode cleanly', flush=True)
                break
            if scsp_enabled and current_task_idx != scsp_last_task_idx:
                _reset_scsp_task_state(scsp_reranker, scsp_distance_cache)
                scsp_last_task_idx = int(current_task_idx)
                scsp_subtask_reset_count += 1
                scsp_boundary_replan_pending = scsp_runtime_mode in {
                    'idea1_only', 'ordered_suffix_mvp'
                }
                print(
                    '[SCSP_RUNTIME] cleared plan state for '
                    f'current_task={current_task_idx}',
                    flush=True,
                )
            t += 1

            t_step_start = time.perf_counter()

            obs = env.get_observation()
            if t == 1:
                obs_tasks = []
                for task in obs.task_observations['tasks']:
                    obs_task = {k: v for k, v in task.items() if k != 'image'}
                    obs_tasks.append(obs_task)
                n_subtasks = len(obs_tasks)   # P6_FALLBACK: sub-task count for boundary guard
                if scsp_enabled and n_subtasks != scsp_ordered_subtasks:
                    raise RuntimeError(
                        'SCSP ordered_subtasks does not match runtime tasks'
                    )
                if category_cooldown.enabled:
                    for task in obs_tasks:
                        cooldown_for_category(
                            category_cooldown.cooldown_by_category,
                            _task_category(task),
                        )
                pprint(obs_tasks)
                runtime_categories = tuple(
                    _task_category(task) for task in obs_tasks
                )
                if scsp_episode_identity is not None and (
                    scsp_episode_identity['scene_id'] != str(scene_id)
                    or scsp_episode_identity['runtime_episode_id']
                    != str(episode_id)
                    or tuple(scsp_episode_identity['ordered_categories'])
                    != runtime_categories
                ):
                    raise RuntimeError(
                        'SCSP episode identity does not match runtime tasks'
                    )
                runtime_context_event = _record_scsp_provider_event(
                    candidate_provider,
                    phase='runtime_context',
                    current_task=current_task_idx,
                    scene_id=scene_id,
                    episode_id=episode_id,
                    ordered_categories=runtime_categories,
                    output_path=scsp_provider_events_path,
                )
                scsp_provider_events.append(runtime_context_event)
                if initial_refresh_pending:
                    (
                        initial_goals,
                        initial_decision,
                        retry_events,
                    ) = _prepare_scsp_refresh(
                        phase='initial_runtime_dispatch',
                        runtime_mode=scsp_runtime_mode,
                        arm=scsp_arm,
                        provider=candidate_provider,
                        provider_event=runtime_context_event,
                        raw_goals=initial_raw_goals,
                        current_task=current_task_idx,
                        origin=np.asarray(
                            episode.start_position, dtype=np.float32
                        ),
                        pathfinder=env.habitat_env.sim.pathfinder,
                        agent=agent,
                        reranker=scsp_reranker,
                        distance_cache=scsp_distance_cache,
                        candidate_cap=scsp_candidate_cap,
                        suffix_weight=scsp_suffix_weight,
                        idea1_config=scsp_idea1_config,
                        provenance=scsp_provenance,
                        episode_identity=scsp_episode_identity,
                        scene_id=scene_id,
                        episode_id=episode_id,
                        ordered_categories=runtime_categories,
                        provider_events_path=scsp_provider_events_path,
                    )
                    scsp_provider_events.extend(retry_events)
                    initial_goals, initial_dispatch_metadata = (
                        _apply_scsp_runtime_order(
                            initial_goals,
                            initial_decision,
                            scsp_runtime_mode,
                        )
                    )
                    initial_dispatch = agent.dispatch_initial_candidates(
                        initial_goals, initial_dispatch_metadata,
                        task_idx=current_task_idx,
                        candidate_batch=initial_decision.get('candidate_batch'),
                    )
                    initial_refresh_applied = initial_dispatch.refresh_applied
                    initial_refresh_error = initial_dispatch.refresh_error
                    initial_realized = initial_dispatch.realized_active_position
                    initial_realized_rank = initial_dispatch.realized_raw_rank
                    initial_decision = _finalize_scsp_dispatch_event(
                        initial_decision,
                        initial_dispatch_metadata,
                        cooldown_event=None,
                        refresh_applied=initial_refresh_applied,
                        realized_active_position=initial_realized,
                        realized_raw_rank=initial_realized_rank,
                        refresh_error=initial_refresh_error,
                    )
                    scsp_decisions.append(
                        _record_scsp_shadow_decision(
                            decision_event=initial_decision,
                            output_path=scsp_decisions_path,
                        )
                    )
                    if not initial_refresh_applied:
                        raise RuntimeError(
                            'SCSP initial candidate refresh failed closed: '
                            f'{initial_refresh_error}'
                        )
                    initial_refresh_pending = False
                    initial_raw_goals = None

            local_step = int(
                agent.sub_task_timesteps[0][current_task_idx]
            )
            if category_cooldown.enabled and local_step > 0:
                realized_active = _active_candidate(agent, current_task_idx)
                external_switch = category_cooldown.observe_active(
                    current_task_idx,
                    realized_active,
                    local_step,
                )
                if external_switch:
                    external_event = {
                        'task_idx': int(current_task_idx),
                        'category': _task_category(
                            obs_tasks[current_task_idx]
                        ),
                        'subtask_step': local_step,
                        'cooldown_steps': cooldown_for_category(
                            category_cooldown.cooldown_by_category,
                            _task_category(obs_tasks[current_task_idx]),
                        ),
                        'reason': 'EXTERNAL_ACTIVE_SWITCH',
                        'realized_switch': 1,
                        'realized_active_position': (
                            None
                            if realized_active is None
                            else [float(value) for value in realized_active]
                        ),
                    }
                    category_cooldown_events.append(external_event)
                    print(
                        '[CATEGORY_COOLDOWN] '
                        f'{json.dumps(external_event, sort_keys=True)}',
                        flush=True,
                    )

            # ── push_live_frame (Phase 5) or push_frame (Phase 4) ──────────
            dt_push = 0.0
            try:
                t0 = time.perf_counter()
                if getattr(candidate_provider, '_live_mode', False):
                    # Phase 5: send actual Habitat obs to ES server
                    _rgb_raw = obs.rgb                                  # (H,W,3) uint8
                    _dep_raw = obs.depth                                # (H,W) float32 m
                    _gp_raw  = np.array(obs.globalpose, dtype=np.float32)  # (7,)
                    candidate_provider.push_live_frame(_rgb_raw, _dep_raw, _gp_raw)
                else:
                    # Phase 4: snapshot-based, send frame index only
                    candidate_provider.push_frame(t - 1)
                dt_push = (time.perf_counter() - t0) * 1000
            except Exception as e:
                print(f'[P5] push_frame warning: {e}', flush=True)

            # ── periodic goal refresh ───────────────────────────────────
            dt_goals = 0.0
            is_upgrade = False
            periodic_refresh_due = t % 10 == 0
            if periodic_refresh_due or scsp_boundary_replan_pending:
                refresh_phase = (
                    'periodic_refresh'
                    if periodic_refresh_due
                    else 'subtask_boundary'
                )
                t0 = time.perf_counter()
                new_goals = None
                query_error = None
                if scsp_boundary_replan_pending and not periodic_refresh_due:
                    try:
                        cached_batch = candidate_provider.get_candidate_batch()
                        if cached_batch is not None:
                            new_goals = {
                                task_idx: [list(position) for position in positions]
                                for task_idx, positions
                                in cached_batch.positions_by_task
                            }
                    except Exception as e:
                        query_error = f'{type(e).__name__}: {e}'
                else:
                    candidate_provider.invalidate_cache()
                    try:
                        new_goals = candidate_provider.get_all_goals()
                    except Exception as e:
                        query_error = f'{type(e).__name__}: {e}'
                        print(f'[P4] goal refresh warning: {e}', flush=True)
                periodic_provider_event = _record_scsp_provider_event(
                    candidate_provider,
                    phase=refresh_phase,
                    current_task=current_task_idx,
                    scene_id=scene_id,
                    episode_id=episode_id,
                    ordered_categories=tuple(
                        _task_category(task) for task in obs_tasks
                    ),
                    output_path=scsp_provider_events_path,
                    query_error=query_error,
                )
                scsp_provider_events.append(periodic_provider_event)
                if scsp_enabled or new_goals:
                    projected_goals = None
                    periodic_decision = None
                    dispatch_metadata = None
                    try:
                        if scsp_enabled:
                            (
                                projected_goals,
                                periodic_decision,
                                retry_events,
                            ) = _prepare_scsp_refresh(
                                phase=refresh_phase,
                                runtime_mode=scsp_runtime_mode,
                                arm=scsp_arm,
                                provider=candidate_provider,
                                provider_event=(
                                    periodic_provider_event or {}
                                ),
                                raw_goals=new_goals or {},
                                current_task=current_task_idx,
                                origin=np.asarray(
                                    obs.globalpose[4:], dtype=np.float32
                                ),
                                pathfinder=(
                                    env.habitat_env.sim.pathfinder
                                ),
                                agent=agent,
                                reranker=scsp_reranker,
                                distance_cache=scsp_distance_cache,
                                candidate_cap=scsp_candidate_cap,
                                suffix_weight=scsp_suffix_weight,
                                idea1_config=scsp_idea1_config,
                                provenance=scsp_provenance,
                                episode_identity=scsp_episode_identity,
                                scene_id=scene_id,
                                episode_id=episode_id,
                                ordered_categories=tuple(
                                    _task_category(task)
                                    for task in obs_tasks
                                ),
                                provider_events_path=(
                                    scsp_provider_events_path
                                ),
                                preserve_existing_on_missing=True,
                            )
                            scsp_provider_events.extend(retry_events)
                            periodic_decision['subtask_boundary_replan'] = bool(
                                scsp_boundary_replan_pending
                            )
                            periodic_decision['periodic_refresh_due'] = bool(
                                periodic_refresh_due
                            )
                            projected_goals, dispatch_metadata = (
                                _apply_scsp_runtime_order(
                                    projected_goals,
                                    periodic_decision,
                                    scsp_runtime_mode,
                                )
                            )
                        else:
                            projected_goals = _project_goals_to_navigable(
                                new_goals, env.habitat_env.sim.pathfinder
                            )
                    except Exception as e:
                        print(f'[P4] goal refresh warning: {e}', flush=True)
                    dispatch = agent.dispatch_candidates(
                        projected_goals, dispatch_metadata,
                        task_idx=current_task_idx,
                        category=_task_category(obs_tasks[current_task_idx]),
                        local_step=local_step,
                        category_cooldown=category_cooldown,
                        candidate_batch=(
                            periodic_decision.get('candidate_batch')
                            if periodic_decision is not None else None
                        ),
                    )
                    projected_goals = dispatch.projected_goals
                    dispatch_metadata = dispatch.metadata
                    cooldown_event = dispatch.cooldown_event
                    refresh_applied = dispatch.refresh_applied
                    refresh_error = dispatch.refresh_error
                    if cooldown_event is not None:
                        category_cooldown_events.append(cooldown_event)
                        print(
                            '[CATEGORY_COOLDOWN] '
                            f'{json.dumps(cooldown_event, sort_keys=True)}',
                            flush=True,
                        )
                    if periodic_decision is not None:
                        realized_active = _active_candidate(
                            agent, current_task_idx
                        )
                        realized_raw_rank = _scsp_raw_rank_for_position(
                            periodic_decision.get('candidate_batch'),
                            current_task_idx,
                            realized_active,
                        )
                        periodic_decision = _finalize_scsp_dispatch_event(
                            periodic_decision,
                            dispatch_metadata,
                            cooldown_event=cooldown_event,
                            refresh_applied=refresh_applied,
                            realized_active_position=realized_active,
                            realized_raw_rank=realized_raw_rank,
                            refresh_error=refresh_error,
                        )
                        scsp_decisions.append(
                            _record_scsp_shadow_decision(
                                decision_event=periodic_decision,
                                output_path=scsp_decisions_path,
                            )
                        )
                dt_goals = (time.perf_counter() - t0) * 1000
                is_upgrade = True
                scsp_boundary_replan_pending = False

            # ── agent act ───────────────────────────────────────────────
            t0 = time.perf_counter()
            action, info = agent.act(obs)
            dt_act = (time.perf_counter() - t0) * 1000
            action_metadata = getattr(agent, 'last_action_metadata', {})
            stop_reason = action_metadata.get('stop_reason')
            planner_error = action_metadata.get('planner_error')
            pathfinder = env.habitat_env.sim.pathfinder
            trapped_event = action_metadata.get('trapped_event')
            random_exploration_requested = action_metadata.get(
                'random_exploration_requested', False
            )
            if trapped_event and action != DiscreteNavigationAction.STOP:
                escape_goal, escape_details = _sample_navmesh_goal(
                    pathfinder, np.asarray(obs.globalpose)[4:], 'escape'
                )
                trapped_event['escape'] = escape_details
                if escape_goal is not None:
                    agent.set_navmesh_override_goal(escape_goal, 'escape')
                    action = DiscreteNavigationAction.TURN_RIGHT
                else:
                    agent.clear_local_navigation_state()
                print(
                    f'[TRAPPED_RECOVERY] {json.dumps(trapped_event, ensure_ascii=False)}',
                    flush=True,
                )
            elif (
                random_exploration_requested
                and action != DiscreteNavigationAction.STOP
            ):
                random_goal, random_details = _sample_navmesh_goal(
                    pathfinder, np.asarray(obs.globalpose)[4:], 'random'
                )
                if random_goal is not None:
                    agent.set_navmesh_override_goal(random_goal, 'random')
                    action = DiscreteNavigationAction.TURN_RIGHT
                print(
                    '[RANDOM_EXPLORATION_GOAL] '
                    f'{json.dumps(random_details, ensure_ascii=False)}',
                    flush=True,
                )
            if (
                action == DiscreteNavigationAction.STOP
                and stop_reason not in {'verified_target', 'max_steps'}
            ):
                raise RuntimeError(
                    f'illegal STOP reason={stop_reason!r} '
                    f'planner_error={planner_error!r}'
                )

            env.apply_action(action, info=info)

            dt_total = (time.perf_counter() - t_step_start) * 1000
            episode_frame_times.append(dt_total)
            subtask_frame_times.setdefault(current_task_idx, []).append(dt_total)

            # ── record timing ───────────────────────────────────────────
            if args.enable_timing:
                timing_rows.append({
                    'scene_id':       scene_id,
                    'episode_id':      episode_id,
                    'subtask_index':   current_task_idx,
                    'frame_index':     t,
                    'snap':            candidate_provider.current_snap,
                    'dt_total_ms':     dt_total,
                    'dt_push_ms':      dt_push,
                    'dt_goals_ms':     dt_goals,
                    'dt_act_ms':       dt_act,
                    'is_upgrade_step': int(is_upgrade),
                })

            pbar.set_description(f'{scene_id}_{episode_id}_{current_task_idx}')
            pbar.update(1)

            if action == DiscreteNavigationAction.STOP:
                all_stop_reasons.append(stop_reason)
                print(
                    f'[STOP_REASON] scene={scene_id} episode={episode_id} '
                    f'task={current_task_idx} reason={stop_reason}',
                    flush=True,
                )
                ep_metrics = evaluator.get_episode_metrics()
                ep_metrics.pop('goat_top_down_map', None)
                path_lengths = evaluator.get_subtask_path_lengths()
                ep_metrics.update(build_subtask_telemetry(
                    ep_metrics,
                    actual_path_length=path_lengths['actual_path_length'],
                    shortest_path_length=path_lengths['shortest_path_length'],
                    steps=len(subtask_frame_times.get(current_task_idx, [])),
                    frame_times_ms=subtask_frame_times.get(current_task_idx, []),
                ))
                evaluator.reset_subtask()
                print('----')
                print(f'{scene_id}_{episode_id}_{current_task_idx}', ep_metrics)
                print('----')
                all_subtask_metrics.append(ep_metrics)
                if not env.episode_over:
                    agent.imagenav_visualizer.set_vis_dir(
                        f'{scene_id}_{episode_id}_{env.habitat_env.task.current_task_idx}')
                    agent.matching.set_vis_dir(
                        f'{scene_id}_{episode_id}_{env.habitat_env.task.current_task_idx}')
                    agent.planner.set_vis_dir(scene_id,
                        f'{episode_id}_{env.habitat_env.task.current_task_idx}')
                    agent.set_lightglue_vis_dir(
                        f'{scene_id}_{episode_id}_{env.habitat_env.task.current_task_idx}')
                    env.visualizer.set_vis_dir(scene_id,
                        f'{episode_id}_{env.habitat_env.task.current_task_idx}')
                    pbar.reset()

        pbar.close()

        # Log ES status
        try:
            status = candidate_provider.status()
            es_status_log.append({'episode_id': episode_id, 't': t, 'status': status})
        except Exception:
            pass

        scene_ep_id = f'{scene_id}_{episode_id}'
        if n_subtasks is None or len(all_subtask_metrics) != n_subtasks:
            raise RuntimeError(
                f'incomplete MON episode {scene_ep_id}: '
                f'captured={len(all_subtask_metrics)} expected={n_subtasks}'
            )
        metrics[scene_ep_id] = {'metrics': all_subtask_metrics}  # ES: no pre-explore, keep all subtasks
        metrics[scene_ep_id]['total_num_steps'] = t
        metrics[scene_ep_id]['sub_task_timesteps'] = agent.sub_task_timesteps[0]  # ES: all subtasks
        metrics[scene_ep_id]['tasks'] = obs_tasks  # ES: all tasks
        metrics[scene_ep_id]['stop_reasons'] = all_stop_reasons
        metrics[scene_ep_id]['trapped_events'] = agent.trapped_events
        metrics[scene_ep_id]['scsp_provider_event_summary'] = {
            'event_count': len(scsp_provider_events),
            'invalid_batch_count': sum(
                event['batch_validation_status'] != 'valid'
                for event in scsp_provider_events
            ),
            'would_invalidate_count': sum(
                bool(event['version_gate'].get('would_invalidate', True))
                for event in scsp_provider_events
            ),
            'jsonl_recorded_count': sum(
                bool(event.get('recorded_to_jsonl', False))
                for event in scsp_provider_events
            ),
            'recording_error_count': sum(
                'recording_error' in event
                for event in scsp_provider_events
            ),
        }
        if scsp_enabled:
            planned_decisions = [
                event
                for event in scsp_decisions
                if event.get('status') == 'planned'
            ]
            decision_summary = {
                'runtime_mode': scsp_runtime_mode,
                'event_count': len(scsp_decisions),
                'planned_count': len(planned_decisions),
                'structured_fallback_count': sum(
                    event.get('status') == 'fallback'
                    for event in scsp_decisions
                ),
                'planner_fallback_count': sum(
                    _fallback_value(
                        event['suffix_plan'].fallback_reason
                    ) != FallbackReason.NONE.value
                    for event in planned_decisions
                    if event.get('suffix_plan') is not None
                ),
                'idea1_selected_changed_count': sum(
                    bool(event['changed']['idea1_from_legacy'])
                    for event in planned_decisions
                ),
                'suffix_selected_changed_count': sum(
                    bool(event['changed']['suffix_from_idea1'])
                    for event in planned_decisions
                ),
                'suffix_eligible_count': sum(
                    bool(event.get('dispatch', {}).get('suffix_eligible'))
                    for event in scsp_decisions
                ),
                'suffix_proposal_count': sum(
                    event.get('proposed_order_source') == 'suffix'
                    for event in scsp_decisions
                ),
                'suffix_dispatch_count': sum(
                    event.get('dispatch_order_source') == 'suffix'
                    and bool(event.get('refresh_applied'))
                    for event in scsp_decisions
                ),
                'realized_suffix_intervention_count': sum(
                    event.get('dispatch_order_source') == 'suffix'
                    and bool(event.get('dispatch', {}).get(
                        'suffix_intervention'
                    ))
                    and not bool(event.get('dispatch', {}).get(
                        'cooldown_vetoed'
                    ))
                    and event.get('selected_raw_ranks', {}).get('dispatched')
                    is not None
                    and event.get('selected_raw_ranks', {}).get('dispatched')
                    == event.get('dispatch', {}).get(
                        'post_cooldown_selected_raw_rank'
                    )
                    and event.get('selected_raw_ranks', {}).get('dispatched')
                    != event.get('selected_raw_ranks', {}).get('idea1')
                    for event in scsp_decisions
                ),
                'idea1_dispatch_count': sum(
                    event.get('dispatch_order_source')
                    in {'idea1', 'idea1_fallback'}
                    and bool(event.get('refresh_applied'))
                    for event in scsp_decisions
                ),
                'cooldown_veto_count': sum(
                    bool(event.get('dispatch', {}).get('cooldown_vetoed'))
                    for event in scsp_decisions
                ),
                'synchronous_requery_count': sum(
                    int(event.get('synchronous_requery_count', 0))
                    for event in scsp_decisions
                ),
                'synchronous_requery_success_count': sum(
                    bool(event.get('synchronous_requery_succeeded'))
                    for event in scsp_decisions
                ),
                'stale_suffix_dispatch_count': sum(
                    int(event.get('dispatch', {}).get(
                        'stale_suffix_dispatch', 0
                    ))
                    for event in scsp_decisions
                ),
                'refresh_applied_count': sum(
                    bool(event.get('refresh_applied'))
                    for event in scsp_decisions
                ),
                'recording_error_count': sum(
                    'recording_error' in event
                    for event in scsp_decisions
                ),
                'distance_cache_entries': len(scsp_distance_cache),
                'subtask_state_reset_count': scsp_subtask_reset_count,
                'subtask_boundary_replan_count': sum(
                    bool(event.get('subtask_boundary_replan'))
                    for event in scsp_decisions
                ),
                'shadow_only': scsp_runtime_mode == 'shadow',
            }
            metrics[scene_ep_id]['scsp_decision_summary'] = decision_summary
            if scsp_runtime_mode == 'shadow':
                metrics[scene_ep_id]['scsp_shadow_decision_summary'] = dict(
                    decision_summary
                )
        metrics[scene_ep_id].update(summarize_episode(
            all_subtask_metrics,
            total_steps=t,
            stop_reasons=all_stop_reasons,
            frame_times_ms=episode_frame_times,
            categories=[_task_category(task) for task in obs_tasks],
        ))
        if category_cooldown.enabled:
            metrics[scene_ep_id]['query_switch_cooldown_by_category'] = dict(
                category_cooldown.cooldown_by_category
            )
            metrics[scene_ep_id]['category_cooldown_events'] = (
                category_cooldown_events
            )

        for metric in (
            DISTANCE_TO_GOAL_KEY,
            SUCCESS_KEY,
            SPL_KEY,
        ):
            metric_summary = _finite_metric_summary(
                metrics[scene_ep_id]['metrics'], metric
            )
            metrics[scene_ep_id][f'{metric}_mean'] = metric_summary['mean']
            metrics[scene_ep_id][f'{metric}_median'] = metric_summary['median']
            if metric == DISTANCE_TO_GOAL_KEY:
                metrics[scene_ep_id][f'{metric}_finite_count'] = (
                    metric_summary['finite_count']
                )
                metrics[scene_ep_id][f'{metric}_unreachable_count'] = (
                    metric_summary['unreachable_count']
                )
            print(
                f'{scene_ep_id} {metric}_mean: '
                f'{metrics[scene_ep_id][f"{metric}_mean"]}'
            )

        print('-' * 60)

        write_json_strict(
            Path(results_dir) / 'per_episode_metrics.json', metrics
        )

        stats = {}
        for metric in (
            DISTANCE_TO_GOAL_KEY,
            SUCCESS_KEY,
            SPL_KEY,
        ):
            metric_summary = _finite_metric_summary(
                [y for ep in metrics.values() for y in ep['metrics']], metric
            )
            stats[f'{metric}_mean'] = metric_summary['mean']
            stats[f'{metric}_median'] = metric_summary['median']
            if metric == DISTANCE_TO_GOAL_KEY:
                stats[f'{metric}_finite_count'] = metric_summary['finite_count']
                stats[f'{metric}_unreachable_count'] = (
                    metric_summary['unreachable_count']
                )
        stats.update(summarize_records(
            metrics,
            frame_times_ms=(
                [row['dt_total_ms'] for row in timing_rows]
                if args.enable_timing
                else None
            ),
        ))
        write_json_strict(Path(results_dir) / 'cumulative_metrics.json', stats)
        if args.enable_timing:
            _write_timing_rows(timing_rows, results_dir)

    # ── Save ES status log ──────────────────────────────────────────────
    with open(os.path.join(results_dir, 'es_status.json'), 'w') as fp:
        json.dump(es_status_log, fp, indent=4)

    # ── Timing summary ──────────────────────────────────────────────────
    if args.enable_timing and timing_rows:
        _print_timing_summary(timing_rows, results_dir, save_csv=True)

    candidate_provider.close()
    print('End time:', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    print('Total time:', round(time.time() - start_time, 1), 's')
