"""BLP runtime binding and passive, replayable calibration records."""
from dataclasses import asdict
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

from pbsp.navigation.blp import BeliefParameters, build_snapshot, solve_snapshot
from pbsp.navigation.blp_parameters import load_frozen_calibration
from pbsp.navigation.planning import active_candidate, subtask_has_acted
from experiments.telemetry import dumps_json_strict


def initialize_blp(planning, agent, directory, *, scene_id=None):
    agent.blp_enabled = True
    planning.blp_directory = Path(directory)
    calibration = planning.blp_config.get('calibration', 'uncalibrated')
    if planning.blp_config.get('enabled') and calibration == 'uncalibrated':
        raise ValueError('Active BLP evaluation requires frozen self-calibration')
    if calibration == 'uncalibrated':
        planning.blp_parameters = BeliefParameters()
        planning.blp_parameter_binding = {'selection': 'uncalibrated_shadow',
                                          'scene_id': scene_id, 'calibration_sha256': None}
    else:
        planning.blp_parameters, planning.blp_parameter_binding = load_frozen_calibration(
            calibration, scene_id)


def append(planning, filename, payload):
    with (planning.blp_directory / filename).open('a') as stream:
        stream.write(dumps_json_strict(payload) + '\n')


def observe_frame(planning, agent, env, obs, t, task_idx):
    import quaternion
    state = env.habitat_env.sim.get_agent_state()
    sensors = state.sensor_states
    key = next(name for name in sensors if 'depth' in name.lower())
    camera = sensors[key]
    rotation = quaternion.as_rotation_matrix(camera.rotation).T
    agent.candidate_evidence.observe_coverage(camera.position, rotation, obs.depth, 79.0)
    previous = planning.blp_context.get('position')
    position = np.asarray(obs.globalpose[4:], dtype=float)
    path = planning.blp_context.get('path_m', 0.0)
    if previous is not None:
        path += float(np.linalg.norm(position - previous))
    planning.blp_context.update(t=int(t), k=int(task_idx), path_m=path,
                                position=position.tolist())
    if planning.blp_context.get('activation_task') != int(task_idx):
        finalize_visit(planning, 'task_boundary')
        planning.blp_context.update(activation_task=int(task_idx), activation_path_m=path,
                                    activation_snapshot_seen=False, empty_activation_open=False)


def observe_frontier(agent, obs, planner_input):
    """Invert the native map-pixel to episode-GPS to world transform."""
    rows, cols = np.nonzero(planner_input['frontier_map'])
    sensor = planner_input['sensor_pose']
    self_pixel = sensor[:2] * 20 - np.array([sensor[5], sensor[3]])
    gps = np.asarray(obs.gps[:2])
    local_gps = np.column_stack((cols, rows)) * 0.05 - self_pixel * 0.05 + gps
    rotation, translation = agent.rot_matrix_v2, agent.tran_v2
    origin = np.asarray(obs.globalpose[4:])
    local_y = (rotation @ origin - translation)[1]
    local = np.column_stack((-local_gps[:, 1], np.full(len(rows), local_y), -local_gps[:, 0]))
    world = (local + translation) @ rotation
    agent.blp_frontier_world = world.tolist()


def compute_blp_event(planning, *, phase, arm, candidate_batch, projected_goals,
                      projected_indices, current_task, origin, pathfinder, agent,
                      distance_cache, version_gate, provenance, episode_identity):
    started = time.perf_counter()
    variant = planning.blp_config.get('variant', 'full')
    # Shadow work must not evict or populate the legacy cache.
    cache = {} if planning.blp_config.get('shadow') else distance_cache
    batch, snapshot = build_snapshot(
        candidate_batch=candidate_batch, projected_goals=projected_goals,
        projected_indices=projected_indices, current_task=current_task, origin=origin,
        geodesic=lambda a, b: planning.geodesic(pathfinder, a, b), agent=agent,
        distance_cache=cache, query_future=variant in {'full', 'suffix_only'},
        project_frontier=pathfinder.snap_point)
    active_rank = None
    active = active_candidate(agent, current_task)
    if active is not None and subtask_has_acted(agent, current_task):
        layer = snapshot['layers'][current_task]
        if layer:
            nearest = min(layer, key=lambda r: np.linalg.norm(
                np.asarray(r['position'])[[0, 2]] - np.asarray(active)[[0, 2]]))
            if np.linalg.norm(np.asarray(nearest['position'])[[0, 2]] -
                              np.asarray(active)[[0, 2]]) <= agent.candidate_evidence.match_radius_m:
                active_rank = nearest['raw_rank']
    result = solve_snapshot(snapshot, planning.blp_parameters, variant, active_rank)
    result['plan_ms'] = (time.perf_counter() - started) * 1000
    identity = {'scene_id': batch.scene_id, 'episode_id': batch.episode_id,
                't': planning.blp_context.get('t', 0), 'k': current_task,
                'path_m': planning.blp_context.get('path_m', 0.0)}
    append(planning, 'blp_snapshots.jsonl', dict(identity, snapshot=snapshot,
           active_raw_rank=active_rank, parameters=asdict(planning.blp_parameters),
           parameter_binding=getattr(planning, 'blp_parameter_binding', None),
           shadow=bool(planning.blp_config.get('shadow'))))
    append(planning, 'blp_decisions.jsonl', dict(identity, **{k: v for k, v in result.items() if k != 'k'}))
    raw_orders = tuple(tuple(r.raw_rank for r in batch.records_for_task(task))
                       for task in range(len(batch.ordered_categories)))
    planning.blp_context['batch'] = batch
    activation = planning.blp_context
    if not activation.get('activation_snapshot_seen'):
        activation['activation_snapshot_seen'] = True
        activation['empty_activation_open'] = not bool(snapshot['layers'][current_task])
        if activation['empty_activation_open']:
            append(planning, 'blp_events.jsonl', dict(identity, event='empty_activation_started'))
    if activation.get('empty_activation_open') and snapshot['layers'][current_task]:
        append(planning, 'blp_events.jsonl', dict(identity, event='empty_activation',
               path_to_first_candidate_m=identity['path_m'] - activation['activation_path_m']))
        activation['empty_activation_open'] = False
    for task in range(current_task, len(batch.ordered_categories)):
        history = planning.blp_context.setdefault('query_record_history', {}).setdefault(task, [])
        for record in batch.records_for_task(task):
            history.append({'position': list(record.position), 'semantic_score': record.semantic_score,
                            'query_serial': batch.query_serial, 't': identity['t']})
            register_candidate(planning, task, record)
    return {'schema_version': 'blp_eq13_v1', 'phase': phase, 'arm': arm,
            'status': 'planned', 'current_task': current_task,
            'candidate_batch': batch, 'provider_candidate_batch': candidate_batch,
            'version_gate': version_gate, 'blp': result,
            'legacy_order': raw_orders, 'idea1_order': raw_orders,
            'selected_raw_ranks': {'blp': result['chosen_raw_rank']},
            'changed': {'idea1_from_legacy': False, 'suffix_from_idea1': False},
            'timing_ms': {'total_shadow': result['plan_ms']},
            'episode_identity': episode_identity, **provenance}


def register_candidate(planning, task, record):
    registry = planning.blp_context.setdefault('candidate_registry', {})
    group = registry.setdefault(task, [])
    point = np.asarray(record.position)
    near = [r for r in group if np.linalg.norm(np.asarray(r['position'])[[0, 2]] - point[[0, 2]]) <= .75]
    if near:
        item = min(near, key=lambda r: np.linalg.norm(np.asarray(r['position'])[[0, 2]] - point[[0, 2]]))
        item.update(position=list(record.position), E=record.semantic_score)
        return item
    item = dict(candidate_id=f'{task}:{len(group)}', position=list(record.position),
                E=record.semantic_score, first_t=planning.blp_context['t'], checked=False)
    group.append(item)
    append(planning, 'blp_events.jsonl', dict(event='candidate_first_appearance', k=task,
           scene_id=planning.blp_context['batch'].scene_id,
           episode_id=planning.blp_context['batch'].episode_id, **item))
    return item


def finalize_visit(planning, reason):
    visit = planning.blp_context.pop('visit', None)
    if visit is None:
        return
    visit.update(event='verification_final', finalized_reason=reason,
                 verification_path_m=max(0., planning.blp_context.get('path_m', 0.) - visit['start_path_m']))
    append(planning, 'blp_events.jsonl', visit)


def prepare_action(planning, agent, task_idx):
    active = active_candidate(agent, task_idx)
    segment = planning.blp_context.get('active_segment')
    unchanged = segment and segment['k'] == task_idx and active is not None and np.linalg.norm(
        np.asarray(segment['position'])[[0, 2]] - np.asarray(active)[[0, 2]]) <= .75
    if not unchanged:
        finalize_visit(planning, 'candidate_changed_before_action')
        planning.blp_context['active_segment'] = None if active is None else {
            'k': task_idx, 'position': list(active),
            'start_path_m': planning.blp_context['path_m']}


def observe_verification(planning, agent, evaluator, metadata):
    event = metadata.get('verification_event')
    batch = planning.blp_context.get('batch')
    if batch is None:
        return
    if event is None:
        # An abandoned check is negative only when its spatial candidate really
        # changes; consecutive non-triggered frames are not independent labels.
        visit = planning.blp_context.get('visit')
        if visit:
            active = active_candidate(agent, visit['k'])
            if active is None or np.linalg.norm(np.asarray(active)[[0, 2]] - np.asarray(visit['position'])[[0, 2]]) > .75:
                finalize_visit(planning, 'candidate_abandoned')
        return
    task, position = event['task_idx'], event['candidate_position']
    if event.get('candidate_source') == 'navmesh_override':
        finalize_visit(planning, 'switched_to_navmesh_override')
        append(planning, 'blp_events.jsonl', {
            'event': 'noncandidate_verification', 'scene_id': batch.scene_id,
            'episode_id': batch.episode_id, 'k': task, 'position': list(position),
            't': planning.blp_context['t'], 'v': int(bool(event['confirmed'])),
            'candidate_source': 'navmesh_override',
            'gt_diagnostic': evaluator.label_candidate(position, task),
            'excluded_from_belief_fit': True})
        return
    records = batch.records_for_task(task)
    matches = [r for r in records if np.linalg.norm(
        np.asarray(r.position)[[0, 2]] - np.asarray(position)[[0, 2]]) <= .75]
    evidence_source = 'current_batch'
    if matches:
        record = min(matches, key=lambda r: np.linalg.norm(
            np.asarray(r.position)[[0, 2]] - np.asarray(position)[[0, 2]]))
    else:
        history = planning.blp_context.get('query_record_history', {}).get(task, [])
        matches = [r for r in history if np.linalg.norm(
            np.asarray(r['position'])[[0, 2]] - np.asarray(position)[[0, 2]]) <= .75]
        if not matches:
            append(planning, 'blp_errors.jsonl', {'error': 'unmatched_query_verification',
                   'scene_id': batch.scene_id, 'episode_id': batch.episode_id,
                   'k': task, 't': planning.blp_context['t'], 'position': list(position)})
            return
        source = max(matches, key=lambda r: r['query_serial'])
        record = SimpleNamespace(position=source['position'], semantic_score=source['semantic_score'])
        evidence_source = 'historical_query_record'
    item = register_candidate(planning, task, record)
    visit = planning.blp_context.get('visit')
    if visit and visit['candidate_id'] != item['candidate_id']:
        finalize_visit(planning, 'candidate_changed')
        visit = None
    if visit is None:
        n_views = agent.candidate_evidence.coverage_views(task, position)
        s, _, _ = planning.blp_parameters.evidence(record.semantic_score, n_views)
        segment = planning.blp_context.get('active_segment') or {}
        visit = dict(candidate_id=item['candidate_id'], scene_id=batch.scene_id,
                     episode_id=batch.episode_id, k=task, position=list(position),
                     E=record.semantic_score, N_views=n_views, S=s,
                     t=planning.blp_context['t'],
                     start_path_m=segment.get('start_path_m', planning.blp_context['path_m']),
                     v=0, y=0)
        planning.blp_context['visit'] = visit
        item['checked'] = True
        append(planning, 'blp_events.jsonl', dict(visit, event='candidate_checked'))
    metrics = evaluator.get_episode_metrics()
    diagnostic = evaluator.label_candidate(position, task)
    n_views = agent.candidate_evidence.coverage_views(task, position)
    s, _, _ = planning.blp_parameters.evidence(record.semantic_score, n_views)
    visit.update(v=int(bool(event['confirmed'])), y=diagnostic['y'],
                 E=record.semantic_score, N_views=n_views, S=s, position=list(position),
                 semantic_evidence_source=evidence_source,
                 gt_diagnostic=diagnostic,
                 y_agent_at_check=int(bool(metrics['goat_sub-task_success'])),
                 last_t=planning.blp_context['t'], verification=event)
    if event['confirmed']:
        finalize_visit(planning, 'confirmed')


def action_target_descriptor(agent, task_idx, position):
    """Copy the selected target before STOP can advance the policy task."""
    point = None if position is None else np.asarray(position, dtype=float)
    if point is not None and (point.shape != (3,) or not np.all(np.isfinite(point))):
        raise ValueError('Action target must be a finite world XYZ point')
    override = getattr(agent, 'navmesh_override_goal', None)
    source = ('frontier' if point is None else 'navmesh_override'
              if override is not None and np.allclose(point, override, rtol=0., atol=1e-6)
              else 'query_candidate')
    return {'schema_version': 'blp_action_target_v1', 'task_idx': int(task_idx),
            'source': source, 'position': None if point is None else point.tolist()}


def bind_action_target(planning, metadata, task_idx, frame_index):
    """Read existing identities without registering candidates or changing rank."""
    target = dict(metadata['blp_action_target'])
    if target['task_idx'] != int(task_idx):
        raise ValueError('Action target belongs to a different pre-STOP task')
    recovery = metadata.get('blp_effective_recovery')
    if recovery is not None:
        target.update(source='navmesh_override', position=list(recovery['position']),
                      runner_recovery_mode=recovery['mode'])
    saved_id = target.get('candidate_id') if recovery is None else None
    target.update(frame_index=int(frame_index), candidate_id=None,
                  query_identity_status='not_query')
    if target['source'] == 'query_candidate':
        point = np.asarray(target['position'], dtype=float)
        registry = planning.blp_context.get('candidate_registry', {}).get(int(task_idx), [])
        if saved_id is not None:
            if not any(r['candidate_id'] == saved_id for r in registry):
                raise ValueError('Confirmed visual parent is absent from episode registry')
            target.update(candidate_id=saved_id, query_identity_status='resolved')
            return target
        distances = [(float(np.linalg.norm((np.asarray(r['position']) - point)[[0, 2]])), r)
                     for r in registry]
        matches = [(distance, r) for distance, r in distances if distance <= .75]
        if matches:
            _, match = min(matches, key=lambda item: (item[0], item[1]['candidate_id']))
            target.update(candidate_id=match['candidate_id'], query_identity_status='resolved')
        else:
            target['query_identity_status'] = 'unresolved'
    return target
