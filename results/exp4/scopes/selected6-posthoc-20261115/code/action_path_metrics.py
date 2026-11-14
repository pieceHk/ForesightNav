"""Attribute accepted-action displacement using the recorded navigation target.

Candidate eligibility follows the distinct checked GT-invalid identities. The
primary distance counts only their pursuit at GT-invalid target coordinates;
all pursuit of those identities is also retained as an explicit upper bound.
"""
from collections import defaultdict
import hashlib
import math
from pathlib import Path

SCHEMA = 'blp_action_target_v1'
GT_CRITERION = 'pinned_GOAT_measures_at_candidate_position'
SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def identity(row):
    return (str(row['scene_id']), str(row['episode_id']), int(row['k']),
            str(row['candidate_id']))


def xyz(pose):
    if len(pose) != 7 or not all(math.isfinite(float(x)) for x in pose):
        raise ValueError('Accepted-action pose is not a finite seven-vector')
    return tuple(map(float, pose[4:]))


def target_xyz(target):
    point = target['position']
    if point is None or len(point) != 3 or not all(math.isfinite(float(x)) for x in point):
        raise ValueError('Query/recovery target must have finite world XYZ')
    return tuple(map(float, point))


def action_paths(events, traces, gt_label, native_stage_paths=None):
    flat = [r for rows in traces.values() for r in rows]
    present = sum('blp_action_target' in r for r in flat)
    if present == 0:
        return None
    if present != len(flat):
        raise ValueError('Cannot combine missing and present action-target records')
    checked, invalid, gt_sets, known = set(), set(), defaultdict(set), set()
    for event in events:
        if event['event'] == 'candidate_first_appearance':
            known.add(identity(event))
        if event['event'] == 'verification_final':
            key = identity(event)
            if event['y'] not in (0, 1):
                raise ValueError('Invalid binary GT label on a checked candidate')
            known.add(key)
            checked.add(key)
            gt_sets[key].add(event['y'])
            if event['y'] == 0:
                invalid.add(key)
    if invalid and gt_label is None:
        raise ValueError('Native post-hoc GT is required for checked-invalid target paths')
    bins, stage_paths = defaultdict(float), defaultdict(float)
    gaps, current_valid_ids = [], set()
    rows_by_source = defaultdict(int)
    gt_rows, last_stop_zero = 0, 0
    verification_recovery_handoffs = 0
    for pair, rows in traces.items():
        scene, episode = map(str, pair)
        if not rows or not rows[-1]['is_stop']:
            raise ValueError('Complete accepted-action trace must end at STOP')
        byframe = {}
        for offset, row in enumerate(rows):
            frame, task = int(row['frame_index']), int(row['subtask_index'])
            if frame != offset + 1 or (str(row['scene_id']), str(row['episode_id'])) != (scene, episode):
                raise ValueError('Accepted-action frame/episode identity mismatch')
            target = row['blp_action_target']
            if target.get('schema_version') != SCHEMA or target['frame_index'] != frame or target['task_idx'] != task:
                raise ValueError('Action-target schema/frame/pre-STOP task mismatch')
            before = xyz(row['world_pose'])
            if 'world_pose_after' in row:
                after = xyz(row['world_pose_after'])
                if offset + 1 < len(rows) and math.dist(after, xyz(rows[offset + 1]['world_pose'])) > 1e-4:
                    raise ValueError('Recorded action endpoint and next observation disagree')
            elif offset + 1 < len(rows):
                after = xyz(rows[offset + 1]['world_pose'])
            else:
                # Native GOAT STOP has no locomotion. The native stage-path
                # comparison below independently verifies the complete sum.
                after = before
                last_stop_zero += 1
            distance = math.dist(before, after)
            stage_paths[scene, episode, task] += distance
            bins['all'] += distance
            source = target['source']
            rows_by_source[source] += 1
            byframe[frame] = target
            if source == 'query_candidate':
                key = (scene, episode, task, str(target.get('candidate_id')))
                target_xyz(target)
                if target.get('candidate_id') is None or target.get('query_identity_status') != 'resolved' or key not in known:
                    gaps.append({'scene_id': scene, 'episode_id': episode, 'k': task,
                                 'frame': frame, 'reason': 'unresolved_or_unknown_query_identity'})
                    bins['unresolved_query'] += distance
                    continue
                if key in invalid:
                    bins['invalid_identity_upper'] += distance
                    label = gt_label(scene, episode, task, target['position'])
                    if label.get('criterion') != GT_CRITERION or label.get('y') not in (0, 1):
                        raise ValueError('Action target did not use the pinned native GT criterion')
                    gt_rows += 1
                    if label['y'] == 0:
                        bins['invalid_point'] += distance
                        part = 'confirmed' if target.get('pursuit_mode') == 'confirmed_visual_goal' else 'approach'
                        bins[part] += distance
                    else:
                        bins['invalid_identity_current_valid'] += distance
                        current_valid_ids.add(key)
                elif key in checked:
                    bins['checked_valid'] += distance
                else:
                    bins['unchecked_query'] += distance
            elif source == 'navmesh_override':
                target_xyz(target)
                if target.get('candidate_id') is not None:
                    raise ValueError('Recovery target was incorrectly assigned a query identity')
                bins['recovery'] += distance
            elif source == 'frontier':
                if target.get('candidate_id') is not None:
                    raise ValueError('Frontier target was incorrectly assigned a query identity')
                bins['frontier'] += distance
            else:
                raise ValueError('Unknown recorded target source')
        for event in events:
            if event['event'] != 'verification_final' or (str(event['scene_id']), str(event['episode_id'])) != (scene, episode):
                continue
            target = byframe.get(int(event['last_t']))
            if target is None:
                raise ValueError('Verification refers to a missing accepted-action frame')
            if (target['task_idx'] == int(event['k']) and target['source'] == 'query_candidate'
                    and target['candidate_id'] == event['candidate_id']):
                continue
            if target.get('runner_recovery_mode') in ('escape', 'random') and target['task_idx'] == int(event['k']):
                verification_recovery_handoffs += 1
            else:
                gaps.append({'scene_id': scene, 'episode_id': episode, 'k': event['k'],
                             'frame': event['last_t'], 'reason': 'verification_and_action_target_identity_disagree'})
    partition = math.fsum(bins[k] for k in ('invalid_identity_upper', 'checked_valid',
        'unchecked_query', 'unresolved_query', 'recovery', 'frontier'))
    if not math.isclose(partition, bins['all'], abs_tol=1e-8, rel_tol=1e-10):
        raise ValueError('Accepted displacement was lost or counted more than once')
    errors = []
    if native_stage_paths is not None:
        if set(native_stage_paths) != set(stage_paths):
            raise ValueError('Accepted-action stages do not cover native path metrics')
        for key, expected in native_stage_paths.items():
            error = abs(stage_paths[key] - expected)
            errors.append(error)
            if not math.isclose(stage_paths[key], expected, abs_tol=1e-3, rel_tol=1e-6):
                raise ValueError(f'Action/native stage path mismatch: {key}, {stage_paths[key]}, {expected}')
    n = len(traces)
    return {
        'invalid_path_source': SCHEMA + '_native_gt',
        'invalid_approach_and_verification_path_m': bins['approach'],
        'invalid_confirmed_approach_tail_m': bins['confirmed'],
        'invalid_visit_path_m': bins['invalid_point'],
        'invalid_visit_path_m_per_ep': bins['invalid_point'] / n,
        'invalid_identity_pursuit_path_m': bins['invalid_identity_upper'],
        'invalid_identity_pursuit_path_m_per_ep': bins['invalid_identity_upper'] / n,
        'checked_invalid_identity_but_current_target_GT_valid_path_m': bins['invalid_identity_current_valid'],
        'checked_invalid_identities_with_current_valid_targets': len(current_valid_ids),
        'mixed_GT_checked_identities': [list(k) for k, values in sorted(gt_sets.items()) if len(values) > 1],
        'invalid_path_association_gaps': gaps,
        'invalid_path_complete': not gaps,
        'invalid_path_definition': 'accepted displacement pursuing a checked-invalid query identity while its recorded target coordinate is native-GT-invalid; confirmed visual continuation retains the parent; recovery/frontier excluded',
        'invalid_identity_pursuit_definition': 'all accepted displacement pursuing identities that had at least one GT-invalid check; explicit upper bound, including any later GT-valid coordinates',
        'accepted_action_path_m': bins['all'],
        'accepted_action_path_partition_m': {k: bins[k] for k in ('invalid_identity_upper',
            'checked_valid', 'unchecked_query', 'unresolved_query', 'recovery', 'frontier')},
        'accepted_action_rows_by_source': dict(rows_by_source),
        'native_target_GT_labelled_action_rows': gt_rows,
        'native_stage_path_max_abs_error_m': max(errors) if errors else None,
        'native_stage_path_check_count': len(errors),
        'native_stage_path_check_tolerance': {'abs_m': 1e-3, 'relative': 1e-6},
        'last_STOP_without_after_pose_assumed_stationary': last_stop_zero,
        'verification_then_effective_recovery_actions': verification_recovery_handoffs,
    }
