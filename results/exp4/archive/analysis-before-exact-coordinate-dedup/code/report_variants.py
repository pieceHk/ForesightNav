"""Postprocess native exp4 runs without changing their runtime source.

Run with PYTHONPATH pointing at the frozen PBSP checkout. All reported native
scores come from experiments.metrics; PPL and prefix conditionals are explicit
supplementary metrics. This script does not launch any simulation.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import asdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics

from experiments.metrics import merge_episode_metric_files, summarize_records
from action_path_metrics import action_paths, SOURCE_SHA256 as ACTION_SOURCE_SHA256
from native_candidate_gt import NativeCandidateGT, SOURCE_SHA256 as GT_SOURCE_SHA256
from pbsp.navigation.blp_parameters import CalibrationDocument, SOURCE_SHA256 as PARAMETER_SOURCE_SHA256

SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def prefix_metrics(stages):
    success = [int(s['goat_sub-task_success']) for s in stages]
    prefix = list(itertools.takewhile(lambda s: bool(s['goat_sub-task_success']), stages))
    if prefix:
        shortest = math.fsum(s['goat_sub-task_shortest_path_length_m'] for s in prefix)
        actual = math.fsum(s['goat_sub-task_path_length_m'] for s in prefix)
        denominator = max(shortest, actual)
        efficiency = shortest / denominator if denominator else 1.
    else:
        efficiency = 0.
    progress = sum(success) / len(success)
    return {'PR': progress, 'PPL': progress * efficiency,
            'PR_prefix': len(prefix) / len(stages),
            'PPL_prefix': len(prefix) / len(stages) * efficiency}


def conditional_rates(vectors):
    if not vectors or len({len(v) for v in vectors}) != 1:
        raise ValueError('Expected nonempty, fixed-length success vectors')
    if any(x not in (0, 1, False, True) for row in vectors for x in row):
        raise ValueError('Success values must be binary')
    numerators, denominators, rates = [], [], []
    for k in range(len(vectors[0])):
        subset = [row for row in vectors if all(row[:k])]
        n = sum(bool(row[k]) for row in subset)
        numerators.append(n)
        denominators.append(len(subset))
        rates.append(n / len(subset) if subset else None)
    joint = 1.
    for rate in rates:
        if rate is None:
            if joint != 0.:
                raise ValueError('Undefined conditional before zero prefix probability')
            break
        joint *= rate
    sr = sum(all(v) for v in vectors) / len(vectors)
    if not math.isclose(joint, sr, abs_tol=1e-12):
        raise ValueError('Conditional product differs from native sequence success')
    return {'rates': rates, 'successes': numerators, 'conditioned_episodes': denominators,
            'Joint': joint, 'definition': 'conditioned on prior successes, not prior execution'}


def summarize_distribution(values):
    if not values:
        return {'n': 0, 'median': None, 'p95': None}
    ordered = sorted(values)
    p = .95 * (len(ordered) - 1)
    lo, hi = math.floor(p), math.ceil(p)
    return {'n': len(ordered), 'median': statistics.median(ordered),
            'p95': ordered[lo] * (hi - p) + ordered[hi] * (p - lo) if hi != lo else ordered[lo]}


def location(row):
    return row['world_pose'][4:]


def trajectory_tail_length(trace, first_frame, last_frame):
    rows = [r for r in trace if first_frame <= r['frame_index'] <= last_frame]
    return math.fsum(math.dist(location(a), location(b)) for a, b in zip(rows, rows[1:]))


def interval_union_length(intervals):
    total = 0.
    stop = None
    for start, end in sorted(intervals):
        if end < start:
            raise ValueError('Visit path interval is reversed')
        total += max(0., end - max(start, stop)) if stop is not None else end - start
        stop = max(stop, end) if stop is not None else end
    return total


def process_metrics(events, decisions, dispatches, traces, episode_keys,
                    gt_label=None, native_stage_paths=None):
    final = [e for e in events if e['event'] == 'verification_final']
    invalid = [e for e in final if e['y'] == 0]
    invalid_distinct = {(e['scene_id'], e['episode_id'], e['k'], e['candidate_id']) for e in invalid}
    checked_distinct = {(e['scene_id'], e['episode_id'], e['k'], e['candidate_id']) for e in final}
    intervals = defaultdict(list)
    for e in invalid:
        intervals[e['scene_id'], e['episode_id']].append(
            (e['start_path_m'], e['start_path_m'] + e['verification_path_m']))
    measured_path = math.fsum(interval_union_length(v) for v in intervals.values())
    confirmed_tail = 0.
    path_gaps = []
    # A confirmed visual goal can require further approach motion before STOP.
    # Attach that observed tail to the stage's last query confirmation, instead
    # of pretending that confirmation itself ended the navigation visit.
    stages = defaultdict(list)
    for event in final:
        stages[(event['scene_id'], event['episode_id'], event['k'])].append(event)
    for (scene, episode, k), rows in stages.items():
        event = max(rows, key=lambda e: e.get('last_t', e['t']))
        for earlier in rows:
            if earlier is not event and earlier['y'] == 0 and earlier['v'] == 1:
                path_gaps.append({'scene_id': scene, 'episode_id': episode, 'k': k,
                                  'reason': 'confirmed_invalid_visit_followed_by_further_checks'})
        if event['y'] != 0 or event['v'] != 1:
            continue
        trace = traces.get((scene, episode), [])
        stops = [r for r in trace if r['subtask_index'] == k and r['is_stop']]
        if stops:
            stop = stops[-1]['frame_index']
            if any(e['event'] == 'noncandidate_verification' and e['scene_id'] == scene
                   and e['episode_id'] == episode and e['k'] == k
                   and event['last_t'] < e['t'] <= stop for e in events):
                path_gaps.append({'scene_id': scene, 'episode_id': episode, 'k': k,
                                  'reason': 'recovery_verification_after_query_confirmation'})
            else:
                confirmed_tail += trajectory_tail_length(trace, event['last_t'], stop)
    eligible = [d for d in decisions if d['nearest_raw_rank'] is not None]
    fallback_decisions = Counter(d['fallback'] for d in decisions if d['fallback'])
    state_counts, state_incidence = Counter(), Counter()
    for d in decisions:
        state_counts.update(d['fallback_counts'])
        state_incidence.update(k for k, v in d['fallback_counts'].items() if v)
    realized = [e for e in dispatches if e.get('refresh_applied')
                and e.get('blp', {}).get('nearest_raw_rank') is not None
                and e.get('selected_raw_ranks', {}).get('dispatched') is not None]
    proposed = sum(d['unconstrained_raw_rank'] != d['nearest_raw_rank'] for d in eligible)
    held = sum(d['chosen_raw_rank'] != d['nearest_raw_rank'] for d in eligible)
    dispatched = sum(e['selected_raw_ranks']['dispatched'] != e['blp']['nearest_raw_rank'] for e in realized)
    percentage = lambda numerator, denominator: 100. * numerator / denominator if denominator else None
    n = len(episode_keys)
    value = {'Invalid_visits_per_ep': len(invalid_distinct) / n,
            'invalid_visit_events_per_ep': len(invalid) / n,
            'checked_distinct_candidates': len(checked_distinct),
            'invalid_approach_and_verification_path_m': measured_path,
            'invalid_confirmed_approach_tail_m': confirmed_tail,
            'invalid_visit_path_m': measured_path + confirmed_tail,
            'invalid_visit_path_m_per_ep': (measured_path + confirmed_tail) / n,
            'invalid_path_association_gaps': path_gaps,
            'invalid_path_complete': not path_gaps,
            'invalid_path_definition': 'union of recorded active-candidate approach/check path intervals plus an unambiguous final confirmation-to-STOP tail',
            'invalid_visit_definition': 'distinct checked (episode, task, persistent spatial identity) with GT-invalid verification; event count also retained',
            'nonquery_recovery_verification_frames': sum(e['event'] == 'noncandidate_verification' for e in events),
            'deviation_from_nearest_proposal_pct': percentage(proposed, len(eligible)),
            'deviation_from_nearest_after_hysteresis_pct': percentage(held, len(eligible)),
            'deviation_from_nearest_realized_pct': percentage(dispatched, len(realized)),
            'eligible_proposals': len(eligible), 'eligible_realized_dispatches': len(realized),
            'planning_ms': summarize_distribution([d['plan_ms'] for d in decisions]),
            'fallback_root_decisions': dict(fallback_decisions),
            'fallback_decision_incidence': dict(state_incidence),
            'fallback_memo_state_counts': dict(state_counts),
            'fallback_count_definition': 'root decisions and per-decision incidence are separate from recursion-state counts',
            'n_computed_snapshots': len(decisions), 'n_recorded_dispatch_events': len(dispatches)}
    paths = action_paths(events, traces, gt_label, native_stage_paths)
    if paths is not None:
        value['legacy_interval_diagnostics'] = {k: value[k] for k in (
            'invalid_visit_path_m', 'invalid_path_complete', 'invalid_path_association_gaps',
            'invalid_path_definition')}
        value.update(paths)
    else:
        value['invalid_path_source'] = 'legacy_visit_intervals'
    return value


def validate_snapshot_parameters(snapshots, document, parameter_sha256, selected):
    if not snapshots:
        raise ValueError('Every selected episode needs active BLP snapshots')
    if any(s.get('shadow') is not False for s in snapshots):
        raise ValueError('Shadow snapshots cannot populate a formal ablation')
    if not isinstance(parameter_sha256, str) or len(parameter_sha256) != 64:
        raise ValueError('Formal snapshots require the frozen parameter-file SHA256')
    for row in snapshots:
        scene = str(row['scene_id'])
        values, binding = document.resolve(scene)
        if row['parameters'] != asdict(values):
            raise ValueError('Observed parameters differ from the frozen scene assignment')
        if row.get('parameter_binding') != dict(binding, calibration_sha256=parameter_sha256):
            raise ValueError('Snapshot parameter source, scene, or held-out-fold binding differs')
    if {(str(s['scene_id']), str(s['episode_id'])) for s in snapshots} != set(selected):
        raise ValueError('Not every selected episode has BLP planning snapshots')


def normalize(variant, runs, expected=None, allow_diagnostic=False, calibration=None,
              parameter_sha256=None):
    records, identities, traces = {}, {}, {}
    events, decisions, dispatches = [], [], []
    task_count = None
    commits = set()
    record_commits = {}
    gt_by_episode, native_gt_runs = {}, []
    if not allow_diagnostic and calibration is None:
        raise ValueError('Formal analysis requires the full frozen self-calibration artifact')
    if not allow_diagnostic and expected is None:
        raise ValueError('Formal analysis requires the frozen episode identity list')
    document = CalibrationDocument.from_mapping(calibration) if calibration is not None else None
    for run in map(Path, runs):
        status = json.loads((run / 'status.json').read_text())
        if status['status'] != 'completed':
            raise ValueError(f'Refuse non-complete run {run}: {status}')
        plan = json.loads((run / 'launch.json').read_text())
        commits.add(plan['source_commit'])
        mode = plan['configuration']['method']['suffix']['mode']
        if not allow_diagnostic and mode != 'blp_eq13':
            raise ValueError('Formal ablation artifacts require an active blp_eq13 run')
        if not allow_diagnostic and plan['configuration']['method']['suffix']['blp_variant'] != variant:
            raise ValueError('Requested variant differs from the actual launch configuration')
        errors = [p for p in run.rglob('blp_errors.jsonl') if p.stat().st_size]
        if errors:
            raise ValueError(f'Unreviewed shadow errors: {errors}')
        selected = {(str(r['scene_id']), str(r['runtime_episode_id'])): r for r in plan['identities']}
        if len(selected) != len(plan['identities']):
            raise ValueError('Duplicate identities in the native launch plan')
        if set(identities) & set(selected):
            raise ValueError('Duplicate native episode assignment')
        identities.update(selected)
        raw = merge_episode_metric_files(run.glob('results_*/**/per_episode_metrics.json'))
        if set(records) & set(raw):
            raise ValueError('Duplicate native episode results')
        if set(raw) != {s + '_' + e for s, e in selected}:
            raise ValueError('Native selected/result identity mismatch')
        records.update(raw)
        record_commits.update({key: plan['source_commit'] for key in raw})
        run_traces = {}
        for path in run.rglob('trajectory.jsonl'):
            identity = json.loads(path.with_name('identity.json').read_text())['identity']
            key = (str(identity['scene_id']), str(identity['runtime_episode_id']))
            if key not in selected or key in traces:
                raise ValueError('Unknown or duplicated trajectory')
            traces[key] = run_traces[key] = read_jsonl(path)
        run_events = [e for p in run.rglob('blp_events.jsonl') for e in read_jsonl(p)]
        events.extend(run_events)
        has_targets = ['blp_action_target' in row for rows in run_traces.values() for row in rows]
        if any(has_targets) and not all(has_targets):
            raise ValueError('A run mixes incompatible action-target recording versions')
        if not allow_diagnostic and (not has_targets or not all(has_targets)):
            raise ValueError('Formal evaluation requires complete accepted-action target records')
        if not allow_diagnostic:
            snapshots = [e for p in run.rglob('blp_snapshots.jsonl') for e in read_jsonl(p)]
            validate_snapshot_parameters(snapshots, document, parameter_sha256, selected)
        if has_targets and all(has_targets):
            native_gt = NativeCandidateGT(run)
            verified = native_gt.verify_events(run_events)
            native_gt_runs.append((run, native_gt, verified))
            gt_by_episode.update({key: native_gt for key in run_traces})
        decisions.extend(e for p in run.rglob('blp_decisions.jsonl') for e in read_jsonl(p))
        dispatches.extend(e for p in run.rglob('scsp_decisions.jsonl') for e in read_jsonl(p))
    if expected is not None:
        wanted = {(str(r['scene_id']), str(r['runtime_episode_id'])) for r in expected}
        if len(wanted) != len(expected):
            raise ValueError('Frozen identity list contains duplicates')
        if set(identities) != wanted:
            raise ValueError('Final output does not cover the frozen evaluation identity set')
        for row in expected:
            observed = identities[str(row['scene_id']), str(row['runtime_episode_id'])]
            for name in ('dataset_manifest_sha256', 'source_episode_ids', 'ordered_categories'):
                if observed[name] != row[name]:
                    raise ValueError(f'Frozen scene/episode/task identity mismatch: {name}')
    if set(traces) != set(identities):
        raise ValueError('Not every complete episode has a trajectory')
    if not allow_diagnostic:
        if any(d['variant'] != variant for d in decisions):
            raise ValueError('Decision variant differs from the claimed variant')
        broken = [e for e in dispatches if e.get('fallback_reason') == 'planner_error'
                  or 'recording_error' in e or e.get('status') == 'telemetry_error']
        if broken:
            raise ValueError('Structural planning/recording errors require review before formal reporting')
    normalized, vectors = [], []
    for (scene, episode), identity in identities.items():
        raw = records[scene + '_' + episode]
        stages = raw['metrics']
        task_count = task_count or len(stages)
        if len(stages) != task_count or len(stages) != len(identity['ordered_categories']):
            raise ValueError('Task cardinality/order mismatch')
        success = [int(s['goat_sub-task_success']) for s in stages]
        vectors.append(success)
        trace = traces[(scene, episode)]
        if len(trace) != raw['total_num_steps']:
            raise ValueError('Trajectory length does not match native step count')
        row = {'method': 'ForesightNav', 'variant': variant, 'task': f'{task_count}-MON',
               'scene_id': scene, 'episode_id': episode, 'goals': list(identity['ordered_categories']),
               'source_episode_ids': list(identity['source_episode_ids']),
               'episode_success': all(success), 'total_steps': raw['total_num_steps'],
               'code_commit': record_commits[scene + '_' + episode],
               'code_commits': sorted(commits), 'stages': [],
               'wallclock_s': raw['frame_timing']['frame_time_sum_ms'] / 1000.,
               'wallclock_definition': 'sum of measured native step wall times, excluding episode reset/setup',
               'native_mon_metrics': raw['mon_metrics'], **prefix_metrics(stages)}
        for k, (category, metric) in enumerate(zip(identity['ordered_categories'], stages)):
            track = [r for r in trace if r['subtask_index'] == k]
            if not track or not track[-1]['is_stop']:
                raise ValueError('Cannot infer a complete stage endpoint without a terminal STOP')
            if len(track) != metric['goat_sub-task_num_steps']:
                raise ValueError('Stage trajectory and evaluator step counts differ')
            row['stages'].append({'k': k + 1, 'goal': category, 'reached': True,
                'success': bool(success[k]), 'steps': metric['goat_sub-task_num_steps'],
                'path_length_m': metric['goat_sub-task_path_length_m'],
                'shortest_path_m': metric['goat_sub-task_shortest_path_length_m'],
                'start_pos': location(track[0]), 'end_pos': location(track[-1]),
                'end_position_observation': 'pre_STOP_pose', 'stop_reason': raw['stop_reasons'][k]})
        normalized.append(row)
    aggregate = summarize_records(records)
    conditional = conditional_rates(vectors)
    native = aggregate['mon_metrics']
    if not math.isclose(conditional['Joint'], native['sr'], abs_tol=1e-12):
        raise ValueError('Conditional product disagrees with native SR')
    scores = {k: 100. * native[v] for k, v in (('SR', 'sr'), ('SPL', 'spl'), ('PR', 'psr'), ('PSR', 'psr'), ('PeSPL', 'pespl'))}
    scores['PPL'] = 100. * statistics.mean(r['PPL'] for r in normalized)
    rates = conditional['rates']
    for k, name in enumerate(('P_C1', 'P_C2_given_C1', 'P_C3_given_C1C2', 'P_C4_given_C1C2C3')):
        scores[name] = 100. * rates[k] if k < len(rates) and rates[k] is not None else None
    scores['Joint'] = 100. * conditional['Joint']
    native_paths = {(r['scene_id'], str(r['episode_id']), k): s['path_length_m']
                    for r in normalized for k, s in enumerate(r['stages'])}
    gt_label = (lambda scene, episode, k, position:
                gt_by_episode[str(scene), str(episode)](scene, episode, k, position)) if gt_by_episode else None
    process = process_metrics(events, decisions, dispatches, traces, set(identities),
                              gt_label=gt_label, native_stage_paths=native_paths)
    if native_gt_runs:
        process['native_target_gt_provenance'] = [
            {'run_directory': str(run), 'recorded_label_verification': verified,
             **native_gt.provenance()} for run, native_gt, verified in native_gt_runs]
    summary = {'method': 'ForesightNav', 'variant': variant, 'task': f'{task_count}-MON',
               'n_episodes': len(normalized), 'n_error': 0, **scores,
               'PPL_implementation': 'new supplementary implementation: all-stage PR times successful-prefix path efficiency',
               'conditional_counts': conditional, 'native_aggregate': aggregate,
               'process': process,
               'analysis_sources_sha256': {'report_variants.py': SOURCE_SHA256,
                   'action_path_metrics.py': ACTION_SOURCE_SHA256, 'native_candidate_gt.py': GT_SOURCE_SHA256,
                   'blp_parameters.py': PARAMETER_SOURCE_SHA256},
               'source_commits': sorted(commits), 'diagnostic_only': bool(allow_diagnostic)}
    if document is not None:
        summary['calibration_binding'] = {'selection': document.selection,
            'parameter_sha256': parameter_sha256,
            'parameters_by_scene': {s: asdict(document.resolve(s)[0]) for s in sorted({s for s, e in identities})}}
    if not allow_diagnostic and not summary['process']['invalid_path_complete']:
        raise ValueError('Ambiguous invalid-visit path association requires action-level audit before publication')
    if not allow_diagnostic and summary['process']['invalid_path_source'] != 'blp_action_target_v1_native_gt':
        raise ValueError('Formal path metrics require native-GT-labelled accepted-action targets')
    return normalized, decisions, summary


def effects_and_pairs(variants):
    names = ('nearest', 'suffix_only', 'myopic', 'full')
    if set(variants) != set(names):
        raise ValueError('All four canonical variants are required')
    if any(value[2].get('diagnostic_only') for value in variants.values()):
        raise ValueError('Diagnostic runs cannot populate the formal comparison table')
    versions = [value[2].get('analysis_sources_sha256') for value in variants.values()]
    if any(versions) and any(v != versions[0] for v in versions):
        raise ValueError('Reanalyze all variants with the same loaded analysis sources')
    bindings = [value[2].get('calibration_binding') for value in variants.values()]
    if any(bindings) and any(b != bindings[0] for b in bindings):
        raise ValueError('Variants must use the same frozen scene-specific parameter assignment')
    identities = [{(r['scene_id'], r['episode_id']) for r in variants[n][0]} for n in names]
    if any(s != identities[0] for s in identities):
        raise ValueError('Paired comparisons require identical episode identities')
    effects = {}
    for metric in ('SR', 'SPL'):
        n, s, m, f = [variants[v][2].get('rates_pct_unrounded', variants[v][2])[metric] for v in names]
        effects[metric] = {'future_main_effect_pp': ((s - n) + (f - m)) / 2,
                           'belief_main_effect_pp': ((m - n) + (f - s)) / 2,
                           'interaction_pp': (f - m) - (s - n)}
    paired = {}
    for a, b in itertools.combinations(names, 2):
        left, right = [{(r['scene_id'], r['episode_id']): r['episode_success'] for r in variants[n][0]} for n in (a, b)]
        counts = Counter((bool(left[key]), bool(right[key])) for key in left)
        paired[a + '_vs_' + b] = {'both_success': counts[True, True],
            'left_only_success': counts[True, False], 'right_only_success': counts[False, True],
            'both_failure': counts[False, False], 'n_paired': len(left)}
    return {'units': 'percentage points', 'effects': effects, 'paired_episode_success': paired}


def format_summary(summary):
    keys = ('SR', 'SPL', 'PR', 'PPL', 'PSR', 'PeSPL', 'P_C1', 'P_C2_given_C1',
            'P_C3_given_C1C2', 'P_C4_given_C1C2C3', 'Joint')
    precise = {key: summary[key] for key in keys if key in summary}
    return {**summary, 'rates_pct_unrounded': precise,
            **{key: round(value, 2) if value is not None else None for key, value in precise.items()}}


def export_comparison(output, variants):
    output = Path(output) / 'summary'
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'effects.json', effects_and_pairs(variants))
    columns = ['method', 'variant', 'task', 'n_episodes', 'n_error', 'future_suffix',
               'validity_belief', 'SR', 'SPL', 'PR', 'PPL', 'P_C1', 'P_C2_given_C1',
               'P_C3_given_C1C2', 'P_C4_given_C1C2C3', 'Joint', 'Invalid_visits_per_ep',
               'deviation_from_nearest_proposal_pct', 'deviation_from_nearest_after_hysteresis_pct',
               'deviation_from_nearest_realized_pct']
    with (output / 'all.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for variant in ('nearest', 'suffix_only', 'myopic', 'full'):
            summary = variants[variant][2]
            row = {**summary, **summary['process'],
                   'future_suffix': variant in ('suffix_only', 'full'),
                   'validity_belief': variant in ('myopic', 'full')}
            writer.writerow({key: '' if row[key] is None else
                             f'{row[key]:.2f}' if isinstance(row[key], float) else row[key]
                             for key in columns})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', required=True)
    parser.add_argument('--run', type=Path, action='append', required=True)
    parser.add_argument('--expected-identities', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--diagnostic-only', action='store_true')
    parser.add_argument('--calibration', type=Path)
    args = parser.parse_args()
    expected = json.loads(args.expected_identities.read_text()) if args.expected_identities else None
    calibration, parameter_hash = None, None
    if args.calibration:
        blob = args.calibration.read_bytes()
        calibration = json.loads(blob)
        parameter_hash = hashlib.sha256(blob).hexdigest()
    episodes, decisions, summary = normalize(args.variant, args.run, expected, args.diagnostic_only,
                                             calibration, parameter_hash)
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / 'summary' / f'{args.variant}.json', format_summary(summary))
    (args.output / 'logs').mkdir(exist_ok=True)
    for kind, rows in (('episodes', episodes), ('decisions', decisions)):
        path = args.output / 'logs' / f'{args.variant}_{kind}.jsonl'
        path.write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in rows))
    print(json.dumps({k: summary[k] for k in ('variant', 'n_episodes', 'SR', 'SPL', 'PR', 'PPL', 'Joint')}, indent=2))


if __name__ == '__main__':
    main()
