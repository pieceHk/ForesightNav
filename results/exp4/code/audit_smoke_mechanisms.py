"""Replay fixed Full inputs for diagnostics; never replaces online ablations."""
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from pbsp.navigation.blp import BeliefParameters, solve_snapshot

ROOT = Path(__file__).resolve().parents[1]


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def key(row):
    return str(row['scene_id']), str(row['episode_id']), row['t'], row['k']


def main():
    ledger = json.loads((ROOT / 'meta/evaluation-smoke-runs.json').read_text())
    counts, differences, violations = Counter(), Counter(), []
    scene_counts, parameters = Counter(), {}
    for job in ledger['jobs']:
        if job['status'] != 'completed':
            continue
        run = Path(job['run_directory'])
        observed = {key(row): row for path in run.rglob('blp_decisions.jsonl') for row in lines(path)}
        snapshots = [row for path in run.rglob('blp_snapshots.jsonl') for row in lines(path)]
        assert len(observed) == len(snapshots)
        for row in snapshots:
            identity = key(row)
            assert row['shadow'] is False
            theta = BeliefParameters(**row['parameters'])
            parameters[job['scene']] = asdict(theta)
            snapshot, k = row['snapshot'], row['k']
            actual = observed[identity]
            replay = solve_snapshot(snapshot, theta, 'full', row['active_raw_rank'])
            if (replay['ordered_raw_ranks'] != actual['ordered_raw_ranks'] or
                    replay['chosen_raw_rank'] != actual['chosen_raw_rank']):
                violations.append({'identity': identity, 'error': 'runtime_replay_order'})
            for left, right in zip(replay['candidates'], actual['candidates']):
                if left['raw_rank'] != right['raw_rank'] or not math.isclose(left['Q'], right['Q'], rel_tol=1e-10, abs_tol=1e-8):
                    violations.append({'identity': identity, 'error': 'runtime_replay_Q'})
            results = {variant: solve_snapshot(snapshot, theta, variant)
                       for variant in ('nearest', 'suffix_only', 'myopic', 'full')}
            counts['snapshots'] += 1
            scene_counts[job['scene']] += 1
            if not snapshot['layers'][k]:
                counts['empty_roots'] += 1
                continue
            counts['nonempty_roots'] += 1
            counts['multiple_current_candidates'] += len(snapshot['layers'][k]) >= 2
            if k < len(snapshot['layers']) - 1:
                counts['pre_final_roots'] += 1
                counts['pre_final_with_any_future_candidates'] += any(snapshot['layers'][k + 1:])
            for variant in ('suffix_only', 'myopic', 'full'):
                differences[variant + '_vs_nearest'] += (results[variant]['unconstrained_raw_rank'] != results['nearest']['unconstrained_raw_rank'])
            differences['full_vs_myopic'] += (results['full']['unconstrained_raw_rank'] != results['myopic']['unconstrained_raw_rank'])
            differences['full_vs_suffix_only'] += (results['full']['unconstrained_raw_rank'] != results['suffix_only']['unconstrained_raw_rank'])
            if k == len(snapshot['layers']) - 1:
                counts['final_task_nonempty_roots'] += 1
                for a, b in [('full', 'myopic'), ('suffix_only', 'nearest')]:
                    qa = {v['raw_rank']: v['Q'] for v in results[a]['candidates']}
                    qb = {v['raw_rank']: v['Q'] for v in results[b]['candidates']}
                    if qa.keys() != qb.keys() or any(not math.isclose(qa[r], qb[r], rel_tol=1e-10, abs_tol=1e-8) for r in qa):
                        violations.append({'identity': identity, 'error': f'final_task_{a}_{b}'})
    payload = {'checked_at_utc': datetime.now(timezone.utc).isoformat(),
               'diagnostic_only': True,
               'definition': 'counterfactual unconstrained proposals on completed Full smoke inputs; not other variants online trajectories or scores',
               'parameters_by_scene': parameters, 'snapshots_by_scene': dict(scene_counts),
               'counts': dict(counts), 'different_proposals_on_nonempty_roots': dict(differences),
               'violations': violations}
    destination = ROOT / 'diagnostics/CP3/mechanism-audit.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps(payload), flush=True)
    if violations:
        raise RuntimeError('Full runtime replay or final-task identity requires review')


if __name__ == '__main__':
    main()
