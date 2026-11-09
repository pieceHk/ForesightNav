"""Inspect live corrected coverage records; no GT, fitting, or policy changes."""
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def rows(path):
    return [json.loads(line) for line in path.read_bytes().split(b'\n')[:-1] if line]


def identity(row):
    return str(row['scene_id']), str(row['episode_id']), row['k'], row['t']


def audit():
    ledger = json.loads((ROOT / 'meta/calibration_runs.json').read_text())
    histograms = {name: Counter() for name in ('current_candidate_occurrences', 'future_candidate_occurrences')}
    counts, violations = Counter(), []
    scenes = {}
    for job in ledger['jobs']:
        if not job.get('run_directory'):
            continue
        run = Path(job['run_directory'])
        snapshots = [r for p in run.rglob('blp_snapshots.jsonl') for r in rows(p)]
        decisions = {identity(r): r for p in run.rglob('blp_decisions.jsonl') for r in rows(p)}
        scenes[job['scene']] = len(snapshots)
        for row in snapshots:
            counts['snapshots'] += 1
            if row['shadow'] is not True:
                violations.append({'identity': identity(row), 'error': 'not_shadow_collection'})
            k, layers = row['k'], row['snapshot']['layers']
            counts['nonempty_current_roots'] += bool(layers[k])
            counts['multiple_current_candidates'] += len(layers[k]) >= 2
            if k < len(layers) - 1:
                counts['pre_final_roots'] += 1
                counts['pre_final_with_any_future_candidates'] += any(layers[k + 1:])
            for m, layer in enumerate(layers[k:], k):
                name = 'current_candidate_occurrences' if m == k else 'future_candidate_occurrences'
                histograms[name].update(c['N_views'] for c in layer)
            decision = decisions.get(identity(row))
            if decision is None:
                counts['snapshots_waiting_for_decision_record'] += 1
                continue
            for candidate in decision['candidates']:
                n, e = candidate['N_views'], candidate['E']
                theta = row['parameters']
                s = -math.expm1(-n / 3.)
                z = theta['a'] * e + theta['b0']
                q = 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))
                belief = s * q + (1 - s) * theta['pi']
                counts['current_q_S_B_records_checked'] += 1
                if (not 0 <= n <= 12 or int(n) != n or
                        any(not math.isclose(candidate[key], value, rel_tol=1e-10, abs_tol=1e-12)
                            for key, value in (('S', s), ('q', q), ('B', belief)))):
                    violations.append({'identity': identity(row), 'raw_rank': candidate['raw_rank'],
                                       'error': 'invalid_coverage_or_belief_record'})
    result = {'checked_at_utc': datetime.now(timezone.utc).isoformat(),
              'scope': 'live a3 shadow inputs; no GT, fitted parameters, or evaluation performance',
              'histogram_unit': 'candidate occurrence in a planning snapshot, not distinct candidates',
              'scene_snapshots': scenes, 'counts': dict(counts),
              'N_views_histograms': {name: dict(sorted(values.items())) for name, values in histograms.items()},
              'violations': violations, 'launches_workers': False}
    destination = ROOT / 'diagnostics/depth-unit-fix/live-coverage-record-audit.json'
    destination.write_text(json.dumps(result, indent=2) + '\n')
    if violations:
        raise ValueError(f'Coverage or belief record violations: {len(violations)}')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    audit()
