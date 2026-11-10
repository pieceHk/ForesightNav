"""Audit completed native Full coverage records without policy changes."""
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

from pbsp.navigation.blp_parameters import CalibrationDocument

ROOT = Path(__file__).resolve().parents[1]


def rows(path):
    return [json.loads(line) for line in path.read_bytes().split(b'\n')[:-1] if line]


def identity(row):
    return str(row['scene_id']), str(row['episode_id']), row['k'], row['t']


def main():
    ledger = json.loads((ROOT / 'meta/evaluation-smoke-runs.json').read_text())
    calibration_path = ROOT / 'meta/calibration.json'
    parameter_sha = hashlib.sha256(calibration_path.read_bytes()).hexdigest()
    document = CalibrationDocument.from_mapping(json.loads(calibration_path.read_text()))
    counts, histograms, scenes, violations = Counter(), {'current': Counter(), 'future': Counter()}, {}, []
    for job in ledger['jobs']:
        if job['status'] != 'completed':
            continue
        run = Path(job['run_directory'])
        snapshots = [row for path in run.rglob('blp_snapshots.jsonl') for row in rows(path)]
        decisions = {identity(row): row for path in run.rglob('blp_decisions.jsonl') for row in rows(path)}
        assert len(decisions) == len(snapshots)
        theta, binding = document.resolve(job['scene'])
        parameters = asdict(theta)
        binding = dict(binding, calibration_sha256=parameter_sha)
        scenes[job['scene']] = len(snapshots)
        for row in snapshots:
            key = identity(row)
            assert row['shadow'] is False and row['parameters'] == parameters and row['parameter_binding'] == binding
            counts['snapshots'] += 1
            k, layers = row['k'], row['snapshot']['layers']
            counts['nonempty_current_roots'] += bool(layers[k])
            counts['multiple_current_candidates'] += len(layers[k]) >= 2
            if k < len(layers) - 1:
                counts['pre_final_roots'] += 1
                counts['pre_final_with_any_future_candidates'] += any(layers[k + 1:])
            for index, layer in enumerate(layers[k:], k):
                histograms['current' if index == k else 'future'].update(candidate['N_views'] for candidate in layer)
            decision = decisions[key]
            for candidate in decision['candidates']:
                n, score = candidate['N_views'], candidate['E']
                coverage = -math.expm1(-n / 3.)
                z = theta.a * score + theta.b0
                q = 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))
                belief = coverage * q + (1 - coverage) * theta.pi
                counts['current_q_S_B_records_checked'] += 1
                if (not 0 <= n <= 12 or int(n) != n or any(
                        not math.isclose(candidate[name], value, rel_tol=1e-10, abs_tol=1e-12)
                        for name, value in (('S', coverage), ('q', q), ('B', belief)))):
                    violations.append({'identity': key, 'raw_rank': candidate['raw_rank'], 'error': 'coverage_or_belief_formula'})
    result = {'checked_at_utc': datetime.now(timezone.utc).isoformat(),
              'scope': 'completed native Full smoke cases only; no GT, fit, or evaluation tuning',
              'completed_scenes': len(scenes), 'snapshots': counts['snapshots'],
              'counts': dict(counts), 'scene_snapshots': scenes,
              'histogram_unit': 'candidate occurrence in planning snapshot, not distinct candidate',
              'N_views_histograms': {key: dict(sorted(value.items())) for key, value in histograms.items()},
              'violations': violations, 'parameter_sha256': parameter_sha,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    destination = ROOT / 'diagnostics/CP3/evidence-coverage-audit.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result))
    assert not violations, 'Completed native coverage records require review'


if __name__ == '__main__':
    main()
