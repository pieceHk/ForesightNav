"""Audit newly completed smoke cases using the formal frozen analysis path."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import traceback

from report_variants import normalize, format_summary, write_json
import report_variants as analysis

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads(path.read_text())


def main():
    ledger = load(ROOT / 'meta/evaluation-smoke-runs.json')
    parameter_file = ROOT / 'meta/calibration.json'
    parameters = load(parameter_file)
    parameter_sha = hashlib.sha256(parameter_file.read_bytes()).hexdigest()
    versions = {'report_variants.py': analysis.SOURCE_SHA256,
                'action_path_metrics.py': analysis.ACTION_SOURCE_SHA256,
                'native_candidate_gt.py': analysis.GT_SOURCE_SHA256,
                'blp_parameters.py': analysis.PARAMETER_SOURCE_SHA256, 'candidate_identities.py': analysis.IDENTITY_SOURCE_SHA256}
    new, behavior = [], []
    for job in ledger['jobs']:
        if job['status'] != 'completed':
            continue
        destination = ROOT / 'diagnostics/CP3/early' / job['scene']
        run = Path(job['run_directory'])
        if not (destination / 'summary.json').exists():
            try:
                episodes, decisions, summary = normalize(
                    'full', [run], expected=job['identities'], calibration=parameters,
                    parameter_sha256=parameter_sha)
                write_json(destination / 'summary.json', format_summary(summary))
                write_json(destination / 'episodes.json', episodes)
            except Exception:
                destination.mkdir(parents=True, exist_ok=True)
                (destination / 'normalization-error.txt').write_text(traceback.format_exc())
                raise
            new.append({'scene_id': job['scene'], 'steps': episodes[0]['total_steps'],
                        'success_vector': [int(stage['success']) for stage in episodes[0]['stages']],
                        **{k: summary[k] for k in ('SR', 'SPL', 'PR', 'PPL')},
                        'invalid_path_complete': summary['process']['invalid_path_complete'],
                        'native_stage_path_max_abs_error_m': summary['process']['native_stage_path_max_abs_error_m']})
        cached = load(destination / 'summary.json')
        assert cached['analysis_sources_sha256'] == versions, 'Reanalyze all cases after an analysis change'
        assert cached['calibration_binding']['parameter_sha256'] == parameter_sha
        assert cached['n_episodes'] == 1 and cached['variant'] == 'full' and not cached['diagnostic_only']
        assert cached['process']['invalid_path_complete']
        episode = load(destination / 'episodes.json')[0]
        assert str(episode['scene_id']) == job['scene']
        events = [json.loads(line) for path in run.rglob('blp_events.jsonl')
                  for line in path.read_text().splitlines() if line]
        stages = []
        for stage in episode['stages']:
            checks = [e for e in events if e.get('event') == 'verification_final'
                      and e.get('k') == stage['k'] - 1]
            item = dict(stage)
            item['last_candidate_checks'] = [{k: e[k] for k in
                ('t', 'candidate_id', 'E', 'S', 'v', 'y', 'position') if k in e}
                for e in checks[-3:]]
            if stage['success']:
                item['outcome_description'] = 'native_success'
            elif stage['stop_reason'] == 'verified_target':
                item['outcome_description'] = 'system_confirmed_but_native_failed'
            elif stage['stop_reason'] == 'max_steps' or 'budget' in stage['stop_reason']:
                item['outcome_description'] = 'budget_stop_native_failed'
            else:
                item['outcome_description'] = 'other_native_failure'
            stages.append(item)
        behavior.append({'scene_id': job['scene'], 'episode_id': episode['episode_id'], 'stages': stages})
    write_json(ROOT / 'diagnostics/CP3/completed-behavior-review.json', {
        'checked_at_utc': datetime.now(timezone.utc).isoformat(), 'diagnostic_only': True,
        'definition': 'observed stop/outcome categories, not causal attribution or evaluation tuning',
        'candidate_y_definition': 'GT at checked candidate coordinate; distinct from final agent success',
        'episodes': behavior})
    print(json.dumps({'completed_scenes': len(behavior), 'newly_audited': new}), flush=True)


if __name__ == '__main__':
    main()
