"""Audit fixed first completed Full episode while its scene keeps running.

This uses existing analysis components, without bypassing the whole-run
normalizer's completion requirement or emitting a formal400 summary.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import report_variants as analysis
from candidate_identities import canonicalize
from experiments.exp4_collect import source_fingerprint
from native_candidate_gt import NativeCandidateGT

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_bytes().split(b'\n')[:-1] if line]


def main():
    ledger = load(ROOT / 'meta/evaluation-full-runs.json')
    assert ledger['status'] == 'running'
    job = ledger['jobs'][0]
    run = Path(job['run_directory'])
    records = analysis.merge_episode_metric_files(run.glob('results_*/**/per_episode_metrics.json'))
    expected = sorted(job['identities'], key=lambda row: int(row['runtime_episode_id']))[0]
    scene, episode = expected['scene_id'], str(expected['runtime_episode_id'])
    key = scene + '_' + episode
    raw = records[key]
    assert len(raw['metrics']) == 3
    trajectories = []
    for path in run.rglob('trajectory.jsonl'):
        identity = load(path.with_name('identity.json'))['identity']
        if identity['scene_id'] == scene and str(identity['runtime_episode_id']) == episode:
            trajectories.append(rows(path))
    assert len(trajectories) == 1
    trace = trajectories[0]
    assert len(trace) == raw['total_num_steps']
    native_paths = {}
    for k, metric in enumerate(raw['metrics']):
        track = [row for row in trace if row['subtask_index'] == k]
        assert track[-1]['is_stop'] and len(track) == metric['goat_sub-task_num_steps']
        native_paths[scene, episode, k] = metric['goat_sub-task_path_length_m']
    def select(name):
        # Shared scene logs use flat BLP identity and nested SCSP identity.
        paths = list(run.rglob(name))
        assert paths, name
        selected = []
        for path in paths:
            for row in rows(path):
                identity = row['episode_identity'] if name == 'scsp_decisions.jsonl' else row
                observed_scene = str(identity['scene_id'])
                observed_episode = str(identity['runtime_episode_id'] if name == 'scsp_decisions.jsonl' else identity['episode_id'])
                if (observed_scene, observed_episode) == (scene, episode):
                    selected.append(row)
        return selected
    events, decisions, dispatches = select('blp_events.jsonl'), select('blp_decisions.jsonl'), select('scsp_decisions.jsonl')
    native_gt = NativeCandidateGT(run)
    verified = native_gt.verify_events(events)
    events, traces, identities = canonicalize(events, {(scene, episode): trace})
    process = analysis.process_metrics(events, decisions, dispatches, traces, {(scene, episode)},
                                      gt_label=native_gt, native_stage_paths=native_paths)
    assert process['invalid_path_complete'] and process['native_stage_path_check_count'] == 3
    assert source_fingerprint() == ledger['source_fingerprint']
    assert hashlib.sha256((ROOT / 'meta/calibration.json').read_bytes()).hexdigest() == ledger['parameter_sha256']
    result = {'checked_at_utc': datetime.now(timezone.utc).isoformat(), 'diagnostic_only': True,
              'scope': 'Fixed first scene/first episode only, observed natively complete; Full400 scene and stage still running',
              'scene_id': scene, 'episode_id': episode, 'source_episode_ids': expected['source_episode_ids'],
              'goals': expected['ordered_categories'], 'steps': raw['total_num_steps'],
              'success_vector': [int(metric['goat_sub-task_success']) for metric in raw['metrics']],
              'native_mon_metrics': raw['mon_metrics'], 'process': process,
              'candidate_event_GT_verification': verified, 'candidate_identity_audit': identities,
              'native_GT_provenance': native_gt.provenance(),
              'analysis_sources_sha256': {'report_variants.py': analysis.SOURCE_SHA256,
                    'action_path_metrics.py': analysis.ACTION_SOURCE_SHA256,
                    'native_candidate_gt.py': analysis.GT_SOURCE_SHA256,
                    'candidate_identities.py': analysis.IDENTITY_SOURCE_SHA256,
                    'blp_parameters.py': analysis.PARAMETER_SOURCE_SHA256},
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'source_and_parameters_unchanged': True, 'native_status_modified': False}
    (ROOT / 'diagnostics/Full400/first-episode-audit.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'scene': scene, 'episode': episode, 'steps': result['steps'],
        'success_vector': result['success_vector'], 'stage_paths_checked': 3,
        'invalid_path_complete': process['invalid_path_complete'],
        'max_path_error_m': process['native_stage_path_max_abs_error_m'], 'stage400_complete': False}))


if __name__ == '__main__':
    main()
