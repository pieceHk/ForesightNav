"""Read-only 300-episode comparison; pending identity source is injected locally."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import report_variants as report
import proposed_candidate_identities as proposed

ROOT = Path('/nvme/khy/ForesightNav_exp4_depthfix/results/exp4')


def main():
    audit = json.loads((ROOT / 'meta/completed-calibration-scene-audits.json').read_text())
    scenes = audit['completed_scenes']
    scenes = list(scenes.values()) if isinstance(scenes, dict) else scenes
    assert audit['complete_episodes'] == 300 and len(scenes) == 10
    runs = [Path(x['run_directory']) for x in scenes]
    expected = [x for x in json.loads((ROOT / 'meta/2mon-identities.json').read_text())
                if x['scene_id'] in {y['scene'] for y in scenes}]
    before = report.normalize('full', runs, expected=expected, allow_diagnostic=True)
    report.canonicalize = proposed.canonicalize
    report.IDENTITY_SOURCE_SHA256 = proposed.SOURCE_SHA256
    after = report.normalize('full', runs, expected=expected, allow_diagnostic=True)
    assert before[0] == after[0] and before[1] == after[1] and len(after[0]) == 300
    metrics = ['SR', 'SPL', 'PR', 'PPL', 'PSR', 'PeSPL', 'P_C1', 'P_C2_given_C1', 'Joint']
    assert all(before[2][k] == after[2][k] for k in metrics)
    a, b = before[2]['process'], after[2]['process']
    # Preserve the completed comparison even if a later invariant fails; native
    # GT query counts can increase when an unchecked pursuit becomes eligible.
    (ROOT / 'code/pending_unchecked_alias/normalization-comparison-last.json').write_text(
        json.dumps({'episode_rows_equal': before[0] == after[0],
                    'decision_rows_equal': before[1] == after[1],
                    'process_before': a, 'process_after': b}, indent=2, allow_nan=False) + '\n')
    assert a['accepted_action_path_m'] == b['accepted_action_path_m']
    assert a['invalid_path_association_gaps'] == b['invalid_path_association_gaps']
    assert len(b['invalid_path_association_gaps']) == 28 and not b['invalid_path_complete']
    for key in ['checked_distinct_candidates', 'native_stage_path_check_count', 'native_stage_path_max_abs_error_m']:
        assert a[key] == b[key], key
    unchanged = []
    for key in a:
        if key in ('checked_candidate_identity', 'native_target_gt_provenance',
                   'native_target_GT_labelled_action_rows',
                   'checked_invalid_identities_with_current_valid_targets') or 'path' in key:
            continue
        assert a[key] == b[key], key
        unchanged.append(key)
    result = {'checked_at_utc': datetime.now(timezone.utc).isoformat(),
              'scope': '300 closed native 2-MON artifacts; analysis-only comparison, no new navigation or fitting',
              'episodes': 300, 'scenes': [x['scene'] for x in scenes],
              'native_episode_rows_identical': True, 'decision_rows_identical': True,
              'navigation_and_conditional_metrics_identical': metrics,
              'other_unchanged_process_fields': unchanged,
              'checked_distinct_candidates_before': a['checked_distinct_candidates'],
              'checked_distinct_candidates_after': b['checked_distinct_candidates'],
              'accepted_physical_path_unchanged': True,
              'raw_association_gaps_retained': b['invalid_path_association_gaps'],
              'invalid_path_complete': False,
              'invalid_visit_path_before_m': a['invalid_visit_path_m'],
              'invalid_visit_path_after_m': b['invalid_visit_path_m'],
              'GT_labelled_action_rows_before': a['native_target_GT_labelled_action_rows'],
              'GT_labelled_action_rows_after': b['native_target_GT_labelled_action_rows'],
              'native_stage_checks': b['native_stage_path_check_count'],
              'native_stage_path_max_abs_error_m': b['native_stage_path_max_abs_error_m'],
              'alias_statistics': b['checked_candidate_identity'],
              'source_before': before[2]['analysis_sources_sha256'],
              'source_after': after[2]['analysis_sources_sha256'],
              'validation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'runtime_source_changed': False, 'raw_data_changed': False,
              'pending_analysis_applied_to_CPU_waiter': False}
    (ROOT / 'diagnostics/unchecked-alias-native300-validation.json').write_text(
        json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('raw_association_gaps_retained', 'other_unchanged_process_fields')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
