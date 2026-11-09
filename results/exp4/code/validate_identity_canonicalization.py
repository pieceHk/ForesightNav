"""Compare complete native calibration artifacts across identity accounting versions."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import report_variants as current

WORK = Path(__file__).resolve().parents[3]
RUNTIME = Path('/nvme/khy/ForesightNav_exp4_depthfix')
ROOT = RUNTIME / 'results/exp4'


def main():
    original_path = WORK / 'results/exp4/code/pending_identity_binding/base_report_variants.py'
    spec = importlib.util.spec_from_file_location('original_report_variants', original_path)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    ledger = json.loads((ROOT / 'meta/calibration_runs.json').read_text())
    scope = {'6s7QHgap2fW', 'TEEsavR23oF', 'QaLdnwvtxbs'}
    jobs = [j for j in ledger['jobs'] if j['status'] == 'completed' and j['scene'] in scope]
    assert len(jobs) == 3
    runs = [Path(j['run_directory']) for j in jobs]
    expected = [r for r in json.loads((ROOT / 'meta/2mon-identities.json').read_text())
                if r['scene_id'] in {j['scene'] for j in jobs}]
    before = original.normalize('full', runs, expected=expected, allow_diagnostic=True)
    after = current.normalize('full', runs, expected=expected, allow_diagnostic=True)
    assert before[0] == after[0]
    assert before[1] == after[1]
    assert len(after[0]) == 90
    names = ['SR', 'SPL', 'PR', 'PPL', 'PSR', 'PeSPL', 'P_C1', 'P_C2_given_C1', 'Joint']
    assert all(before[2][k] == after[2][k] for k in names)
    bp, ap = before[2]['process'], after[2]['process']
    assert bp['accepted_action_path_m'] == ap['accepted_action_path_m']
    assert bp['invalid_path_association_gaps'] == ap['invalid_path_association_gaps']
    assert not ap['invalid_path_complete'] and len(ap['invalid_path_association_gaps']) == 10
    for k in ['native_stage_path_check_count', 'native_stage_path_max_abs_error_m', 'invalid_visit_path_m']:
        assert bp[k] == ap[k], k
    assert ap['checked_candidate_identity']['identities_merged'] == 1
    assert bp['checked_distinct_candidates'] - ap['checked_distinct_candidates'] == 1
    native = ap['native_target_gt_provenance']
    result = {'checked_at_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'closed a3 90-episode native artifact accounting; no parameter fitting or native rerun',
        'episodes': len(after[0]), 'scenes': [j['scene'] for j in jobs],
        'normalized_native_episode_rows_identical': True,
        'all_decision_rows_identical': True, 'native_SR_SPL_PR_PPL_and_conditionals_identical': True,
        'accepted_physical_path_identical': True, 'invalid_GT_point_path_identical': True,
        'raw_gaps_retained': ap['invalid_path_association_gaps'],
        'path_complete': ap['invalid_path_complete'],
        'native_stage_path_checks': ap['native_stage_path_check_count'],
        'native_stage_path_max_abs_error_m': ap['native_stage_path_max_abs_error_m'],
        'raw_checked_registry_identities': bp['checked_distinct_candidates'],
        'canonical_checked_identities': ap['checked_distinct_candidates'],
        'identity_rule': ap['checked_candidate_identity'],
        'candidate_GT_checks': sum(r['recorded_label_verification']['candidate_verifications'] for r in native),
        'recovery_GT_checks': sum(r['recorded_label_verification']['recovery_verifications'] for r in native),
        'max_GT_distance_error_m': max(r['recorded_label_verification']['max_distance_error_m'] for r in native),
        'source_sha256': {'old_report': original.SOURCE_SHA256, **after[2]['analysis_sources_sha256'],
                          'validation': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        'raw_data_changed': False, 'actual_fitting_started': False}
    path = WORK / 'results/exp4/diagnostics/identity-canonicalization-native-validation.json'
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

