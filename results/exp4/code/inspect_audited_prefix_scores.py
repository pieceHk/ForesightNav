from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

import report_variants as analysis
from experiments.exp4_collect import source_fingerprint

root = Path('/nvme/khy/ForesightNav_exp4_depthfix/results/exp4')
def load(path):
    return json.loads(path.read_text())
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
ledger = load(root / 'meta/evaluation-full-runs.json')
prefix = load(root / 'diagnostics/completed-prefix/full/summary.json')
cp2 = load(root / 'meta/CP2b_acceptance.json')
fingerprint = source_fingerprint()
assert fingerprint == ledger['source_fingerprint'] == cp2['source_fingerprint']
assert digest(root / 'meta/calibration.json') == ledger['parameter_sha256'] == prefix['bindings']['parameter_sha256']
assert prefix['all_paths_complete'] and prefix['diagnostic_only'] and not prefix['formal400_complete']
versions = {'report_variants.py': analysis.SOURCE_SHA256, 'action_path_metrics.py': analysis.ACTION_SOURCE_SHA256,
    'native_candidate_gt.py': analysis.GT_SOURCE_SHA256, 'candidate_identities.py': analysis.IDENTITY_SOURCE_SHA256,
    'blp_parameters.py': analysis.PARAMETER_SOURCE_SHA256}
assert versions == prefix['bindings']['analysis_sources_sha256']
selected, vectors, live_records = {}, [], 0
for job in ledger['jobs']:
    if not job.get('run_directory'):
        continue
    exported = analysis.merge_episode_metric_files(Path(job['run_directory']).glob('results_*/**/per_episode_metrics.json'))
    live_records += len(exported)
    expected = {job['scene'] + '_' + str(row['runtime_episode_id']): row for row in job['identities']}
    assert set(exported) <= set(expected)
    for key, raw in exported.items():
        identity = expected[key]
        cache = root / 'diagnostics/completed-prefix/full' / job['scene'] / (str(identity['runtime_episode_id']) + '.json')
        if not cache.is_file():
            continue
        audited = load(cache)
        assert audited['bindings'] == prefix['bindings']
        assert audited['raw_native_record_sha256'] == hashlib.sha256(json.dumps(raw, sort_keys=True, allow_nan=False).encode()).hexdigest()
        assert audited['process']['invalid_path_complete'] and audited['process']['native_stage_path_check_count'] == 3
        assert len(raw['metrics']) == 3 and [row['category'] for row in raw['tasks']] == identity['ordered_categories']
        assert audited['goals'] == identity['ordered_categories'] and audited['source_episode_ids'] == identity['source_episode_ids']
        vector = [int(row['goat_sub-task_success']) for row in raw['metrics']]
        assert vector == audited['success_vector']
        assert key not in selected
        selected[key] = raw
        vectors.append(vector)
assert len(selected) == prefix['completed_native_metric_records_checked']
native = analysis.summarize_records(selected)['mon_metrics']
conditional = analysis.conditional_rates(vectors)
assert math.isclose(native['sr'], conditional['Joint'], abs_tol=1e-12)
scores = {key: 100 * native[source] for key, source in [('SR', 'sr'), ('SPL', 'spl'), ('PR', 'psr')]}
scores['PPL'] = 100 * statistics.mean(analysis.prefix_metrics(raw['metrics'])['PPL'] for raw in selected.values())
assert source_fingerprint() == fingerprint and digest(root / 'meta/calibration.json') == ledger['parameter_sha256']
print(json.dumps({'checked_at_utc': datetime.now(timezone.utc).isoformat(), 'stage': 'full',
    'diagnostic_only': True, 'formal400_complete': False, 'audited_episodes': len(selected),
    'current_native_completed_records': live_records, 'successful_sequences': sum(all(v) for v in vectors),
    'successful_subtasks': sum(sum(v) for v in vectors), 'rates_pct_unrounded': scores,
    'conditional_counts': conditional, 'native_source_entries': len(fingerprint),
    'parameter_sha256': ledger['parameter_sha256'], 'common_analysis_sha256': versions,
    'source_and_parameters_unchanged': True, 'native_states_modified': False, 'writes_outputs': False,
    'scope': 'Interim scores over exactly the currently GT-audited native episode exports. Reuses unchanged native score aggregation and supplementary PPL; no GT recomputation, fitting, simulation or formal stage finalization.'}))
