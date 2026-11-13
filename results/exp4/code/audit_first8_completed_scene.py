"""Audit one fully completed scene early using the frozen CPU-only native GT path."""
from datetime import datetime,timezone
from pathlib import Path
import json,os,sys
os.environ['CUDA_VISIBLE_DEVICES']=''
from evaluate_sequence import OUT,load,digest,source_fingerprint,audit_scene
from report_variants import normalize,format_summary

root=OUT/'scopes/first8';scope=load(root/'meta/scope.json')
stage=sys.argv[1] if len(sys.argv)>1 else 'myopic';assert stage in ('myopic','suffix_only')
assert source_fingerprint()==scope['source_fingerprint']
assert digest(OUT/'meta/calibration.json')==scope['parameter_sha256']
ledger_path=root/f'meta/evaluation-{stage}-runs.json'
ledger=load(ledger_path) if ledger_path.exists() else {'jobs':[]}
jobs=[j for j in ledger['jobs'] if j['status']=='completed' and j.get('returncode')==0]
if not jobs:
    print(json.dumps({'ready':False,'reason':'No scene has a completed Native launcher with actual returncode0 yet','native_states_modified':False}));raise SystemExit(0)
job=next(j for scene in scope['ordered_scenes'] for j in jobs if j['scene']==scene)
folder=root/'diagnostics/early-completed-scene-audits'/stage/job['scene'];proof=folder/'result.json'
if proof.exists():
    value=load(proof)
    for name,sha in value['artifact_sha256'].items():assert digest(folder/name)==sha
    print(json.dumps({'reused_existing_audit':True,**value}));raise SystemExit(0)
assert not folder.exists(),'Inspect an incomplete audit directory before retrying this CPU analysis'
audit_scene(Path(job['run_directory']),job)
expected=[r for r in scope['identities'] if r['scene_id']==job['scene']]
assert len(expected)==20 and job['identities']==expected
rows,decisions,summary=normalize(stage,[job['run_directory']],expected=expected,
    calibration=load(OUT/'meta/calibration.json'),parameter_sha256=scope['parameter_sha256'])
assert summary['n_episodes']==20 and summary['n_error']==0
assert summary['process']['native_stage_path_check_count']==60 and summary['process']['invalid_path_complete']
assert summary['analysis_sources_sha256']==load(root/'summary/full.json')['analysis_sources_sha256']
folder.mkdir(parents=True)
summary.update(coverage='First completed scene only; remaining fixed scenes continue unchanged',complete_stage160=False)
(folder/'summary.json').write_text(json.dumps(format_summary(summary),indent=2)+'\n')
for kind,values in (('episodes',rows),('decisions',decisions)):
    (folder/(kind+'.jsonl')).write_text(''.join(json.dumps(row,allow_nan=False)+'\n' for row in values))
assert source_fingerprint()==scope['source_fingerprint']
value={'at_utc':datetime.now(timezone.utc).isoformat(),'stage':stage,'scene':job['scene'],
 'completed_native_episodes':20,'audited_stage_paths':60,'max_native_path_abs_error_m':summary['process']['native_stage_path_max_abs_error_m'],
 'invalid_paths_complete':True,'n_error':0,'native_launcher_returncode':job['returncode'],'exit_status_proof':job['exit_status_proof'],
 'source_and_parameters_unchanged':True,'native_states_modified':False,'new_Native_workers_started':False,
 'normalization_uses_CPU_PathFinder_only':True,'full_stage160_complete':False,
 'artifact_sha256':{name:digest(folder/name) for name in ('summary.json','episodes.jsonl','decisions.jsonl')}}
proof.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps({'proof':str(proof.relative_to(OUT)),**value}))
