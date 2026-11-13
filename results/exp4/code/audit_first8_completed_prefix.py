"""Audit only completed scene jobs with real OS exits, using frozen CPU GT.

This does not turn a partial prefix into a completed 160-episode variant.  It
reuses existing scene audits and never writes canonical stage summaries.
"""
from datetime import datetime,timezone
from pathlib import Path
import fcntl,json,os,sys
os.environ['CUDA_VISIBLE_DEVICES']=''
from evaluate_sequence import OUT,load,digest,source_fingerprint,audit_scene
from report_variants import normalize,format_summary

root=OUT/'scopes/first8';scope=load(root/'meta/scope.json')
stage=sys.argv[1] if len(sys.argv)>1 else 'myopic'
assert stage in ('myopic','suffix_only')
assert source_fingerprint()==scope['source_fingerprint']
assert digest(OUT/'meta/calibration.json')==scope['parameter_sha256']
ledger_path=root/f'meta/evaluation-{stage}-runs.json'
ledger_bytes=ledger_path.read_bytes();ledger=json.loads(ledger_bytes)
assert ledger['source_fingerprint']==scope['source_fingerprint'] and ledger['parameter_sha256']==scope['parameter_sha256']
folder=root/'diagnostics/early-completed-scene-audits'/stage;folder.mkdir(parents=True,exist_ok=True)
with (folder/'prefix-audit.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    jobs=[j for scene in scope['ordered_scenes'] for j in ledger['jobs'] if j['scene']==scene and j['status']=='completed' and j.get('returncode')==0]
    proofs=[]
    for job in jobs:
        scene=job['scene'];directory=folder/scene;proof=directory/'result.json'
        assert job.get('exit_status_proof'),'A completed native status alone is insufficient'
        expected=[r for r in scope['identities'] if r['scene_id']==scene]
        assert len(expected)==20 and job['identities']==expected
        run=Path(job['run_directory']);audit_scene(run,job)
        launch=load(run/'launch.json')
        assert launch['identities']==expected and launch['configuration']['seed']==1
        assert launch['configuration']['method']['suffix']['blp_variant']==stage
        if proof.exists():
            value=load(proof)
            assert value['scene']==scene and value['stage']==stage and value['completed_native_episodes']==20
            assert value['audited_stage_paths']==60 and value['n_error']==0
            for name,sha in value['artifact_sha256'].items():assert digest(directory/name)==sha
        else:
            assert not directory.exists(),'Inspect an incomplete CPU audit before retrying analysis'
            mem={k:int(v.split()[0]) for k,v in (s.split(':',1) for s in Path('/proc/meminfo').read_text().splitlines())}
            assert mem['MemAvailable']>=8*1024**2,'Defer CPU GT audit during severe memory pressure'
            rows,decisions,summary=normalize(stage,[str(run)],expected=expected,
                calibration=load(OUT/'meta/calibration.json'),parameter_sha256=scope['parameter_sha256'])
            assert summary['n_episodes']==20 and summary['n_error']==0
            assert summary['process']['native_stage_path_check_count']==60 and summary['process']['invalid_path_complete']
            assert summary['analysis_sources_sha256']==load(root/'summary/full.json')['analysis_sources_sha256']
            directory.mkdir()
            summary.update(coverage='One complete scene in an incomplete fixed eight-scene stage',complete_stage160=False)
            (directory/'summary.json').write_text(json.dumps(format_summary(summary),indent=2)+'\n')
            for kind,values in (('episodes',rows),('decisions',decisions)):
                (directory/(kind+'.jsonl')).write_text(''.join(json.dumps(row,allow_nan=False)+'\n' for row in values))
            value={'at_utc':datetime.now(timezone.utc).isoformat(),'stage':stage,'scene':scene,
                'completed_native_episodes':20,'audited_stage_paths':60,
                'max_native_path_abs_error_m':summary['process']['native_stage_path_max_abs_error_m'],
                'invalid_paths_complete':True,'n_error':0,'native_launcher_returncode':0,'exit_status_proof':job['exit_status_proof'],
                'source_and_parameters_unchanged':True,'native_states_modified':False,'new_Native_workers_started':False,
                'normalization_uses_CPU_PathFinder_only':True,'full_stage160_complete':False,
                'helper_sha256':digest(Path(__file__)),
                'artifact_sha256':{name:digest(directory/name) for name in ('summary.json','episodes.jsonl','decisions.jsonl')}}
            proof.write_text(json.dumps(value,indent=2)+'\n')
        proofs.append({'scene':scene,'proof':str(proof.relative_to(OUT)),'proof_sha256':digest(proof),
            'native_run':str(run),'launch_sha256':digest(run/'launch.json'),'status_sha256':digest(run/'status.json'),
            'max_native_path_abs_error_m':value['max_native_path_abs_error_m']})
        print(json.dumps({'scene':scene,'audited_native_episodes':20,'audited_paths':60,'n_error':0}),flush=True)
    assert source_fingerprint()==scope['source_fingerprint'] and digest(OUT/'meta/calibration.json')==scope['parameter_sha256']
    index={'at_utc':datetime.now(timezone.utc).isoformat(),'stage':stage,'fully_completed_scenes_audited':len(proofs),
        'audited_native_episodes':20*len(proofs),'audited_stage_paths':60*len(proofs),'n_error':0,
        'full_stage160_complete':False,'canonical_summary_modified':False,'native_workers_modified':False,
        'source_fingerprint':scope['source_fingerprint'],'parameter_sha256':scope['parameter_sha256'],
        'ledger_snapshot_sha256':__import__('hashlib').sha256(ledger_bytes).hexdigest(),'proofs':proofs,
        'helper_sha256':digest(Path(__file__))}
    target=folder/'prefix-current.json';temporary=target.with_name(target.name+f'.tmp-{os.getpid()}')
    temporary.write_text(json.dumps(index,indent=2)+'\n');temporary.replace(target)
    print(json.dumps({'index':str(target.relative_to(OUT)),'scenes':len(proofs),'episodes':20*len(proofs),'paths':60*len(proofs),
        'full_stage160_complete':False,'frozen_sources_and_parameters_unchanged':True}),flush=True)
