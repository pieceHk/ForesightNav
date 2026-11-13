"""CPU-only delivery preparation; never starts or changes a Native evaluation."""
from datetime import datetime,timezone
from pathlib import Path
import fcntl,json,os,subprocess,sys,time
from evaluate_sequence import OUT,load,digest,source_fingerprint

ROOT=OUT/'scopes/first8'
STATUS=ROOT/'meta/final-report-watch-status.json'

def save(value):
    value={'at_utc':datetime.now(timezone.utc).isoformat(),**value}
    temporary=STATUS.with_name(STATUS.name+f'.tmp-{os.getpid()}')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(STATUS)

with (ROOT/'meta/final-report-watch.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    began=time.monotonic();scope=load(ROOT/'meta/scope.json');previous_pairs=-1
    try:
        while time.monotonic()-began<43200:
            assert source_fingerprint()==scope['source_fingerprint']
            assert digest(OUT/'meta/calibration.json')==scope['parameter_sha256']
            existing=ROOT/'meta/CP5-report-prepared.json'
            if existing.exists():
                record=load(existing);assert digest(ROOT/'reports/CP5.md')==record['report_sha256']
                save({'status':'report_prepared_pending_actual_visual_QA','native_states_modified':False});break
            pair_count=0
            for stage in ('full','nearest','myopic','suffix_only'):
                path=ROOT/f'meta/evaluation-{stage}-runs.json'
                if not path.exists():continue
                for job in load(path)['jobs']:
                    if not job.get('run_directory'):continue
                    run=Path(job['run_directory'])
                    if (run/'mapping_environment.json').exists() and list(run.glob('navigation_environment*.json')):
                        pair_count+=1
            if pair_count!=previous_pairs:
                result=subprocess.run([sys.executable,str(OUT/'code/record_first8_environments.py')],cwd=OUT.parents[1],
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),text=True,capture_output=True)
                assert result.returncode==0,result.stderr[-1500:]
                previous_pairs=pair_count
            progress=load(ROOT/'meta/progress.json')
            archives=ROOT/'meta/raw-archives/manifest.json'
            archived=load(archives) if archives.exists() else {}
            if progress.get('status')=='completed_pending_delivery' and archived.get('complete') and pair_count==32:
                save({'status':'preparing_report','environment_pairs':pair_count,'native_states_modified':False})
                with (ROOT/'logs/final-report-prepare.log').open('x') as output:
                    result=subprocess.run([sys.executable,str(OUT/'code/report_first8_final.py')],cwd=OUT.parents[1],
                        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',MPLCONFIGDIR='/tmp/exp4-first8-matplotlib'),
                        stdout=output,stderr=subprocess.STDOUT)
                assert result.returncode==0,'Inspect final-report-prepare.log; no Native retry is required'
                record=load(existing);assert digest(ROOT/'reports/CP5.md')==record['report_sha256']
                save({'status':'report_prepared_pending_actual_visual_QA','report_sha256':record['report_sha256'],
                    'figure_sha256':record['figure_sha256'],'native_states_modified':False});break
            save({'status':'waiting_for_complete_evidence','environment_pairs':pair_count,
                'archived_runs':archived.get('archived_runs',0),'evaluation_status':progress.get('status'),
                'native_states_modified':False})
            time.sleep(20)
        else:save({'status':'time_limit_reached','native_states_modified':False})
    except BaseException as error:
        save({'status':'error','error':f'{type(error).__name__}: {error}','native_states_modified':False})
        raise
