"""Archive each completed scoped Native run while other GPU runs continue."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
from pathlib import Path
import argparse,fcntl,json,os,shutil,time
import archive_evaluation_logs as raw
import evaluate_sequence as base
import first8_evaluate as analysis

ROOT=base.OUT;SCOPE=ROOT/'scopes/first8';SELF=Path(__file__).resolve()

def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f'.tmp-{os.getpid()}')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temporary.replace(path)

def one(stage,job,scope):
    assert job['scene'] in scope['ordered_scenes'] and job['status']=='completed' and job['returncode']==0
    assert job['identities']==[r for r in scope['identities'] if r['scene_id']==job['scene']]
    run=Path(job['run_directory']);base.audit_scene(run,job)
    index=SCOPE/f'meta/raw-archives/{stage}/{job["scene"]}.json'
    if index.exists():
        row=base.load(index);target=ROOT/row['archive']
        assert target.stat().st_size==row['archive_bytes'] and base.digest(target)==row['archive_sha256']
        assert row['launch_sha256']==base.digest(run/'launch.json') and row['status_sha256']==base.digest(run/'status.json')
        return row
    target=SCOPE/f'logs/native/{stage}/{job["scene"]}.tar.gz'
    assert not target.exists(),'Inspect an existing unindexed archive; never overwrite it blindly'
    target.parent.mkdir(parents=True,exist_ok=True)
    reused=False
    if stage=='full' and (ROOT/'meta/evaluation-full-log-archive.json').exists():
        old=base.load(ROOT/'meta/evaluation-full-log-archive.json')
        assert old['source_fingerprint']==scope['source_fingerprint'] and old['parameter_sha256']==scope['parameter_sha256']
        source=next(row for row in old['runs'] if row['scene_id']==job['scene'])
        original=ROOT/source['archive']
        assert original.stat().st_size==source['archive_bytes'] and base.digest(original)==source['archive_sha256']
        assert source['native_run']==str(run)
        shutil.copy2(original,target)
        row={**source,'archive':str(target.relative_to(ROOT))};reused=True
        assert base.digest(target)==source['archive_sha256']
        # Verify the original bytes against the completed live source again.
        for entry in row['files']:
            assert raw.file_digest(run/entry['path'])==entry['sha256']
    else:
        row=raw.archive_run(run,target)
    row.update(scope='first8',stage=stage,scene_id=job['scene'],n_episodes=20,
        launch_sha256=base.digest(run/'launch.json'),status_sha256=base.digest(run/'status.json'),
        source_fingerprint=scope['source_fingerprint'],parameter_sha256=scope['parameter_sha256'],
        archive_helper_sha256=base.digest(ROOT/'code/archive_evaluation_logs.py'),
        reused_original_full_archive=reused,checked_at_utc=datetime.now(timezone.utc).isoformat())
    save(index,row)
    return row

def collect():
    scope=base.load(SCOPE/'meta/scope.json')
    assert base.source_fingerprint()==scope['source_fingerprint']
    assert base.digest(ROOT/'meta/calibration.json')==scope['parameter_sha256']
    eligible=[]
    for stage in analysis.STAGES:
        path=SCOPE/f'meta/evaluation-{stage}-runs.json'
        if not path.exists():continue
        ledger=base.load(path)
        assert ledger['scope']=='first8' and ledger['source_fingerprint']==scope['source_fingerprint']
        assert ledger['parameter_sha256']==scope['parameter_sha256']
        for job in ledger['jobs']:
            if job['status']=='completed' and job.get('returncode')==0:eligible.append((stage,job))
    with ThreadPoolExecutor(max_workers=2) as pool:
        rows=list(pool.map(lambda pair:one(pair[0],pair[1],scope),eligible))
    assert len(rows)==len({(r['stage'],r['scene_id']) for r in rows})
    value={'scope':'first8','at_utc':datetime.now(timezone.utc).isoformat(),'archived_runs':len(rows),'target_runs':32,
        'complete':len(rows)==32,'by_stage':{stage:sum(r['stage']==stage for r in rows) for stage in analysis.STAGES},
        'source_fingerprint':scope['source_fingerprint'],'parameter_sha256':scope['parameter_sha256'],
        'source_sha256':base.digest(SELF),'runs':rows,'native_states_modified':False,'signals_sent':False}
    save(SCOPE/'meta/raw-archives/manifest.json',value)
    return value

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--watch-seconds',type=int,default=0);args=parser.parse_args()
    assert 0<=args.watch_seconds<=86400
    with (SCOPE/'meta/raw-archive.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        began=time.monotonic()
        while True:
            result=collect()
            print(json.dumps({k:result[k] for k in ('at_utc','archived_runs','target_runs','complete','by_stage')}),flush=True)
            if result['complete'] or not args.watch_seconds or time.monotonic()-began>=args.watch_seconds:break
            time.sleep(30)

if __name__=='__main__':main()
