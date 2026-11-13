"""Recover a CUDA-OOM scene without restarting other Native launchers.

This is a CPU coordination adapter over the frozen eight-GPU dispatcher.  All
twenty identities of the one failed scene are retried; its thirteen old exports
are retained and excluded together.  Scientific runtime sources remain frozen.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, copy, fcntl, gzip, hashlib, json, os, subprocess, sys, time

SELF=Path(__file__).resolve()
ENGINE_SHA='3d47c382318af84c80ff010da87c6a2a85fc0511797af9c727083fab156690ff'
PARAMETER='e1d75f3c9503c7439467748fc97be6ae2ad450c429f40dc7235b1564ff7852ae'
FAILED_SCENE='DYehNKdT76V'
FAILED_PID=2236371
TRACED={2236372:'35702460',2236373:'35702460',2236375:'35702462'}

def validated_receipt(data,pid,birth):
    if data.get('pid')!=pid or str(data.get('start_ticks'))!=str(birth) or data.get('uid')!=1001:
        raise ValueError('Exit receipt does not match the original launcher identity')
    status=data.get('actual_wait_status')
    if not isinstance(status,int) or data.get('exit_event_status')!=status:
        raise ValueError('Actual wait status and ptrace exit event must agree')
    code=os.waitstatus_to_exitcode(status)
    if data.get('returncode')!=code or data.get('source')!='real os.waitpid of traced launcher after PTRACE_EVENT_EXIT':
        raise ValueError('Exit receipt is not a real matching OS wait')
    return code

def validate_retry_assignment(jobs,scope):
    expected={s:[r for r in scope['identities'] if r['scene_id']==s] for s in scope['ordered_scenes']}
    seen=set()
    for job in jobs:
        scene=job['scene']
        if scene in seen or scene not in expected or job['identities']!=expected[scene]:
            raise ValueError('Recovery cannot duplicate scenes or change frozen identities')
        seen.add(scene)
    if set(expected)-seen!={FAILED_SCENE} or any(len(v)!=20 for v in expected.values()):
        raise ValueError('Exactly the whole twenty-episode failed scene must be retried')

def runtime():
    global base, legacy, engine, process, ROOT, SCOPE
    import evaluate_sequence as base
    import first8_evaluate as legacy
    import first8_evaluate8 as engine
    from watch_evaluation import process
    ROOT=base.OUT;SCOPE=ROOT/'scopes/first8'
    assert base.digest(ROOT/'code/first8_evaluate8.py')==ENGINE_SHA
    assert base.digest(ROOT/'meta/calibration.json')==PARAMETER

def save(path,value):
    legacy.save(path,value)

def checked_receipt(pid,birth,tracer,directory):
    path=Path(tracer['directory'])/('pid-'+str(pid)+'-exit.json')
    if not path.exists():return None,None
    assert path.stat().st_uid==0 and not path.stat().st_mode & 0o022
    assert path.parent.stat().st_uid==0 and not path.parent.stat().st_mode & 0o022
    data=base.load(path);code=validated_receipt(data,pid,birth)
    retained=directory/'actual-exits'/path.name;retained.parent.mkdir(exist_ok=True)
    retained.write_bytes(path.read_bytes())
    return code,{'receipt':str(retained.relative_to(ROOT)),'sha256':base.digest(retained),**data}

def prepare():
    from archive_evaluation_logs import archive_run
    scope=engine.binding();current=base.load(SCOPE/'meta/controller.json')
    assert current['pid']==2236364 and current['start_ticks']=='35702398'
    assert not process(current['pid']).get('live')
    assert current['source_sha256']==ENGINE_SHA
    assert current['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    engine.checked_cpu(base.load(ROOT/current['handoff_plan'])['old_controller'],stopped=True)
    ledger=base.load(SCOPE/'meta/evaluation-myopic-runs.json')
    assert ledger['controller_pid']==current['pid'] and ledger['source_fingerprint']==scope['source_fingerprint']
    failed=next(j for j in ledger['jobs'] if j['scene']==FAILED_SCENE)
    assert failed['pid']==FAILED_PID and not process(FAILED_PID).get('live')
    run=Path(failed['run_directory']);status=base.load(run/'status.json')
    assert status['status']=='failed' and status['timestamp']=='2026-11-13T07:52:50.654451+00:00'
    error=base.load(SCOPE/'meta/error.json')
    assert error['error']=='GateError: Native scene failed: DYehNKdT76V, returncode=1'
    assert 'CUDA out of memory' in (run/'logs/mapping.log').read_text()
    assert not (SCOPE/'meta/infrastructure-recovery-20261113.json').exists(),'Inspect existing recovery rather than duplicate it'
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    directory=SCOPE/'diagnostics/infrastructure-recovery'/stamp;directory.mkdir(parents=True)
    snapshot=directory/'before';snapshot.mkdir()
    for path in (SCOPE/'meta/controller.json',SCOPE/'meta/progress.json',SCOPE/'meta/error.json',SCOPE/'meta/evaluation-myopic-runs.json'):
        (snapshot/path.name).write_bytes(path.read_bytes())
    tracer=base.load(SCOPE/'meta/orphan-exit-tracer.json')
    recovered=[]
    for original in ledger['jobs']:
        if original['scene']==FAILED_SCENE:continue
        job=copy.deepcopy(original)
        if job['status']!='completed':
            if job['pid'] in TRACED:
                code,proof=checked_receipt(job['pid'],job['pid_start_ticks'],tracer,directory)
                if code is None:assert process(job['pid']).get('live') and process(job['pid'])['start_ticks']==job['pid_start_ticks']
            else:
                code,proof=engine.adopted_returncode(job['pid'],job['pid_start_ticks'])
                assert Path('/proc',str(job['pid'])).stat().st_uid==1001
            if code is not None:
                assert code==0
                job.update(status='completed',returncode=0,audit=base.audit_scene(Path(job['run_directory']),job),exit_status_proof=proof)
        else:
            assert job['returncode']==0 and job.get('exit_status_proof')
            base.audit_scene(Path(job['run_directory']),job)
        recovered.append(job)
    validate_retry_assignment(recovered,scope)
    # Preserve all failed-run Native evidence before changing any canonical job.
    archived=archive_run(run,directory/'failed-DY-entire-attempt.tar.gz')
    logs=[]
    for path in sorted((run/'logs').glob('*.log')):
        dest=directory/(path.name+'.gz')
        with path.open('rb') as source,gzip.open(dest,'wb',compresslevel=3) as output:
            for chunk in iter(lambda:source.read(1024*1024),b''):output.write(chunk)
        with gzip.open(dest,'rb') as source:assert hashlib.sha256(source.read()).hexdigest()==base.digest(path)
        logs.append({'source':str(path),'source_sha256':base.digest(path),'archive':str(dest.relative_to(ROOT)),'archive_sha256':base.digest(dest)})
    original_plan=base.load(ROOT/current['handoff_plan'])
    new_ledger={**ledger,'jobs':recovered}
    save(directory/'ledger-before.json',new_ledger)
    plan={**original_plan,'created_at_utc':legacy.now(),'source_sha256':base.digest(SELF),
        'ledger_before_sha256':base.digest(directory/'ledger-before.json'),
        'failed_controller':current,'tracer':tracer,'failed_scene':FAILED_SCENE,'failed_job':failed,
        'failed_scene_native_archive':archived,'failed_worker_logs':logs,
        'recovery_scope':'Retain seven unaffected scene jobs; rerun all twenty fixed identities of infrastructure-failed DY scene; exclude all thirteen old exports together.',
        'outcome_selection_used':False,'full_scene_retry_count':20,'failed_prefix_exports_excluded':13,
        'scientific_parameters_or_sources_changed':False,'living_Native_restarted':False,
        'snapshot_sha256':{p.name:base.digest(p) for p in snapshot.iterdir()}}
    save(directory/'plan.json',plan)
    save(SCOPE/'meta/infrastructure-recovery-20261113.json',{'at_utc':legacy.now(),'plan':str((directory/'plan.json').relative_to(ROOT)),
        'plan_sha256':base.digest(directory/'plan.json'),'state':'prepared_not_activated','retained_jobs':7,'full_scene_retry_count':20})
    print(json.dumps({'prepared':True,'plan':str(directory/'plan.json'),'retained_jobs':[(j['scene'],j['status']) for j in recovered],
        'failed_archive_bytes':archived['archive_bytes'],'failed_archive_roundtrip':True,'Native_started':False}))

def activate(plan_path):
    plan=base.load(plan_path);directory=plan_path.parent
    assert base.digest(SELF)==plan['source_sha256']
    current=base.load(SCOPE/'meta/controller.json');assert current==plan['failed_controller']
    assert not process(current['pid']).get('live')
    for name,sha in plan['snapshot_sha256'].items():assert base.digest(SCOPE/'meta'/name)==sha
    claim=directory/'activation-claim.json'
    with claim.open('x') as stream:json.dump({'at_utc':legacy.now(),'plan_sha256':base.digest(plan_path)},stream)
    argv=[sys.executable,'-u',str(SELF),'--run-plan',str(plan_path)]
    log=directory/'controller.log'
    with log.open('xb') as output:
        child=subprocess.Popen(argv,cwd=base.REPO,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),
            stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
    ready=engine.wait_json(directory/'ready.json',child,seconds=100)
    assert ready['pid']==child.pid and process(child.pid)['start_ticks']==ready['start_ticks']
    record={**current,'pid':child.pid,'start_ticks':ready['start_ticks'],'command':argv,
        'source_sha256':base.digest(SELF),'log':str(log),'started_at_utc':legacy.now(),
        'handoff_plan':str(plan_path.relative_to(ROOT)),
        'predecessor_controller':{'pid':current['pid'],'start_ticks':current['start_ticks'],'status':'failed after confirmed Native CUDA OOM; all original files retained'},
        'living_native_restarted':False,'infrastructure_full_scene_retry':FAILED_SCENE}
    save(SCOPE/'meta/controller.json',record)
    assert base.digest(SCOPE/'meta/error.json')==plan['snapshot_sha256']['error.json']
    (SCOPE/'meta/error.json').rename(directory/'retired-error.json')
    save(directory/'activate.json',{'pid':child.pid,'start_ticks':ready['start_ticks'],'plan_sha256':base.digest(plan_path),'at_utc':legacy.now()})
    active=engine.wait_json(directory/'activated.json',child,seconds=100)
    save(SCOPE/'meta/infrastructure-recovery-20261113.json',{'at_utc':legacy.now(),'plan':str(plan_path.relative_to(ROOT)),
        'plan_sha256':base.digest(plan_path),'state':'activated','controller_pid':child.pid,'controller_start_ticks':ready['start_ticks'],
        'living_native_restarted':False,'other_user_signals_sent':False,'scientific_sources_or_parameters_changed':False})
    print(json.dumps({'activated':True,'controller':record,'activation':active,'no_resources_threshold_lowered':True}))

def run(plan_path):
    plan=base.load(plan_path);directory=plan_path.parent
    original_exit=engine.adopted_returncode
    original_retire=engine.retire_parent
    original_launch=legacy.launch_scene
    def exit_status(pid,birth):
        if pid in TRACED:
            code,proof=checked_receipt(pid,birth,plan['tracer'],directory)
            if code is not None:return code,proof
            state=process(pid)
            if not state.get('live') or state.get('start_ticks')!=str(birth):
                raise base.GateError('Traced launcher vanished without an actual exit receipt')
            return None,None
        return original_exit(pid,birth)
    def retire_after_all_retained(p,jobs,d):
        retained={j['scene'] for j in base.load(directory/'ledger-before.json')['jobs']}
        if any(j['scene'] in retained and j['status']!='completed' for j in jobs):return
        original_retire(p,jobs,d)
    def launch(stage,scene,identities,gpu):
        if stage!='myopic' or scene!=FAILED_SCENE:return original_launch(stage,scene,identities,gpu)
        assert identities==plan['failed_job']['identities'] and len(identities)==20
        argv=base.command(stage,scene,identities,gpu,'a4-first8-infrastructure-20261113')
        log=directory/'DY-full20-retry.controller.log';handle=log.open('x')
        env=dict(os.environ);env.pop('CUDA_VISIBLE_DEVICES',None)
        child=subprocess.Popen(argv,cwd=base.REPO,env=env,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
        return {'scene':scene,'identities':identities,'gpu':gpu,'pid':child.pid,'pid_start_ticks':process(child.pid)['start_ticks'],
            'command':argv,'log':str(log),'status':'running','infrastructure_retry_plan':str(plan_path.relative_to(ROOT)),
            'excluded_failed_run':plan['failed_job']['run_directory']},child,handle
    engine.SELF=SELF
    engine.adopted_returncode=exit_status
    engine.retire_parent=retire_after_all_retained
    legacy.launch_scene=launch
    engine.control(plan_path)

if __name__=='__main__':
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare',action='store_true');group.add_argument('--activate',type=Path);group.add_argument('--run-plan',type=Path)
    args=parser.parse_args();runtime()
    if args.run_plan:run(args.run_plan)
    else:
        with (SCOPE/'meta/handoff.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if args.prepare:prepare()
            else:activate(args.activate)
