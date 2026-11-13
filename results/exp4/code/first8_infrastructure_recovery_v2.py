"""Recover a CUDA-OOM scene without restarting other Native launchers.

This CPU adapter preserves all five completed scenes and the healthy TE run.
DY and 6s each repeat their full twenty identities after proven CUDA OOM; all
exports of their failed attempts remain archived and excluded together.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, copy, fcntl, gzip, hashlib, json, math, os, signal, subprocess, sys, time

SELF=Path(__file__).resolve()
ENGINE_SHA='3d47c382318af84c80ff010da87c6a2a85fc0511797af9c727083fab156690ff'
PARAMETER='e1d75f3c9503c7439467748fc97be6ae2ad450c429f40dc7235b1564ff7852ae'
FAILED_SCENE='DYehNKdT76V'
FAILED_PID=2236371
FAILED_SCENES={FAILED_SCENE,'6s7QHgap2fW'}
PREDECESSOR_SHA='a3a6d638c76f8953a09ae705437e4b03bb185ebe21c1d9e31d7b4226098362df'
TRACED={2236372:'35702460',2236373:'35702460',2236375:'35702462'}

def startup_host_ready(available_gib,recent_launches):
    if isinstance(available_gib,bool) or not isinstance(available_gib,(int,float)):
        return False
    if not math.isfinite(available_gib) or isinstance(recent_launches,bool) or not isinstance(recent_launches,int) or recent_launches<0:return False
    # Preserve the existing64GiB floor and reserve16GiB for each new launch
    # while allocations settle.  This prevents an eight-launch admission race.
    return available_gib>=64+16*(recent_launches+1)

def mapping_is_unusable(text):
    value=text.lower()
    return 'mapping rebuild failed; incomplete maps cannot be used' in value and 'cuda out of memory' in value

def cleanup_target(job,state,uid,command):
    if uid!=1001 or not state.get('live') or state.get('start_ticks')!=str(job['pid_start_ticks']) or command!=job['command']:
        return None
    return job['pid']

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
    if set(expected)-seen!=FAILED_SCENES or any(len(v)!=20 for v in expected.values()):
        raise ValueError('Exactly the two whole twenty-episode OOM scenes must be retried')

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
    assert current['pid']==2304501 and current['start_ticks']=='36672358'
    engine.checked_cpu(current,stopped=True)
    assert current['source_sha256']==PREDECESSOR_SHA
    assert current['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    previous=base.load(ROOT/current['handoff_plan'])
    engine.checked_cpu(previous['old_controller'],stopped=True)
    from available_gpu_evaluate_sequence32 import assert_no_live_children
    assert_no_live_children(current['pid'])
    ledger=base.load(SCOPE/'meta/evaluation-myopic-runs.json')
    assert ledger['controller_pid']==current['pid'] and ledger['source_fingerprint']==scope['source_fingerprint']
    cleanup=base.load(SCOPE/'meta/6s-oom-cleanup-20261113.json')
    assert cleanup['cleanup_request_sent'] and cleanup['CPU_controller_stopped_only']
    failed=next(j for j in ledger['jobs'] if j['scene']=='6s7QHgap2fW')
    assert failed==cleanup['job'] and failed['pid']==2197440 and failed['pid_start_ticks']=='35305299'
    code,actual_exit=engine.adopted_returncode(failed['pid'],failed['pid_start_ticks'])
    assert code is not None and code!=0,'Wait for the same cleanup launcher actual exit'
    run=Path(failed['run_directory']);status=base.load(run/'status.json');assert status['status']=='failed'
    assert 'CUDA out of memory' in (run/'logs/mapping.log').read_text()
    assert not (SCOPE/'meta/infrastructure-recovery-v2-20261113.json').exists(),'Inspect existing recovery rather than duplicate it'
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    directory=SCOPE/'diagnostics/infrastructure-recovery'/stamp;directory.mkdir(parents=True)
    snapshot=directory/'before';snapshot.mkdir()
    for path in (SCOPE/'meta/controller.json',SCOPE/'meta/progress.json',SCOPE/'meta/error.json',SCOPE/'meta/evaluation-myopic-runs.json',SCOPE/'meta/6s-oom-cleanup-20261113.json'):
        if path.exists():(snapshot/path.name).write_bytes(path.read_bytes())
    save(directory/'6s-original-actual-exit.json',actual_exit)
    tracer=base.load(SCOPE/'meta/orphan-exit-tracer.json')
    recovered=[]
    for original in ledger['jobs']:
        if original['scene'] in FAILED_SCENES:continue
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
    archived=archive_run(run,directory/'failed-6s-entire-attempt.tar.gz')
    observed=set()
    for path in run.glob('results_6s7QHgap2fW/*/per_episode_metrics.json'):
        observed.update(k for k,v in base.load(path).items() if isinstance(v,dict) and isinstance(v.get('mon_metrics'),dict))
    assert observed and len(observed)<=20
    logs=[]
    for path in sorted((run/'logs').glob('*.log')):
        dest=directory/(path.name+'.gz')
        with path.open('rb') as source,gzip.open(dest,'wb',compresslevel=3) as output:
            for chunk in iter(lambda:source.read(1024*1024),b''):output.write(chunk)
        with gzip.open(dest,'rb') as source:assert hashlib.sha256(source.read()).hexdigest()==base.digest(path)
        logs.append({'source':str(path),'source_sha256':base.digest(path),'archive':str(dest.relative_to(ROOT)),'archive_sha256':base.digest(dest)})
    original_plan=previous
    new_ledger={**ledger,'jobs':recovered}
    save(directory/'ledger-before.json',new_ledger)
    plan={**original_plan,'created_at_utc':legacy.now(),'source_sha256':base.digest(SELF),
        'ledger_before_sha256':base.digest(directory/'ledger-before.json'),
        'failed_controller':current,'tracer':tracer,
        'failed_scene':FAILED_SCENE,'failed_job':previous['failed_job'],
        'failed_scene_native_archive':previous['failed_scene_native_archive'],'failed_worker_logs':previous['failed_worker_logs'],
        'failed_scene_jobs':{FAILED_SCENE:previous['failed_job'],'6s7QHgap2fW':failed},
        'failed_attempts':[
            {'scene':FAILED_SCENE,'job':previous['failed_job'],'native_archive':previous['failed_scene_native_archive'],'worker_logs':previous['failed_worker_logs'],'exported_records_excluded':13},
            {'scene':'6s7QHgap2fW','job':failed,'native_archive':archived,'worker_logs':logs,'exported_records_excluded':len(observed),'actual_exit':actual_exit,'operator_stopped_after_confirmed_OOM':True}],
        'recovery_scope':'Retain five complete scene jobs and healthy TE; retry all twenty fixed identities of each of the two infrastructure-failed scenes; exclude both failed prefixes in full.',
        'outcome_selection_used':False,'full_scene_retry_count':20,'total_fixed_native_retry_episodes':40,'failed_prefix_exports_excluded':13,
        'failed_prefix_exports_excluded_by_scene':{FAILED_SCENE:13,'6s7QHgap2fW':len(observed)},
        'startup_host_floor_gib':64,'startup_allocation_reserve_gib':16,'startup_reservation_seconds':60,
        'scientific_parameters_or_sources_changed':False,'living_Native_restarted':False,
        'snapshot_sha256':{p.name:base.digest(p) for p in snapshot.iterdir()}}
    save(directory/'plan.json',plan)
    save(SCOPE/'meta/infrastructure-recovery-v2-20261113.json',{'at_utc':legacy.now(),'plan':str((directory/'plan.json').relative_to(ROOT)),
        'plan_sha256':base.digest(directory/'plan.json'),'state':'prepared_not_activated','retained_jobs':6,'full_scene_retry_count_each':20,'fixed_retry_total':40})
    print(json.dumps({'prepared':True,'plan':str(directory/'plan.json'),'retained_jobs':[(j['scene'],j['status']) for j in recovered],
        'failed_archive_bytes':archived['archive_bytes'],'failed_archive_roundtrip':True,'Native_started':False}))

def activate(plan_path):
    plan=base.load(plan_path);directory=plan_path.parent
    assert base.digest(SELF)==plan['source_sha256']
    current=base.load(SCOPE/'meta/controller.json');assert current==plan['failed_controller']
    engine.checked_cpu(current,stopped=True)
    for name,sha in plan['snapshot_sha256'].items():assert base.digest(SCOPE/'meta'/name)==sha
    claim=directory/'activation-claim.json'
    with claim.open('x') as stream:json.dump({'at_utc':legacy.now(),'plan_sha256':base.digest(plan_path)},stream)
    # Both adapters use the same exclusive dispatcher lock.  The stopped
    # predecessor has no living Popen children, so retire only that CPU before
    # asking the replacement to acquire the shared lock.  Native TE is orphaned
    # under init and has an independent real-exit tracer.
    from available_gpu_evaluate_sequence32 import assert_no_live_children
    engine.checked_cpu(current,stopped=True);assert_no_live_children(current['pid'])
    retirement={'at_utc':legacy.now(),'pid':current['pid'],'start_ticks':current['start_ticks'],
        'source_sha256':current['source_sha256'],'no_living_children_verified':True,
        'reason':'Release shared dispatcher lock before replacement starts; only stopped CPU has no living children',
        'Native_processes_signaled':False,'retirement_signal_sent':False}
    save(directory/'previous-adapter-retired.json',retirement)
    os.kill(current['pid'],signal.SIGKILL)
    deadline=time.monotonic()+10
    while process(current['pid']).get('live'):
        assert time.monotonic()<deadline,'Inspect the same predecessor; do not launch another replacement'
        time.sleep(.05)
    retirement.update(retirement_signal_sent=True,actual_old_CPU_no_longer_live=True,retired_at_utc=legacy.now())
    save(directory/'previous-adapter-retired.json',retirement)
    argv=[sys.executable,'-u',str(SELF),'--run-plan',str(plan_path)]
    log=directory/'controller.log'
    with log.open('xb') as output:
        child=subprocess.Popen(argv,cwd=base.REPO,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),
            stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
    save(directory/'replacement-launcher.json',{'at_utc':legacy.now(),'pid':child.pid,
        'start_ticks':process(child.pid)['start_ticks'],'command':argv,'source_sha256':base.digest(SELF)})
    ready=engine.wait_json(directory/'ready.json',child,seconds=100)
    assert ready['pid']==child.pid and process(child.pid)['start_ticks']==ready['start_ticks']
    record={**current,'pid':child.pid,'start_ticks':ready['start_ticks'],'command':argv,
        'source_sha256':base.digest(SELF),'log':str(log),'started_at_utc':legacy.now(),
        'handoff_plan':str(plan_path.relative_to(ROOT)),
        'predecessor_controller':{'pid':current['pid'],'start_ticks':current['start_ticks'],'status':'CPU stopped after confirmed mapper CUDA OOM; retired after replacement activation'},
        'living_native_restarted':False,'infrastructure_full_scene_retry':sorted(FAILED_SCENES),
        'startup_host_floor_gib':64,'startup_allocation_reserve_gib':16,'startup_reservation_seconds':60,
        'mapper_fatal_error_guard':'Graceful stop of only the matching own launcher, then stop own CPU to preserve all child wait statuses for review'}
    save(SCOPE/'meta/controller.json',record)
    if 'error.json' in plan['snapshot_sha256']:
        assert base.digest(SCOPE/'meta/error.json')==plan['snapshot_sha256']['error.json']
        (SCOPE/'meta/error.json').rename(directory/'retired-error.json')
    save(directory/'activate.json',{'pid':child.pid,'start_ticks':ready['start_ticks'],'plan_sha256':base.digest(plan_path),'at_utc':legacy.now()})
    active=engine.wait_json(directory/'activated.json',child,seconds=100)
    assert not process(current['pid']).get('live')
    save(SCOPE/'meta/infrastructure-recovery-v2-20261113.json',{'at_utc':legacy.now(),'plan':str(plan_path.relative_to(ROOT)),
        'plan_sha256':base.digest(plan_path),'state':'activated','controller_pid':child.pid,'controller_start_ticks':ready['start_ticks'],
        'living_native_restarted':False,'other_user_signals_sent':False,'scientific_sources_or_parameters_changed':False})
    print(json.dumps({'activated':True,'controller':record,'activation':active,'no_resources_threshold_lowered':True}))

def run(plan_path):
    plan=base.load(plan_path);directory=plan_path.parent
    original_exit=engine.adopted_returncode
    original_retire=engine.retire_parent
    original_launch=legacy.launch_scene;original_eligible=engine.eligible;original_binding=engine.binding;recent=[]
    import psutil
    excluded={j['run_directory'] for j in plan['failed_scene_jobs'].values()}
    def guarded_binding():
        scope=original_binding()
        for stage in ('myopic','suffix_only'):
            path=SCOPE/f'meta/evaluation-{stage}-runs.json'
            if not path.exists():continue
            for job in base.load(path)['jobs']:
                if job['status']!='running' or not job.get('run_directory') or job['run_directory'] in excluded:continue
                log=Path(job['run_directory'])/'logs/mapping.log'
                if not log.exists():continue
                with log.open('rb') as stream:
                    stream.seek(max(0,log.stat().st_size-65536));tail=stream.read().decode(errors='replace')
                if not mapping_is_unusable(tail):continue
                state=process(job['pid']);target=None
                if state.get('live'):
                    actor=psutil.Process(job['pid'])
                    target=cleanup_target(job,state,actor.uids().real,actor.cmdline())
                fault={'at_utc':legacy.now(),'controller_pid':os.getpid(),'controller_source_sha256':base.digest(SELF),
                    'job':job,'mapping_log':str(log),'mapping_tail':tail[-12000:],
                    'reason':'Confirmed fatal mapper CUDA OOM; fresh navigation frames cannot validate an unusable map',
                    'matching_own_launcher_to_cleanup':target,'other_user_signals_sent':False,
                    'own_CPU_will_stop_to_preserve_all_actual_child_exit_statuses':True,'scientific_sources_or_parameters_changed':False}
                save(directory/'mapper-fatal-guard.json',fault)
                save(SCOPE/'meta/error.json',{'at_utc':legacy.now(),'error':'Fatal mapper CUDA OOM in '+job['scene']+'; own CPU held for review; do not resume blindly',
                    'fault_proof':str((directory/'mapper-fatal-guard.json').relative_to(ROOT)),
                    'native_workers_left_running_for_inspection':True,'no_outcome_selected_retry':True})
                if target is not None:
                    os.kill(target,signal.SIGTERM)
                    save(directory/'mapper-fatal-guard-cleanup-sent.json',{'at_utc':legacy.now(),'pid':target,
                        'start_ticks':job['pid_start_ticks'],'signal':'SIGTERM','uid':1001,'other_Native_workers_signaled':False})
                # Keeping this parent stopped retains actual wait statuses for
                # every in-flight Popen child.  Never orphan them on a guard error.
                while True:
                    os.kill(os.getpid(),signal.SIGSTOP)
                    time.sleep(.1)
        return scope
    def guarded_eligible(inventory,allowed=engine.ALLOWED):
        for gpu in original_eligible(inventory,allowed):
            recent[:]=[t for t in recent if time.monotonic()-t<60]
            if not startup_host_ready(psutil.virtual_memory().available/1024**3,len(recent)):continue
            fresh=engine.gpu_inventory()
            if gpu in original_eligible(fresh,allowed):yield gpu
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
        if stage!='myopic' or scene not in FAILED_SCENES:
            result=original_launch(stage,scene,identities,gpu);recent.append(time.monotonic());return result
        assert identities==plan['failed_scene_jobs'][scene]['identities'] and len(identities)==20
        argv=base.command(stage,scene,identities,gpu,'a5-first8-infrastructure-20261113')
        log=directory/(scene+'-full20-retry.controller.log');handle=log.open('x')
        env=dict(os.environ);env.pop('CUDA_VISIBLE_DEVICES',None)
        child=subprocess.Popen(argv,cwd=base.REPO,env=env,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
        recent.append(time.monotonic())
        return {'scene':scene,'identities':identities,'gpu':gpu,'pid':child.pid,'pid_start_ticks':process(child.pid)['start_ticks'],
            'command':argv,'log':str(log),'status':'running','infrastructure_retry_plan':str(plan_path.relative_to(ROOT)),
            'excluded_failed_run':plan['failed_scene_jobs'][scene]['run_directory']},child,handle
    engine.SELF=SELF
    engine.adopted_returncode=exit_status
    engine.retire_parent=retire_after_all_retained
    engine.eligible=guarded_eligible
    engine.binding=guarded_binding
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
