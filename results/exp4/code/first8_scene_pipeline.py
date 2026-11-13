"""Optional per-scene Myopic-to-Suffix scheduling, requiring explicit approval.

Native commands, fixed identities, seeds and calibration do not change.  The
current coordinator's two live children keep their stopped parent until their
actual exit statuses have been captured.  TE retains its existing exit tracer.
"""
from datetime import datetime,timezone
from pathlib import Path
import argparse,copy,fcntl,json,os,signal,subprocess,sys,time

SELF=Path(__file__).resolve()
PINS={'first8_evaluate.py':'c6d68b83f784224894677484cf6e88420864fb71e6b1612ddb802ca77a21b3de',
    'first8_evaluate8.py':'3d47c382318af84c80ff010da87c6a2a85fc0511797af9c727083fab156690ff',
    'first8_infrastructure_recovery_v2.py':'9d16337ef903a0a0e46de102c7eb2c02f1a44bceb973289b0406e23ed8f7cc40',
    'audit_first8_completed_prefix.py':'8ff552d47b3f3655b02990675ff6335e8c5b20e31a7822907ea4d2dd5ee89424'}
APPROVAL_SCOPE='per_scene_Myopic20_then_Suffix20_keep_fixed_first8_seed_and_frozen_parameters'

def approved(value,scope_sha,parameter_sha,boot):
    return bool(value.get('approved') is True and value.get('scope')==APPROVAL_SCOPE and
        value.get('scope_sha256')==scope_sha and value.get('parameter_sha256')==parameter_sha and value.get('boot_id')==boot and
        value.get('source')=='explicit_user_chat_reply' and value.get('user_reply_text'))

def scene_ready(job,proof,expected):
    return bool(job.get('status')=='completed' and job.get('returncode')==0 and job.get('identities')==expected and len(expected)==20 and
        proof.get('scene')==job['scene'] and proof.get('completed_native_episodes')==20 and proof.get('audited_stage_paths')==60 and
        proof.get('n_error')==0 and proof.get('invalid_paths_complete') is True and proof.get('source_and_parameters_unchanged') is True)

def suffix_ready_scenes(ordered_scenes,myopic_jobs,suffix_jobs,proofs,identities):
    require_scenes={j['scene'] for j in suffix_jobs};myopic={j['scene']:j for j in myopic_jobs}
    if len(myopic)!=len(myopic_jobs) or len(require_scenes)!=len(suffix_jobs):raise ValueError('Duplicate scene assignments')
    return [scene for scene in ordered_scenes if scene not in require_scenes and scene in myopic and scene in proofs and
        scene_ready(myopic[scene],proofs[scene],[r for r in identities if r['scene_id']==scene])]

def runtime():
    global base,legacy,engine,safety,process,ROOT,SCOPE,psutil,run_path
    import evaluate_sequence as base
    import first8_evaluate as legacy
    import first8_evaluate8 as engine
    import first8_infrastructure_recovery_v2 as safety
    safety.runtime()
    import psutil
    from watch_evaluation import process
    from experiments.exp4_collect import run_path
    ROOT=base.OUT;SCOPE=ROOT/'scopes/first8'
    for name,sha in PINS.items():assert base.digest(ROOT/'code'/name)==sha

def save(path,value):legacy.save(path,value)
def now():return datetime.now(timezone.utc).isoformat()

def binding():
    scope=engine.binding()
    assert base.digest(SELF)==base.load(SCOPE/'meta/controller.json')['source_sha256']
    return scope

def scene_proofs(jobs,scope):
    proofs={}
    for job in jobs:
        if job['status']!='completed' or job.get('returncode')!=0:continue
        path=SCOPE/f'diagnostics/early-completed-scene-audits/myopic/{job["scene"]}/result.json'
        if not path.exists():continue
        value=base.load(path)
        for name,sha in value['artifact_sha256'].items():assert base.digest(path.parent/name)==sha
        summary=base.load(path.parent/'summary.json')
        assert summary['analysis_sources_sha256']==base.load(SCOPE/'summary/full.json')['analysis_sources_sha256']
        expected=[r for r in scope['identities'] if r['scene_id']==job['scene']]
        assert scene_ready(job,value,expected)
        proofs[job['scene']]=value
    return proofs

def readiness(require_approval=False):
    scope=engine.binding();old=base.load(SCOPE/'meta/controller.json')
    assert old['pid']==2311693 and old['start_ticks']=='36789196'
    assert old['source_sha256']==PINS['first8_infrastructure_recovery_v2.py']
    engine.checked_cpu(old)
    assert process(old['pid'])['state'] not in ('T','t') and not (SCOPE/'meta/error.json').exists()
    assert not (SCOPE/'meta/evaluation-suffix_only-runs.json').exists(),'Original coordinator has already advanced; do not overlap a started stage'
    ledger=base.load(SCOPE/'meta/evaluation-myopic-runs.json');assert ledger['status']=='running' and ledger['controller_pid']==old['pid']
    assert len(ledger['jobs'])==8 and {j['scene'] for j in ledger['jobs']}==set(scope['ordered_scenes'])
    assert ledger['source_fingerprint']==scope['source_fingerprint'] and ledger['parameter_sha256']==scope['parameter_sha256']
    for job in ledger['jobs']:
        expected=[r for r in scope['identities'] if r['scene_id']==job['scene']]
        assert job['identities']==expected and len(expected)==20 and job['gpu'] in range(8)
        assert job['status'] in ('running','completed')
        if job['status']=='completed':assert job['returncode']==0 and job.get('exit_status_proof');base.audit_scene(Path(job['run_directory']),job)
    assert any(j['status']=='running' for j in ledger['jobs'])
    legacy.completed('nearest',scope)
    proofs=scene_proofs(ledger['jobs'],scope)
    scenes=suffix_ready_scenes(scope['ordered_scenes'],ledger['jobs'],[],proofs,scope['identities'])
    inventory=engine.gpu_inventory();busy={j['gpu'] for j in ledger['jobs'] if j['status']=='running'}
    available=[g for g in engine.eligible(inventory) if g not in busy]
    permission=SCOPE/'meta/scene-pipeline-order-approval.json'
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    consent=(permission.exists() and not permission.is_symlink() and permission.stat().st_uid==1001 and
        approved(base.load(permission),base.digest(SCOPE/'meta/scope.json'),scope['parameter_sha256'],boot))
    if require_approval:assert consent,'Explicit user approval to override global stage order is still required'
    return scope,old,ledger,proofs,scenes,inventory,available,bool(consent)

def check():
    scope,old,ledger,proofs,scenes,inventory,available,consent=readiness()
    value={'at_utc':now(),'proposal_only':True,'approval_present':consent,'Native_workers_started':False,'signals_sent':False,
        'current_controller':{'pid':old['pid'],'start_ticks':old['start_ticks'],'source_sha256':old['source_sha256']},
        'source_sha256':base.digest(SELF),'scope_sha256':base.digest(SCOPE/'meta/scope.json'),'parameter_sha256':scope['parameter_sha256'],
        'ready_completed_scenes':scenes,'verified_ready_episodes':20*len(scenes),'verified_ready_GT_paths':60*len(scenes),
        'available_idle_GPUs':available,'available_memory_gib':psutil.virtual_memory().available/1024**3,
        'actual_change':'Only the scheduling order: each fixed scene starts Suffix after its entire Myopic20 and60 GT paths pass, while other Myopic scenes continue.',
        'current_design_order':'Nearest160 -> Myopic160 -> Suffix160','fixed_scope':'Same original first8 scenes and20 identities each, seed1; no algorithm/configuration/calibration change',
        'design_reference':{'file':'exp4_blp_2x2.md','sha256':'382d3c679441d237d2648989a514171064075acca33225a1b2c0499b660be1ee','order_lines':[179,213]},
        'required_decision':'Approve this per-scene scheduling exception, or keep the original global order'}
    save(SCOPE/'meta/scene-pipeline-proposal.json',value);print(json.dumps(value))

def adopt():
    with (SCOPE/'meta/handoff.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        scope,old,ledger,proofs,scenes,inventory,available,consent=readiness(require_approval=True)
        claim=SCOPE/'meta/scene-pipeline-handoff-claim.json'
        with claim.open('x') as stream:json.dump({'at_utc':now(),'source_sha256':base.digest(SELF)},stream)
        directory=SCOPE/'diagnostics/scene-pipeline-handoff'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');directory.mkdir(parents=True)
        originals={p:p.read_bytes() for p in (SCOPE/'meta/controller.json',SCOPE/'meta/progress.json',SCOPE/'meta/evaluation-myopic-runs.json')}
        child=None;paused=False;activated=False
        try:
            engine.checked_cpu(old);os.kill(old['pid'],signal.SIGSTOP);paused=True
            for _ in range(30):
                if process(old['pid'])['state'] in ('T','t'):break
                time.sleep(.1)
            engine.checked_cpu(old,stopped=True)
            ledger=base.load(SCOPE/'meta/evaluation-myopic-runs.json');assert ledger['controller_pid']==old['pid']
            assert not (SCOPE/'meta/evaluation-suffix_only-runs.json').exists()
            assert not (SCOPE/'meta/error.json').exists()
            parents={}
            for job in ledger['jobs']:
                if job['status']=='completed':assert job['returncode']==0;continue
                p=psutil.Process(job['pid']);state=process(job['pid'])
                assert p.uids().real==1001 and state['start_ticks']==job['pid_start_ticks']
                if state['live']:assert p.cmdline()==job['command']
                if p.ppid()==old['pid']:parents[str(job['pid'])]={'pid':old['pid'],'start_ticks':old['start_ticks']}
                else:assert job['pid']==2236375 and job['pid_start_ticks']=='35702462'
            actual={p.pid for p in psutil.Process(old['pid']).children() if p.status()!=psutil.STATUS_ZOMBIE}
            assert actual<={int(p) for p in parents}
            for path,data in originals.items():(directory/('before-'+path.name)).write_bytes(data)
            save(directory/'myopic-before.json',ledger)
            infrastructure_plan=base.load(ROOT/old['handoff_plan'])
            original_parent=infrastructure_plan['old_controller']
            approval=SCOPE/'meta/scene-pipeline-order-approval.json'
            plan={'at_utc':now(),'boot_id':old['boot_id'],'source_sha256':base.digest(SELF),
                'scope_sha256':base.digest(SCOPE/'meta/scope.json'),'source_fingerprint':scope['source_fingerprint'],'parameter_sha256':scope['parameter_sha256'],
                'old_controller':old,'parents':parents,'myopic_before_sha256':base.digest(directory/'myopic-before.json'),
                'historical_original_parent':original_parent,'historical_failed_6s_exit':next(a['actual_exit'] for a in infrastructure_plan['failed_attempts'] if a['scene']=='6s7QHgap2fW'),
                'approval_sha256':base.digest(approval),'approval':base.load(approval),
                'tracer':base.load(SCOPE/'meta/orphan-exit-tracer.json'),'living_Native_restarted':False,
                'initial_inventory':engine.public_inventory(inventory),'initial_available_GPUs':available}
            save(directory/'plan.json',plan)
            argv=[sys.executable,'-u',str(SELF),'--run-plan',str(directory/'plan.json')]
            log=directory/'controller.log'
            with log.open('xb') as output:
                child=subprocess.Popen(argv,cwd=base.REPO,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdin=subprocess.DEVNULL,
                    stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
            ready=engine.wait_json(directory/'ready.json',child,seconds=100)
            record={**old,'pid':child.pid,'start_ticks':ready['start_ticks'],'command':argv,'source_sha256':base.digest(SELF),
                'log':str(log),'started_at_utc':now(),'handoff_plan':str((directory/'plan.json').relative_to(ROOT)),
                'execution_order':'per_scene_Myopic20_then_Suffix20','predecessor_controller':{'pid':old['pid'],'start_ticks':old['start_ticks'],'status':'stopped parent retained for actual child exits'},
                'approval_sha256':base.digest(approval),'living_native_restarted':False}
            save(SCOPE/'meta/controller.json',record)
            save(directory/'activate.json',{'pid':child.pid,'start_ticks':ready['start_ticks'],'plan_sha256':base.digest(directory/'plan.json')});activated=True
            active=engine.wait_json(directory/'activated.json',child,seconds=100)
            save(SCOPE/'meta/scene-pipeline-handoff.json',{'at_utc':now(),'controller':record,'plan':str((directory/'plan.json').relative_to(ROOT)),
                'plan_sha256':base.digest(directory/'plan.json'),'retained_old_parent':old['pid'],'living_Native_restarted':False,
                'foreign_tasks_signaled':False,'scientific_sources_or_parameters_changed':False})
            print(json.dumps({'activated':True,'pid':child.pid,'start_ticks':ready['start_ticks'],'activation':active,'old_CPU_retained':old['pid'],'living_Native_restarted':False}))
        except BaseException:
            if not activated:
                if child and child.poll() is None:child.terminate();child.wait(timeout=10)
                for path,data in originals.items():path.write_bytes(data)
                if paused:engine.checked_cpu(old,stopped=True);os.kill(old['pid'],signal.SIGCONT)
            raise

def run(plan_path):
    from available_gpu_evaluate_sequence32 import adopted_returncode,assert_no_live_children
    plan=base.load(plan_path);directory=plan_path.parent
    assert base.digest(SELF)==plan['source_sha256'] and base.digest(SCOPE/'meta/scope.json')==plan['scope_sha256']
    assert base.digest(directory/'myopic-before.json')==plan['myopic_before_sha256']
    assert base.digest(SCOPE/'meta/scene-pipeline-order-approval.json')==plan['approval_sha256']
    assert approved(plan['approval'],plan['scope_sha256'],plan['parameter_sha256'],plan['boot_id'])
    assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==plan['boot_id']
    with (SCOPE/'meta/scene-pipeline-controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=process(os.getpid());save(directory/'ready.json',{'pid':os.getpid(),'start_ticks':state['start_ticks']})
        limit=time.monotonic()+120
        while not (directory/'activate.json').exists():assert time.monotonic()<limit;time.sleep(.1)
        activation=base.load(directory/'activate.json');assert activation['pid']==os.getpid() and activation['plan_sha256']==base.digest(plan_path)
        scope=binding();legacy.binding=binding
        jobs={'myopic':copy.deepcopy(base.load(directory/'myopic-before.json')['jobs']),'suffix_only':[]};owned={};handles={};finalized=set();recent=[];last_health=0
        def record():
            for stage in jobs:
                if stage in finalized:continue
                save(SCOPE/f'meta/evaluation-{stage}-runs.json',{'scope':'first8','stage':stage,'status':'running','at_utc':now(),'jobs':jobs[stage],
                    'controller_pid':os.getpid(),'n_scenes':8,'episodes_per_variant':160,'allowed_gpus':list(range(8)),
                    'source_fingerprint':scope['source_fingerprint'],'parameter_sha256':scope['parameter_sha256'],'execution_order':'per_scene_Myopic20_then_Suffix20'})
            save(SCOPE/'meta/progress.json',{'scope':'first8','status':'running','stage':'pipelined_myopic_suffix','at_utc':now(),
                'execution_order':'per_scene_Myopic20_then_Suffix20','controller_pid':os.getpid(),'allowed_gpus':list(range(8)),
                'completed_scenes':{s:sum(j['status']=='completed' for j in rows) for s,rows in jobs.items()},'source_fingerprint':scope['source_fingerprint'],'parameter_sha256':scope['parameter_sha256']})
        def hold(error):
            save(SCOPE/'meta/error.json',{'at_utc':now(),'error':str(error),'controller_held_to_preserve_actual_child_exits':True,
                'Native_not_restarted':True,'no_outcome_selected_retry':True})
            while True:os.kill(os.getpid(),signal.SIGSTOP);time.sleep(.1)
        record();save(directory/'activated.json',{'pid':os.getpid(),'at_utc':now(),'retained_native':sum(j['status']=='running' for j in jobs['myopic']),'living_Native_restarted':False})
        try:
            historical=plan['historical_original_parent']
            if process(historical['pid']).get('live'):
                engine.checked_cpu(historical,stopped=True)
                old_exits={}
                for pid,birth in ((2197428,'35305272'),(2197432,'35305284'),(2197440,'35305299')):
                    code,proof=adopted_returncode(pid,birth);assert code is not None
                    old_exits[str(pid)]={'returncode':code,'proof':proof}
                assert [old_exits[str(p)]['returncode'] for p in (2197428,2197432,2197440)]==[0,0,143]
                assert_no_live_children(historical['pid']);os.kill(historical['pid'],signal.SIGKILL)
                save(directory/'historical-parent-retired.json',{'at_utc':now(),'pid':historical['pid'],'start_ticks':historical['start_ticks'],
                    'actual_original_launcher_exits':old_exits,'failed_6s_whole_attempt_excluded':True,'Native_processes_signaled':False})
            while len(finalized)<2:
                binding()
                disk=os.statvfs(SCOPE);assert disk.f_bavail*disk.f_frsize>=30*1024**3
                for stage,rows in jobs.items():
                    for job in rows:
                        if job['status']!='running':continue
                        run_dir=run_path(Path(job['log']))
                        if run_dir:job['run_directory']=str(run_dir)
                        if run_dir:
                            log=run_dir/'logs/mapping.log'
                            if log.exists():
                                with log.open('rb') as stream:stream.seek(max(0,log.stat().st_size-65536));tail=stream.read().decode(errors='replace')
                                if safety.mapping_is_unusable(tail):
                                    actor=psutil.Process(job['pid']);target=safety.cleanup_target(job,process(job['pid']),actor.uids().real,actor.cmdline())
                                    save(directory/'mapper-fatal-guard.json',{'at_utc':now(),'job':job,'mapping_tail':tail[-12000:],'matching_own_launcher':target,'other_PIDs_signaled':False})
                                    if target:os.kill(target,signal.SIGTERM)
                                    hold('Fatal mapper CUDA OOM in '+job['scene'])
                            if any(p.stat().st_size for p in run_dir.rglob('blp_errors.jsonl')):hold('Native BLP recording error: '+job['scene'])
                        if job['pid'] in owned:
                            code=owned[job['pid']].poll();proof={'source':'real subprocess.Popen.poll','at_utc':now(),'pid':job['pid'],'start_ticks':job['pid_start_ticks']}
                        elif str(job['pid']) in plan['parents']:
                            engine.checked_cpu(plan['old_controller'],stopped=True);code,proof=adopted_returncode(job['pid'],job['pid_start_ticks'])
                        else:
                            assert job['pid']==2236375 and job['pid_start_ticks']=='35702462'
                            code,proof=safety.checked_receipt(job['pid'],job['pid_start_ticks'],plan['tracer'],directory)
                            if code is None:assert process(job['pid']).get('live') and process(job['pid'])['start_ticks']==job['pid_start_ticks']
                        if code is None:continue
                        if job['pid'] in handles:handles.pop(job['pid']).close()
                        if code!=0 or not run_dir:hold('Native scene exited unsuccessfully: '+job['scene']+', code='+str(code))
                        job.update(status='completed',returncode=0,audit=base.audit_scene(run_dir,job),exit_status_proof=proof);record()
                adopted=set(map(int,plan['parents']))
                if not (directory/'predecessor-retired.json').exists() and all(j['status']=='completed' for j in jobs['myopic'] if j['pid'] in adopted):
                    engine.checked_cpu(plan['old_controller'],stopped=True);assert_no_live_children(plan['old_controller']['pid'])
                    os.kill(plan['old_controller']['pid'],signal.SIGKILL)
                    save(directory/'predecessor-retired.json',{'at_utc':now(),'pid':plan['old_controller']['pid'],'start_ticks':plan['old_controller']['start_ticks'],
                        'reason':'All its retained Native child exits captured0 and no living children; TE has independent tracer','Native_processes_signaled':False})
                proofs=scene_proofs(jobs['myopic'],scope)
                if any(j['status']=='completed' and j['scene'] not in proofs for j in jobs['myopic']):
                    record()
                    output=directory/('scene-GT-audit-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.log')
                    with output.open('x') as stream:
                        result=subprocess.run([sys.executable,str(ROOT/'code/audit_first8_completed_prefix.py'),'myopic'],cwd=base.REPO,
                            env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=stream,stderr=subprocess.STDOUT)
                    if result.returncode:hold('Completed-scene CPU GT audit requires review; no result-selected retry')
                    proofs=scene_proofs(jobs['myopic'],scope)
                ready=suffix_ready_scenes(scope['ordered_scenes'],jobs['myopic'],jobs['suffix_only'],proofs,scope['identities'])
                inventory=engine.gpu_inventory();busy={j['gpu'] for rows in jobs.values() for j in rows if j['status']=='running'}
                for gpu in engine.eligible(inventory):
                    if gpu in busy or not ready:continue
                    recent[:]=[t for t in recent if time.monotonic()-t<60]
                    if not safety.startup_host_ready(psutil.virtual_memory().available/1024**3,len(recent)):break
                    if gpu not in engine.eligible(engine.gpu_inventory()):continue
                    scene=ready.pop(0);identities=[r for r in scope['identities'] if r['scene_id']==scene]
                    job,child,handle=legacy.launch_scene('suffix_only',scene,identities,gpu)
                    jobs['suffix_only'].append(job);owned[child.pid]=child;handles[child.pid]=handle;recent.append(time.monotonic());busy.add(gpu);record()
                for stage in ('myopic','suffix_only'):
                    if stage not in finalized and len(jobs[stage])==8 and all(j['status']=='completed' for j in jobs[stage]):
                        stage_result=legacy.finalize(stage,jobs[stage],scope)
                        stage_result.update(execution_order='per_scene_Myopic20_then_Suffix20',order_override_approval_sha256=plan['approval_sha256'])
                        save(SCOPE/f'meta/evaluation-{stage}-runs.json',stage_result);finalized.add(stage)
                record()
                if time.monotonic()-last_health>=600:
                    save(directory/('health-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json'),{'at_utc':now(),
                        'available_memory_gib':psutil.virtual_memory().available/1024**3,'gpu_inventory':engine.public_inventory(inventory),
                        'pressure':{k:Path('/proc/pressure',k).read_text() for k in ('cpu','memory','io')},'Native_restarted':False})
                    last_health=time.monotonic()
                if len(finalized)<2:time.sleep(10)
            legacy.comparison()
            save(SCOPE/'meta/progress.json',{'scope':'first8','status':'completed_pending_delivery','at_utc':now(),'execution_order':'per_scene_Myopic20_then_Suffix20',
                'n_episodes_by_variant':{s:160 for s in legacy.STAGES},'allowed_gpus':list(range(8)),'remaining':'Final32 archives,6paired CSVs,decision audit,actual figure QA,CP5 and GitHub roundtrip'})
        except BaseException as error:hold(type(error).__name__+': '+str(error))

if __name__=='__main__':
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--check',action='store_true');group.add_argument('--adopt',action='store_true');group.add_argument('--run-plan',type=Path)
    args=parser.parse_args();runtime()
    if args.check:check()
    elif args.adopt:adopt()
    else:run(args.run_plan)
