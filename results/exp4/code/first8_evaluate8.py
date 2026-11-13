"""Eight-GPU scheduling handoff; retains original Native launchers and wait statuses."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import copy
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
import psutil

import evaluate_sequence as base
import first8_evaluate as legacy
from available_gpu_evaluate_sequence32 import gpu_inventory, adopted_returncode, assert_no_live_children
from experiments.exp4_collect import run_path
from watch_evaluation import process

ROOT=base.OUT
SCOPE=ROOT/'scopes/first8'
SELF=Path(__file__).resolve()
OLD_SHA='c6d68b83f784224894677484cf6e88420864fb71e6b1612ddb802ca77a21b3de'
PARAMETER='e1d75f3c9503c7439467748fc97be6ae2ad450c429f40dc7235b1564ff7852ae'
ALLOWED=list(range(8))
POLICY={'allowed_gpus':ALLOWED,'reserved_gpus':[],'min_free_mib':32768,'max_existing_compute_mib':11264}
save=legacy.save
now=legacy.now

def eligible(inventory,allowed=ALLOWED):
    output=[]
    for gpu in inventory['gpus']:
        index=int(gpu['index'])
        if index not in allowed:continue
        apps=[a for a in inventory['compute_apps'] if a['gpu_uuid']==gpu['uuid']]
        try:free=int(gpu['memory.free']);memory=[int(a['used_gpu_memory']) for a in apps]
        except (KeyError,ValueError,TypeError):continue
        if free>=POLICY['min_free_mib'] and all(v>=0 for v in memory) and sum(memory)<=POLICY['max_existing_compute_mib']:
            output.append(index)
    return output

def public_inventory(inventory):
    return {'checked_at_utc':inventory['checked_at_utc'],'gpus':inventory['gpus'],
        'compute_apps':[{'gpu_uuid':a['gpu_uuid'],'used_gpu_memory':a['used_gpu_memory']} for a in inventory['compute_apps']],
        'external_process_names_and_PIDs_omitted':True}

def checked_cpu(record,stopped=False):
    state=process(record['pid'])
    if not state.get('live') or state.get('start_ticks')!=record['start_ticks']:
        raise base.GateError('Controller PID/birth no longer matches')
    p=psutil.Process(record['pid'])
    assert p.uids().real==os.getuid() and p.cmdline()==record['command']
    if stopped:assert state['state'] in ('T','t')
    return state

def binding():
    scope=base.load(SCOPE/'meta/scope.json')
    scenes,identities=legacy.select_first8(base.load(ROOT/'meta/3mon-identities.json'))
    assert scope['ordered_scenes']==scenes and scope['identities']==identities
    assert base.source_fingerprint()==scope['source_fingerprint']
    assert base.digest(ROOT/'meta/calibration.json')==scope['parameter_sha256']==PARAMETER
    allocation=base.load(SCOPE/'meta/gpu-allocation.json')
    assert all(allocation[k]==v for k,v in POLICY.items())
    assert allocation['user_authorized_all_eight'] is True
    return scope

def validate_retained(ledger,scope):
    assert ledger['stage']=='myopic' and ledger['status']=='running'
    seen=set();running_gpus=set()
    for job in ledger['jobs']:
        scene=job['scene'];assert scene in scope['ordered_scenes'] and scene not in seen
        seen.add(scene)
        assert job['identities']==[r for r in scope['identities'] if r['scene_id']==scene]
        assert job['status'] in ('running','completed')
        if job['status']=='completed':
            assert job.get('returncode')==0
            base.audit_scene(Path(job['run_directory']),job)
        else:
            assert job['gpu'] in ALLOWED and job['gpu'] not in running_gpus
            running_gpus.add(job['gpu'])
            assert process(job['pid'])['start_ticks']==job['pid_start_ticks']
            assert psutil.Process(job['pid']).uids().real==os.getuid()
            assert psutil.Process(job['pid']).cmdline()==job['command']
    return [s for s in scope['ordered_scenes'] if s not in seen]

def readiness():
    scope=base.load(SCOPE/'meta/scope.json');old=base.load(SCOPE/'meta/controller.json')
    assert Path(old['command'][2])==ROOT/'code/first8_evaluate.py'
    assert old['source_sha256']==base.digest(ROOT/'code/first8_evaluate.py')==OLD_SHA
    assert old['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    checked_cpu(old)
    assert base.source_fingerprint()==scope['source_fingerprint']==old['source_fingerprint']
    assert base.digest(ROOT/'meta/calibration.json')==PARAMETER
    ledger=base.load(SCOPE/'meta/evaluation-myopic-runs.json')
    assert ledger['controller_pid']==old['pid'] and not (SCOPE/'meta/error.json').exists()
    pending=validate_retained(ledger,scope)
    legacy.completed('nearest',scope)
    busy={j['gpu'] for j in ledger['jobs'] if j['status']=='running'}
    inventory=gpu_inventory();available=[g for g in eligible(inventory) if g not in busy]
    assert len(available)>=len(pending),'Wait for released capacity; never displace other tasks'
    assert psutil.virtual_memory().available>=((16*len(pending))+48)*1024**3
    return scope,old,ledger,inventory,available

def wait_json(path,child,seconds=60):
    deadline=time.monotonic()+seconds
    while not path.exists():
        assert child.poll() is None,'Inspect the same failed handoff, never launch a duplicate'
        assert time.monotonic()<deadline,'Inspect this same initializing PID'
        time.sleep(.1)
    return base.load(path)

def adopt():
    with (SCOPE/'meta/handoff.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        scope,old,ledger,inventory,available=readiness()
        claim=SCOPE/'meta/eight-gpu-handoff-claim.json'
        with claim.open('x') as stream:json.dump({'at_utc':now(),'source_sha256':base.digest(SELF)},stream)
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        directory=SCOPE/'diagnostics/eight-gpu-handoff'/stamp;directory.mkdir(parents=True)
        originals={p:p.read_bytes() if p.exists() else None for p in
            (SCOPE/'meta/controller.json',SCOPE/'meta/gpu-allocation.json',ROOT/'meta/evaluation-available-gpu-policy.json')}
        for i,(path,data) in enumerate(originals.items()):
            if data is not None:(directory/f'original-{i}-{path.name}').write_bytes(data)
        paused=False;activated=False;child=None
        try:
            checked_cpu(old);os.kill(old['pid'],signal.SIGSTOP);paused=True
            for _ in range(30):
                if process(old['pid'])['state'] in ('T','t'):break
                time.sleep(.1)
            checked_cpu(old,stopped=True)
            ledger=base.load(SCOPE/'meta/evaluation-myopic-runs.json')
            pending=validate_retained(ledger,scope)
            actual={p.pid for p in psutil.Process(old['pid']).children() if p.status()!=psutil.STATUS_ZOMBIE}
            assert actual<={j['pid'] for j in ledger['jobs'] if j['status']=='running'}
            adopted=[]
            for job in ledger['jobs']:
                if job['status']=='running':
                    assert psutil.Process(job['pid']).ppid()==old['pid']
                    adopted.append({'scene':job['scene'],'pid':job['pid'],'start_ticks':job['pid_start_ticks'],
                        'parent_pid':old['pid'],'parent_start_ticks':old['start_ticks']})
            save(directory/'ledger-before.json',ledger)
            allocation={**POLICY,'user_authorized_all_eight':True,'at_utc':now(),
                'user_instruction':'我准备先暂停Exp2，先8卡全速跑完Exp4，我现在已经释放GPU，请你持续每10分钟进行监督，避免OOM、CPU等待等问题出现，确保11/13日晚上能完成Exp4的实验结果。',
                'other_task_termination_authorized':False,'Exp2_data_preserved':True}
            plan={'created_at_utc':now(),'boot_id':old['boot_id'],'source_sha256':base.digest(SELF),
                'legacy_source_sha256':OLD_SHA,'scope_sha256':base.digest(SCOPE/'meta/scope.json'),
                'ledger_before_sha256':base.digest(directory/'ledger-before.json'),
                'old_controller':old,'adopted_launchers':adopted,'allocation':allocation,
                'source_fingerprint':scope['source_fingerprint'],'parameter_sha256':PARAMETER,
                'native_restarted':False,'native_signaled':False,'initial_inventory':public_inventory(inventory)}
            save(directory/'plan.json',plan)
            log=directory/'controller.log';argv=[sys.executable,'-u',str(SELF),'--run-plan',str(directory/'plan.json')]
            with log.open('xb') as output:
                child=subprocess.Popen(argv,cwd=base.REPO,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),
                    stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
            ready=wait_json(directory/'ready.json',child)
            assert ready['pid']==child.pid and process(child.pid)['start_ticks']==ready['start_ticks']
            record={'scope':'first8','pid':child.pid,'start_ticks':ready['start_ticks'],'command':argv,
                'source_sha256':base.digest(SELF),'source_fingerprint':scope['source_fingerprint'],
                'parameter_sha256':PARAMETER,'log':str(log),'started_at_utc':now(),'boot_id':old['boot_id'],
                'allowed_gpus':ALLOWED,'handoff_plan':str((directory/'plan.json').relative_to(ROOT)),
                'predecessor_controller':{'pid':old['pid'],'start_ticks':old['start_ticks'],
                    'status':'stopped CPU only; original native launchers continue until actual exit capture'}}
            save(SCOPE/'meta/gpu-allocation.json',allocation)
            save(ROOT/'meta/evaluation-available-gpu-policy.json',allocation)
            save(SCOPE/'meta/controller.json',record)
            save(directory/'activate.json',{'pid':child.pid,'start_ticks':ready['start_ticks'],
                'plan_sha256':base.digest(directory/'plan.json'),'at_utc':now()});activated=True
            active=wait_json(directory/'activated.json',child)
            proof={'at_utc':now(),'controller_pid':child.pid,'controller_start_ticks':ready['start_ticks'],
                'source_sha256':base.digest(SELF),'plan':str((directory/'plan.json').relative_to(ROOT)),
                'old_controller_stopped':old['pid'],'native_launchers_retained':adopted,
                'allowed_gpus':ALLOWED,'activation':active,'native_restarted':False,'native_signaled':False,
                'other_user_signals':False,'source_and_parameters_unchanged':base.source_fingerprint()==scope['source_fingerprint']}
            save(directory/'handoff.json',proof);save(SCOPE/'meta/eight-gpu-handoff.json',proof)
            print(json.dumps(proof))
        except BaseException:
            if not activated:
                if child and child.poll() is None:child.terminate();child.wait(timeout=10)
                for path,data in originals.items():
                    if data is None:path.unlink(missing_ok=True)
                    else:path.write_bytes(data)
                if paused:checked_cpu(old,stopped=True);os.kill(old['pid'],signal.SIGCONT)
            raise

def retire_parent(plan,jobs,directory):
    if not plan or (directory/'predecessor-retired.json').exists():return
    adopted={row['scene'] for row in plan['adopted_launchers']}
    if any(j['scene'] in adopted and j['status']!='completed' for j in jobs):return
    assert all(j['returncode']==0 for j in jobs if j['scene'] in adopted)
    old=plan['old_controller'];checked_cpu(old,stopped=True);assert_no_live_children(old['pid'])
    os.kill(old['pid'],signal.SIGKILL)
    save(directory/'predecessor-retired.json',{'at_utc':now(),'pid':old['pid'],'start_ticks':old['start_ticks'],
        'reason':'All original native launcher OS exit statuses captured0; stopped CPU has no living children',
        'native_processes_signaled':False})

def run_stage(stage,scope,plan=None,directory=None):
    legacy.completed(legacy.STAGES[legacy.STAGES.index(stage)-1],scope)
    if plan:
        assert base.digest(directory/'ledger-before.json')==plan['ledger_before_sha256']
        ledger=base.load(directory/'ledger-before.json');jobs=copy.deepcopy(ledger['jobs'])
        parents={row['scene']:row for row in plan['adopted_launchers']}
    else:
        assert not (SCOPE/f'meta/evaluation-{stage}-runs.json').exists()
        jobs=[];parents={}
    pending=[s for s in scope['ordered_scenes'] if s not in {j['scene'] for j in jobs}]
    active={j['gpu']:(j,None,None) for j in jobs if j['status']=='running'}
    last_health=0
    def record(status='running',**extra):
        value={'scope':'first8','stage':stage,'status':status,'at_utc':now(),'jobs':jobs,
            'controller_pid':os.getpid(),'n_scenes':8,'episodes_per_variant':160,'allowed_gpus':ALLOWED,
            'source_fingerprint':scope['source_fingerprint'],'parameter_sha256':PARAMETER,**extra}
        save(SCOPE/f'meta/evaluation-{stage}-runs.json',value);save(SCOPE/'meta/progress.json',value)
    record()
    if plan:save(directory/'activated.json',{'pid':os.getpid(),'at_utc':now(),'retained_native':len(active),
        'pending_scenes':pending,'native_restarted':False})
    while pending or active:
        binding();assert base.digest(SELF)==base.load(SCOPE/'meta/controller.json')['source_sha256']
        for gpu,(job,child,handle) in list(active.items()):
            run=run_path(Path(job['log']))
            if run:
                job['run_directory']=str(run)
                if any(p.stat().st_size for p in run.rglob('blp_errors.jsonl')):raise base.GateError('Native BLP errors require inspection')
            if child is None:
                checked_cpu(plan['old_controller'],stopped=True)
                code,exit_proof=adopted_returncode(job['pid'],job['pid_start_ticks'])
            else:
                code=child.poll();exit_proof={'source':'subprocess.Popen.poll','at_utc':now()} if code is not None else None
            if code is None:continue
            if handle:handle.close()
            if code!=0 or run is None:raise base.GateError(f'Native scene failed: {job["scene"]}, returncode={code}')
            job.update(status='completed',returncode=code,audit=base.audit_scene(run,job),exit_status_proof=exit_proof)
            del active[gpu]
        retire_parent(plan,jobs,directory)
        inventory=gpu_inventory();disk=os.statvfs(SCOPE)
        assert disk.f_bavail*disk.f_frsize>=30*1024**3
        for gpu in eligible(inventory):
            if gpu in active or not pending:continue
            if psutil.virtual_memory().available<64*1024**3:break
            scene=pending.pop(0);identities=[r for r in scope['identities'] if r['scene_id']==scene]
            job,child,handle=legacy.launch_scene(stage,scene,identities,gpu)
            jobs.append(job);active[gpu]=(job,child,handle);record()
        pressures={name:Path('/proc/pressure',name).read_text() for name in ('cpu','memory','io')}
        observation={'at_utc':now(),'stage':stage,'gpu_inventory':public_inventory(inventory),
            'host_available_memory_mib':psutil.virtual_memory().available//1024**2,'pressure':pressures,
            'cpu_load':os.getloadavg(),'active_scenes':[j['scene'] for j,_,_ in active.values()],
            'output_free_bytes':disk.f_bavail*disk.f_frsize,'signals_sent':False}
        with (SCOPE/'logs/eight-gpu-resource-samples.jsonl').open('a') as stream:stream.write(json.dumps(observation)+'\n')
        record(gpu_inventory=public_inventory(inventory),pending_scenes=pending)
        if time.monotonic()-last_health>=600:
            save(SCOPE/'diagnostics'/('health-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json'),observation)
            last_health=time.monotonic()
        if pending or active:time.sleep(10)
    return legacy.finalize(stage,jobs,scope)

def control(plan_path):
    plan=base.load(plan_path);directory=plan_path.parent
    assert base.digest(SELF)==plan['source_sha256']
    assert base.digest(ROOT/'code/first8_evaluate.py')==plan['legacy_source_sha256']==OLD_SHA
    assert base.digest(SCOPE/'meta/scope.json')==plan['scope_sha256']
    assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==plan['boot_id']
    with (SCOPE/'meta/eight-gpu-controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=process(os.getpid());save(directory/'ready.json',{'pid':os.getpid(),'start_ticks':state['start_ticks']})
        deadline=time.monotonic()+120
        while not (directory/'activate.json').exists():
            assert time.monotonic()<deadline;time.sleep(.1)
        activation=base.load(directory/'activate.json')
        assert activation['pid']==os.getpid() and activation['start_ticks']==state['start_ticks']
        assert activation['plan_sha256']==base.digest(plan_path)
        scope=binding()
        # Adapt only this coordination module's resource binding, never Native algorithms.
        legacy.binding=binding
        try:
            run_stage('myopic',scope,plan,directory)
            run_stage('suffix_only',scope)
            legacy.comparison()
            save(SCOPE/'meta/progress.json',{'scope':'first8','status':'completed_pending_delivery','at_utc':now(),
                'n_episodes_by_variant':{s:160 for s in legacy.STAGES},'allowed_gpus':ALLOWED,
                'remaining':'32 scoped raw/config archives, actual figure QA, report and GitHub completion audit'})
        except BaseException as error:
            save(SCOPE/'meta/error.json',{'at_utc':now(),'error':f'{type(error).__name__}: {error}',
                'native_workers_left_running_for_inspection':True,'no_outcome_selected_retry':True})
            raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--check',action='store_true');group.add_argument('--adopt',action='store_true')
    group.add_argument('--run-plan',type=Path);args=parser.parse_args()
    if args.check:
        scope,old,ledger,inventory,available=readiness()
        print(json.dumps({'ready':True,'old_controller':old['pid'],'retained_scenes':len(ledger['jobs']),
            'available_additional_gpus':available,'available_host_memory_gib':psutil.virtual_memory().available/1024**3,
            'gpu_inventory':public_inventory(inventory),'native_workers_started':False}))
    elif args.adopt:adopt()
    else:control(args.run_plan)
