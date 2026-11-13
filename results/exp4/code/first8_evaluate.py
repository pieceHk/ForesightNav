"""User-authorized eight-scene Exp4 scope; original 400-case evidence is immutable."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import csv
import fcntl
import itertools
import json
import os
import signal
import subprocess
import sys
import time

import evaluate_sequence as base
from available_gpu_evaluate_sequence32 import gpu_inventory, eligible_gpus, public_inventory
from experiments.exp4_collect import run_path
from report_variants import normalize, format_summary, export_comparison
from watch_evaluation import process

ROOT = base.OUT
SCOPE = ROOT/'scopes/first8'
SELF = Path(__file__).resolve()
PARAMETER = 'e1d75f3c9503c7439467748fc97be6ae2ad450c429f40dc7235b1564ff7852ae'
SCHEDULER = 'f941c06d450ba05bcbd2d45041e454ed727cc0232289f2fcc694771e46bf2c96'
STAGES = ('full','nearest','myopic','suffix_only')

def now():
    return datetime.now(timezone.utc).isoformat()

def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+f'.tmp-{os.getpid()}')
    temporary.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2)+'\n')
    temporary.replace(path)

def select_first8(identities):
    scenes = list(dict.fromkeys(row['scene_id'] for row in identities))[:8]
    selected = [row for row in identities if row['scene_id'] in scenes]
    keys = [(r['scene_id'],str(r['runtime_episode_id'])) for r in selected]
    if len(scenes)!=8 or len(selected)!=160 or len(set(keys))!=160:
        raise base.GateError('Eight distinct scenes and 160 distinct fixed identities are required')
    if any(sum(r['scene_id']==s for r in selected)!=20 for s in scenes):
        raise base.GateError('Every selected scene must retain all twenty episodes')
    if any(len(r['source_episode_ids'])!=3 or len(r['ordered_categories'])!=3 for r in selected):
        raise base.GateError('Three ordered task/source identities must be retained')
    return scenes, selected

def validate_jobs(jobs, scope):
    wanted = {s:[r for r in scope['identities'] if r['scene_id']==s] for s in scope['ordered_scenes']}
    if len(jobs)!=8 or {j['scene'] for j in jobs}!=set(wanted):
        raise base.GateError('A scoped result requires exactly the eight selected scenes')
    for job in jobs:
        if job['status']!='completed' or job.get('returncode')!=0 or job['identities']!=wanted[job['scene']]:
            raise base.GateError('Incomplete process or changed assignment in scoped result')
        base.audit_scene(Path(job['run_directory']),job)

def binding():
    scope = base.load(SCOPE/'meta/scope.json')
    scenes, identities = select_first8(base.load(ROOT/'meta/3mon-identities.json'))
    assert scope['ordered_scenes']==scenes and scope['identities']==identities
    assert scope['original_identity_sha256']==base.digest(ROOT/'meta/3mon-identities.json')
    assert base.source_fingerprint()==scope['source_fingerprint']
    assert base.digest(ROOT/'meta/calibration.json')==scope['parameter_sha256']==PARAMETER
    assert base.digest(ROOT/'code/available_gpu_evaluate_sequence32.py')==SCHEDULER
    policy = base.load(ROOT/'meta/evaluation-available-gpu-policy.json')
    assert policy['allowed_gpus']==[5,6,7] and policy['reserved_gpus']==[0,3]
    assert policy['min_free_mib']==32768 and policy['max_existing_compute_mib']==11264
    return scope

def initialize():
    scope_path = SCOPE/'meta/scope.json'
    if scope_path.exists():
        print(json.dumps({'scope_exists':True,'scope':str(scope_path),'binding_verified':bool(binding())}))
        return
    proof = base.load(ROOT/'meta/first8-scope-cutoff.json')
    assert proof['remaining_target_pids']==[] and proof['raw_records_deleted'] is False
    assert proof['source_and_parameters_unchanged'] and proof['episodes_per_variant']==160
    old = base.load(ROOT/'meta/nearest-start-a3.json')
    assert not process(old['pid']).get('live')
    scenes, selected = select_first8(base.load(ROOT/'meta/3mon-identities.json'))
    assert scenes==proof['selected_scenes']
    base.validate_gate('full')
    original = {}
    for stage in ('full','nearest'):
        ledger = base.load(ROOT/f'meta/evaluation-{stage}-runs.json')
        jobs = [job for job in ledger['jobs'] if job['scene'] in scenes]
        validate_jobs(jobs,{'ordered_scenes':scenes,'identities':selected})
        assert ledger['source_fingerprint']==base.source_fingerprint()
        assert ledger['parameter_sha256']==PARAMETER
        original[stage]={'jobs':jobs,'original_ledger_sha256':base.digest(ROOT/f'meta/evaluation-{stage}-runs.json')}
    scope = {'scope_id':'first8','created_at_utc':now(),'user_instruction':'我们只跑前8个场景，不需要跑全量实验了',
             'ordered_scenes':scenes,'identities':selected,'n_scenes':8,'n_episodes_per_variant':160,
             'original_identity_sha256':base.digest(ROOT/'meta/3mon-identities.json'),
             'source_fingerprint':base.source_fingerprint(),'parameter_sha256':PARAMETER,
             'scope_cutoff_proof_sha256':base.digest(ROOT/'meta/first8-scope-cutoff.json'),
             'reused_native_runs':original,'allowed_gpus':[5,6,7],'reserved_gpus':[0,3],
             'native_sources_or_scientific_parameters_changed':False,
             'interpretation':'User reduced scope after partial outcomes existed; fixed original order, not an independent blind benchmark; original Full400 and excluded native records retained.'}
    SCOPE.mkdir(parents=True,exist_ok=True)
    (SCOPE/'meta').mkdir(exist_ok=True)
    with scope_path.open('x') as stream:
        stream.write(json.dumps(scope,ensure_ascii=False,indent=2)+'\n')
    for name in ('logs','summary','diagnostics','reports'):
        (SCOPE/name).mkdir(exist_ok=True)
    save(ROOT/'meta/active-evaluation-scope.json',{'scope':'first8','scope_path':str(scope_path),
         'n_scenes':8,'n_episodes_per_variant':160,'original_400_target_cancelled':True,'at_utc':now()})
    print(json.dumps({'initialized':True,'scope':str(scope_path),'episodes_per_variant':160,'scenes':scenes}))

def completed(stage, scope):
    ledger = base.load(SCOPE/f'meta/evaluation-{stage}-runs.json')
    assert ledger['scope']=='first8' and ledger['status']=='completed' and ledger['n_episodes']==160
    assert ledger['source_fingerprint']==scope['source_fingerprint'] and ledger['parameter_sha256']==PARAMETER
    validate_jobs(ledger['jobs'],scope)
    for relative, sha in ledger['artifact_sha256'].items():
        assert base.digest(SCOPE/relative)==sha
    summary = base.load(SCOPE/f'summary/{stage}.json')
    assert summary['n_episodes']==160 and summary['n_error']==0 and not summary['diagnostic_only']
    assert summary['process']['native_stage_path_check_count']==480
    rows = [json.loads(line) for line in (SCOPE/f'logs/{stage}_episodes.jsonl').read_text().splitlines()]
    assert {(r['scene_id'],str(r['episode_id'])) for r in rows} == {(r['scene_id'],str(r['runtime_episode_id'])) for r in scope['identities']}
    return ledger

def finalize(stage, jobs, scope):
    validate_jobs(jobs,scope)
    save(SCOPE/'meta/progress.json',{'status':'normalizing','stage':stage,'at_utc':now(),'episodes_per_variant':160})
    rows, decisions, summary = normalize(stage,[j['run_directory'] for j in jobs],expected=scope['identities'],
        calibration=base.load(ROOT/'meta/calibration.json'),parameter_sha256=PARAMETER)
    assert len(rows)==summary['n_episodes']==160 and summary['n_error']==0
    assert summary['process']['native_stage_path_check_count']==480
    summary.update(scope='first8',n_scenes=8,original_400_scope_complete=False)
    save(SCOPE/f'summary/{stage}.json',format_summary(summary))
    for kind, values in (('episodes',rows),('decisions',decisions)):
        (SCOPE/f'logs/{stage}_{kind}.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in values))
    ledger={'scope':'first8','stage':stage,'status':'completed','n_episodes':160,'n_error':0,
        'jobs':jobs,'source_fingerprint':scope['source_fingerprint'],'parameter_sha256':PARAMETER,
        'completed_at_utc':now(),'artifact_sha256':{p:base.digest(SCOPE/p) for p in
        (f'summary/{stage}.json',f'logs/{stage}_episodes.jsonl',f'logs/{stage}_decisions.jsonl')}}
    save(SCOPE/f'meta/evaluation-{stage}-runs.json',ledger)
    completed(stage,scope)
    if summary['SR']==0 or summary['SR']>90:
        raise base.GateError('Scoped anomalous score requires review before advancing')
    return ledger

def launch_scene(stage, scene, identities, gpu):
    tag_attempt='a3-first8'
    argv=base.command(stage,scene,identities,gpu,tag_attempt)
    log=SCOPE/f'logs/{stage}-{scene}.controller.log'
    handle=log.open('x')
    env=dict(os.environ);env.pop('CUDA_VISIBLE_DEVICES',None)
    child=subprocess.Popen(argv,cwd=base.REPO,env=env,stdin=subprocess.DEVNULL,
        stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
    job={'scene':scene,'identities':identities,'gpu':gpu,'pid':child.pid,
         'pid_start_ticks':process(child.pid)['start_ticks'],'command':argv,'log':str(log),'status':'running'}
    return job,child,handle

def run_stage(stage, scope):
    completed(STAGES[STAGES.index(stage)-1],scope)
    ledger_path=SCOPE/f'meta/evaluation-{stage}-runs.json'
    if ledger_path.exists():
        raise base.GateError('Existing scoped stage: inspect the same controller; never rerun')
    pending=list(scope['ordered_scenes']);jobs=[];active={}
    def record(status='running', **extra):
        value={'scope':'first8','stage':stage,'status':status,'at_utc':now(),'jobs':jobs,
            'controller_pid':os.getpid(),'n_scenes':8,'episodes_per_variant':160,
            'source_fingerprint':scope['source_fingerprint'],'parameter_sha256':PARAMETER,**extra}
        save(ledger_path,value);save(SCOPE/'meta/progress.json',value)
    record()
    while pending or active:
        binding()
        assert base.digest(SELF)==base.load(SCOPE/'meta/controller.json')['source_sha256']
        for gpu,(job,child,handle) in list(active.items()):
            run=run_path(Path(job['log']))
            if run:
                job['run_directory']=str(run)
                if any(p.stat().st_size for p in run.rglob('blp_errors.jsonl')):
                    raise base.GateError('Native BLP errors require inspection')
            code=child.poll()
            if code is None:
                continue
            handle.close()
            if code!=0 or run is None:
                raise base.GateError(f'Native scoped scene exited unsuccessfully: {job["scene"]}, returncode={code}')
            job.update(status='completed',returncode=code,audit=base.audit_scene(run,job),
                exit_status_proof={'source':'subprocess.Popen.poll','at_utc':now()})
            del active[gpu]
        inventory=gpu_inventory()
        stat=os.statvfs(SCOPE)
        assert stat.f_bavail*stat.f_frsize>=30*1024**3
        for gpu in eligible_gpus(inventory,scope['allowed_gpus']):
            if gpu in active or not pending:
                continue
            scene=pending.pop(0)
            identities=[r for r in scope['identities'] if r['scene_id']==scene]
            job,child,handle=launch_scene(stage,scene,identities,gpu)
            jobs.append(job);active[gpu]=(job,child,handle)
            record()
        record(gpu_inventory=public_inventory(inventory),pending_scenes=pending)
        with (SCOPE/'logs/resource-samples.jsonl').open('a') as stream:
            stream.write(json.dumps({'at_utc':now(),'stage':stage,'gpu_inventory':public_inventory(inventory),
                'memory_pressure':Path('/proc/pressure/memory').read_text(),
                'output_free_bytes':stat.f_bavail*stat.f_frsize},allow_nan=False)+'\n')
        if pending or active:
            time.sleep(10)
    record(status='normalizing')
    return finalize(stage,jobs,scope)

def comparison():
    variants={}
    for stage in STAGES:
        completed(stage,binding())
        rows=[json.loads(line) for line in (SCOPE/f'logs/{stage}_episodes.jsonl').read_text().splitlines()]
        variants[stage]=(rows,[],base.load(SCOPE/f'summary/{stage}.json'))
    export_comparison(SCOPE,variants)
    folder=SCOPE/'summary/paired';folder.mkdir(exist_ok=True)
    pairs=[]
    for left,right in itertools.combinations(('nearest','suffix_only','myopic','full'),2):
        a={(r['scene_id'],str(r['episode_id'])):r for r in variants[left][0]}
        b={(r['scene_id'],str(r['episode_id'])):r for r in variants[right][0]}
        target=folder/f'{left}_vs_{right}.csv'
        with target.open('w',newline='') as stream:
            fields=['scene_id','episode_id','left','right','left_success','right_success','left_SR','right_SR','left_SPL','right_SPL','left_PR','right_PR','left_PPL','right_PPL']
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
            for key in sorted(a):
                x,y=a[key],b[key]
                assert x['goals']==y['goals'] and x['source_episode_ids']==y['source_episode_ids']
                writer.writerow({'scene_id':key[0],'episode_id':key[1],'left':left,'right':right,
                    'left_success':int(x['episode_success']),'right_success':int(y['episode_success']),
                    **{prefix+'_'+metric:row['native_mon_metrics'][metric.lower()] if metric in ('SR','SPL') else row[metric]
                       for prefix,row in (('left',x),('right',y)) for metric in ('SR','SPL','PR','PPL')}})
        with target.open() as stream:
            assert len(list(csv.DictReader(stream)))==160
        pairs.append({'path':str(target.relative_to(SCOPE)),'n_paired':160,'sha256':base.digest(target)})
    save(folder/'manifest.json',{'scope':'first8','n_episodes_each_pair':160,'pairs':pairs})

def control():
    with (SCOPE/'meta/controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        identity=process(os.getpid())
        save(SCOPE/'meta/ready.json',{'pid':os.getpid(),'start_ticks':identity['start_ticks'],'at_utc':now()})
        deadline=time.monotonic()+120
        while not (SCOPE/'meta/activate.json').exists():
            assert time.monotonic()<deadline
            time.sleep(.1)
        record=base.load(SCOPE/'meta/controller.json')
        assert record['pid']==os.getpid() and record['start_ticks']==identity['start_ticks']
        assert record['source_sha256']==base.digest(SELF)
        scope=binding()
        try:
            for stage in ('full','nearest'):
                finalize(stage,scope['reused_native_runs'][stage]['jobs'],scope)
            for stage in ('myopic','suffix_only'):
                run_stage(stage,scope)
            comparison()
            save(SCOPE/'meta/progress.json',{'scope':'first8','status':'completed_pending_delivery',
                'at_utc':now(),'n_episodes_by_variant':{stage:160 for stage in STAGES},
                'remaining':'32 scoped raw/config archives, actual figure QA, report and GitHub completion audit'})
        except BaseException as error:
            save(SCOPE/'meta/error.json',{'at_utc':now(),'error':f'{type(error).__name__}: {error}',
                'native_workers_left_running_for_inspection':True,'no_outcome_selected_retry':True})
            raise

def launch():
    scope=binding()
    claim=SCOPE/'meta/controller-launch-claim.json'
    with claim.open('x') as stream:
        json.dump({'at_utc':now(),'scope':'first8','source_sha256':base.digest(SELF)},stream)
    log=SCOPE/'logs/pipeline-controller.log'
    argv=[sys.executable,'-u',str(SELF),'--run']
    with log.open('xb') as out:
        child=subprocess.Popen(argv,cwd=base.REPO,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),
            stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
    deadline=time.monotonic()+60
    while not (SCOPE/'meta/ready.json').exists():
        assert child.poll() is None, 'Inspect the failed launch, never duplicate it'
        assert time.monotonic()<deadline, 'Inspect the same initializing PID'
        time.sleep(.2)
    ready=base.load(SCOPE/'meta/ready.json')
    assert ready['pid']==child.pid and process(child.pid)['start_ticks']==ready['start_ticks']
    record={'scope':'first8','pid':child.pid,'start_ticks':ready['start_ticks'],'command':argv,
            'source_sha256':base.digest(SELF),'source_fingerprint':scope['source_fingerprint'],
            'parameter_sha256':PARAMETER,'log':str(log),'started_at_utc':now(),
            'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
    save(SCOPE/'meta/controller.json',record)
    save(SCOPE/'meta/activate.json',{'pid':child.pid,'at_utc':now()})
    print(json.dumps(record))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--initialize',action='store_true')
    group.add_argument('--launch',action='store_true')
    group.add_argument('--run',action='store_true')
    args=parser.parse_args()
    if args.initialize: initialize()
    elif args.launch: launch()
    else: control()
