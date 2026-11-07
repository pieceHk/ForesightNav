#!/usr/bin/env python3
"""Run PBSP MFMON in a fixed directory with explicit worker partitions."""
import argparse
from dataclasses import replace
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parents[1]
GPU_PAIRS=((0,2),(3,5),(6,7))
ADAPTER_REVISION='mfmon_navigation_mapping_v5'
sys.path.insert(0,str(ROOT))
from experiments.mfmon_protocol import atomic, aggregate
from experiments.mfmon_audit import freeze,verify,geometry

def proc(pid):
    try:
        f=Path('/proc/%s/stat'%pid).read_text().rsplit(')',1)[1].split()
        return dict(pid=pid,start_ticks=f[19],state=f[0])
    except OSError:return None

def main():
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['start','status','verify','_run','_part'])
    p.add_argument('dataset',choices=['hm3d_mfmon'])
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--part',type=int,default=0)
    p.add_argument('--gpu-pairs',default=';'.join(','.join(map(str,pair)) for pair in GPU_PAIRS))
    p.add_argument('--smoke-ids',default='0,1',help='Explicit diagnostic IDs; ignored only when not smoke')
    p.add_argument('--geometry-gpu',type=int,default=0)
    p.add_argument('--run-id',default=None,help=argparse.SUPPRESS)
    a=p.parse_args()
    pairs=tuple(tuple(map(int,s.split(','))) for s in a.gpu_pairs.split(';'))
    assert pairs and all(len(x)==2 and min(x)>=0 for x in pairs)
    smoke_ids=tuple(map(int,a.smoke_ids.split(',')))
    assert len(set(smoke_ids))==len(smoke_ids) and all(0<=i<927 for i in smoke_ids)
    cli_args=['--gpu-pairs',a.gpu_pairs,'--smoke-ids',a.smoke_ids]
    out=ROOT/'experiments/results'/('hm3d_mfmon_smoke' if a.smoke else a.dataset)
    def result_rows():
        return [json.loads(f.read_text()) for f in sorted(out.glob('workers/*/episodes/episode_*.json'))]
    if a.action=='start':
        lock=(ROOT/'experiments/.mfmon.lock').open('a+')
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if out.exists() and any(out.iterdir()):raise RuntimeError('Existing fixed result directory; archive before a fresh run')
        out.mkdir(parents=True,exist_ok=True)
        run_id=str(uuid.uuid4())
        cmd=[sys.executable,__file__,'_run',a.dataset,'--run-id',run_id]+cli_args+(['--smoke'] if a.smoke else [])
        with (out/'launcher.log').open('wb') as log:
            child=subprocess.Popen(cmd,stdout=log,stderr=log,stdin=subprocess.DEVNULL,start_new_session=True)
        atomic(out/'launch.json',dict(**proc(child.pid),run_id=run_id,started_epoch=time.time()))
        print(json.dumps(dict(directory=str(out),pid=child.pid)))
    elif a.action=='status':
        r=json.loads((out/'run.json').read_text()) if (out/'run.json').exists() else {}
        r['observed_completed']=len(result_rows());r['process']=proc(r.get('pid',-1));print(json.dumps(r,indent=2))
    elif a.action=='_part':
        from pbsp.runtime.config import load_experiment
        from pbsp.runtime.launch import launch,source_info
        from pbsp.runtime.plan import prepare_plan
        from pbsp.runtime.resources import validate_resources
        exp=load_experiment(ROOT/'configs/experiments/hm3d_mfmon.yaml',ROOT/'runtime.local.yaml')
        pair=pairs[a.part]
        exp=replace(exp,runtime=replace(exp.runtime,data=Path('/data/3DONav_datasets'),
                    mapping_gpu=pair[0],navigation_gpu=pair[1]))
        plan=prepare_plan(exp,root=ROOT,source_commit=source_info(ROOT)['commit'])
        selected=[r for r in plan.identities if
                  (int(r.runtime_episode_id) in smoke_ids[a.part::len(pairs)] if a.smoke
                   else int(r.runtime_episode_id)%len(pairs)==a.part)]
        keys={(r.scene_id,r.runtime_episode_id) for r in selected}
        scenes=tuple(replace(s,episode_ids=tuple(i for i in s.episode_ids if (s.scene_id,i) in keys))
                     for s in plan.scenes if any((s.scene_id,i) in keys for i in s.episode_ids))
        plan=replace(plan,identities=tuple(selected),scenes=scenes)
        plan=replace(plan,run_directory=out/'workers'/str(a.part))
        checks=validate_resources(plan.resources,exp.runtime)
        launch(plan,resource_checks=checks)
    elif a.action=='_run':
        lock=(ROOT/'experiments/.mfmon.run.lock').open('a+')
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        r=dict(**proc(os.getpid()),run_id=a.run_id or str(uuid.uuid4()),status='starting',started_epoch=time.time(),
               expected_episodes=len(smoke_ids) if a.smoke else 927,goal_information='full_sequence',children=[],
               adapter_revision=ADAPTER_REVISION,expected_ids=sorted(smoke_ids) if a.smoke else list(range(927)),
               gpu_pairs=pairs)
        atomic(out/'run.json',r)
        children=[]; logs=[]
        def interrupted(*args):raise RuntimeError('Requested stop')
        signal.signal(signal.SIGTERM,interrupted)
        try:
            manifest=Path('/data/3DONav_datasets/PBSP/hm3d_mfmon/dataset_manifest.json')
            models=json.loads((ROOT/'configs/resources/models.json').read_text())
            extra=[Path('/nvme/khy/weights/pbsp')/row['path'] for row in models['files']+models['refs']]
            r['input_inventory_sha256']=freeze(ROOT,out,manifest,extra)
            r['protocol']=dict(max_episode_steps=1500,metric='posterior',success_distance=1.5,vertical_gate=1.1,
                goal_information='full_sequence',target_memory='persistent',sensor_width=640,sensor_height=480,
                sensor_hfov=79.,camera_height=.88,navmesh_radius=.1,planning='ordered_suffix_mvp')
            atomic(out/'run.json',r)
            for part in range(min(len(smoke_ids),len(pairs)) if a.smoke else len(pairs)):
                for gpu in set(pairs[part]):
                    used=int(subprocess.check_output(['nvidia-smi','--id='+str(gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True))
                    if used>1800:raise RuntimeError('GPU occupied: '+str(gpu))
                log=(out/('worker_'+str(part)+'.log')).open('wb');logs.append(log)
                cmd=[sys.executable,__file__,'_part',a.dataset,'--part',str(part)]+cli_args+(['--smoke'] if a.smoke else [])
                child=subprocess.Popen(cmd,stdout=log,stderr=log,start_new_session=True);children.append(child)
                r['children'].append(dict(**proc(child.pid),part=part,gpus=pairs[part]))
            while any(c.poll() is None for c in children):
                if any(c.poll() not in (None,0) for c in children):raise RuntimeError('PBSP worker failed')
                r.update(status='running',completed_episodes=len(result_rows()),updated_epoch=time.time());atomic(out/'run.json',r)
                time.sleep(15)
            if any(c.returncode for c in children):raise RuntimeError('PBSP worker failed')
            rows=result_rows();assert len(rows)==r['expected_episodes']
            atomic(out/'metrics.json',aggregate(rows))
            r.update(status='completed',completed_episodes=len(rows),finished_epoch=time.time())
        except BaseException as e:
            r.update(status='failed',error=repr(e));raise
        finally:
            for c in children:
                if c.poll() is None:os.killpg(c.pid,signal.SIGTERM)
            for c in children:
                try:c.wait(timeout=45)
                except subprocess.TimeoutExpired:os.killpg(c.pid,signal.SIGKILL);c.wait()
            for log in logs:log.close()
            r['cleanup_finished_epoch']=time.time();atomic(out/'run.json',r)
    else:
        report=verify(out,'PBSP',a.smoke)
        from pbsp.runtime.source_snapshot import verify_source_snapshot
        for worker in (out/'workers').iterdir():
            prov=json.loads((worker/'provenance.json').read_text())
            proof=verify_source_snapshot(worker,prov['source_snapshot'],prov,lambda p:json.loads(p.read_text()))
            assert proof['complete']
        os.environ['CUDA_VISIBLE_DEVICES']=str(a.geometry_gpu)
        print(json.dumps(geometry(out,'PBSP')))
        print(json.dumps(report['metrics'],indent=2))

if __name__=='__main__':main()
