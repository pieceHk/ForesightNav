"""Read-only memory sampling for an existing native evaluation stage."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
import psutil
from evaluate_sequence import OUT, load, digest
from available_gpu_evaluate_sequence import gpu_inventory
from watch_evaluation import process

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('nearest','myopic','suffix_only'),required=True)
    parser.add_argument('--watch-seconds',type=int,default=86400)
    args=parser.parse_args();assert 1<=args.watch_seconds<=86400
    start=load(OUT/f'meta/{args.stage}-start-a3.json')
    allowed=start['allowed_gpus'];assert allowed and set(allowed)<={1,2,4,5,6,7}
    assert start['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    end=time.monotonic()+args.watch_seconds;baseline={}
    while time.monotonic()<end:
        ledger=load(OUT/f'meta/evaluation-{args.stage}-runs.json')
        assert ledger['controller_pid']==start['pid'] and ledger['boot_id']==start['boot_id']
        assert digest(OUT/'meta/calibration.json')==start['parameter_sha256']
        identity=process(start['pid'])
        if ledger['status']!='running':
            print(json.dumps({'at':datetime.now(timezone.utc).isoformat(),'stage':args.stage,'event':'stage_left_running','status':ledger['status']}),flush=True);return
        assert identity['live'] and identity['start_ticks']==start['start_ticks']
        runs=[job.get('run_directory') for job in ledger['jobs'] if job.get('run_directory')]
        inv=gpu_inventory();cards=[]
        for gpu in inv['gpus']:
            idx=int(gpu['index'])
            if idx not in allowed: continue
            own=other=0;own_count=other_count=0
            for app in inv['compute_apps']:
                if app['gpu_uuid']!=gpu['uuid']:continue
                try:
                    proc=psutil.Process(int(app['pid']));cmd=' '.join(proc.cmdline())
                    matched=proc.uids().real==os.getuid() and any(run in cmd for run in runs)
                except (psutil.NoSuchProcess,psutil.AccessDenied):matched=False
                memory=int(app['used_gpu_memory'])
                if matched:own+=memory;own_count+=1
                else:other+=memory;other_count+=1
            baseline.setdefault(idx,other)
            free=int(gpu['memory.free'])
            cards.append({'gpu':idx,'free_mib':free,'experiment_compute_mib':own,
                'experiment_compute_processes':own_count,'other_compute_mib':other,
                'other_compute_processes':other_count,'low_headroom':free<3072,
                'other_compute_growth_mib':other-baseline[idx]})
        print(json.dumps({'at':datetime.now(timezone.utc).isoformat(),'stage':args.stage,
            'controller_pid':start['pid'],'controller_start_ticks':start['start_ticks'],
            'gpus':cards,'signals_sent':False,'native_states_modified':False,
            'scope':'Discrete memory samples, not a proven peak-memory bound'}),flush=True)
        time.sleep(20)
    print(json.dumps({'at':datetime.now(timezone.utc).isoformat(),'event':'bounded_watch_complete'}),flush=True)

if __name__=='__main__':main()
