from datetime import datetime,timezone
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
WORKER="\nfrom datetime import datetime, timezone\nfrom pathlib import Path\nimport hashlib,json,os,re,signal,subprocess,sys,time\npath=Path(sys.argv[1])\nassert re.fullmatch(r'/tmp/exp4-gpu3-release-plan-[0-9TZ]+\\.json',str(path)) and not path.is_symlink()\np=json.loads(path.read_text())\nassert path.stat().st_uid==p['operator_uid'] and p['operator_uid']!=0\nassert p['authorized_gpu']==3 and p['signals_sent'] is False\nassert (datetime.now(timezone.utc)-datetime.fromisoformat(p['created_at_utc'])).total_seconds()<180\nassert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==p['boot_id']\nhas_pidfd=hasattr(os,'pidfd_open') and hasattr(signal,'pidfd_send_signal')\ndef membership():\n    def rows(kind,fields):\n        return [line.strip().split(', ') for line in subprocess.check_output(['nvidia-smi','--query-'+kind+'='+fields,'--format=csv,noheader,nounits'],text=True).splitlines()]\n    ids={uuid:int(index) for index,uuid in rows('gpu','index,uuid')}\n    return [(ids[uuid],int(pid)) for uuid,pid in rows('compute-apps','gpu_uuid,pid')]\ndef matching(t,require_gpu=True):\n    proc=Path('/proc')/str(t['pid'])\n    try:\n        raw=(proc/'stat').read_text();f=raw[raw.rfind(')')+2:].split()\n        if f[0]=='Z' or f[19]!=t['start_ticks'] or proc.stat().st_uid!=t['uid']:\n            return False\n        args=(proc/'cmdline').read_bytes().rstrip(b'\\0')\n        if hashlib.sha256(args).hexdigest()!=t['command_sha256']:\n            return False\n        devices={gpu for gpu,pid in membership() if pid==t['pid']}\n        return (3 in devices and devices<={1,2,3}) if require_gpu else not devices or devices<={1,2,3}\n    except (FileNotFoundError,ProcessLookupError):\n        return False\nproof=path.with_name(path.stem.replace('-plan-','-result-')+'.json')\nresult={'checked_at_utc':datetime.now(timezone.utc).isoformat(),'plan':str(path),'authorized_gpu':3,'events':[],\n    'protected_gpus_not_targeted':[0,4,5,6,7],'parent_process_groups_signaled':False}\nfd_proof=os.open(proof,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)\nos.close(fd_proof);os.chown(proof,p['operator_uid'],-1)\ndef save():\n    with proof.open('r+') as stream:\n        stream.seek(0);stream.write(json.dumps(result,indent=2)+'\\n');stream.truncate();stream.flush();os.fsync(stream.fileno())\nsave()\nfor t in p['targets']:\n    if not t.get('eligible_gpu3_other_user') or t['uid']==p['operator_uid'] or not matching(t):\n        result['events'].append({'pid':t['pid'],'action':'skipped_scope_or_identity_changed'});save();continue\n    fd=os.pidfd_open(t['pid']) if has_pidfd else None\n    try:\n        if not matching(t):\n            continue\n        if fd is None: os.kill(t['pid'],signal.SIGTERM)\n        else: signal.pidfd_send_signal(fd,signal.SIGTERM)\n        result['events'].append({'pid':t['pid'],'start_ticks':t['start_ticks'],'uid':t['uid'],'action':'SIGTERM','gpu_indices_checked':t['gpu_indices']});save()\n        end=time.monotonic()+10\n        while time.monotonic()<end and matching(t,False):\n            time.sleep(.5)\n        if matching(t):\n            if fd is None: os.kill(t['pid'],signal.SIGKILL)\n            else: signal.pidfd_send_signal(fd,signal.SIGKILL)\n            result['events'].append({'pid':t['pid'],'action':'SIGKILL_after_TERM_and_fresh_GPU3_scope_recheck'});save()\n    finally:\n        if fd is not None: os.close(fd)\nresult['completed_at_utc']=datetime.now(timezone.utc).isoformat();save()\nprint(json.dumps({'proof':str(proof),'SIGTERM_count':sum(x['action']=='SIGTERM' for x in result['events']),\n    'SIGKILL_count':sum(x['action'].startswith('SIGKILL') for x in result['events']),\n    'skipped_count':sum(x['action'].startswith('skipped') for x in result['events']),\n    'protected_gpus_not_targeted':result['protected_gpus_not_targeted'],'parent_process_groups_signaled':False}))\n"
control=Path(sys.argv[1]);cfg=json.loads(control.read_text());started=time.monotonic()
assert os.geteuid()==0 and cfg['authorized_gpu']==3
boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
def query(kind,fields):
    return [line.strip().split(', ') for line in subprocess.check_output(['nvidia-smi','--query-'+kind+'='+fields,'--format=csv,noheader,nounits'],text=True).splitlines()]
print(json.dumps({'at':datetime.now(timezone.utc).isoformat(),'event':'guard_started','physical_gpu':3,'operator_uid':cfg['operator_uid'],'max_seconds':cfg['max_seconds']}),flush=True)
while time.monotonic()-started<cfg['max_seconds']:
    current=json.loads(control.read_text())
    if not current['enabled'] or boot!=Path('/proc/sys/kernel/random/boot_id').read_text().strip(): break
    terminal=control.parent/'evaluation-suffix_only-runs.json'
    if terminal.exists() and json.loads(terminal.read_text()).get('status')=='completed': break
    ids={uuid:int(idx) for idx,uuid in query('gpu','index,uuid')}
    pairs=[(ids[uuid],int(pid),int(mem)) for uuid,pid,mem in query('compute-apps','gpu_uuid,pid,used_gpu_memory')]
    targets=[]
    for gpu,pid,mem in pairs:
        if gpu!=3: continue
        proc=Path('/proc')/str(pid)
        try:
            uid=proc.stat().st_uid;raw=(proc/'stat').read_text();fields=raw[raw.rfind(')')+2:].split()
            if uid<1000 or uid==cfg['operator_uid'] or fields[0]=='Z': continue
            devices=sorted({g for g,p,m in pairs if p==pid})
            if not set(devices)<={1,2,3}: continue
            targets.append({'pid':pid,'start_ticks':fields[19],'uid':uid,'gpu_indices':devices,'command_sha256':hashlib.sha256((proc/'cmdline').read_bytes().rstrip(b'\0')).hexdigest(),'eligible_gpu3_other_user':True})
        except (FileNotFoundError,ProcessLookupError): continue
    if targets:
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');plan=Path('/tmp')/('exp4-gpu3-release-plan-'+stamp+'.json')
        p={'created_at_utc':datetime.now(timezone.utc).isoformat(),'boot_id':boot,'operator_uid':cfg['operator_uid'],'authorized_gpu':3,'targets':targets,'signals_sent':False}
        fd=os.open(plan,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,'w') as stream: json.dump(p,stream)
        os.chown(plan,cfg['operator_uid'],-1)
        original=sys.argv;sys.argv=['gpu3_checked_release',str(plan)]
        try: exec(compile(WORKER,'<gpu3-birth-checked-release>','exec'),{'__name__':'__main__'})
        finally: sys.argv=original
    print(json.dumps({'at':datetime.now(timezone.utc).isoformat(),'event':'poll','eligible_foreign_gpu3_processes':len(targets)}),flush=True)
    time.sleep(cfg['interval_seconds'])
print(json.dumps({'at':datetime.now(timezone.utc).isoformat(),'event':'guard_stopped'}),flush=True)
