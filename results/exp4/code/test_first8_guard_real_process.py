"""Exercise the new fault guard with two disposable CPU processes, never GPUs."""
from datetime import datetime,timezone
from pathlib import Path
import hashlib,json,os,signal,subprocess,sys,tempfile,time
from evaluate_sequence import OUT,load,digest,source_fingerprint

source=OUT/'code/first8_infrastructure_recovery_v2.py';expected='9d16337ef903a0a0e46de102c7eb2c02f1a44bceb973289b0406e23ed8f7cc40'
assert os.getuid()==1001 and digest(source)==expected
scope=load(OUT/'scopes/first8/meta/scope.json');assert source_fingerprint()==scope['source_fingerprint']
assert digest(OUT/'meta/calibration.json')==scope['parameter_sha256']
def stat(pid):
    p=Path('/proc')/str(pid);raw=(p/'stat').read_text();fields=raw[raw.rfind(')')+2:].split()
    return {'pid':pid,'state':fields[0],'start_ticks':fields[19],'uid':p.stat().st_uid,
        'wait_status':int(fields[49]),'stat_sha256':hashlib.sha256(raw.encode()).hexdigest()}

PROGRAM=r'''
from pathlib import Path
from types import SimpleNamespace
import hashlib,json,os,subprocess,sys,time
import first8_infrastructure_recovery_v2 as mod
root=Path(sys.argv[1]);(root/'meta').mkdir();run=root/'dummy-native';(run/'logs').mkdir(parents=True)
dummy_ready=root/'dummy-ready'
command=[sys.executable,'-c','import signal,sys,time;from pathlib import Path;signal.signal(signal.SIGTERM,lambda *_:sys.exit(143));Path(sys.argv[1]).write_text("ready");time.sleep(60)',str(dummy_ready)]
child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
deadline=time.monotonic()+8
while not dummy_ready.exists():
    assert child.poll() is None and time.monotonic()<deadline;time.sleep(.01)
def state(pid):
    p=Path('/proc')/str(pid);raw=(p/'stat').read_text();parts=raw[raw.rfind(')')+2:].split()
    return {'live':parts[0]!='Z','state':parts[0],'start_ticks':parts[19]}
job={'pid':child.pid,'pid_start_ticks':state(child.pid)['start_ticks'],'command':command,'scene':'CPU_DUMMY_ONLY','status':'running','run_directory':str(run)}
(root/'ready.json').write_text(json.dumps({'dummy':{'pid':child.pid,'start_ticks':job['pid_start_ticks']},'coordinator':os.getpid()}))
(run/'logs/mapping.log').write_text("RuntimeError: Mapping rebuild failed; incomplete maps cannot be used: OutOfMemoryError('CUDA out of memory.')")
(root/'meta/evaluation-myopic-runs.json').write_text(json.dumps({'jobs':[job]}))
plan=root/'plan.json';plan.write_text(json.dumps({'failed_scene_jobs':{},'tracer':{}}))
def save(path,value):Path(path).write_text(json.dumps(value))
engine=SimpleNamespace(ALLOWED=list(range(8)),adopted_returncode=lambda *_:None,retire_parent=lambda *_:None,
    eligible=lambda *_:[],binding=lambda:{},control=lambda _:engine.binding())
mod.base=SimpleNamespace(load=lambda path:json.loads(Path(path).read_text()),digest=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest())
mod.legacy=SimpleNamespace(launch_scene=lambda *_:None,save=save,now=lambda:__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat())
mod.engine=engine;mod.ROOT=root;mod.SCOPE=root;mod.process=state
mod.run(plan)
raise RuntimeError('Guard returned instead of preserving the stopped parent')
'''
with tempfile.TemporaryDirectory(prefix='exp4-guard-cpu-test-') as folder:
    root=Path(folder);log=root/'test.log'
    with log.open('wb') as output:
        coordinator=subprocess.Popen([sys.executable,'-u','-c',PROGRAM,str(root)],cwd=OUT/'code',
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),
            stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
    birth=stat(coordinator.pid)['start_ticks'];info=None;dummy=None
    try:
        end=time.monotonic()+15
        while time.monotonic()<end:
            if (root/'ready.json').exists():info=load(root/'ready.json')
            parent=stat(coordinator.pid)
            if info:
                dummy=stat(info['dummy']['pid'])
                if parent['state'] in ('T','t') and dummy['state']=='Z':break
            assert coordinator.poll() is None,log.read_text()
            time.sleep(.02)
        assert info and parent['state'] in ('T','t') and parent['start_ticks']==birth
        assert dummy['state']=='Z' and dummy['start_ticks']==info['dummy']['start_ticks'] and dummy['uid']==1001
        actual=os.waitstatus_to_exitcode(dummy['wait_status']);assert actual==143
        guard=load(root/'mapper-fatal-guard.json');sent=load(root/'mapper-fatal-guard-cleanup-sent.json')
        assert guard['matching_own_launcher_to_cleanup']==dummy['pid'] and not guard['other_user_signals_sent']
        assert sent['pid']==dummy['pid'] and sent['signal']=='SIGTERM'
        result={'at_utc':datetime.now(timezone.utc).isoformat(),'passed':True,'test_kind':'Two disposable CPU-only processes; no Native or GPU workers',
            'source_sha256':expected,'real_parent_stopped_before_reaping':True,'dummy_child_actual_exit':actual,
            'dummy_child_exit_stat':dummy,'coordinator_birth':birth,'fault_record':guard,'cleanup_record':sent,
            'actual_wait_status_preserved_while_parent_stopped':True,'foreign_tasks_signaled':False,'Native_workers_started':False}
    finally:
        if info:
            try:
                current=stat(info['dummy']['pid'])
                if current['uid']==1001 and current['start_ticks']==info['dummy']['start_ticks'] and current['state']!='Z':os.kill(current['pid'],signal.SIGTERM)
            except FileNotFoundError:pass
        if coordinator.poll() is None:
            current=stat(coordinator.pid);assert current['uid']==1001 and current['start_ticks']==birth
            coordinator.kill();coordinator.wait(timeout=10)
    assert source_fingerprint()==scope['source_fingerprint'] and digest(OUT/'meta/calibration.json')==scope['parameter_sha256']
    result['frozen_sources_and_parameters_unchanged']=True
    target=OUT/('scopes/first8/diagnostics/recovery-v2-real-CPU-guard-test-'+expected[:12]+'.json');target.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'proof':str(target.relative_to(OUT)),'passed':True,'real_parent_stopped':True,'actual_dummy_exit':actual,
        'actual_child_wait_status_preserved':True,'Native_workers_started':False,'frozen_sources_and_parameters_unchanged':True}))
