from datetime import datetime,timezone
from pathlib import Path
import json,os,subprocess,sys,time
root=Path('/nvme/khy/ForesightNav_exp4_depthfix/results/exp4')
repo=root.parents[1]
stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
log=root/('logs/calibration-observer-a3-'+stamp+'.jsonl')
meta={'observer_pid':os.getpid(),'started_at_utc':datetime.now(timezone.utc).isoformat(),
 'max_polls':20,'interval_seconds':45,'log':str(log),
 'scope':'bounded read-only supervision of controller31036 and CPU122348; stop on new closed scene'}
(root/'meta/last-calibration-observer.json').write_text(json.dumps(meta,indent=2)+'\n')
initial=None
with log.open('x') as stream:
 for index in range(meta['max_polls']):
  run=subprocess.run([sys.executable,str(root/'code/watch_calibration.py')],cwd=repo,capture_output=True,text=True)
  if run.returncode:
   out={'poll':index+1,'status':'observation_error','returncode':run.returncode,'stderr':run.stderr[-2000:]}
   stream.write(json.dumps(out)+'\n');stream.flush();print(json.dumps(out),flush=True);break
  audit=json.loads(run.stdout)
  state=json.loads((root/'meta/calibration-postprocess.json').read_text())
  proc=Path('/proc')/str(state['pid'])
  live=False
  if proc.exists():
   stat=(proc/'stat').read_text().rsplit(')',1)[1].split()
   live=stat[0]!='Z' and stat[19]==str(state['start_ticks']) and (proc/'cwd').resolve()==repo
  completed={j['scene'] for j in audit['jobs'] if j['status']=='completed'}
  if initial is None:initial=completed
  valid=audit['boot_id_matches'] and audit['frozen_files_unchanged'] and (
      audit['controller_status']!='running' or audit['controller_live_and_matching'])
  out={'poll':index+1,'checked_at_utc':audit['checked_at_utc'],'completed':audit['completed_episodes'],
   'controller_live_matching':audit['controller_live_and_matching'],'boot_matches':audit['boot_id_matches'],
   'source_unchanged':audit['frozen_files_unchanged'],
   'active_jobs':[{'scene':j['scene'],'gpu':j['gpu'],'pid':j['pid'],'completed':j['completed_episodes'],
       'accepted_actions':j['accepted_actions'],'snapshots':j['snapshots'],
       'live_matching':j['launcher_live_and_matching'],'shadow_error_bytes':j['shadow_error_bytes']}
       for j in audit['jobs'] if j['status']=='running'],
   'CPU_pid':state['pid'],'CPU_live_matching':live,'CPU_status':state['status']}
  reason=None
  if not valid:reason='controller_or_source_requires_review'
  elif state['status']=='stopped_for_review':reason='CPU_postprocess_requires_review'
  elif audit['controller_status']=='running' and not live:reason='CPU_waiter_not_live'
  elif completed-initial:reason='new_complete_scene'
  elif audit['controller_status']!='running':reason='collection_state_changed'
  if reason:out['observation_finished_reason']=reason
  stream.write(json.dumps(out)+'\n');stream.flush();print(json.dumps(out),flush=True)
  if reason:break
  if index+1<meta['max_polls']:time.sleep(meta['interval_seconds'])

