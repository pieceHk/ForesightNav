"""Independent structural, metric and frozen-input checks for MFMON results."""
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def write(path,value):
    p=Path(path);q=p.with_suffix(p.suffix+'.tmp')
    q.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');q.replace(p)


def freeze(root,out,manifest,extra=()):
    """Record scene, dataset, policy and model bytes before worker startup."""
    root,out,manifest=map(Path,(root,out,manifest));m=json.loads(manifest.read_text())
    inputs={manifest,*map(Path,(r['path'] for r in m['input_files']))}
    scene_root=Path('/data/3DONav_datasets/LifelongCrossNav/scene_datasets')
    inputs.add(scene_root/'hm3d_v0.2/hm3d_annotated_basis.scene_dataset_config.json')
    for e in m['entries']:
        p=scene_root/e['scene_id'];assert p.is_file()
        inputs.update(q for q in p.parent.glob(p.name.split('.')[0]+'*') if q.is_file())
    inputs.update(map(Path,extra))
    source=set()
    raw=subprocess.check_output(['git','ls-files','-co','--exclude-standard','-z'],cwd=root)
    for item in raw.split(b'\0'):
        if not item:continue
        p=root/os.fsdecode(item)
        if p.is_file() and p.suffix in {'.py','.cpp','.h','.hpp','.yaml','.yml','.json','.xml','.launch','.sh','.txt'} and out.parent not in p.parents:source.add(p)
    def records(paths):return [dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(paths)]
    value=dict(created_epoch=time.time(),manifest_sha256=sha(manifest),inputs=records(inputs),sources=records(source))
    write(out/'input_inventory.json',value)
    return sha(out/'input_inventory.json')


def alive(row):
    try:
        s=(Path('/proc')/str(row['pid'])/'stat').read_text().rsplit(')',1)[1].split()
        return s[19]==str(row['start_ticks']) and s[0]!='Z'
    except (OSError,KeyError):return False


def close(a,b,tol=1e-9):
    if a is None or b is None:assert a is b,(a,b)
    else:assert math.isclose(a,b,rel_tol=tol,abs_tol=tol),(a,b)


def metric(stages):
    successful=[s for s in stages if s['success']]
    ds=math.fsum(s['shortest_m'] for s in successful)
    dr=math.fsum(s['traveled_m'] for s in successful)
    efficiency=ds/max(ds,dr) if max(ds,dr)>0 else 0.
    full=int(len(successful)==3)
    return dict(SR=full,SPL=full*efficiency,PR=len(successful)/3,PPL=len(successful)/3*efficiency)


def verify(out,project,smoke=False,check_inputs=True):
    out=Path(out);run=json.loads((out/'run.json').read_text())
    assert run['status']=='completed' and run.get('cleanup_finished_epoch')
    assert not any(alive(r) for r in [run]+run.get('children',[]))
    pattern='workers/*/episodes/episode_*.json' if project=='PBSP' else 'episodes/episode_[0-9][0-9][0-9].json'
    paths=list(out.glob(pattern));rows=sorted((json.loads(p.read_text()) for p in paths),key=lambda r:r['global_id'])
    manifest_path=next(out.glob('workers/*/dataset_manifest.json')) if project=='PBSP' else out/'dataset_manifest.json'
    m=json.loads(manifest_path.read_text())
    expected_ids=run.get('expected_ids',list(range(2))) if smoke else list(range(927))
    n=len(expected_ids)
    assert len(m['entries'])==927 and len({e['scene_id'] for e in m['entries']})==36
    assert len(m['cfr_ids'])==len(set(m['cfr_ids']))==288
    assert len(rows)==n and [r['global_id'] for r in rows]==sorted(expected_ids)
    budget_prefix=[]
    for r in rows:
        e=m['entries'][r['global_id']]
        for key in ('global_id','content_index','source_file','scene_id','start_position','start_rotation','goals'):assert r[key]==e[key],(r['global_id'],key)
        assert r['cfr']==(r['global_id'] in m['cfr_ids'])
        assert r['goal_information']==('full_sequence' if project=='PBSP' else 'current_goal_only')
        stages=r['stages'];assert 1<=len(stages)<=3
        assert all(s['success']==1 for s in stages[:-1])
        if len(stages)<3 and stages[-1]['success']:
            assert stages[-1]['budget_exhausted'] and r['total_steps']==1500
            budget_prefix.append(r['global_id'])
        prev=e['start_position'];step_sum=0
        for i,s in enumerate(stages):
            assert s['stage']==i and s['category']==e['goals'][i]
            assert s['success'] in (0,1) and type(s['steps']) is int and s['steps']>0
            poses=s['poses'];assert len(poses)==s['steps']+1
            assert all(len(p)==3 and all(math.isfinite(v) for v in p) for p in poses)
            assert math.dist(prev,poses[0])<(.05 if i==0 else 1e-6),(r['global_id'],i,'discontinuity')
            prev=poses[-1];step_sum+=s['steps']
            dist=math.fsum(math.dist(a,b) for a,b in zip(poses,poses[1:]));close(dist,s['traveled_m'])
            ds=s['shortest_m'];assert ds is None or math.isfinite(ds) and ds>=0
            if s['success']:
                assert s['called_found'] and s['terminal_action']=='stop' and ds is not None
                assert s['agent_distance_m'] is not None and 0<=s['agent_distance_m']<1.5
                assert s['detection_distance_m'] is not None and 0<=s['detection_distance_m']<1.5
                assert s['detection_position'] is not None
            spl=s['success']*ds/max(ds,dist) if ds is not None and max(ds,dist)>0 else 0.
            close(spl,s['spl'])
            if s['terminal_action']!='stop':assert i==len(stages)-1 and r['total_steps']==1500 and not s['success']
        assert step_sum==r['total_steps'] and 1<=step_sum<=1500
        if project=='ApexNav':
            actions=[json.loads(l) for l in (out/'episodes'/('episode_%03d.actions.jsonl'%r['global_id'])).read_text().splitlines()]
            assert len(actions)==step_sum
            offset=0
            for i,s in enumerate(stages):
                for j in range(s['steps']):
                    a=actions[offset+j];assert a['step']==offset+j+1 and a['stage']==i
                    assert a['position']==s['poses'][j] and a['position_after']==s['poses'][j+1]
                offset+=s['steps']
        if project=='PBSP':
            source_path=next(p for p in paths if int(p.stem.split('_')[-1])==r['global_id'])
            worker=source_path.parent.parent;scene=Path(r['scene_id']).name.split('.')[0]
            trace=worker/('results_'+scene)/'hm3d_mfmon/episodes'/scene/str(r['global_id'])/'trajectory.jsonl'
            actions=[json.loads(line) for line in trace.read_text().splitlines()]
            assert len(actions)==step_sum
            offset=0
            for i,stage in enumerate(stages):
                for j in range(stage['steps']):
                    action=actions[offset+j]
                    assert action['frame_index']==offset+j+1 and action['subtask_index']==i and action['task_step']==j+1
                    assert math.dist(action['world_pose'][4:],stage['poses'][j])<1e-6
                    assert math.dist(action['world_pose_after'][4:],stage['poses'][j+1])<1e-6
                final=actions[offset+stage['steps']-1]
                assert final['action_name'].lower()==stage['terminal_action']
                assert (final['is_stop'] and final['stop_reason']=='verified_target')==stage['called_found']
                offset+=stage['steps']
        if 'metrics' in r:
            for k,v in metric(stages).items():close(v,r['metrics'][k])
    saved=json.loads((out/'metrics.json').read_text())
    for name,group in [('all',rows),('cross_floor_required',[r for r in rows if r['cfr']])]:
        if not group:assert saved[name] is None;continue
        g=saved[name];assert g['episodes']==len(group)
        for k in ('SR','SPL','PR','PPL'):close(math.fsum(metric(r['stages'])[k] for r in group)/len(group),g[k])
        for i in range(3):
            reached=[r['stages'][i] for r in group if len(r['stages'])>i]
            successes=sum(s['success'] for s in reached);spl=math.fsum(s['spl'] for s in reached)
            assert g['stages_reached'][i]==len(reached) and g['stage_successes'][i]==successes
            close(successes/len(reached) if reached else None,g['conditional_SR'][i])
            close(spl/len(reached) if reached else None,g['conditional_SPL'][i])
            close(successes/len(group),g['global_SR'][i]);close(spl/len(group),g['global_SPL'][i])
    inventory_path=out/'input_inventory.json'
    if check_inputs:
        assert inventory_path.exists()
        close_digest=run.get('input_inventory_sha256');assert close_digest==sha(inventory_path)
        inv=json.loads(inventory_path.read_text())
        assert inv['manifest_sha256']==sha(manifest_path)
        for row in inv['inputs']+inv['sources']:assert sha(row['path'])==row['sha256'],row['path']
    report=dict(status='passed',episodes=n,scenes=len({r['scene_id'] for r in rows}),
                cfr_episodes=sum(r['cfr'] for r in rows),successful_budget_prefixes=budget_prefix,
                evaluated_epoch=time.time(),metrics=saved,result_files=[dict(path=str(p),sha256=sha(p)) for p in sorted(paths)])
    write(out/'verification.json',report)
    return report


def geometry(out,project):
    """Recompute posterior shortest paths and terminal geometry after exit."""
    import numpy as np
    if project=='PBSP':from experiments.mfmon_protocol import World,box_center,box_sizes
    else:from mfmon_protocol import World,box_center,box_sizes
    out=Path(out)
    pattern='workers/*/episodes/episode_*.json' if project=='PBSP' else 'episodes/episode_[0-9][0-9][0-9].json'
    rows=sorted((json.loads(p.read_text()) for p in out.glob(pattern)),key=lambda r:r['global_id'])
    world=World();checked=0
    try:
        for r in rows:
            world.reset(r)
            for i,s in enumerate(r['stages']):
                world.stage=i
                close(world.shortest_distance(r['goals'][i],s['poses'][0]),s['shortest_m'],tol=1e-5)
                camera=s.get('camera_position',list(np.asarray(s['poses'][-1])+[0.,world.camera_height,0.]))
                for p,field in [(camera,'agent_distance_m'),(s['detection_position'],'detection_distance_m')]:
                    distances=[]
                    if p is not None:
                        for o in world.objects[r['goals'][i]]:
                            c=np.asarray(box_center(o.aabb));z=np.asarray(box_sizes(o.aabb))
                            if abs(p[1]-c[1])>1.1:continue
                            dx=max(c[0]-z[0]/2-p[0],0.,p[0]-c[0]-z[0]/2)
                            dz=max(c[2]-z[2]/2-p[2],0.,p[2]-c[2]-z[2]/2)
                            distances.append(math.hypot(dx,dz))
                    close(min(distances) if distances else None,s[field],tol=1e-5)
                checked+=1
    finally:world.close()
    report=dict(status='passed',episodes=len(rows),stages=checked,evaluated_epoch=time.time())
    write(out/'geometry_verification.json',report)
    return report
