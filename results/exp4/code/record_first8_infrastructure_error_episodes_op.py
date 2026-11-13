"""Preserve separate, unscored error episodes from the two failed archives."""
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import gzip,json,os,tarfile
from evaluate_sequence import OUT,load,digest,source_fingerprint

root=OUT/'scopes/first8';scope=load(root/'meta/scope.json')
assert source_fingerprint()==scope['source_fingerprint'] and digest(OUT/'meta/calibration.json')==scope['parameter_sha256']
recovery=load(root/'meta/infrastructure-recovery-v2-20261113.json');assert digest(OUT/recovery['plan'])==recovery['plan_sha256']
plan=load(OUT/recovery['plan']);rows=[]
for attempt in plan['failed_attempts']:
    archive=attempt['native_archive'];assert digest(OUT/archive['archive'])==archive['archive_sha256']
    expected=attempt['job']['identities'];assert len(expected)==20
    with tarfile.open(OUT/archive['archive'],'r:gz') as tar:
        names=tar.getnames();launch=json.load(tar.extractfile('launch.json'));status=json.load(tar.extractfile('status.json'))
        assert status['status']=='failed' and launch['identities']==expected
        completed={}
        for name in names:
            if name.endswith('/per_episode_metrics.json'):completed.update(json.load(tar.extractfile(name)))
        assert len(completed)==attempt['exported_records_excluded']
        prefix={attempt['scene']+'_'+str(r['runtime_episode_id']) for r in expected[:len(completed)]}
        assert set(completed)==prefix,'Only an uninterrupted original prefix can identify the next reset target'
        incomplete=[]
        for name in names:
            if not name.endswith('/trajectory.jsonl'):continue
            identity_path=str(Path(name).with_name('identity.json'));identity=json.load(tar.extractfile(identity_path))['identity']
            if attempt['scene']+'_'+str(identity['runtime_episode_id']) in completed:continue
            trace=[json.loads(line) for line in tar.extractfile(name).read().splitlines() if line]
            incomplete.append((identity,trace,name,identity_path))
        assert len(incomplete)<=1
        inferred=not incomplete
        if incomplete:identity,trace,trace_path,identity_path=incomplete[0]
        else:identity,trace,trace_path,identity_path=expected[len(completed)],[],None,None
        assert identity==expected[len(completed)]
        if inferred:
            navigation=[r for r in attempt['worker_logs'] if Path(r['source']).name.startswith('navigation_')];assert len(navigation)==1
            log=navigation[0];assert digest(OUT/log['archive'])==log['archive_sha256']
            text=gzip.decompress((OUT/log['archive']).read_bytes()).decode(errors='replace')
            assert 'reset_episode failed' in text and 'CUDA out of memory' in text
        counts=Counter(r['subtask_index'] for r in trace);failing_stage=trace[-1]['subtask_index'] if trace else 0
        stages=[]
        for k,goal in enumerate(identity['ordered_categories']):
            stages.append({'k':k+1,'goal':goal,'reached':bool(counts[k]),'success':None,'steps':counts[k],
                'path_length_m':None,'shortest_path_m':None,'start_pos':None,'end_pos':None,
                'stop_reason':'error' if k==failing_stage else None})
        actual_exit=attempt.get('actual_exit')
        rows.append({'method':'ForesightNav','variant':'myopic','task':'3-MON','episode_id':str(identity['runtime_episode_id']),
            'scene_id':identity['scene_id'],'goals':identity['ordered_categories'],'source_episode_ids':identity['source_episode_ids'],
            'stages':stages,'episode_success':None,'total_steps':len(trace),'wallclock_s':None,'code_commit':launch['source_commit'],
            'stop_reason':'error','error_kind':'infrastructure_mapper_CUDA_OOM','native_status':status,
            'actual_launcher_wait_proof':actual_exit,'canonical_score_included':False,'metrics_unavailable_for_failed_episode':True,
            'identity_status':'next_reset_target_inferred_from_exact_completed_prefix_and_reset_traceback' if inferred else 'observed_in_incomplete_native_identity_and_trajectory',
            'accepted_action_trajectory_present':bool(trace),'trajectory_path':trace_path,'identity_path':identity_path,
            'native_run':archive['native_run'],'native_archive':archive['archive'],'native_archive_sha256':archive['archive_sha256'],
            'completed_prefix_exports_excluded_in_full':len(completed),
            'worker_log_archives':attempt['worker_logs'],
            'interpretation':'All scores and unmeasured paths remain null; raw actions after mapper failure do not validate the map. Both full20 replacements retain the same identities and seed.'})
assert len(rows)==2 and sum(r['completed_prefix_exports_excluded_in_full'] for r in rows)==29
assert {(r['scene_id'],r['episode_id']) for r in rows}=={('DYehNKdT76V','13'),('6s7QHgap2fW','16')}
target=root/'logs/infrastructure_failed_episodes.jsonl';data=''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in rows)
if target.exists():assert target.read_text()==data,'Preserve existing error episode evidence rather than overwrite it'
else:target.write_text(data)
assert [json.loads(line) for line in target.read_text().splitlines()]==rows
record={'at_utc':datetime.now(timezone.utc).isoformat(),'error_episode_log':str(target.relative_to(OUT)),'error_episode_log_sha256':digest(target),
    'failed_native_attempts':2,'unscored_error_episode_records':2,'completed_prefix_exports_excluded_in_full':29,
    'complete_fixed_replacements_each':20,'recovery_plan':recovery['plan'],'recovery_plan_sha256':recovery['plan_sha256'],
    'Native_or_GPU_workers_started':False,'formal_scores_modified':False,'unavailable_metrics_imputed':False}
(root/'meta/infrastructure-error-episode-log.json').write_text(json.dumps(record,indent=2)+'\n')
section='''
## 失败原生尝试的独立错误回合日志

<!-- exp4-first8-explicit-error-episodes-20261113 -->

| 位置 | 规定 | 实际做法 | 原因 |
| --- | --- | --- | --- |
| 中断回合日志 | 通用约定第3节要求中断或报错回合记录stop_reason=error并单列 | `scopes/first8/logs/infrastructure_failed_episodes.jsonl`单列两个未计分错误回合：DY的下一reset目标13、6s的已观察回合16。前者在创建轨迹前报错，其身份由严格连续的已导出前缀和reset异常推得，日志明确标为推断。后者保留1499条实际动作及阶段索引引用。未知成功值、未核查路径和耗时保留null，不补零、不计入当前640条正式结果。 | 两次原生尝试的状态和worker异常已完整留档；独立错误日志补齐统一episode级入口。两个失败前缀的29条导出仍全部排除，两场景各完整补跑原20条，不挑选导航成败。 |
'''
for path in (OUT/'deviations.md',root/'deviations.md'):
    text=path.read_text()
    if '<!-- exp4-first8-explicit-error-episodes-20261113 -->' not in text:path.write_text(text.rstrip()+'\n'+section)
assert source_fingerprint()==scope['source_fingerprint'] and digest(OUT/'meta/calibration.json')==scope['parameter_sha256']
print(json.dumps(record),flush=True)
