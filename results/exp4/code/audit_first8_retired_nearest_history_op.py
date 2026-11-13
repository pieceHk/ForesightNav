"""Audit the earlier discarded Nearest cohort and expose its error episode."""
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import hashlib,json,tarfile
from evaluate_sequence import OUT,load,digest,source_fingerprint
from first8_evaluate import completed

root=OUT/'scopes/first8';scope=load(root/'meta/scope.json')
assert source_fingerprint()==scope['source_fingerprint'] and digest(OUT/'meta/calibration.json')==scope['parameter_sha256']
current=completed('nearest',scope)
info=load(OUT/'meta/nearest-infrastructure-restart.json');manifest_path=OUT/info['retired_attempt_manifest']
assert digest(manifest_path)==info['retired_attempt_manifest_sha256']
old=load(manifest_path)
assert old['source_fingerprint']==scope['source_fingerprint'] and old['parameter_sha256']==scope['parameter_sha256']
assert old['exported_records_total']==info['retired_all_exported_records']==57
assert old['exported_records_by_scene']=={'4ok3usBNeis':20,'5cdEh9F2hJL':20,'6s7QHgap2fW':17}
for name,sha in old['evidence_sha256'].items():assert digest(OUT/name)==sha
audits=[];errors=[]
for record in old['native_run_archives']:
    archive=OUT/record['archive'];assert archive.stat().st_size==record['archive_bytes'] and digest(archive)==record['archive_sha256']
    with tarfile.open(archive,'r:gz') as tar:
        assert tar.getnames()==[r['path'] for r in record['files']]
        metrics={}
        for entry in record['files']:
            raw=tar.extractfile(entry['path']).read();assert len(raw)==entry['bytes'] and hashlib.sha256(raw).hexdigest()==entry['sha256']
            if entry['path'].endswith('/per_episode_metrics.json'):metrics.update(json.loads(raw))
        launch=json.load(tar.extractfile('launch.json'));status=json.load(tar.extractfile('status.json'))
        ids=launch['identities'];scene=ids[0]['scene_id'];expected=[r for r in scope['identities'] if r['scene_id']==scene]
        assert ids==expected and len(ids)==20 and launch['configuration']['seed']==1
        assert launch['configuration']['method']['suffix']['blp_variant']=='nearest'
        assert len(metrics)==old['exported_records_by_scene'][scene]
        assert set(metrics)=={scene+'_'+str(r['runtime_episode_id']) for r in ids[:len(metrics)]}
        new=next(r for r in current['jobs'] if r['scene']==scene);assert new['run_directory']!=record['native_run']
        audits.append({'scene':scene,'retired_native_run':record['native_run'],'archive':record['archive'],
            'archive_sha256':record['archive_sha256'],'inner_files_SHA_verified':len(record['files']),
            'all_exports_excluded':len(metrics),'old_native_status':status['status'],'current_native_run':new['run_directory']})
        if status['status']!='failed':continue
        incomplete=[]
        for name in tar.getnames():
            if not name.endswith('/trajectory.jsonl'):continue
            identity_path=str(Path(name).with_name('identity.json'));identity=json.load(tar.extractfile(identity_path))['identity']
            if scene+'_'+str(identity['runtime_episode_id']) in metrics:continue
            trace=[json.loads(line) for line in tar.extractfile(name).read().splitlines() if line]
            incomplete.append((identity,trace,name))
        assert len(incomplete)<=1
        identity,trace,trace_path=incomplete[0] if incomplete else (ids[len(metrics)],[],None)
        assert identity==ids[len(metrics)]
        counts=Counter(r['subtask_index'] for r in trace);last=trace[-1]['subtask_index'] if trace else 0
        errors.append({'method':'ForesightNav','variant':'nearest','task':'3-MON','scene_id':scene,
            'episode_id':str(identity['runtime_episode_id']),'goals':identity['ordered_categories'],'source_episode_ids':identity['source_episode_ids'],
            'stages':[{'k':k+1,'goal':goal,'reached':bool(counts[k]),'success':None,'steps':counts[k],
                'path_length_m':None,'shortest_path_m':None,'start_pos':None,'end_pos':None,'stop_reason':'error' if k==last else None}
                for k,goal in enumerate(identity['ordered_categories'])],
            'episode_success':None,'total_steps':len(trace),'wallclock_s':None,'code_commit':launch['source_commit'],
            'stop_reason':'error','error_kind':'retired_Nearest_mapper_CUDA_OOM','native_status':status,
            'actual_launcher_wait_proof':None,'canonical_score_included':False,'metrics_unavailable_for_failed_episode':True,
            'identity_status':'observed_in_incomplete_native_identity_and_trajectory' if incomplete else 'next_reset_target_inferred_from_exact_completed_prefix_and_reset_traceback',
            'trajectory_path':trace_path,'native_archive':record['archive'],'native_archive_sha256':record['archive_sha256'],
            'entire_retired_cohort_exports_excluded':57,'failed_scene_exports_excluded':len(metrics)})
worker=OUT/old['worker_log_archive'];worker_sha=old['evidence_sha256'][old['worker_log_archive']];assert digest(worker)==worker_sha
with tarfile.open(worker,'r:gz') as tar:
    assert tar.getnames()==[r['name'] for r in old['worker_log_entries']]
    texts=[]
    for entry in old['worker_log_entries']:
        raw=tar.extractfile(entry['name']).read();assert len(raw)==entry['bytes'] and hashlib.sha256(raw).hexdigest()==entry['sha256']
        texts.append(raw)
assert any(b'CUDA out of memory' in raw for raw in texts) and any(b'reset_episode failed' in raw for raw in texts)
assert len(audits)==3 and sum(r['all_exports_excluded'] for r in audits)==57
assert len(errors)==1 and errors[0]['scene_id']=='6s7QHgap2fW' and errors[0]['episode_id']=='17'
error_path=root/'logs/retired_nearest_failed_episodes.jsonl'
text=''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in errors)
if error_path.exists():assert error_path.read_text()==text
else:error_path.write_text(text)
proof={'at_utc':datetime.now(timezone.utc).isoformat(),'retired_manifest':info['retired_attempt_manifest'],
    'retired_manifest_sha256':info['retired_attempt_manifest_sha256'],'native_archives_deep_verified':audits,
    'worker_log_archive':old['worker_log_archive'],'worker_log_archive_sha256':worker_sha,
    'all57_exports_excluded_together':True,'error_episode_log':str(error_path.relative_to(OUT)),
    'error_episode_log_sha256':digest(error_path),'error_episode_records':1,'current_nearest_episodes':160,
    'old_launcher_exit_codes_not_invented':True,'formal_scores_modified':False,'Native_states_modified':False}
proof_path=root/'meta/retired-nearest-history-audit.json';proof_path.write_text(json.dumps(proof,indent=2)+'\n')
section='''
## Nearest早期废弃尝试与历史执行记录

<!-- exp4-first8-retired-nearest-history-20261113 -->

Nearest在缩减范围前有一次因GPU3显存故障而废弃的尝试，共57条导出，来自两个完整20条场景和一个17条场景前缀。57条全部保留并整体排除，当前Nearest160条来自同一固定身份、seed和冻结版本重新完成后的8个场景。三个旧原生档案及worker日志已重新逐文件SHA核验；旧启动器缺失的真实退出码继续保留为未知。

错误回合另记于[Nearest未计分错误日志](logs/retired_nearest_failed_episodes.jsonl)，stop_reason为error，缺失成功值、路径和耗时保留null。6s下一reset目标17的身份由连续0..16已导出前缀和reset异常推得，日志已标明推断来源。

证据：[本次历史核查](meta/retired-nearest-history-audit.json)、[原Nearest故障报告](../../reports/Nearest_infrastructure_restart.md)、[更早的完整执行偏差](../../deviations.md)。当前Myopic两场景错误日志继续单列，不改写正式成绩。
'''
deviation=root/'deviations.md';before=deviation.read_text()
if '<!-- exp4-first8-retired-nearest-history-20261113 -->' not in before:deviation.write_text(before.rstrip()+'\n'+section)
assert source_fingerprint()==scope['source_fingerprint'] and digest(OUT/'meta/calibration.json')==scope['parameter_sha256']
print(json.dumps({'proof':str(proof_path.relative_to(OUT)),'retired_Nearest_native_archives_deep_verified':3,
    'all_exported_records_excluded':57,'separate_unscored_error_episode':errors[0]['episode_id'],
    'error_identity_source':errors[0]['identity_status'],'formal_scores_modified':False,'Native_states_modified':False}))
