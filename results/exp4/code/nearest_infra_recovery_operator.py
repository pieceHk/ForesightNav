"""Retire one entire infrastructure-failed Nearest attempt; never select episodes."""
from datetime import datetime, timezone
from pathlib import Path
import fcntl
import gzip
import hashlib
import json
import os
import shutil
import tarfile

from archive_evaluation_logs import archive_run, file_digest, stream_digest
from evaluate_sequence import OUT, load, source_fingerprint
from report_variants import merge_episode_metric_files
from watch_evaluation import process

def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')

with (OUT/'meta/evaluation-launch.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    start=load(OUT/'meta/nearest-start-a3.json')
    ledger=load(OUT/'meta/evaluation-nearest-runs.json')
    observer=load(OUT/'meta/nearest-observer-a3.json')
    assert start['pid']==1062224 and start['start_ticks']=='22632144'
    assert ledger['status']=='stopped_for_review' and '6s7QHgap2fW, returncode=1' in ledger['error']
    assert not process(start['pid'])['live'] and not process(observer['observer_pid'])['live']
    assert all(not process(job['pid'])['live'] for job in ledger['jobs'])
    fingerprint=source_fingerprint();parameter=file_digest(OUT/'meta/calibration.json')
    assert fingerprint==ledger['source_fingerprint']==start['source_fingerprint']
    assert parameter==ledger['parameter_sha256']==start['parameter_sha256']
    assert len(fingerprint)==161 and parameter=='e1d75f3c9503c7439467748fc97be6ae2ad450c429f40dc7235b1564ff7852ae'
    assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==start['boot_id']
    assert all(not (OUT/path).exists() for path in ('summary/nearest.json','logs/nearest_episodes.jsonl','logs/nearest_decisions.jsonl'))
    backup=OUT/'meta/full-observer-complete-a3.json'
    assert backup.read_bytes()==(OUT/'meta/full-observer-a3.json').read_bytes()
    assert load(backup)['stage']=='full' and load(OUT/'meta/last-evaluation-observer.json')['stage']=='nearest'
    assert load(OUT/'meta/CP4_reported.json')['reported'] is True
    prefix='diagnostics/infrastructure-restarts/nearest-20261111T172644'
    target=OUT/prefix
    assert not target.exists(), 'Inspect existing infrastructure recovery; never duplicate a retirement'
    target.mkdir(parents=True)
    canonical=[OUT/p for p in ('meta/evaluation-nearest-runs.json','meta/nearest-start-a3.json',
        'meta/nearest-observer-a3.json','meta/nearest-shared-gpu-launch-a3.json',
        'meta/last-evaluation-observer.json','logs/evaluation-nearest-controller-a3.log')]
    canonical.extend(Path(job['log']) for job in ledger['jobs'])
    canonical.extend(Path(observer[key]) for key in ('log','stderr_log'))
    assert len(canonical)==len(set(canonical)) and all(p.is_file() for p in canonical)
    evidence={}
    for path in canonical:
        relative=path.relative_to(OUT).as_posix()
        copied=target/'snapshot'/relative;copied.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,copied)
        assert file_digest(copied)==file_digest(path)
        evidence[copied.relative_to(OUT).as_posix()]=file_digest(copied)
    runs=[];logs=[];counts={}
    for job in ledger['jobs']:
        run=Path(job['run_directory'])
        metrics=list(run.glob('results_*/**/per_episode_metrics.json'))
        count=len(merge_episode_metric_files(metrics));counts[job['scene']]=count
        row=archive_run(run,target/'native'/(job['scene']+'.tar.gz'))
        row.update(scene_id=job['scene'],exported_native_records=count)
        runs.append(row);evidence[row['archive']]=row['archive_sha256']
        for path in sorted((run/'logs').glob('*.log')):
            logs.append({'source':path,'name':'native/'+job['scene']+'/'+path.relative_to(run).as_posix(),
                'bytes':path.stat().st_size,'sha256':file_digest(path)})
    assert counts=={'4ok3usBNeis':20,'5cdEh9F2hJL':20,'6s7QHgap2fW':17}
    assert sum(counts.values())==57
    archive=target/'worker-logs.tar.gz'
    with archive.open('xb') as out:
        with gzip.GzipFile(fileobj=out,filename='',mtime=0,compresslevel=6,mode='wb') as compressed:
            with tarfile.open(fileobj=compressed,mode='w|',format=tarfile.PAX_FORMAT) as tar:
                for row in logs:
                    info=tarfile.TarInfo(row['name']);info.size=row['bytes'];info.mode=0o644
                    info.uid=info.gid=info.mtime=0
                    with row['source'].open('rb') as source: tar.addfile(info,source)
    with tarfile.open(archive,'r:gz') as tar:
        assert tar.getnames()==[row['name'] for row in logs]
        for row in logs:
            with tar.extractfile(row['name']) as stream: assert stream_digest(stream)==row['sha256']
            assert file_digest(row['source'])==row['sha256']
    assert archive.stat().st_size<95*1024*1024
    evidence[archive.relative_to(OUT).as_posix()]=file_digest(archive)
    result={'recorded_at_utc':datetime.now(timezone.utc).isoformat(),'stage':'nearest',
        'reason':'Mapper CUDA OOM on physical GPU3; original worker failed and in-place resume unsupported',
        'scope':'Entire failed infrastructure attempt retired, all57 exports retained and excluded together from formal Nearest; a fresh fixed400 is required regardless of previous outcomes',
        'source_fingerprint':fingerprint,'parameter_sha256':parameter,'old_start':start,
        'exported_records_by_scene':counts,'exported_records_total':57,'native_run_archives':runs,
        'worker_log_archive':archive.relative_to(OUT).as_posix(),
        'worker_log_entries':[{k:v for k,v in row.items() if k!='source'} for row in logs],
        'evidence_sha256':evidence,'operator_source_sha256':file_digest(Path(__file__)),
        'archive_round_trip_verified':True,'original_native_runs_retained':True,
        'canonical_retired':False,'new_native_workers_started':False,'formal400_complete':False}
    write(target/'manifest.json',result)
    moves=[]
    for path in canonical:
        relative=path.relative_to(OUT).as_posix();destination=target/'retired'/relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        expected=file_digest(path);assert not destination.exists()
        path.rename(destination);assert file_digest(destination)==expected
        moves.append({'old_relative_path':relative,'retired_path':destination.relative_to(OUT).as_posix(),'sha256':expected})
    with (OUT/'meta/last-evaluation-observer.json').open('xb') as stream: stream.write(backup.read_bytes())
    assert file_digest(OUT/'meta/last-evaluation-observer.json')==file_digest(backup)
    assert source_fingerprint()==fingerprint and file_digest(OUT/'meta/calibration.json')==parameter
    result.update(canonical_retired=True,canonical_moves=moves,
        restored_predecessor_observer_sha256=file_digest(backup),completed_at_utc=datetime.now(timezone.utc).isoformat())
    write(target/'manifest.json',result)
    write(OUT/'meta/nearest-infrastructure-restart.json',{'stage':'nearest','retired_attempt_manifest':prefix+'/manifest.json',
        'retired_attempt_manifest_sha256':file_digest(target/'manifest.json'),'retired_all_exported_records':57,
        'new_fixed_episodes':400,'reason':result['reason'],'source_and_parameters_unchanged':True,
        'restart_criterion':'Infrastructure worker failed, not a navigation success threshold','new_native_workers_started':False})
    report=OUT/'reports/Nearest_infrastructure_restart.md'
    report.write_text('# Nearest 显存故障与整组重跑记录\n\n'
        '2026年11月12日的首次Nearest运行在物理GPU3上发生CUDA显存不足。建图进程申请1.39 GiB时报告空闲1.41 GiB，随后地图重建失败。导航进程在切换回合、调用reset_episode时退出，调度器停止。日志中的GPU0是进程内逻辑编号，原启动参数为物理GPU3、3。\n\n'
        '本次共导出57条：4ok3usBNeis与5cdEh9F2hJL各20条，6s7QHgap2fW为17条。失败场景的第16号回合已经出现失效地图响应，导出条数不能视为健康评测条数。启动器不支持原地恢复，另一个场景在调度器退出后结束，也没有保留下来的Popen退出记录。\n\n'
        '本次尝试的57条整体保留并归档，全部不进入正式Nearest统计。后续按相同固定400条身份、seed和冻结参数重新运行整个Nearest，不拼接旧样本，也不按导航成败挑选重跑回合。Full400与已发布CP4保持原结果。\n\n'
        f'原始身份、轨迹、指标、BLP/SCSP记录及角色环境见[归档清单](../{prefix}/manifest.json)，其中也列出了建图与导航stdout日志的原始SHA及压缩包往返核验。原始运行目录未删除；旧canonical元数据和日志已可逆移入清单列出的retired路径。\n\n'
        '用户已授权清理物理GPU3上的其他用户任务并持续监督。清理按进程身份与实际GPU范围复查，GPU0及GPU4至7不作为信号目标，父进程组不作为清理目标。资源监控按20秒间隔运行，最长24小时，并可通过其控制文件关闭。\n')
    schedule=OUT/'reports/gpu_scheduling.md'
    schedule.write_text(schedule.read_text().rstrip()+'\n\nNearest首次尝试因物理GPU3的mapper CUDA OOM中断。此次57条整体归档，随后完整400条重新运行；具体记录和原始SHA见[恢复记录](Nearest_infrastructure_restart.md)。用户另行授权GPU3的其他用户进程清理与20秒周期监督，GPU0及4至7不作为信号目标。\n')
    print(json.dumps({'manifest':prefix+'/manifest.json','manifest_sha256':file_digest(target/'manifest.json'),
        'raw_records_retired_together':57,'native_run_archives':3,'round_trip_verified':True,
        'canonical_files_retired':len(moves),'next_fixed_episode_count':400,
        'source_and_parameters_unchanged':True,'new_native_workers_started':False,
        'report':'reports/Nearest_infrastructure_restart.md'}))
