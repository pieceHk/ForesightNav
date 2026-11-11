"""Preserve the user-interrupted GPU0 initialization as excluded raw evidence."""
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import tarfile

from archive_evaluation_logs import file_digest, stream_digest
from experiments.exp4_collect import source_fingerprint
from report_variants import merge_episode_metric_files
from watch_evaluation import process

ROOT = Path(__file__).resolve().parents[1]


def main():
    release_path = ROOT/'meta/gpu0-withdrawal-a3.json'
    release = json.loads(release_path.read_text())
    assert release['gpu0_native_completed_at_withdrawal'] == 0
    assert release['raw_logs_retained'] and release['source_and_parameters_unchanged']
    for entry in release['signals']:
        current = process(entry['pid'])
        assert not (current.get('live') and current.get('start_ticks') == entry['start_ticks'])
    run = Path(release['job']['run_directory'])
    assert run.resolve().is_relative_to((ROOT/'runs').resolve())
    accepted = json.loads((ROOT/'meta/CP2b_acceptance.json').read_text())
    fingerprint = source_fingerprint()
    assert fingerprint == accepted['source_fingerprint']
    parameter_sha = file_digest(ROOT/'meta/calibration.json')
    assert parameter_sha == accepted['parameter_sha256']
    metrics = list(run.glob('results_*/**/per_episode_metrics.json'))
    assert not merge_episode_metric_files(metrics)
    names = ('launch.json', 'status.json', 'source_snapshot.json', 'mon_metrics.json',
        'trajectory.jsonl', 'identity.json', 'blp_events.jsonl', 'blp_snapshots.jsonl',
        'blp_decisions.jsonl', 'blp_errors.jsonl', 'scsp_decisions.jsonl',
        'per_episode_metrics.json', 'mapping_environment.json', 'navigation_environment_*.json')
    selected = set()
    for name in names:
        selected.update(run.rglob(name))
    for required in ('launch.json','status.json','source_snapshot.json'):
        assert (run/required) in selected
    assert all(path.is_file() and path.resolve().is_relative_to(run.resolve()) for path in selected)
    entries = [{'path':path.relative_to(run).as_posix(), 'bytes':path.stat().st_size,
                'sha256':file_digest(path)} for path in sorted(selected)]
    relative = 'logs/native/resource_interruptions/full-cvZr5TUy5C5-gpu0-initialization.tar.gz'
    target = ROOT/relative
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('xb') as output:
        with gzip.GzipFile(fileobj=output,filename='',mtime=0,compresslevel=6,mode='wb') as compressed:
            with tarfile.open(fileobj=compressed,mode='w|',format=tarfile.PAX_FORMAT) as archive:
                for entry in entries:
                    info = tarfile.TarInfo(entry['path'])
                    info.size,info.mode = entry['bytes'],0o644
                    info.uid,info.gid,info.mtime = 0,0,0
                    with (run/entry['path']).open('rb') as source:
                        archive.addfile(info,source)
    with tarfile.open(target,mode='r:gz') as archive:
        assert archive.getnames() == [entry['path'] for entry in entries]
        for entry in entries:
            with archive.extractfile(entry['path']) as stream:
                assert stream_digest(stream) == entry['sha256']
            assert file_digest(run/entry['path']) == entry['sha256']
    assert source_fingerprint() == fingerprint and file_digest(ROOT/'meta/calibration.json') == parameter_sha
    result = {'generated_at_utc':datetime.now(timezone.utc).isoformat(),
        'scope':'User-withdrawn GPU0 initialization only; excluded from formal400; all original files retained',
        'diagnostic_only':True,'formal400_complete':False,'n_completed_native_episodes':0,
        'scene_id':release['scene'],'native_run':str(run),'release_path':str(release_path.relative_to(ROOT)),
        'release_sha256':file_digest(release_path),'archive':relative,'archive_sha256':file_digest(target),
        'archive_bytes':target.stat().st_size,'raw_bytes':sum(entry['bytes'] for entry in entries),
        'files':entries,'all_round_trip_sha256_match':True,'source_sha256':file_digest(Path(__file__)),
        'native_source_entries':len(fingerprint),'source_and_parameters_unchanged':True,
        'parameter_sha256':parameter_sha,'native_workers_signaled':False,
        'excluded':'Images, model/map files, private runtime YAML and raw terminal output; originals remain on server'}
    path = ROOT/'meta/gpu0-initialization-log-archive.json'
    with path.open('x') as stream:
        stream.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:result[key] for key in ('archive','archive_sha256','archive_bytes','raw_bytes',
        'n_completed_native_episodes','all_round_trip_sha256_match','source_and_parameters_unchanged')}))


if __name__ == '__main__':
    main()
