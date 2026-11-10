"""Archive complete native evaluation evidence, preserving exact file bytes."""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import tarfile

from experiments.exp4_collect import source_fingerprint

ROOT = Path(__file__).resolve().parents[1]


def stream_digest(stream):
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
        digest.update(chunk)
    return digest.hexdigest()


def file_digest(path):
    with path.open('rb') as stream:
        return stream_digest(stream)


def archive_run(run, target):
    names = ('trajectory.jsonl', 'identity.json', 'blp_events.jsonl', 'blp_snapshots.jsonl',
             'blp_decisions.jsonl', 'blp_errors.jsonl', 'scsp_decisions.jsonl',
             'per_episode_metrics.json', 'mapping_environment.json', 'navigation_environment_*.json')
    selected = {run / name for name in ('launch.json', 'status.json', 'source_snapshot.json')}
    for name in names:
        selected.update(run.rglob(name))
    assert all(path.is_file() and path.resolve().is_relative_to(run.resolve()) for path in selected)
    for required in ('trajectory.jsonl', 'blp_snapshots.jsonl', 'blp_decisions.jsonl', 'per_episode_metrics.json'):
        assert any(path.name == required for path in selected), required
    entries = [{'path': path.relative_to(run).as_posix(), 'bytes': path.stat().st_size,
                'sha256': file_digest(path)} for path in sorted(selected)]
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('wb') as output:
        with gzip.GzipFile(fileobj=output, filename='', mtime=0, compresslevel=6, mode='wb') as compressed:
            with tarfile.open(fileobj=compressed, mode='w|', format=tarfile.PAX_FORMAT) as archive:
                for entry in entries:
                    info = tarfile.TarInfo(entry['path'])
                    info.size, info.mode = entry['bytes'], 0o644
                    info.uid, info.gid, info.mtime = 0, 0, 0
                    with (run / entry['path']).open('rb') as source:
                        archive.addfile(info, source)
    with tarfile.open(target, mode='r:gz') as archive:
        assert archive.getnames() == [entry['path'] for entry in entries]
        for entry in entries:
            with archive.extractfile(entry['path']) as stream:
                assert stream_digest(stream) == entry['sha256']
            assert file_digest(run / entry['path']) == entry['sha256']
    assert target.stat().st_size < 95 * 1024 * 1024, 'Split large evidence before GitHub upload'
    return {'native_run': str(run), 'archive': target.relative_to(ROOT).as_posix(),
            'archive_sha256': file_digest(target), 'archive_bytes': target.stat().st_size,
            'raw_bytes': sum(entry['bytes'] for entry in entries), 'files': entries,
            'all_round_trip_sha256_match': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('smoke', 'full', 'nearest', 'myopic', 'suffix_only'), default='smoke')
    args = parser.parse_args()
    ledger_path = ROOT / f'meta/evaluation-{args.stage}-runs.json'
    ledger = json.loads(ledger_path.read_text())
    assert ledger['status'] in ('completed', 'completed_requires_score_review')
    assert len(ledger['jobs']) == 20 and all(job['status'] == 'completed' for job in ledger['jobs'])
    assert ledger['source_fingerprint'] == source_fingerprint()
    assert ledger['parameter_sha256'] == file_digest(ROOT / 'meta/calibration.json')
    assert ledger['n_episodes'] == (20 if args.stage == 'smoke' else 400) and ledger['n_error'] == 0
    prefix = 'logs/CP3/raw' if args.stage == 'smoke' else 'logs/native/' + args.stage
    runs = []
    for job in ledger['jobs']:
        entry = archive_run(Path(job['run_directory']), ROOT / prefix / (job['scene'] + '.tar.gz'))
        entry['scene_id'] = job['scene']
        runs.append(entry)
    assert source_fingerprint() == ledger['source_fingerprint']
    assert file_digest(ROOT / 'meta/calibration.json') == ledger['parameter_sha256']
    result = {'generated_at_utc': datetime.now(timezone.utc).isoformat(), 'stage': args.stage,
              'scope': 'Exact native launch/status/source metadata, episode identities, metrics, trajectories, BLP and SCSP records, and role environments; original files retained',
              'excluded': 'Images, model/map assets, private runtime YAML and unrelated host diagnostics',
              'gzip': {'level': 6, 'mtime': 0, 'filename': ''},
              'tar_headers': {'mtime': 0, 'uid': 0, 'gid': 0, 'mode': '0644'},
              'ledger_sha256': file_digest(ledger_path), 'parameter_sha256': ledger['parameter_sha256'],
              'source_fingerprint': ledger['source_fingerprint'], 'source_sha256': file_digest(Path(__file__)),
              'native_runs': len(runs), 'n_episodes': ledger['n_episodes'],
              'raw_bytes_total': sum(entry['raw_bytes'] for entry in runs),
              'archive_bytes_total': sum(entry['archive_bytes'] for entry in runs), 'runs': runs}
    destination = ROOT / f'meta/evaluation-{args.stage}-log-archive.json'
    destination.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ('stage', 'native_runs', 'n_episodes', 'raw_bytes_total', 'archive_bytes_total')}))


if __name__ == '__main__':
    main()
