"""Deterministically compress the six complete calibration logs for GitHub."""
import gzip
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def digest(stream):
    value = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
        value.update(chunk)
    return value.hexdigest()


def main():
    ledger = json.loads((ROOT / 'meta/calibration_runs.json').read_text())
    assert ledger['status'] == 'completed_three_folds_pending_CP2a'
    entries = []
    for f in range(3):
        for kind in ('events', 'snapshots'):
            relative = f'logs/calibration_fold{f}_{kind}.jsonl'
            source = ROOT / relative
            target = source.with_suffix(source.suffix + '.gz')
            with source.open('rb') as raw:
                raw_hash = digest(raw)
            with source.open('rb') as raw, target.open('wb') as compressed:
                with gzip.GzipFile(fileobj=compressed, mode='wb', filename='', mtime=0, compresslevel=6) as archive:
                    shutil.copyfileobj(raw, archive)
            with gzip.open(target, 'rb') as unpacked:
                assert digest(unpacked) == raw_hash
            with target.open('rb') as compressed:
                compressed_hash = digest(compressed)
            entries.append({'raw': relative, 'gzip': relative + '.gz',
                            'raw_bytes': source.stat().st_size, 'gzip_bytes': target.stat().st_size,
                            'raw_sha256': raw_hash, 'gzip_sha256': compressed_hash,
                            'round_trip_sha256_matches': True})
    result = {'scope': 'complete three-fold calibration events and replay snapshots; raw files retained',
              'gzip_settings': {'level': 6, 'mtime': 0, 'filename': ''},
              'raw_bytes_total': sum(x['raw_bytes'] for x in entries),
              'gzip_bytes_total': sum(x['gzip_bytes'] for x in entries), 'files': entries}
    (ROOT / 'meta/calibration-log-archive.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'files'}, indent=2))


if __name__ == '__main__':
    main()
