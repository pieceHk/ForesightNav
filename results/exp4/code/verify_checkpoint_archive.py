"""Verify a published complete checkpoint archive and restore exact evidence bytes."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--bytes', required=True, type=int)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--files', required=True, type=int)
    parser.add_argument('--checkpoint', required=True, choices=('CP4', 'CP5'))
    parser.add_argument('--destination', required=True, type=Path)
    args = parser.parse_args()
    assert args.archive.stat().st_size == args.bytes and digest(args.archive) == args.sha256
    relative_manifest = f'meta/{args.checkpoint}-publication-files.json'
    with tarfile.open(args.archive, 'r:') as archive:
        members = archive.getmembers()
        assert len({m.name for m in members}) == len(members)
        for member in members:
            assert member.isfile() and not Path(member.name).is_absolute() and '..' not in Path(member.name).parts
        manifest = json.load(archive.extractfile(relative_manifest))
        names = ['full'] if args.checkpoint == 'CP4' else ['full', 'nearest', 'myopic', 'suffix_only']
        assert manifest['checkpoint'] == args.checkpoint and manifest['formal400_complete'] is True
        assert manifest['publication_commit'] == args.commit and manifest['remote_verified'] is True
        assert manifest['completed_variants'] == names and manifest['n_episodes_by_variant'] == {n: 400 for n in names}
        assert len(manifest['files']) == args.files
        assert set(m.name for m in members) == set(manifest['files']) | {relative_manifest}
        # Verify every archived member before writing any evidence to the mirror.
        for relative in manifest['files']:
            value = hashlib.sha256()
            with archive.extractfile(relative) as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    value.update(chunk)
            assert value.hexdigest() == manifest['sha256'][relative], relative
        destination = args.destination.resolve()
        destination.mkdir(parents=True, exist_ok=True)
        for member in members:
            target = destination / member.name
            assert not target.is_symlink() and target.resolve().is_relative_to(destination)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + '.checkpoint-tmp')
            assert not temporary.is_symlink()
            with archive.extractfile(member) as source, temporary.open('wb') as output:
                shutil.copyfileobj(source, output)
            temporary.replace(target)
    for relative, expected in manifest['sha256'].items():
        assert digest(destination / relative) == expected, relative
    for name in names:
        summary = json.loads((destination / f'summary/{name}.json').read_text())
        assert summary['n_episodes'] == 400 and summary['n_error'] == 0
        assert summary['process']['native_stage_path_check_count'] == 1200 and summary['process']['invalid_path_complete']
        with (destination / f'logs/{name}_episodes.jsonl').open() as stream:
            assert sum(bool(line.strip()) for line in stream) == 400
    print(json.dumps({'checkpoint': args.checkpoint, 'commit': args.commit, 'verified_files': args.files,
        'archive_bytes': args.bytes, 'archive_sha256': args.sha256, 'formal400_complete': True,
        'n_episodes_by_variant': manifest['n_episodes_by_variant'], 'all_local_files_SHA_verified': True}))


if __name__ == '__main__':
    main()
