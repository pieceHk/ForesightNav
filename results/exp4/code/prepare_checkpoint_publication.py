"""Stage complete, verified CP4/CP5 evidence in the publication checkout."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

from complete_checkpoint_evidence import archive_valid
from evaluate_sequence import GateError
from experiments.exp4_collect import source_fingerprint
from report_full_evaluation import verify

ROOT = Path(__file__).resolve().parents[1]
PUBLICATION = Path('/nvme/khy/ForesightNav_exp4_evalprep')
LIMIT = 95 * 1024 * 1024


def load(path):
    return json.loads(path.read_text())


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def safe_source(relative):
    path = ROOT / relative
    require(not Path(relative).is_absolute() and '..' not in Path(relative).parts,
            'Unsafe publication path: ' + relative)
    require('.private' not in path.name and path.is_file() and not path.is_symlink()
            and path.resolve().is_relative_to(ROOT.resolve()), 'Unsafe or missing evidence: ' + relative)
    require(path.stat().st_size < LIMIT, 'Evidence requires a lossless publication representation before staging: ' + relative)
    return path


def evidence(checkpoint):
    _, _, inputs = verify(checkpoint)
    names = ('full',) if checkpoint == 'CP4' else ('full', 'nearest', 'myopic', 'suffix_only')
    execution_path = ROOT / f'diagnostics/{checkpoint}/complete-evidence-execution.json'
    require(execution_path.is_file(), 'Complete checkpoint evidence has not been executed')
    execution = load(execution_path)
    prepared = load(ROOT / f'meta/{checkpoint}_report_prepared.json')
    fingerprint = source_fingerprint()
    parameter = digest(ROOT / 'meta/calibration.json')
    require(execution['input_sha256'] == prepared['input_sha256'] == inputs, 'Stale checkpoint inputs')
    require(execution['checkpoint'] == prepared['checkpoint'] == checkpoint, 'Wrong checkpoint')
    require(execution['source_and_parameters_unchanged'] and not execution['native_states_modified']
            and not execution['launches_workers'] and not execution['authorizes_next_stage'], 'Unexpected execution scope')
    require(execution['source_fingerprint'] == fingerprint and execution['parameter_sha256'] == parameter,
            'Runtime source or frozen calibration differs')
    require(digest(ROOT / execution['source']) == execution['source_sha256'], 'Evidence runner source differs')
    for step in execution['steps']:
        require(step['exit_code'] == 0 and digest(ROOT / step['script']) == step['script_sha256'], 'Failed or changed evidence step')
    files = set(inputs) | set(execution['output_sha256']) | set(prepared['output_sha256'])
    for relative, expected in {**inputs, **execution['output_sha256'], **prepared['output_sha256']}.items():
        require(digest(safe_source(relative)) == expected, 'Checkpoint artifact changed: ' + relative)
    report = f'reports/{checkpoint}.md'
    require(digest(ROOT / report) == execution['report_sha256'], 'Checkpoint report changed')
    files.update((report, execution_path.relative_to(ROOT).as_posix(), f'meta/{checkpoint}_report_prepared.json',
                  'code/prepare_checkpoint_publication.py', 'code/verify_checkpoint_archive.py', execution['source'],
                  'code/archive_evaluation_logs.py', 'code/report_environment_index_parallel.py'))
    for stage in names:
        ledger = load(ROOT / f'meta/evaluation-{stage}-runs.json')
        require(all(job['status'] == 'completed' and job['returncode'] == 0 for job in ledger['jobs']), 'Incomplete native exits')
        require(archive_valid(stage, fingerprint, parameter), 'Raw log archive is not verified: ' + stage)
        archive = load(ROOT / f'meta/evaluation-{stage}-log-archive.json')
        files.add(f'meta/evaluation-{stage}-log-archive.json')
        files.update(row['archive'] for row in archive['runs'])
    if checkpoint == 'CP5':
        qa_path = ROOT / 'diagnostics/CP5/figure-visual-review.json'
        require(qa_path.is_file(), 'Actual CP5 figure inspection has not been recorded')
        qa = load(qa_path)
        require(qa['actual_visual_inspection'] is True and qa['report_sha256'] == digest(ROOT / report), 'Stale or missing figure inspection')
        expected = {'figures/CP5/SR_SPL.png', 'figures/CP5/effects.png'}
        require(set(qa['figure_sha256']) == expected, 'Incomplete figure review coverage')
        for relative, value in qa['figure_sha256'].items():
            require(digest(ROOT / relative) == value, 'Reviewed figure changed: ' + relative)
        files.add(qa_path.relative_to(ROOT).as_posix())
    previous = load(ROOT / 'meta/full-prefix-sync-files.json')
    require(previous['remote_verified'] and not previous['formal400_complete'], 'Previous diagnostic publication is unverified')
    files.update(previous['files'])
    # Keep the exact diagnostic prefix snapshot; CP4/CP5 are the formal results.
    for relative in ('meta/evaluation-gpu-policy.json', 'meta/full-start-a3.json', 'meta/full-observer-a3.json'):
        files.add(relative)
    for relative in files:
        safe_source(relative)
    require(source_fingerprint() == fingerprint and digest(ROOT / 'meta/calibration.json') == parameter,
            'Source or calibration changed during publication verification')
    return files, inputs, names, fingerprint, parameter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', choices=('CP4', 'CP5'), required=True)
    parser.add_argument('--parent')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    try:
        checked = evidence(args.checkpoint)
    except (GateError, FileNotFoundError, ValueError, KeyError) as error:
        if not args.check:
            raise
        print(json.dumps({'checkpoint': args.checkpoint, 'ready': False, 'reason': str(error),
                          'writes_outputs': False, 'launches_workers': False, 'native_states_modified': False}))
        return
    if args.check:
        print(json.dumps({'checkpoint': args.checkpoint, 'ready': True, 'writes_outputs': False,
                          'launches_workers': False, 'native_states_modified': False}))
        return
    require(args.parent is not None and re.fullmatch('[a-f0-9]{40}', args.parent), 'Exact publication parent is required')
    files, inputs, names, fingerprint, parameter = checked
    git = lambda *parts: subprocess.check_output(['git', *parts], cwd=PUBLICATION, text=True).strip()
    require(git('rev-parse', 'HEAD') == args.parent, 'Publication parent changed; inspect before retrying')
    require(git('branch', '--show-current') == 'exp4/blp-2x2-a3', 'Unexpected publication branch')
    require(not git('status', '--porcelain', '--untracked-files=no'), 'Publication checkout has tracked changes')
    out = PUBLICATION / 'results/exp4'
    for relative in sorted(files):
        target = out / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(safe_source(relative), target)
        require(digest(target) == digest(ROOT / relative), 'Publication copy failed: ' + relative)
    manifest = {'generated_at_utc': datetime.now(timezone.utc).isoformat(), 'checkpoint': args.checkpoint,
        'branch': 'exp4/blp-2x2-a3', 'publication_parent': args.parent,
        'commit_message': 'results(exp4): publish complete ' + args.checkpoint + ' evaluation evidence',
        'archive_path': '/tmp/exp4-a3-' + args.checkpoint + '-sync.tar', 'formal400_complete': True,
        'completed_variants': list(names), 'n_episodes_by_variant': {name: 400 for name in names},
        'input_sha256': inputs, 'source_fingerprint': fingerprint, 'parameter_sha256': parameter,
        'files': sorted(files), 'sha256': {relative: digest(out / relative) for relative in sorted(files)},
        'remote_verified': False, 'native_states_modified': False, 'launches_workers': False,
        'reported_in_chat': False, 'authorizes_next_stage': False,
        'source': 'code/prepare_checkpoint_publication.py', 'source_sha256': digest(Path(__file__))}
    require(source_fingerprint() == fingerprint and digest(ROOT / 'meta/calibration.json') == parameter,
            'Source or calibration changed during publication preparation')
    path = ROOT / f'meta/{args.checkpoint}-publication-files.json'
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'checkpoint': args.checkpoint, 'prepared_files': len(files), 'formal400_complete': True,
        'n_episodes_by_variant': manifest['n_episodes_by_variant'], 'manifest': path.relative_to(ROOT).as_posix(),
        'native_states_modified': False, 'launches_workers': False, 'authorizes_next_stage': False}))


if __name__ == '__main__':
    main()
