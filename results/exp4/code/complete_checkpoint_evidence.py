"""Prepare complete CP4/CP5 evidence without launching or authorizing stages."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from experiments.exp4_collect import source_fingerprint
from evaluate_sequence import GateError
from report_full_evaluation import verify

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
SELF = Path(__file__)


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def load(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def archive_valid(stage, fingerprint, parameter_sha):
    path = ROOT / f'meta/evaluation-{stage}-log-archive.json'
    if not path.is_file():
        return False
    value = load(path)
    assert value['stage'] == stage and value['native_runs'] == 20 and value['n_episodes'] == 400
    assert value['source_fingerprint'] == fingerprint and value['parameter_sha256'] == parameter_sha
    assert value['source_sha256'] == digest(ROOT / 'code/archive_evaluation_logs.py')
    assert value['ledger_sha256'] == digest(ROOT / f'meta/evaluation-{stage}-runs.json')
    assert len(value['runs']) == 20 and len({row['scene_id'] for row in value['runs']}) == 20
    for row in value['runs']:
        assert row['all_round_trip_sha256_match']
        target = ROOT / row['archive']
        assert target.stat().st_size == row['archive_bytes'] < 95 * 1024 * 1024
        assert digest(target) == row['archive_sha256']
        run = Path(row['native_run'])
        for entry in row['files']:
            assert digest(run / entry['path']) == entry['sha256']
    return True


def complete(checkpoint, evidence):
    names = ('full',) if checkpoint == 'CP4' else ('full', 'nearest', 'myopic', 'suffix_only')
    fingerprint = source_fingerprint()
    parameter_sha = digest(ROOT / 'meta/calibration.json')
    reporter_validation = load(ROOT / 'diagnostics/reporter-scope-validation.json')
    assert reporter_validation['test_exit_code'] == 0
    assert reporter_validation['reporter_sha256'] == digest(ROOT / 'code/report_full_evaluation.py')
    for stage in names:
        ledger = load(ROOT / f'meta/evaluation-{stage}-runs.json')
        assert len(ledger['jobs']) == 20 and all(job['returncode'] == 0 for job in ledger['jobs'])
    destination = ROOT / 'diagnostics' / checkpoint / 'complete-evidence-execution.json'
    if destination.exists():
        old = load(destination)
        assert old['source_sha256'] == digest(SELF) and old['input_sha256'] == evidence
        assert old['source_and_parameters_unchanged'] and not old['native_states_modified']
        for relative, expected in old['output_sha256'].items():
            assert digest(ROOT / relative) == expected, relative
        assert all(archive_valid(stage, fingerprint, parameter_sha) for stage in names)
        return dict(old, reused_existing_execution=True)

    started = datetime.now(timezone.utc)
    work = destination.parent / 'execution-logs' / started.strftime('%Y%m%dT%H%M%S%fZ')
    work.mkdir(parents=True, exist_ok=False)
    steps, outputs, reused_archives = [], set(), []

    def execute(script, arguments, label):
        stdout, stderr = work / (label + '.stdout.log'), work / (label + '.stderr.log')
        with stdout.open('xb') as out, stderr.open('xb') as err:
            result = subprocess.run([sys.executable, str(ROOT / 'code' / script), *arguments],
                cwd=REPO, stdout=out, stderr=err)
        row = {'script': 'code/' + script, 'script_sha256': digest(ROOT / 'code' / script),
            'arguments': arguments, 'exit_code': result.returncode,
            'stdout': stdout.relative_to(ROOT).as_posix(), 'stderr': stderr.relative_to(ROOT).as_posix(),
            'stdout_sha256': digest(stdout), 'stderr_sha256': digest(stderr)}
        steps.append(row)
        outputs.update((row['stdout'], row['stderr']))
        if result.returncode:
            write(work / 'failure.json', {'checkpoint': checkpoint, 'failed_step': row,
                'native_states_modified': False, 'launches_workers': False})
            raise RuntimeError(f'Evidence step failed: {label}; inspect {row["stderr"]}')
        return stdout.read_text()

    environment = json.loads(execute('report_environment_index_parallel.py', [], 'environment'))
    indexed = load(ROOT / environment['proof'])
    assert indexed['source_and_parameters_unchanged'] and not indexed['workers_or_native_states_modified']
    for stage in names:
        coverage = indexed['stages'][stage]
        assert coverage['recorded_role_pairs'] == 20 and not coverage['pending']
        assert coverage['started_scene_assignments'] == 20 and coverage['unlaunched_scenes'] == 0
        assert all(case['gpu'] in (1, 2, 3) for case in coverage['cases'])
    outputs.update((environment['proof'], 'meta/env.txt'))

    for stage in names:
        if archive_valid(stage, fingerprint, parameter_sha):
            reused_archives.append(stage)
        else:
            execute('archive_evaluation_logs.py', ['--stage', stage], 'archive-' + stage)
            assert archive_valid(stage, fingerprint, parameter_sha)
        outputs.add(f'meta/evaluation-{stage}-log-archive.json')

    prepared = json.loads(execute('report_full_evaluation.py', ['--checkpoint', checkpoint], 'report'))
    assert prepared['checkpoint'] == checkpoint and prepared['input_sha256'] == evidence
    assert not prepared['reported_in_chat'] and not prepared['authorizes_next_stage'] and not prepared['launches_workers']
    report = ROOT / f'reports/{checkpoint}.md'
    if checkpoint == 'CP5':
        paired = json.loads(execute('export_paired_episodes.py', [], 'paired-episodes'))
        assert paired['input_sha256'] == evidence and paired['n_pairs'] == 6
        assert paired['n_episodes_each_pair'] == 400
        outputs.add('summary/paired/manifest.json')
        outputs.update(row['path'] for row in paired['pairs'].values())
        report.write_text(report.read_text().rstrip() + '\n\n'
            '六组逐回合配对CSV、各400条，见[配对清单](../summary/paired/manifest.json)。'
            'SPL、PR与PPL保留原始未舍入的0至1小数，成功标记为0或1；'
            '每组四格计数与effects.json一致，导出后已逐值核对CSV往返读取。\n')
    resource = ROOT / 'reports/gpu_scheduling.md'
    assert resource.is_file()
    report.write_text(report.read_text().rstrip() + '\n\n'
        '本次运行期间的GPU调度、共享占用、CPU拥堵与恢复记录见[资源记录](gpu_scheduling.md)。'
        '完整保留这些阶段的回合结果与规划时延。规划耗时按实际经过时间记录，比较各变体时需同时考虑运行条件；'
        '本次差异和2×2效应不额外作为已经隔离资源影响的独立因果证据。\n')
    prepared['output_sha256'][report.relative_to(ROOT).as_posix()] = digest(report)
    prepared['resource_context_augmentation'] = {
        'source': SELF.relative_to(ROOT).as_posix(), 'source_sha256': digest(SELF),
        'resource_report': 'reports/gpu_scheduling.md', 'resource_report_sha256': digest(resource)}
    prepared_path = ROOT / f'meta/{checkpoint}_report_prepared.json'
    write(prepared_path, prepared)
    outputs.update(prepared['output_sha256'])
    outputs.update((prepared_path.relative_to(ROOT).as_posix(), 'reports/gpu_scheduling.md'))
    assert source_fingerprint() == fingerprint and digest(ROOT / 'meta/calibration.json') == parameter_sha
    result = {'checkpoint': checkpoint, 'started_at_utc': started.isoformat(),
        'completed_at_utc': datetime.now(timezone.utc).isoformat(),
        'source': SELF.relative_to(ROOT).as_posix(), 'source_sha256': digest(SELF),
        'input_sha256': evidence, 'source_fingerprint': fingerprint, 'parameter_sha256': parameter_sha,
        'steps': steps, 'archives_reused_after_SHA_checks': reused_archives,
        'output_sha256': {relative: digest(ROOT / relative) for relative in sorted(outputs)},
        'report_sha256': digest(report), 'environment_proof': environment['proof'],
        'native_runs_archived_by_stage': {stage: 20 for stage in names},
        'source_and_parameters_unchanged': True, 'native_states_modified': False,
        'reported_in_chat': False, 'authorizes_next_stage': False, 'launches_workers': False,
        'figures_require_visual_review': checkpoint == 'CP5'}
    write(destination, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', choices=('CP4', 'CP5'), required=True)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    try:
        _, _, evidence = verify(args.checkpoint)
    except (GateError, FileNotFoundError, ValueError, KeyError) as error:
        if not args.check:
            raise
        print(json.dumps({'checkpoint': args.checkpoint, 'ready': False, 'reason': str(error),
            'writes_outputs': False, 'native_states_modified': False, 'launches_workers': False}))
        return
    if args.check:
        print(json.dumps({'checkpoint': args.checkpoint, 'ready': True,
            'writes_outputs': False, 'native_states_modified': False, 'launches_workers': False}))
        return
    print(json.dumps(complete(args.checkpoint, evidence)))


if __name__ == '__main__':
    main()
