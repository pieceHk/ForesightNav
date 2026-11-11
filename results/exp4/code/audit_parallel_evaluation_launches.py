"""Audit launched evaluation scenes without changing native state or inputs."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from evaluate_sequence import (OUT, audit_scene, build_jobs, expected_identities,
                               source_fingerprint)
from watch_evaluation import process


def load(path):
    return Path(path).read_bytes()


def read(path):
    return json.loads(load(path))


def sha(path):
    return hashlib.sha256(load(path)).hexdigest()


def identity_map(rows):
    indexed = {(str(row['scene_id']), str(row['runtime_episode_id'])): row for row in rows}
    assert len(indexed) == len(rows), 'Duplicate native episode identity'
    return indexed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('full', 'nearest', 'myopic', 'suffix_only'), required=True)
    args = parser.parse_args()
    stage, root = args.stage, OUT
    ledger_path = root / f'meta/evaluation-{stage}-runs.json'
    ledger_bytes = load(ledger_path)
    ledger = json.loads(ledger_bytes)
    assert ledger['stage'] == stage and ledger['attempt'] == 'a3'
    assert ledger['status'] in ('running', 'completed_pending_process_audit',
                                'completed', 'completed_requires_score_review')
    parameters_before = sha(root / 'meta/calibration.json')
    native_before = source_fingerprint()
    accepted = read(root / 'meta/CP3_acceptance.json')
    assert accepted['accepted'] is True
    assert native_before == ledger['source_fingerprint'] == accepted['source_fingerprint']
    assert parameters_before == ledger['parameter_sha256'] == accepted['parameter_sha256']
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    assert boot == ledger['boot_id']
    start = read(root / f'meta/{stage}-start-a3.json')
    assert start['pid'] == ledger['controller_pid'] and start['boot_id'] == boot
    controller = process(ledger['controller_pid'])
    if ledger['status'] == 'running':
        assert controller['live'] and controller['start_ticks'] == start['start_ticks']
        assert f'--stage {stage}' in controller['command']
    expected = {job['scene']: identity_map(job['identities'])
                for job in build_jobs(stage, expected_identities(stage))}
    seen, records = set(), []
    for job in ledger['jobs']:
        scene, gpu = job['scene'], job['gpu']
        assert scene in expected and scene not in seen and gpu in ledger.get('allowed_gpus', (2, 3))
        seen.add(scene)
        wanted = expected[scene]
        assert identity_map(job['identities']) == wanted
        argv = job['command']
        assert argv[argv.index('--gpus') + 1] == f'{gpu},{gpu}'
        record = {'scene': scene, 'gpu': gpu, 'ledger_status': job['status']}
        run = Path(job['run_directory']) if job.get('run_directory') else None
        if run is None or not (run / 'launch.json').exists():
            assert job['status'] == 'running'
            record['state'] = 'waiting_for_native_launch_metadata'
            records.append(record)
            continue
        plan = read(run / 'launch.json')
        observed = identity_map(plan['identities'])
        assert set(observed) == set(wanted) and len(observed) == 20
        for identity, row in observed.items():
            for key in ('ordered_categories', 'source_episode_ids'):
                assert row[key] == wanted[identity][key], (identity, key)
        assert plan['source_commit'] == start['source_commit']
        configuration = plan['configuration']
        assert configuration['seed'] == 1
        assert configuration['method']['suffix']['blp_variant'] == stage
        launcher = process(job['pid'])
        if job['status'] == 'completed':
            assert job['returncode'] == 0
            record['native_completion_audit'] = audit_scene(run, job)
            record['state'] = 'completed'
        elif job['status'] == 'running' and launcher['live']:
            assert f'--tag exp4-{stage}-a3-{scene}' in launcher['command']
            record.update(state='running', launcher_start_ticks=launcher['start_ticks'])
        else:
            assert job['status'] == 'running'
            record['native_completion_audit'] = audit_scene(run, job)
            record['state'] = 'native_complete_waiting_controller_reap'
        record.update(native_identities=20, categories_and_source_ids_match=True,
                      seed=1, source_commit=plan['source_commit'], launcher_pid=job['pid'],
                      launcher_live_at_read=launcher['live'], launch_sha256=sha(run / 'launch.json'))
        records.append(record)
    assert source_fingerprint() == native_before
    assert sha(root / 'meta/calibration.json') == parameters_before
    now = datetime.now(timezone.utc)
    proof = {'checked_at_utc': now.isoformat(), 'stage': stage, 'diagnostic_only': True,
             'scope': 'Actual launched scene plans, live launchers and completed native scenes; does not certify aggregate stage metrics',
             'ledger_sha256_at_read': hashlib.sha256(ledger_bytes).hexdigest(),
             'auditor_sha256': sha(__file__), 'parameter_sha256': parameters_before,
             'source_and_parameters_unchanged': True, 'controller_identity_at_read': controller,
             'expected_stage_scenes': 20, 'expected_stage_episodes': 400,
             'checked_native_identities': sum(row.get('native_identities', 0) for row in records),
             'scene_launches': records, 'native_states_modified': False, 'allowed_gpus': ledger.get('allowed_gpus', [2, 3]),
             'scheduling': ledger.get('scheduling')}
    output = root / 'diagnostics' / stage / 'parallel-scene-launch-audits' / (now.strftime('%Y%m%dT%H%M%S%fZ') + '.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(proof, indent=2) + '\n')
    print(json.dumps({'proof': str(output.relative_to(root)), **proof}))


if __name__ == '__main__':
    main()
