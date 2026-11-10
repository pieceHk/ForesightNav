"""Index recorded role environments without importing frameworks or changing workers."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

from audit_smoke_environment import signature
from experiments.exp4_collect import source_fingerprint

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    now = datetime.now(timezone.utc)
    cp2 = load(ROOT / 'meta/CP2b_acceptance.json')
    cp3 = load(ROOT / 'meta/CP3_acceptance.json')
    fingerprint = source_fingerprint()
    parameter_sha = sha(ROOT / 'meta/calibration.json')
    assert cp2['accepted'] is True and cp3['accepted'] is True
    assert fingerprint == cp2['source_fingerprint'] == cp3['source_fingerprint']
    assert parameter_sha == cp2['parameter_sha256'] == cp3['parameter_sha256']
    reference_path = ROOT / 'meta/environment-smoke.json'
    assert sha(reference_path) == cp3['evidence_sha256']['meta/environment-smoke.json']
    reference = load(reference_path)
    assert reference['completed_cases_checked'] == 20
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    assert boot == reference['boot_id'] and head == reference['runtime_commit']
    hardware = subprocess.check_output(['nvidia-smi',
        '--query-gpu=index,uuid,name,driver_version,memory.total',
        '--format=csv,noheader,nounits'], text=True).splitlines()
    hardware = [line for line in hardware if int(line.split(',')[0]) in (2, 3)]
    assert hardware == reference['gpu_hardware_and_driver']
    stages = {}
    for stage in ('smoke', 'full', 'nearest', 'myopic', 'suffix_only'):
        identities = load(ROOT / 'meta' / ('smoke-20-identities.json' if stage == 'smoke' else '3mon-identities.json'))
        expected_scenes = {row['scene_id'] for row in identities}
        assert len(expected_scenes) == 20
        ledger_path = ROOT / f'meta/evaluation-{stage}-runs.json'
        if not ledger_path.exists():
            stages[stage] = {'status': 'not_launched', 'recorded_role_pairs': 0,
                'expected_scenes': 20, 'started_scene_assignments': 0, 'unlaunched_scenes': 20, 'pending': []}
            continue
        ledger = load(ledger_path)
        assert ledger['source_fingerprint'] == fingerprint
        assert ledger['parameter_sha256'] == parameter_sha and ledger['boot_id'] == boot
        assigned = {job['scene'] for job in ledger['jobs']}
        assert assigned <= expected_scenes and len(assigned) == len(ledger['jobs'])
        cases, pending = [], []
        for job in ledger['jobs']:
            run = Path(job['run_directory']) if job.get('run_directory') else None
            if run is None or not (run / 'launch.json').is_file():
                pending.append({'scene': job['scene'], 'reason': 'awaiting_launch_metadata'})
                continue
            launch = load(run / 'launch.json')
            assert launch['source_commit'] == head and launch['configuration']['seed'] == 1
            artifacts, roles = {}, {}
            for path in sorted(run.glob('*environment*.json')):
                value = load(path)
                role = value['role']
                assert role in ('mapping', 'navigation') and role not in roles
                assert value['launch_source_commit'] == head
                assert value['configuration_sha256'] == launch['configuration_sha256']
                assert value['identity_sha256'] == launch['identity_sha256']
                assert value['cuda_visible_devices'] == str(job['gpu'])
                roles[role] = signature(value)
                assert roles[role] == reference['role_signatures'][role]
                artifacts[path.name] = sha(path)
            if set(roles) != {'mapping', 'navigation'}:
                pending.append({'scene': job['scene'], 'reason': 'awaiting_role_environment_records'})
                continue
            cases.append({'scene': job['scene'], 'gpu': job['gpu'],
                'native_status_at_read': job['status'], 'launch_sha256': sha(run / 'launch.json'),
                'role_signatures': roles, 'environment_artifact_sha256': artifacts})
        stages[stage] = {'status': ledger['status'], 'ledger_sha256_at_read': sha(ledger_path),
            'expected_scenes': len(expected_scenes), 'started_scene_assignments': len(assigned),
            'unlaunched_scenes': len(expected_scenes - assigned), 'recorded_role_pairs': len(cases),
            'recorded_environment_files': sum(len(row['environment_artifact_sha256']) for row in cases),
            'cases': cases, 'pending': pending}
    assert source_fingerprint() == fingerprint and sha(ROOT / 'meta/calibration.json') == parameter_sha
    relative = f'diagnostics/environment-index/{now.strftime("%Y%m%dT%H%M%S%fZ")}.json'
    result = {'recorded_at_utc': now.isoformat(), 'runtime_commit': head, 'boot_id': boot,
        'source_fingerprint_entries': len(fingerprint), 'parameter_sha256': parameter_sha,
        'source_and_parameters_unchanged': True, 'workers_or_native_states_modified': False,
        'reference_path': 'meta/environment-smoke.json', 'reference_sha256': sha(reference_path),
        'reference_recorded_at_utc': reference['recorded_at_utc'],
        'source_sha256': sha(Path(__file__)),
        'role_signature_implementation_sha256': sha(ROOT / 'code/audit_smoke_environment.py'),
        'gpu_hardware_and_driver_checked_at_utc': now.isoformat(),
        'gpu_hardware_and_driver': hardware, 'stages': stages,
        'scope': 'Recorded native role inventories available at this time; pending and unlaunched scenes are not certified. Framework versions reuse the accepted CPU-only smoke probe; no frameworks reloaded.'}
    destination = ROOT / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x') as handle:
        handle.write(json.dumps(result, indent=2) + '\n')
    lines = ['Exp4 environment index', f'Generated UTC: {now.isoformat()}',
        f'Runtime commit: {head}', f'Boot ID: {boot}',
        f'Frozen calibration SHA256: {parameter_sha}',
        'Seed: 1', 'Physical GPUs: 2, 3; mapping/navigation colocated per scene',
        f'Accepted environment reference: meta/environment-smoke.json (SHA256 {sha(reference_path)})',
        f'Framework probe recorded UTC: {reference["recorded_at_utc"]}',
        'Framework import probe was CPU-only; native GPU assignments are checked from role records.',
        'Frameworks:']
    for role, values in reference['frameworks'].items():
        lines.append(f'  {role}: Python {values["python_version"]}; torch {values["torch_version"]}; torch CUDA {values["torch_cuda_version"]}; executable {values["python_executable"]}')
        lines.append(f'  {role} Python/lock/package inventory signature: {reference["role_signatures"][role]}')
    lines += [f'GPU hardware/driver checked UTC: {now.isoformat()}', *['  ' + row for row in hardware],
        'Recorded environment coverage (not episode completion):']
    for stage, value in stages.items():
        lines.append(f'  {stage}: {value["status"]}; {value["recorded_role_pairs"]} role pairs; '
            f'{value["started_scene_assignments"]}/{value["expected_scenes"]} fixed scenes started; '
            f'{value["unlaunched_scenes"]} unlaunched; {len(value["pending"])} pending environment records')
    lines += [f'Exact recorded-file SHA256 and per-scene evidence: {relative}',
        f'Evidence SHA256: {sha(destination)}',
        'Scope: available recorded environments only. Refresh after later stages; Full400 is not certified by this index.',
        'Native workers, runtime source, frozen parameters and checkpoint acceptances were not changed.']
    (ROOT / 'meta/env.txt').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'proof': relative, 'env_path': 'meta/env.txt', 'env_sha256': sha(ROOT / 'meta/env.txt'),
        'recorded_role_pairs_by_stage': {stage: value['recorded_role_pairs'] for stage, value in stages.items()},
        'source_and_parameters_unchanged': True, 'workers_or_native_states_modified': False}))


if __name__ == '__main__':
    main()
