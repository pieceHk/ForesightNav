"""Inventory completed smoke workers using their actual recorded environments."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def signature(value):
    packages = sorted(value['packages'], key=lambda item: json.dumps(item, sort_keys=True))
    selected = {key: value[key] for key in ('python_executable', 'python_version', 'prefix', 'base_prefix', 'lock_sha256')}
    selected['packages'] = packages
    return hashlib.sha256(json.dumps(selected, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def main():
    ledger = json.loads((ROOT / 'meta/evaluation-smoke-runs.json').read_text())
    start = json.loads((ROOT / 'meta/smoke_start.json').read_text())
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    assert ledger['boot_id'] == boot == start['boot_id']
    signatures, exemplars, cases, seeds = {'mapping': set(), 'navigation': set()}, {}, [], set()
    for job in ledger['jobs']:
        if job['status'] != 'completed':
            continue
        run = Path(job['run_directory'])
        launch = json.loads((run / 'launch.json').read_text())
        assert launch['source_commit'] == start['head']
        seeds.add(launch['configuration']['seed'])
        artifacts, found, role_signatures = {}, set(), {}
        for path in sorted(run.glob('*environment*.json')):
            value = json.loads(path.read_text())
            role = value['role']
            assert role in signatures and role not in found
            assert value['launch_source_commit'] == start['head']
            assert value['configuration_sha256'] == launch['configuration_sha256']
            assert value['identity_sha256'] == launch['identity_sha256']
            assert value['cuda_visible_devices'] == str(job['gpu'])
            fingerprint = signature(value)
            signatures[role].add(fingerprint)
            role_signatures[role] = fingerprint
            exemplars.setdefault(role, value)
            artifacts[str(path.relative_to(run))] = sha(path)
            found.add(role)
        assert found == {'mapping', 'navigation'}
        cases.append({'scene_id': job['scene'], 'run_directory': str(run), 'gpu': job['gpu'],
                      'seed': launch['configuration']['seed'], 'configuration_sha256': launch['configuration_sha256'],
                      'identity_sha256': launch['identity_sha256'], 'role_signatures': role_signatures,
                      'environment_artifact_sha256': artifacts})
    assert cases and len(seeds) == 1
    uniform = {role: len(values) == 1 for role, values in signatures.items()}
    assert all(uniform.values()), 'Native role environments changed between smoke cases'
    fingerprints = {role: next(iter(values)) for role, values in signatures.items()}
    probe_path = ROOT / 'meta/environment-smoke-probe-a3.json'
    if probe_path.exists():
        probe = json.loads(probe_path.read_text())
        assert probe['boot_id'] == boot and probe['runtime_commit'] == start['head'] and probe['role_signatures'] == fingerprints
    else:
        frameworks = {}
        code = 'import json,sys,site,torch; print(json.dumps({"python_executable":sys.executable,"python_version":sys.version.split()[0],"torch_version":torch.__version__,"torch_cuda_version":torch.version.cuda,"cuda_available":torch.cuda.is_available(),"user_site_enabled":site.ENABLE_USER_SITE}))'
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        for role, value in exemplars.items():
            if value['user_site_enabled']:
                environment.pop('PYTHONNOUSERSITE', None)
            else:
                environment['PYTHONNOUSERSITE'] = '1'
            actual = json.loads(subprocess.check_output([value['python_executable'], '-c', code], env=environment, text=True))
            assert actual['python_executable'] == value['python_executable'] and actual['python_version'] == value['python_version']
            assert actual['user_site_enabled'] == value['user_site_enabled']
            assert actual['cuda_available'] is False
            frameworks[role] = actual
        hardware = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,driver_version,memory.total', '--format=csv,noheader,nounits'], text=True)
        hardware = [line for line in hardware.splitlines() if int(line.split(',')[0]) in (2, 3)]
        probe = {'recorded_at_utc': datetime.now(timezone.utc).isoformat(), 'boot_id': boot,
                 'runtime_commit': start['head'], 'role_signatures': fingerprints,
                 'frameworks': frameworks, 'gpu_hardware_and_driver': hardware,
                 'framework_import_probe_has_no_visible_GPU': True}
        probe_path.write_text(json.dumps(probe, indent=2) + '\n')
    result = {**probe, 'recorded_at_utc': datetime.now(timezone.utc).isoformat(), 'stage': 'smoke',
              'seed': next(iter(seeds)), 'worker_placement': 'mapping/navigation colocated on physical GPU2 or GPU3',
              'completed_cases_checked': len(cases), 'uniform_python_lock_and_package_inventory_by_role': uniform,
              'cases': cases, 'scope': 'actual completed cases only; refresh after all20',
              'source_sha256': sha(Path(__file__))}
    (ROOT / 'meta/environment-smoke.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ('completed_cases_checked', 'seed', 'runtime_commit', 'uniform_python_lock_and_package_inventory_by_role', 'frameworks')}))


if __name__ == '__main__':
    main()
