"""Schedule unchanged Exp4 scene commands on the current user-approved available GPU pool.

An in-progress Full stage can hand over its CPU controller without restarting
native launchers. The old controller remains stopped until its children have
exited, so their real wait statuses can be read before retiring that parent.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from evaluate_sequence import (OUT, REPO, GateError, audit_scene, build_jobs,
    command, digest, expected_identities, finalize, load, source_fingerprint,
    validate_gate, write)
from experiments.exp4_collect import run_path
from watch_evaluation import process

ATTEMPT = 'a3'
PREVIOUS = {'nearest': 'full', 'myopic': 'nearest', 'suffix_only': 'myopic'}
SELF = Path(__file__).resolve()
BASE = OUT / 'code/evaluate_sequence.py'
MEMORY_POLICY = {'min_free_mib': 34 * 1024, 'max_existing_compute_mib': 10 * 1024}
SAFE_GPU_SET = {1, 2, 4, 5, 6, 7}



def now():
    return datetime.now(timezone.utc).isoformat()


def claim(path, value):
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def boot():
    return Path('/proc/sys/kernel/random/boot_id').read_text().strip()


def checked_process(pid, ticks, *, stopped=False):
    value = process(pid)
    if not value.get('live') or value.get('start_ticks') != str(ticks):
        raise GateError(f'PID identity no longer matches: {pid}')
    if stopped and value['state'] not in ('T', 't'):
        raise GateError(f'Predecessor controller is not stopped: {pid}')
    return value


def gpu_inventory():
    def query(kind, fields):
        lines = subprocess.check_output(['nvidia-smi', '--query-' + kind + '=' + ','.join(fields),
            '--format=csv,noheader,nounits'], text=True).splitlines()
        return [dict(zip(fields, (part.strip() for part in line.split(',')))) for line in lines]
    gpus = query('gpu', ['index', 'uuid', 'name', 'memory.total', 'memory.free'])
    apps = query('compute-apps', ['gpu_uuid', 'pid', 'used_gpu_memory'])
    return {'checked_at_utc': now(), 'gpus': gpus, 'compute_apps': apps}


def eligible_gpus(inventory, allowed):
    eligible = []
    for gpu in inventory['gpus']:
        index = int(gpu['index'])
        if index not in allowed or index not in SAFE_GPU_SET:
            continue
        apps = [row for row in inventory['compute_apps'] if row['gpu_uuid'] == gpu['uuid']]
        try:
            free = int(gpu['memory.free'])
            occupied = [int(row['used_gpu_memory']) for row in apps]
        except (KeyError, TypeError, ValueError):
            continue
        if free >= MEMORY_POLICY['min_free_mib'] and all(value >= 0 for value in occupied) and sum(occupied) <= MEMORY_POLICY['max_existing_compute_mib']:
            eligible.append(index)
    return eligible


def public_inventory(inventory):
    gpus = [dict(row) for row in inventory['gpus'] if int(row['index']) in SAFE_GPU_SET]
    uuids = {row['uuid'] for row in gpus}
    return {'checked_at_utc':inventory.get('checked_at_utc'), 'gpus':gpus,
            'compute_apps':[{'gpu_uuid':row['gpu_uuid'], 'used_gpu_memory':row.get('used_gpu_memory')}
                            for row in inventory['compute_apps'] if row['gpu_uuid'] in uuids],
            'external_usernames_and_PIDs_omitted':True}


def recorded_parent_births(old):
    records = [old, old.get('predecessor_controller', {})]
    if old.get('scheduling_plan'):
        plan = load(OUT / old['scheduling_plan'])
        if plan['controller_sha256'] != old['controller_sha256'] or plan['boot_id'] != old['boot_id']:
            raise GateError('Predecessor scheduling plan does not match its recorded source or boot')
        records.extend(plan['parent_controllers'])
    births = {}
    for record in records:
        if record.get('pid') is None:
            continue
        pid, ticks = int(record['pid']), str(record['start_ticks'])
        if pid in births and births[pid] != ticks:
            raise GateError('Conflicting controller birth records')
        births[pid] = ticks
    return births


def policy_allowed(requested):
    policy = load(OUT/'meta/evaluation-available-gpu-policy.json')
    allowed = policy['allowed_gpus']
    if not allowed or len(allowed) != len(set(allowed)) or not set(allowed) <= SAFE_GPU_SET:
        raise GateError('Only the current available GPU pool is authorized; physical0 and3 are reserved')
    return [gpu for gpu in requested if gpu in allowed]


def resource_resumed_scene(stage):
    if stage != 'full':
        return None
    return load(OUT/'meta/gpu0-withdrawal-a3.json')['scene']


def keep_authorized_jobs(stage, jobs):
    if stage != 'full':
        return jobs
    proof = load(OUT/'meta/gpu0-withdrawal-a3.json')
    if proof['gpu0_native_completed_at_withdrawal'] != 0 or not proof['raw_logs_retained']:
        raise GateError('GPU0 interruption requires separate review if it contains completed results')
    withdrawn = proof['job']
    matches = [job for job in jobs if job['pid'] == withdrawn['pid']]
    assert len(matches) <= 1
    if matches:
        assert matches[0]['scene'] == withdrawn['scene'] and matches[0]['gpu'] == 0
    retained = [job for job in jobs if job['pid'] != withdrawn['pid']]
    if any(job['status'] == 'running' and job['gpu'] not in (1,2,3) for job in retained):
        raise GateError('Retained running scene is outside GPU1/2/3')
    return retained


def assert_no_live_children(parent_pid):
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            raw = path.read_text()
            fields = raw[raw.rfind(')')+2:].split()
            if int(fields[1]) == parent_pid and fields[0] != 'Z':
                raise GateError('Refusing to retire a CPU parent with a living child')
        except (FileNotFoundError, ProcessLookupError):
            pass


def pending_assignments(stage, jobs):
    expected = build_jobs(stage, expected_identities(stage))
    wanted = {job['scene']: job for job in expected}
    seen = set()
    for job in jobs:
        scene = job['scene']
        if scene in seen or scene not in wanted or job['identities'] != wanted[scene]['identities']:
            raise GateError('Existing assignments are duplicated or differ from frozen identities')
        if job['status'] not in ('running', 'completed'):
            raise GateError('Existing scene is not eligible for a scheduling handoff')
        seen.add(scene)
    return [job for job in expected if job['scene'] not in seen]


def adopted_returncode(pid, ticks):
    """Completed native status alone is insufficient: require the OS exit status."""
    path = Path('/proc') / str(pid) / 'stat'
    if not path.exists():
        raise GateError('Adopted launcher vanished before its real exit status was captured')
    raw = path.read_text()
    fields = raw[raw.rfind(')') + 2:].split()
    if fields[19] != str(ticks):
        raise GateError('Adopted launcher PID was reused')
    if fields[0] != 'Z':
        return None, None
    wait_status = int(fields[49])
    return os.waitstatus_to_exitcode(wait_status), {
        'pid': pid, 'start_ticks': str(ticks), 'wait_status': wait_status,
        'stat_sha256': hashlib.sha256(raw.encode()).hexdigest(),
        'captured_at_utc': now(), 'source': '/proc/PID/stat field52; original parent stopped'}


def readiness(stage, allowed, adopt=False):
    reasons = []
    gate = None
    try:
        gate = validate_gate(stage)
    except (GateError, AssertionError) as error:
        reasons.append(str(error))
    inventory = gpu_inventory()
    eligible = eligible_gpus(inventory, allowed)
    if not eligible and not adopt:
        reasons.append('No authorized GPU meets the bounded sharing headroom limits')
    ledger_path = OUT / f'meta/evaluation-{stage}-runs.json'
    if adopt:
        if stage != 'full' or not ledger_path.exists():
            reasons.append('Only the existing Full stage can be adopted')
        else:
            ledger = load(ledger_path)
            ledger['jobs'] = keep_authorized_jobs(stage, ledger['jobs'])
            start = load(OUT / 'meta/full-start-a3.json')
            if ledger['status'] != 'running' or ledger['controller_pid'] != start['pid']:
                reasons.append('Existing Full controller/ledger is not an adoptable running stage')
            checked_process(start['pid'], start['start_ticks'])
            pending = pending_assignments(stage, ledger['jobs'])
            if any((OUT / f'logs/evaluation-{stage}-a3-{job["scene"]}.controller.log').exists() and job['scene'] != resource_resumed_scene(stage) for job in pending):
                reasons.append('An unassigned scene already has a controller log; inspect before adoption')
    else:
        for path in (ledger_path, OUT / f'meta/{stage}-start-a3.json',
                     OUT / f'logs/evaluation-{stage}-controller-a3.log'):
            if path.exists():
                reasons.append('Existing target artifacts: inspect, never duplicate a stage')
        if stage not in PREVIOUS:
            reasons.append('A fresh stage must be nearest, myopic or suffix_only')
        elif load(OUT / 'meta/last-evaluation-observer.json')['stage'] != PREVIOUS[stage]:
            reasons.append('Predecessor observer does not match the fixed stage order')
    disk = os.statvfs(OUT)
    free = disk.f_bavail * disk.f_frsize
    if free < 30 * 1024**3:
        reasons.append('Less than 30GiB of output space remains')
    return {'checked_at_utc': now(), 'stage': stage, 'ready': not reasons, 'reasons': reasons,
        'allowed_gpus': allowed, 'eligible_gpus_at_check': eligible, 'gpu_memory_policy': MEMORY_POLICY, 'gpu_inventory': public_inventory(inventory),
        'scenes': 20, 'episodes': 400, 'adopt_existing_full': adopt,
        'native_workers_started': False, 'native_source_entries': len(source_fingerprint()),
        'source_sha256': digest(SELF), 'base_controller_sha256': digest(BASE),
        'parameter_sha256': digest(OUT / 'meta/calibration.json'), 'output_free_bytes': free}, gate


def verify_no_unrecorded_children(parent_pid, jobs):
    expected = {int(job['pid']) for job in jobs if job['status'] == 'running'}
    actual = set()
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            raw = path.read_text()
            fields = raw[raw.rfind(')') + 2:].split()
            if int(fields[1]) == parent_pid:
                actual.add(int(path.parent.name))
        except (FileNotFoundError, ProcessLookupError):
            pass
    if not actual <= expected:
        raise GateError(f'Controller children not fully recorded: actual={sorted(actual)}, expected={sorted(expected)}')


def wait_for(path, child, seconds=90):
    deadline = time.monotonic() + seconds
    while not path.exists():
        if child.poll() is not None:
            raise GateError('Replacement controller exited; inspect its log and claim')
        if time.monotonic() >= deadline:
            raise GateError('Controller handoff initialization timed out; inspect this same PID')
        time.sleep(.2)
    return load(path)


def launch(stage, allowed, adopt):
    lock_path = OUT / 'meta/evaluation-launch.lock'
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        check, gate = readiness(stage, allowed, adopt)
        if not check['ready']:
            raise GateError('; '.join(check['reasons']))
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        directory = OUT / 'diagnostics/gpu-resource-sharing' / stamp
        directory.mkdir(parents=True, exist_ok=False)
        start_path = OUT / f'meta/{stage}-start-a3.json'
        ledger_path = OUT / f'meta/evaluation-{stage}-runs.json'
        takeover_claim = OUT / f'meta/{stage}-shared-gpu-launch-a3.json'
        claim(takeover_claim, {'status': 'preparing', 'stage': stage, 'directory': str(directory),
                              'authorization': 'User chose to preserve Exp2 and move Exp4 to other GPUs; physical0 and3 remain reserved', 'at_utc': now()})
        old = load(start_path) if adopt else None
        paused, activated, child = False, False, None
        was_stopped = bool(old and process(old['pid']).get('state') in ('T','t'))
        original_bytes = {}
        try:
            if adopt:
                checked_process(old['pid'], old['start_ticks'])
                assert boot() == old['boot_id']
                os.kill(old['pid'], signal.SIGSTOP)
                paused = True
                for _ in range(30):
                    if process(old['pid']).get('state') in ('T', 't'):
                        break
                    time.sleep(.1)
                checked_process(old['pid'], old['start_ticks'], stopped=True)
                ledger = load(ledger_path)
                assert ledger['status'] == 'running' and ledger['controller_pid'] == old['pid']
                verify_no_unrecorded_children(old['pid'], ledger['jobs'])
                ledger['jobs'] = keep_authorized_jobs(stage, ledger['jobs'])
                ledger['resource_interruptions'] = [load(OUT/'meta/gpu0-withdrawal-a3.json')]
                pending_assignments(stage, ledger['jobs'])
                for name in (f'meta/evaluation-{stage}-runs.json', f'meta/{stage}-start-a3.json',
                             f'meta/{stage}-observer-a3.json', 'meta/last-evaluation-observer.json'):
                    data = (OUT / name).read_bytes()
                    original_bytes[name] = data
                    (directory / Path(name).name).write_bytes(data)
                adopted = []
                for job in ledger['jobs']:
                    if job['status'] != 'running':
                        continue
                    value = process(job['pid'])
                    assert value.get('start_ticks') and (value.get('live') or value.get('state') == 'Z')
                    assert f'--tag exp4-{stage}-a3-{job["scene"]}' in value.get('command', '') or value.get('state') == 'Z'
                    raw = (Path('/proc')/str(job['pid'])/'stat').read_text()
                    fields = raw[raw.rfind(')')+2:].split()
                    parent_pid = int(fields[1])
                    permitted_parents = recorded_parent_births(old)
                    assert parent_pid in permitted_parents
                    parent = checked_process(parent_pid, permitted_parents[parent_pid], stopped=True)
                    adopted.append({'scene': job['scene'], 'pid': job['pid'], 'gpu': job['gpu'],
                                    'start_ticks': value['start_ticks'], 'parent_pid': parent_pid,
                                    'parent_start_ticks': parent['start_ticks']})
            else:
                ledger = {**{key: value for key, value in gate.items() if key != 'parameters'},
                          'stage': stage, 'attempt': ATTEMPT, 'status': 'running',
                          'boot_id': boot(), 'jobs': []}
                adopted = []
            assert ledger['source_fingerprint'] == gate['source_fingerprint'] == source_fingerprint()
            assert ledger['parameter_sha256'] == gate['parameter_sha256']
            write(directory / 'ledger-before.json', ledger)
            plan = {'stage': stage, 'attempt': ATTEMPT, 'created_at_utc': now(), 'boot_id': boot(),
                'allowed_gpus': allowed, 'source_fingerprint': gate['source_fingerprint'],
                'parameter_sha256': gate['parameter_sha256'], 'controller_sha256': digest(SELF),
                'base_controller_sha256': digest(BASE), 'old_controller': old,
                'adopted_launchers': adopted, 'resource_withdrawal': 'meta/gpu0-withdrawal-a3.json' if adopt else None,
                'parent_controllers': list({row['parent_pid']: {'pid':row['parent_pid'], 'start_ticks':row['parent_start_ticks']} for row in adopted}.values()),
                'ledger_snapshot_sha256': digest(directory / 'ledger-before.json'),
                'preflight': check, 'native_commands_and_inputs_unchanged': True,
                'gpu_memory_policy': MEMORY_POLICY,
                'scope': 'CPU scheduling with bounded GPU-memory sharing; native commands and running launchers are unchanged'}
            write(directory / 'plan.json', plan)
            log = directory / 'controller.log' if adopt else OUT / f'logs/evaluation-{stage}-controller-a3.log'
            env = dict(os.environ)
            env.pop('CUDA_VISIBLE_DEVICES', None)
            argv = [sys.executable, '-u', str(SELF), '--stage', stage, '--attempt', ATTEMPT,
                    '--controller', '--plan', str(directory / 'plan.json')]
            with log.open('xb') as out:
                child = subprocess.Popen(argv, cwd=REPO, env=env, stdin=subprocess.DEVNULL,
                    stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
            ready = wait_for(directory / 'ready.json', child)
            checked_process(child.pid, ready['start_ticks'])
            assert ready['pid'] == child.pid and ready['plan_sha256'] == digest(directory / 'plan.json')
            record = dict(old) if adopt else {'stage': stage, 'attempt': ATTEMPT, 'started_at_utc': now(),
                'boot_id': boot(), 'source_commit': subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
                'source_fingerprint': gate['source_fingerprint'], 'parameter_sha256': gate['parameter_sha256'],
                'CP3_acceptance_sha256': digest(OUT / 'meta/CP3_acceptance.json')}
            record.update(status='running', pid=child.pid, start_ticks=ready['start_ticks'], command=argv,
                log=str(log), allowed_gpus=allowed, scheduling='bounded_memory_scene_parallel',
                controller_sha256=digest(SELF), base_controller_sha256=digest(BASE),
                scheduling_plan=str((directory / 'plan.json').relative_to(OUT)),
                scheduling_started_at_utc=now(), source_fingerprint=gate['source_fingerprint'],
                CUDA_VISIBLE_DEVICES_in_native_parent=None, detached_session=True)
            if adopt:
                record['predecessor_controller'] = {'pid': old['pid'], 'start_ticks': old['start_ticks'],
                    'state': 'stopped_only_cpu_controller; original native children continue',
                    'snapshot': str((directory / 'full-start-a3.json').relative_to(OUT))}
                write(start_path, record)
                for relative in ('meta/full-observer-a3.json', 'meta/last-evaluation-observer.json'):
                    value = load(OUT / relative)
                    assert value['stage'] == 'full'
                    value.update(controller_pid=child.pid, controller_start_ticks=ready['start_ticks'],
                        scheduling_handoff_plan=record['scheduling_plan'])
                    write(OUT / relative, value)
            else:
                record['predecessor_ledger_sha256'] = digest(OUT / f'meta/evaluation-{PREVIOUS[stage]}-runs.json')
                if stage == 'nearest':
                    record['CP4_reported_sha256'] = digest(OUT / 'meta/CP4_reported.json')
                claim(start_path, record)
            write(directory / 'activate.json', {'pid': child.pid, 'start_ticks': ready['start_ticks'],
                'plan_sha256': digest(directory / 'plan.json'), 'activated_at_utc': now()})
            activated = True
            active = wait_for(directory / 'activated.json', child)
            if adopt:
                observer = load(OUT / 'meta/full-observer-a3.json')
                checked_process(observer['observer_pid'], observer['start_ticks'])
            else:
                from start_remaining_evaluation import attach_observer
                observer = attach_observer(stage, record)
                if observer.get('observer_pid'):
                    record.update(observer_pid=observer['observer_pid'], observer_start_ticks=observer['start_ticks'])
                    write(start_path, record)
            proof = {'status': 'running', 'stage': stage, 'at_utc': now(), 'controller_pid': child.pid,
                'controller_start_ticks': ready['start_ticks'], 'plan': record['scheduling_plan'],
                'observer_pid': observer.get('observer_pid'), 'adopted_launchers': adopted,
                'native_workers_restarted': False, 'source_and_parameters_unchanged':
                source_fingerprint() == gate['source_fingerprint'] and digest(OUT/'meta/calibration.json') == gate['parameter_sha256'],
                'activation': active, 'gpu_inventory_at_handoff': check['gpu_inventory']}
            write(takeover_claim, proof)
            write(directory / 'handoff.json', proof)
            print(json.dumps(proof))
        except BaseException as error:
            failure = {'status': 'inspect_activated_controller' if activated else 'rolled_back_before_activation',
                       'at_utc': now(), 'activated': activated, 'error': f'{type(error).__name__}: {error}',
                       'controller_pid': child.pid if child else None, 'directory': str(directory)}
            if not activated:
                if child and child.poll() is None:
                    child.terminate()
                    child.wait(timeout=10)
                for relative, data in original_bytes.items():
                    (OUT / relative).write_bytes(data)
                if paused and not was_stopped:
                    checked_process(old['pid'], old['start_ticks'], stopped=True)
                    os.kill(old['pid'], signal.SIGCONT)
            write(takeover_claim, failure)
            write(directory / 'failure.json', failure)
            raise


def control(plan_path):
    plan = load(plan_path)
    directory = plan_path.parent
    stage = plan['stage']
    gate = validate_gate(stage)
    assert boot() == plan['boot_id']
    assert digest(SELF) == plan['controller_sha256'] and digest(BASE) == plan['base_controller_sha256']
    assert plan['gpu_memory_policy'] == MEMORY_POLICY
    assert gate['source_fingerprint'] == plan['source_fingerprint']
    assert gate['parameter_sha256'] == plan['parameter_sha256']
    assert digest(directory/'ledger-before.json') == plan['ledger_snapshot_sha256']
    ledger = load(directory / 'ledger-before.json')
    pending = pending_assignments(stage, ledger['jobs'])
    identity = process(os.getpid())
    write(directory/'ready.json', {'pid': os.getpid(), 'start_ticks': identity['start_ticks'],
        'plan_sha256': digest(plan_path), 'at_utc': now(), 'native_workers_started': False})
    deadline = time.monotonic() + 120
    while not (directory/'activate.json').exists():
        if time.monotonic() >= deadline:
            raise GateError('No activation received; no worker or ledger mutation performed')
        time.sleep(.1)
    activation = load(directory/'activate.json')
    assert activation['pid'] == os.getpid() and activation['start_ticks'] == identity['start_ticks']
    assert activation['plan_sha256'] == digest(plan_path)
    ledger.update(controller_pid=os.getpid(), boot_id=boot(), allowed_gpus=plan['allowed_gpus'],
        scheduling={'mode': 'bounded_memory_scene_parallel', 'plan': str(plan_path.relative_to(OUT)),
                    'controller_sha256': digest(SELF), 'base_controller_sha256': digest(BASE)}, status='running')
    ledger_path = OUT / f'meta/evaluation-{stage}-runs.json'
    active, owned, handles = {}, {}, {}
    adopted = {row['scene']: row for row in plan['adopted_launchers']}
    for job in ledger['jobs']:
        if job['status'] == 'running':
            row = adopted[job['scene']]
            assert row['pid'] == job['pid'] and row['gpu'] == job['gpu'] and job['gpu'] not in active
            job['pid_start_ticks'] = row['start_ticks']
            job['adopted_without_native_restart'] = True
            active[job['gpu']] = job
    write(ledger_path, ledger)
    write(directory/'activated.json', {'pid': os.getpid(), 'at_utc': now(),
        'adopted_native_launchers': list(adopted.values()), 'pending_scenes': [job['scene'] for job in pending],
        'ledger_controller_pid': ledger['controller_pid']})
    def stop(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    retired = set()
    try:
        while pending or active:
            if source_fingerprint() != plan['source_fingerprint'] or digest(OUT/'meta/calibration.json') != plan['parameter_sha256']:
                raise GateError('Frozen native inputs changed during parallel evaluation')
            if digest(SELF) != plan['controller_sha256'] or digest(BASE) != plan['base_controller_sha256']:
                raise GateError('Scheduling source changed during parallel evaluation')
            for gpu, job in list(active.items()):
                run = run_path(Path(job['log']))
                if run:
                    job['run_directory'] = str(run)
                    if any(path.stat().st_size for path in run.rglob('blp_errors.jsonl')):
                        raise GateError('Live BLP error; inspect original raw scene')
                if job['scene'] in owned:
                    code = owned[job['scene']].poll()
                    exit_proof = {'source': 'subprocess.Popen.poll', 'at_utc': now()} if code is not None else None
                else:
                    parent = adopted[job['scene']]
                    checked_process(parent['parent_pid'], parent['parent_start_ticks'], stopped=True)
                    code, exit_proof = adopted_returncode(job['pid'], job['pid_start_ticks'])
                if code is None:
                    continue
                if code != 0 or run is None:
                    raise GateError(f'Native scene exited unsuccessfully: {job["scene"]}, returncode={code}')
                result = audit_scene(run, job)
                if job['scene'] in handles:
                    handles.pop(job['scene']).close()
                job.update(status='completed', returncode=code, audit=result, exit_status_proof=exit_proof)
                del active[gpu]
            for old in plan['parent_controllers']:
                if old['pid'] in retired or any(job['scene'] in adopted and adopted[job['scene']]['parent_pid'] == old['pid'] for job in active.values()):
                    continue
                checked_process(old['pid'], old['start_ticks'], stopped=True)
                assert all(job['status'] == 'completed' and job['returncode'] == 0 for job in ledger['jobs'] if job['scene'] in adopted and adopted[job['scene']]['parent_pid'] == old['pid'])
                assert_no_live_children(old['pid'])
                os.kill(old['pid'], signal.SIGKILL)
                write(directory/f'predecessor-retired-{old["pid"]}.json', {'pid': old['pid'], 'start_ticks': old['start_ticks'],
                    'at_utc': now(), 'signal': 'SIGKILL to stopped CPU controller only after all retained native children exited0; user-withdrawn GPU0 child already exited',
                    'native_workers_signaled': False})
                retired.add(old['pid'])
            inventory = gpu_inventory()
            for gpu in eligible_gpus(inventory, policy_allowed(plan['allowed_gpus'])):
                if gpu in active or not pending:
                    continue
                assignment = pending.pop(0)
                scene = assignment['scene']
                resume = scene == resource_resumed_scene(stage)
                suffix = '-resource-resume1' if resume else ''
                log = OUT / f'logs/evaluation-{stage}-a3-{scene}{suffix}.controller.log'
                argv = command(stage, scene, assignment['identities'], gpu, ATTEMPT)
                if resume:
                    argv[argv.index('--tag')+1] += suffix
                intent_path = directory / f'launch-{scene}.json'
                claim(intent_path, {'scene': scene, 'gpu': gpu, 'command': argv,
                    'status': 'launch_intent', 'at_utc': now(), 'gpu_inventory': public_inventory(inventory)})
                handle = log.open('x')
                env = dict(os.environ)
                env.pop('CUDA_VISIBLE_DEVICES', None)
                child = subprocess.Popen(argv, cwd=REPO, env=env, stdout=handle,
                    stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
                observed = process(child.pid)
                assert observed['live'] and f'--tag exp4-{stage}-a3-{scene}' in observed['command']
                job = dict(assignment, gpu=gpu, pid=child.pid, pid_start_ticks=observed['start_ticks'],
                           log=str(log), command=argv, status='running')
                write(intent_path, {**job, 'at_utc': now(), 'gpu_inventory_before_launch': public_inventory(inventory)})
                ledger['jobs'].append(job)
                active[gpu], owned[scene], handles[scene] = job, child, handle
                write(ledger_path, ledger)
            write(ledger_path, ledger)
            if pending or active:
                time.sleep(10)
        ledger['status'] = 'completed_pending_process_audit'
        write(ledger_path, ledger)
        finalize(stage, ledger)
        write(ledger_path, ledger)
    except BaseException as error:
        ledger.update(status='stopped_for_review', error=f'{type(error).__name__}: {error}',
            scheduling_error_at_utc=now(), existing_native_workers_left_running_for_diagnosis=True)
        write(ledger_path, ledger)
        raise
    finally:
        for handle in handles.values():
            handle.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('full','nearest','myopic','suffix_only'), required=True)
    parser.add_argument('--attempt', choices=('a3',), default='a3')
    parser.add_argument('--gpus', default='5,6,7')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--start', action='store_true')
    mode.add_argument('--adopt-full', action='store_true')
    mode.add_argument('--controller', action='store_true')
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--check-adoption', action='store_true')
    args = parser.parse_args()
    assert args.stage in PREVIOUS and not args.adopt_full, 'This version only starts the remaining stages'
    allowed = [int(value) for value in args.gpus.split(',')]
    assert allowed and len(set(allowed)) == len(allowed) and set(allowed) <= SAFE_GPU_SET
    assert allowed == policy_allowed(allowed)
    if args.controller:
        assert args.plan
        control(args.plan)
    elif args.check:
        print(json.dumps(readiness(args.stage, allowed, args.check_adoption)[0]))
    else:
        launch(args.stage, allowed, args.adopt_full)


if __name__ == '__main__':
    main()
