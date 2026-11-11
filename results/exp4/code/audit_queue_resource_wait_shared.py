"""Read the live scene queue and GPU launch guard without changing native jobs."""
from datetime import datetime, timezone
from pathlib import Path
import json

from shared_gpu_evaluate_sequence import (
    OUT, MEMORY_POLICY, digest, gpu_inventory, eligible_gpus, keep_authorized_jobs, load,
    pending_assignments, policy_allowed, process, source_fingerprint, write,
)


def main():
    started = datetime.now(timezone.utc).isoformat()
    ledger = load(OUT/'meta/evaluation-full-runs.json')
    start = load(OUT/'meta/full-start-a3.json')
    observer = load(OUT/'meta/full-observer-a3.json')
    assert ledger['status'] == 'running' and ledger['controller_pid'] == start['pid']
    live = process(start['pid'])
    watching = process(observer['observer_pid'])
    assert live['live'] and live['start_ticks'] == start['start_ticks']
    assert watching['live'] and watching['start_ticks'] == observer['start_ticks']
    assert Path('/proc/sys/kernel/random/boot_id').read_text().strip() == start['boot_id']
    controller = OUT/'code/shared_gpu_evaluate_sequence.py'
    assert digest(controller) == start['controller_sha256']
    assert source_fingerprint() == ledger['source_fingerprint']
    assert digest(OUT/'meta/calibration.json') == ledger['parameter_sha256']
    jobs = keep_authorized_jobs('full', ledger['jobs'])
    active = [job for job in jobs if job['status'] == 'running']
    pending = pending_assignments('full', jobs)
    launchers = {job['pid']: job for job in active}
    for pid, job in launchers.items():
        p = process(pid)
        assert p.get('start_ticks') == job['pid_start_ticks'] and (p.get('live') or p.get('state') == 'Z')

    def belongs_to_active_launcher(pid):
        for _ in range(64):
            if pid in launchers:
                return True
            if pid <= 1:
                return False
            try:
                raw = (Path('/proc')/str(pid)/'stat').read_text()
                pid = int(raw[raw.rfind(')')+2:].split()[1])
            except (FileNotFoundError, ProcessLookupError):
                return False
        return False

    allowed = policy_allowed([1,2,3])
    inventory = gpu_inventory()
    eligible = eligible_gpus(inventory, allowed)
    context = []
    for gpu in inventory['gpus']:
        index = int(gpu['index'])
        if index not in allowed:
            continue
        apps = [a for a in inventory['compute_apps'] if a['gpu_uuid'] == gpu['uuid']]
        own = [a for a in apps if belongs_to_active_launcher(int(a['pid']))]
        external = [a for a in apps if a not in own]
        context.append({'gpu':index, 'free_mib':int(gpu['memory.free']),
            'experiment_compute_processes':len(own), 'external_compute_processes':len(external),
            'external_compute_memory_mib':sum(int(a['used_gpu_memory']) for a in external),
            'has_active_scene':any(j['gpu'] == index for j in active),
            'eligible_under_current_controller':index in eligible,
            'guard_requires_no_compute_processes':False, 'guard_min_free_mib':MEMORY_POLICY['min_free_mib'],
            'launch_guard_max_preexisting_compute_mib':MEMORY_POLICY['max_existing_compute_mib']})
    record = {'observed_at_utc':inventory['checked_at_utc'], 'audit_started_at_utc':started,
        'stage':'full', 'diagnostic_only':True, 'formal400_complete':False,
        'allowed_gpus':allowed, 'gpu_memory_policy':MEMORY_POLICY, 'gpu_context':context,
        'active_scenes':[{'scene':j['scene'], 'gpu':j['gpu']} for j in active],
        'completed_scenes':sum(j['status'] == 'completed' for j in jobs),
        'pending_scenes':[j['scene'] for j in pending],
        'idle_scene_slots_waiting_for_resources':[g['gpu'] for g in context
            if not g['has_active_scene'] and not g['eligible_under_current_controller'] and pending],
        'controller_live':True, 'observer_live':True, 'source_and_parameters_unchanged':True,
        'controller_sha256':digest(controller), 'parameter_sha256':ledger['parameter_sha256'],
        'source_path':'code/audit_queue_resource_wait_shared.py', 'source_sha256':digest(Path(__file__)),
        'signals_sent':False, 'native_states_modified':False, 'external_usernames_and_PIDs_omitted':True,
        'scope':'Discrete queue and GPU-guard observation; no launch, parameter change, or causal timing claim'}
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    target = OUT/'diagnostics/resource-context'/f'{stamp}-shared-queue.json'
    write(target,record)
    print(json.dumps({'path':str(target.relative_to(OUT)), 'sha256':digest(target), **record}))


if __name__ == '__main__':
    main()
