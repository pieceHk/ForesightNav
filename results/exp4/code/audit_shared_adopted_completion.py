"""Read-only audit of an actually completed adopted Full scene after CPU retirement."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

from evaluate_sequence import OUT, audit_scene, digest, load, source_fingerprint, write
from watch_evaluation import process


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', required=True)
    args = parser.parse_args()
    ledger = load(OUT / 'meta/evaluation-full-runs.json')
    start = load(OUT / 'meta/full-start-a3.json')
    plan_path = OUT / start['scheduling_plan']
    plan = load(plan_path)
    assert ledger['stage'] == start['stage'] == plan['stage'] == 'full'
    assert ledger['scheduling']['plan'] == start['scheduling_plan']
    controller_path = OUT / 'code/shared_gpu_evaluate_sequence.py'
    assert digest(controller_path) == plan['controller_sha256'] == start['controller_sha256']
    assert digest(OUT / 'code/evaluate_sequence.py') == plan['base_controller_sha256']
    assert Path('/proc/sys/kernel/random/boot_id').read_text().strip() == start['boot_id'] == plan['boot_id']
    current = process(start['pid'])
    assert current['live'] and current['start_ticks'] == start['start_ticks']
    assert ledger['controller_pid'] == start['pid']
    fingerprint = source_fingerprint()
    parameter_sha = digest(OUT / 'meta/calibration.json')
    assert fingerprint == plan['source_fingerprint'] == ledger['source_fingerprint']
    assert parameter_sha == plan['parameter_sha256'] == ledger['parameter_sha256']
    adopted = {row['scene']: row for row in plan['adopted_launchers']}
    assert args.scene in adopted, 'This reader only audits genuinely adopted scenes'
    row = adopted[args.scene]
    job = next(job for job in ledger['jobs'] if job['scene'] == args.scene)
    retirement_path = plan_path.parent / f'predecessor-retired-{row["parent_pid"]}.json'
    if job['status'] != 'completed' or not retirement_path.is_file():
        print(json.dumps({'ready': False, 'scene': args.scene,
            'reason': 'Native completion and original CPU retirement record are both required',
            'native_workers_signaled': False}))
        return
    assert job['returncode'] == 0 and job['adopted_without_native_restart']
    assert job['pid'] == row['pid'] and job['pid_start_ticks'] == row['start_ticks']
    captured = job['exit_status_proof']
    assert captured['pid'] == row['pid'] and captured['start_ticks'] == row['start_ticks']
    assert captured['wait_status'] == 0 and os.waitstatus_to_exitcode(captured['wait_status']) == 0
    assert captured['source'] == '/proc/PID/stat field52; original parent stopped'
    assert len(captured['stat_sha256']) == 64
    retired = load(retirement_path)
    assert retired['pid'] == row['parent_pid'] and retired['start_ticks'] == row['parent_start_ticks']
    assert retired['native_workers_signaled'] is False
    original_parent = process(row['parent_pid'])
    parent_live_at_birth = original_parent['live'] and original_parent.get('start_ticks') == row['parent_start_ticks']
    assert not parent_live_at_birth
    run = Path(job['run_directory'])
    status_sha = digest(run / 'status.json')
    native = audit_scene(run, job)
    assert native == {'completed_episodes': 20, 'identity_count': 20}
    independent = False
    reason = 'Original launcher record no longer retained after original CPU parent retirement'
    proc_path = Path('/proc') / str(row['pid']) / 'stat'
    if proc_path.is_file():
        raw = proc_path.read_text()
        fields = raw[raw.rfind(')') + 2:].split()
        if fields[19] == row['start_ticks'] and fields[0] == 'Z' and int(fields[1]) == row['parent_pid']:
            assert int(fields[49]) == 0 and hashlib.sha256(raw.encode()).hexdigest() == captured['stat_sha256']
            independent = True
            reason = 'Exact original retained zombie record independently matches the controller capture'
    assert source_fingerprint() == fingerprint and digest(OUT / 'meta/calibration.json') == parameter_sha
    assert digest(run / 'status.json') == status_sha
    now = datetime.now(timezone.utc)
    proof = {'checked_at_utc': now.isoformat(), 'stage': 'full', 'diagnostic_only': True,
        'formal400_complete': False, 'scene': args.scene, 'gpu': job['gpu'],
        'scope': 'Actual completed adopted scene, native coverage, controller-captured OS exit and CPU retirement; independent OS recheck availability stated separately',
        'native_completion_audit': native, 'native_status_sha256': status_sha,
        'native_returncode': job['returncode'], 'controller_recorded_OS_exit': captured,
        'independent_proc_stat_recheck_available': independent, 'independent_recheck_reason': reason,
        'retirement_path': str(retirement_path.relative_to(OUT)), 'retirement_sha256': digest(retirement_path),
        'retirement': retired, 'original_parent_live_at_original_birth': False,
        'native_source_entries': len(fingerprint), 'parameter_sha256': parameter_sha,
        'reader_source': str(Path(__file__).relative_to(OUT)), 'reader_sha256': digest(Path(__file__)),
        'controller_sha256': plan['controller_sha256'], 'plan': str(plan_path.relative_to(OUT)),
        'plan_sha256': digest(plan_path), 'controller_pid': start['pid'], 'controller_start_ticks': start['start_ticks'],
        'source_and_parameters_unchanged': True, 'native_states_modified': False,
        'signals_sent_by_this_reader': False}
    destination = plan_path.parent / 'native-completion-audits' / (now.strftime('%Y%m%dT%H%M%S%fZ') + '-' + args.scene + '.json')
    destination.parent.mkdir(exist_ok=True)
    assert not destination.exists()
    write(destination, proof)
    print(json.dumps({'proof': str(destination.relative_to(OUT)), **proof}))


if __name__ == '__main__':
    main()
