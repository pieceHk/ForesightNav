"""Verify a real adopted scene exit against the retained Linux zombie record."""
import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import subprocess

from evaluate_sequence import OUT, audit_scene, digest, load, source_fingerprint, write
from watch_evaluation import process


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    ledger = load(OUT/'meta/evaluation-full-runs.json')
    plan_path = OUT/ledger['scheduling']['plan']
    plan = load(plan_path)
    adopted = {row['scene']:row for row in plan['adopted_launchers']}
    candidates = []
    for job in ledger['jobs']:
        if job['scene'] not in adopted or job['status'] != 'completed':
            continue
        row = adopted[job['scene']]
        parent = process(row['parent_pid'])
        if (Path('/proc')/str(job['pid'])/'stat').is_file() and parent.get('state') in ('T','t'):
            candidates.append(job)
    if not candidates or args.check:
        import json
        print(json.dumps({'ready':bool(candidates),'completed_adopted_scenes_with_retained_parent':[job['scene'] for job in candidates],
            'scope':'Read-only readiness; does not certify a scene exit','native_workers_signaled':False}))
        return
    fingerprint = source_fingerprint()
    parameter_sha = digest(OUT/'meta/calibration.json')
    assert fingerprint == plan['source_fingerprint'] == ledger['source_fingerprint']
    assert parameter_sha == plan['parameter_sha256'] == ledger['parameter_sha256']
    assert digest(OUT/'code/three_gpu_evaluate_sequence.py') == plan['controller_sha256']
    assert digest(OUT/'code/evaluate_sequence.py') == plan['base_controller_sha256']
    records = []
    for job in candidates:
        row = adopted[job['scene']]
        assert row['pid'] == job['pid'] and row['start_ticks'] == job['pid_start_ticks']
        assert job['returncode'] == 0 and job['adopted_without_native_restart']
        status = job['exit_status_proof']
        raw = (Path('/proc')/str(job['pid'])/'stat').read_text()
        fields = raw[raw.rfind(')')+2:].split()
        assert fields[0] == 'Z' and fields[19] == row['start_ticks']
        assert int(fields[1]) == row['parent_pid'] and int(fields[49]) == 0
        assert status['wait_status'] == 0 and status['stat_sha256'] == hashlib.sha256(raw.encode()).hexdigest()
        parent = process(row['parent_pid'])
        assert parent['state'] in ('T','t') and parent['start_ticks'] == row['parent_start_ticks']
        ps = subprocess.check_output(['ps','-o','stat=,ppid=','-p',str(job['pid'])],text=True).split()
        assert ps[0].startswith('Z') and int(ps[1]) == row['parent_pid']
        native = audit_scene(Path(job['run_directory']),job)
        assert native['completed_episodes'] == 20 and native['identity_count'] == 20
        records.append({'scene':job['scene'],'gpu':job['gpu'],'pid':job['pid'],
            'start_ticks':row['start_ticks'],'parent_pid':row['parent_pid'],
            'parent_start_ticks':row['parent_start_ticks'],'proc_stat_exact':raw,
            'ps_state_and_parent':ps,'controller_exit_status_proof':status,
            'native_completion_audit':native,'native_status_sha256':digest(Path(job['run_directory'])/'status.json')})
    assert source_fingerprint() == fingerprint and digest(OUT/'meta/calibration.json') == parameter_sha
    now = datetime.now(timezone.utc)
    proof = {'checked_at_utc':now.isoformat(),'scope':'Actual complete adopted scenes with independently read retained OS exit records; not a synthetic test or Full400 completion',
        'actual_OS_exit_status_verified':True,'source_and_parameters_unchanged':True,
        'native_workers_signaled':False,'native_source_entries':len(fingerprint),
        'parameter_sha256':parameter_sha,'source_sha256':digest(Path(__file__)),
        'controller_sha256':plan['controller_sha256'],'scheduling_plan_sha256':digest(plan_path),
        'scenes':records,'next_jobs_on_same_gpus':[{'scene':job['scene'],'gpu':job['gpu'],
            'pid':job['pid'],'status':job['status']} for job in ledger['jobs']
            if job['status']=='running' and job['gpu'] in {row['gpu'] for row in records}]}
    destination = plan_path.parent/'native-exit-audits'/(now.strftime('%Y%m%dT%H%M%S%fZ')+'.json')
    destination.parent.mkdir(exist_ok=True)
    assert not destination.exists()
    write(destination,proof)
    import json
    print(json.dumps({'proof':str(destination.relative_to(OUT)),**proof}))


if __name__ == '__main__':
    main()
