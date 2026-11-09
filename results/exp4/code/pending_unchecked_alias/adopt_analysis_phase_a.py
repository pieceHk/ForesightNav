"""Stop only the idle CPU waiter, archive its binding, and install tested analysis.

Run once after inspecting the recorded validations. Native inputs and workers
are never changed or signalled. A separate launch creates the new CPU binding.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from experiments.exp4_collect import source_fingerprint

REPO = Path('/nvme/khy/ForesightNav_exp4_depthfix')
ROOT = REPO / 'results/exp4'
PENDING = ROOT / 'code/pending_unchecked_alias'
STATE = ROOT / 'meta/calibration-postprocess.json'
RECORD = ROOT / 'meta/unchecked-alias-handoff.json'
BEFORE = 'a209d9e9f1ffab098198c8466c5b64b81dfd9577e105a89df448c91ba9d7a24d'
AFTER = 'a311b37d45d59f379ce00b58d1350b5eb308da0b9a24d34ad2facdb495d9991d'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    temp.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temp, path)


def proc(pid):
    path = Path('/proc') / str(pid)
    if not path.exists():
        return None
    stat = (path / 'stat').read_text().rsplit(')', 1)[1].split()
    return {'pid': pid, 'state': stat[0], 'start_ticks': stat[19],
            'cwd': str((path / 'cwd').resolve()), 'uid': path.stat().st_uid,
            'command': (path / 'cmdline').read_bytes().replace(b'\0', b' ').decode()}


def main():
    assert not RECORD.exists(), 'Inspect the existing handoff instead of repeating it'
    normalized = json.loads((ROOT / 'diagnostics/unchecked-alias-native300-validation.json').read_text())
    replay = json.loads((ROOT / 'diagnostics/unchecked-alias-XB-validation.json').read_text())
    assert normalized['episodes'] == 300 and normalized['native_episode_rows_identical']
    assert normalized['decision_rows_identical'] and len(normalized['raw_association_gaps_retained']) == 28
    assert normalized['source_after']['candidate_identities.py'] == AFTER
    assert replay['all_closed_calibration']['q_pi_Cexp_inputs_identical']
    assert replay['all_closed_calibration']['episodes'] == 300
    assert replay['source_sha256']['proposed_identity_analysis'] == AFTER
    assert sha(PENDING / 'proposed_candidate_identities.py') == AFTER
    assert sha(ROOT / 'code/candidate_identities.py') == BEFORE
    assert sha(PENDING / 'base_candidate_identities.py') == BEFORE
    assert 'Ran 6 tests' in (PENDING / 'tests.txt').read_text() and (PENDING / 'tests.txt').read_text().rstrip().endswith('OK')
    assert 'Ran 51 tests' in (PENDING / 'existing-analysis-report-tests.txt').read_text() and (PENDING / 'existing-analysis-report-tests.txt').read_text().rstrip().endswith('OK')
    old = json.loads(STATE.read_text())
    assert old['pid'] == 122348 and str(old['start_ticks']) == '2658437'
    assert old['status'] == 'waiting_for_complete_collection' and not old['completed_steps']
    assert not (ROOT / 'meta/calibration-fit-binding.json').exists()
    assert not (ROOT / 'meta/calibration.json').exists()
    assert len(old['script_sha256']) == 11 and all(sha(ROOT / 'code'/name) == value for name,value in old['script_sha256'].items())
    assert len(source_fingerprint()) == 160 and source_fingerprint() == old['source_fingerprint']
    assert Path('/proc/sys/kernel/random/boot_id').read_text().strip() == old['boot_id']
    ledger = json.loads((ROOT / 'meta/calibration_runs.json').read_text())
    assert ledger['status'] == 'running'
    cp = proc(old['pid'])
    assert cp and cp['state'] != 'Z' and cp['start_ticks'] == str(old['start_ticks'])
    assert cp['cwd'] == str(REPO) and cp['uid'] == os.getuid()
    assert 'finish_calibration.py --wait' in cp['command']
    observer = json.loads((ROOT / 'meta/last-calibration-observer.json').read_text())
    op = proc(observer['observer_pid'])
    assert observer['observer_pid'] == 279117 and op and op['start_ticks'] == '7306148'
    audit = json.loads((ROOT / 'meta/current-live-audit.json').read_text())
    assert audit['controller_live_and_matching'] and audit['boot_id_matches'] and audit['frozen_files_unchanged']
    assert audit['completed_episodes'] < 360
    native = [proc(31036)] + [proc(j['pid']) for j in audit['jobs'] if j['status'] == 'running']
    assert all(p and p['state'] != 'Z' for p in native)
    value = {'started_at_utc': datetime.now(timezone.utc).isoformat(),
             'phase': 'stopping_idle_CPU_waiter', 'reason': 'checked-point aliases for accurate invalid pursuit paths',
             'old_CPU_state': old, 'old_CPU_process': cp, 'old_observer': observer,
             'old_observer_process': op, 'native_processes_before': native,
             'source_before_sha256': BEFORE, 'source_after_sha256': AFTER,
             'native_process_signals_sent': False, 'external_process_signals_sent': False,
             'fits_started': False, 'validation_sha256': {
                 'normalized300': sha(ROOT / 'diagnostics/unchecked-alias-native300-validation.json'),
                 'XB_replay': sha(ROOT / 'diagnostics/unchecked-alias-XB-validation.json'),
                 'six_alias_tests': sha(PENDING / 'tests.txt'),
                 'existing_51_tests': sha(PENDING / 'existing-analysis-report-tests.txt')}}
    dump(RECORD, value)
    os.kill(cp['pid'], signal.SIGTERM)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and (current := proc(cp['pid'])) and current['state'] != 'Z' and current['start_ticks'] == cp['start_ticks']:
        time.sleep(.2)
    current = proc(cp['pid'])
    assert current is None or current['state'] == 'Z' or current['start_ticks'] != cp['start_ticks']
    stopped = json.loads(STATE.read_text())
    assert stopped['pid'] == cp['pid'] and stopped['status'] == 'stopped_for_review' and not stopped['completed_steps']
    value['phase'] = 'idle_CPU_stopped_waiting_for_readonly_observer'
    value['CPU_signal'] = 'SIGTERM to exact owned waiting PID only'
    dump(RECORD, value)
    print(json.dumps({'phase': value['phase'], 'native_workers_unchanged': True}), flush=True)
    deadline = time.monotonic() + 50
    while time.monotonic() < deadline and (current := proc(op['pid'])) and current['state'] != 'Z' and current['start_ticks'] == op['start_ticks']:
        time.sleep(.5)
    current = proc(op['pid'])
    assert current is None or current['state'] == 'Z' or current['start_ticks'] != op['start_ticks'], 'Inspect the readonly observer before continuing'
    rows = [json.loads(x) for x in Path(observer['log']).read_text().splitlines()]
    assert rows[-1]['observation_finished_reason'] == 'CPU_postprocess_requires_review'
    assert rows[-1]['CPU_status'] == 'stopped_for_review'
    assert all(all(row[k] for k in ['controller_live_matching','boot_matches','source_unchanged']) for row in rows)
    assert all(j['live_matching'] and j['shadow_error_bytes'] == 0 for row in rows for j in row['active_jobs'])
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    archive = ROOT / ('archive/calibration-postprocess-before-unchecked-alias-' + stamp)
    archive.mkdir(parents=True)
    dump(archive / 'state-before-stop.json', old)
    STATE.rename(archive / 'state-stopped.json')
    value['archived_CPU_state'] = str((archive / 'state-stopped.json').relative_to(ROOT))
    value['old_observer_last_record'] = rows[-1]
    value['old_observer_tool_exit_code'] = None
    target = ROOT / 'code/candidate_identities.py'
    temporary = target.with_name(target.name + '.validated-alias.tmp')
    temporary.write_bytes((PENDING / 'proposed_candidate_identities.py').read_bytes())
    temporary.chmod(target.stat().st_mode & 0o777)
    os.replace(temporary, target)
    assert sha(target) == AFTER
    differences = [name for name,value in old['script_sha256'].items() if sha(ROOT/'code'/name) != value]
    assert differences == ['candidate_identities.py'], differences
    assert source_fingerprint() == old['source_fingerprint']
    for before in native:
        after = proc(before['pid'])
        assert after and after['state'] != 'Z' and after['start_ticks'] == before['start_ticks']
    value.update(phase='ready_to_launch_new_CPU_waiter', completed_at_utc=datetime.now(timezone.utc).isoformat(),
                 changed_CPU_scripts=differences, runtime160_unchanged=True,
                 native_processes_after=[proc(x['pid']) for x in native])
    dump(RECORD, value)
    print(json.dumps({'phase': value['phase'], 'changed_CPU_scripts': differences,
                      'archived_state': value['archived_CPU_state'], 'native160_unchanged': True}), flush=True)


if __name__ == '__main__':
    main()
