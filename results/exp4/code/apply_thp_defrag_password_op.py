"""One reversible, explicitly approved host setting; never signal a worker."""
import subprocess
import sys

ROOT_WORKER = r'''
from datetime import datetime, timezone
from pathlib import Path
import hashlib, json, os, re, sys
TARGET = Path('/sys/kernel/mm/transparent_hugepage/defrag')
REPO = Path('/nvme/khy/ForesightNav_exp4_depthfix')
OUT = REPO / 'results/exp4'
def active():
    text = TARGET.read_text().strip()
    selected = re.findall(r'\[([^\]]+)\]', text)
    assert len(selected) == 1
    return text, selected[0]
before_text, before = active()
if sys.argv[1] == 'check':
    assert before == 'madvise', 'Original policy changed; review before approval'
    print(json.dumps({'status':'read_only_preflight_passed','target':str(TARGET),
        'before':before,'proposed':'defer','rollback':'madvise',
        'available_modes':before_text,'settings_changed':False,'signals_sent':False,
        'scope':'Host-wide THP defrag policy only. Explicit user approval is required before apply.'}))
    raise SystemExit(0)
assert os.geteuid() == 0
path = Path(sys.argv[1])
assert re.fullmatch(r'/tmp/exp4-thp-defrag-plan-[0-9TZ]+\.json', str(path)) and not path.is_symlink()
plan = json.loads(path.read_text())
assert path.stat().st_uid == plan['operator_uid'] == 1001
assert plan['explicit_user_approval'] is True
assert plan['approved_scope'] == 'temporary_host_thp_defrag_madvise_to_defer'
assert plan['target'] == str(TARGET)
assert plan['boot_id'] == Path('/proc/sys/kernel/random/boot_id').read_text().strip()
age = (datetime.now(timezone.utc) - datetime.fromisoformat(plan['created_at_utc'])).total_seconds()
assert 0 <= age <= 180
action = plan['action']
assert action in ('apply', 'restore')
expected, desired = ('madvise','defer') if action == 'apply' else ('defer','madvise')
assert plan['expected_value'] == expected and plan['desired_value'] == desired
assert before == expected, 'Host policy changed; do not overwrite another operator'
if action == 'apply':
    start = json.loads((OUT / 'meta/nearest-start-a3.json').read_text())
    assert start['pid'] == plan['controller_pid'] and start['start_ticks'] == plan['controller_start_ticks']
    proc = Path('/proc') / str(start['pid'])
    raw = (proc / 'stat').read_text(); fields = raw[raw.rfind(')')+2:].split()
    assert proc.stat().st_uid == plan['operator_uid'] and fields[19] == start['start_ticks'] and fields[0] != 'Z'
    assert hashlib.sha256((OUT / 'meta/calibration.json').read_bytes()).hexdigest() == plan['parameter_sha256'] == start['parameter_sha256']
    source = Path(start['command'][2]); assert source.resolve().is_relative_to(OUT.resolve())
    assert hashlib.sha256(source.read_bytes()).hexdigest() == start['controller_sha256'] == plan['controller_sha256']
with TARGET.open('w') as stream:
    stream.write(desired + '\n')
after_text, after = active()
assert after == desired
print(json.dumps({'status':'applied' if action == 'apply' else 'restored',
    'at_utc':datetime.now(timezone.utc).isoformat(),'target':str(TARGET),
    'before':before,'after':after,'after_text':after_text,'boot_id':plan['boot_id'],
    'only_one_sysfs_setting_written':True,'signals_sent':False,
    'native_sources_changed':False,'native_parameters_changed':False,
    'scope':'Host-wide memory-allocation policy. Retain measurements and restore original policy at experiment end.'}))
'''

if sys.argv[1] == 'check':
    exec(compile(ROOT_WORKER, '<exp4-thp-readonly-check>', 'exec'))
else:
    result = subprocess.run(['sudo', '-S', '-p', '__EXP4_THP_SUDO__',
                             sys.executable, '-c', ROOT_WORKER, sys.argv[1]])
    raise SystemExit(result.returncode)
