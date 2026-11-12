"""Generate a fresh plan only after a separately recorded explicit user approval."""
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import re
import sys
from evaluate_sequence import OUT, load, digest, source_fingerprint
from watch_evaluation import process

approval = load(OUT / 'meta/thp-defrag-user-approval.json')
assert approval['approved'] is True
assert approval['scope'] == 'temporary_host_thp_defrag_madvise_to_defer'
action = sys.argv[1]
assert action in ('apply', 'restore')
target = Path('/sys/kernel/mm/transparent_hugepage/defrag')
current = re.findall(r'\[([^\]]+)\]', target.read_text())[0]
expected, desired = ('madvise','defer') if action == 'apply' else ('defer','madvise')
assert current == expected
start = load(OUT / 'meta/nearest-start-a3.json')
if action == 'apply':
    value = process(start['pid'])
    assert value['live'] and value['start_ticks'] == start['start_ticks']
    assert source_fingerprint() == start['source_fingerprint']
    assert digest(OUT / 'meta/calibration.json') == start['parameter_sha256']
stamp = datetime.now(timezone.utc)
plan = {'created_at_utc':stamp.isoformat(),'action':action,'target':str(target),
    'expected_value':expected,'desired_value':desired,'operator_uid':os.getuid(),
    'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
    'explicit_user_approval':True,'approved_scope':approval['scope'],
    'approval_sha256':digest(OUT / 'meta/thp-defrag-user-approval.json'),
    'controller_pid':start['pid'],'controller_start_ticks':start['start_ticks'],
    'controller_sha256':start['controller_sha256'],'parameter_sha256':start['parameter_sha256'],
    'signals_sent':False}
path = Path('/tmp') / (stamp.strftime('exp4-thp-defrag-plan-%Y%m%dT%H%M%SZ.json'))
fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, 'w') as stream:
    json.dump(plan, stream, indent=2)
print(json.dumps({'plan_path':str(path),'action':action,'expected_value':expected,
                  'desired_value':desired,'settings_changed':False}))
