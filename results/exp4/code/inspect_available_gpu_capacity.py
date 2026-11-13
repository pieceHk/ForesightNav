from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from available_gpu_evaluate_sequence import gpu_inventory
from evaluate_sequence import OUT, digest, load, source_fingerprint

stage = sys.argv[1] if len(sys.argv) > 1 else load(OUT/'meta/last-evaluation-observer.json')['stage']
assert stage in ('nearest', 'myopic', 'suffix_only')
start = load(OUT/f'meta/{stage}-start-a3.json')
plan = load(OUT/start['scheduling_plan'])
policy = load(OUT/'meta/evaluation-available-gpu-policy.json')
assert start['controller_sha256'] == plan['controller_sha256'] == digest(Path(start['command'][2]))
assert start['source_fingerprint'] == source_fingerprint()
assert start['parameter_sha256'] == digest(OUT/'meta/calibration.json')
assert policy['reserved_gpus'] == [0, 3]
active = plan['gpu_memory_policy']
assert all(policy[key] == active[key] for key in ('min_free_mib', 'max_existing_compute_mib'))
candidates = sorted({1, 2, 4, 5, 6, 7} - set(policy['allowed_gpus']))
inventory = gpu_inventory()
rows = []
for gpu in inventory['gpus']:
    index = int(gpu['index'])
    apps = [a for a in inventory['compute_apps'] if a['gpu_uuid'] == gpu['uuid']]
    try:
        occupied = [int(a['used_gpu_memory']) for a in apps]
        compute = sum(occupied)
        free = int(gpu['memory.free'])
        known = free >= 0 and all(value >= 0 for value in occupied)
    except (KeyError, TypeError, ValueError):
        free = compute = None
        known = False
    rows.append({'gpu': index, 'free_mib': free, 'all_compute_mib': compute,
                 'compute_processes': len(apps), 'reserved': index in (0, 3),
                 'memory_values_known': known,
                 'extra_idle_card_at_baseline_gate':
                 known and index in candidates and free >= 34816 and compute <= 10240,
                 'extra_idle_card_at_active_gate': known and index in candidates
                 and free >= active['min_free_mib'] and compute <= active['max_existing_compute_mib']})
value = {'checked_at_utc': datetime.now(timezone.utc).isoformat(), 'stage': stage,
    'active_controller_source_sha256': start['controller_sha256'],
    'gpus': rows, 'extra_card_gate': {'free_mib_at_least': 34816,
        'all_compute_mib_at_most': 10240, 'candidate_gpus': candidates},
    'active_extra_card_gate': {'free_mib_at_least': active['min_free_mib'],
        'all_compute_mib_at_most': active['max_existing_compute_mib'], 'candidate_gpus': candidates},
    'active_allowed_gpus': policy['allowed_gpus'],
    'eligible_additional_candidates': [row['gpu'] for row in rows if row['extra_idle_card_at_active_gate']],
    'scope': 'Read-only prospective capacity check using the active dispatcher admission rule; no GPU allocation change',
    'source_and_parameters_unchanged': True, 'native_states_modified': False, 'signals_sent': False}
stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
path = OUT/f'diagnostics/resource-disclosure/available-extra-gpu-capacity-{stamp}.json'
with path.open('x') as stream: json.dump(value, stream, indent=2)
print(json.dumps({'proof': path.relative_to(OUT).as_posix(), **value}))
