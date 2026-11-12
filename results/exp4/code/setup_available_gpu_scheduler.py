from datetime import datetime, timezone
from pathlib import Path
import hashlib
import importlib.util
import json
import subprocess
import sys
import unittest

from evaluate_sequence import OUT, digest, load, source_fingerprint
from watch_evaluation import process

fingerprint=source_fingerprint();parameter=digest(OUT/'meta/calibration.json')
assert len(fingerprint)==161 and fingerprint==load(OUT/'meta/CP2b_acceptance.json')['source_fingerprint']
assert digest(OUT/'code/shared_gpu_evaluate_sequence.py')=='e4f9b37823c4ccfc9a884ae496bf044133f6d12a4582f7bf30d20ff5ca779eb4'
assert digest(OUT/'code/evaluate_sequence.py')=='d238350a2733b56d258e774e5ca4f1537f09818cd34a313a638b9ec483c0b82f'
assert not (OUT/'meta/nearest-start-a3.json').exists()
assert not (OUT/'meta/evaluation-nearest-runs.json').exists()
guard=load(OUT/'meta/gpu3-guard-start.json')
assert not load(OUT/'meta/gpu3-guard-control.json')['enabled'] and not process(guard['pid'])['live']

source=(OUT/'code/shared_gpu_evaluate_sequence.py').read_text()
source=source.replace('with bounded GPU-memory sharing.', 'on the current user-approved available GPU pool.',1)
source=source.replace("MEMORY_POLICY = {'min_free_mib': 38 * 1024, 'max_existing_compute_mib': 8 * 1024}",
    "MEMORY_POLICY = {'min_free_mib': 34 * 1024, 'max_existing_compute_mib': 10 * 1024}\nSAFE_GPU_SET = {1, 2, 4, 5, 6, 7}")
source=source.replace('index not in (1, 2, 3)', 'index not in SAFE_GPU_SET')
source=source.replace("int(row['index']) in (1,2,3)","int(row['index']) in SAFE_GPU_SET")
source=source.replace("meta/evaluation-gpu-policy.json", "meta/evaluation-available-gpu-policy.json")
source=source.replace('set(allowed) <= {1,2,3}', 'set(allowed) <= SAFE_GPU_SET')
source=source.replace("Only GPUs1/2/3 are authorized by the latest user instruction", "Only the current available GPU pool is authorized; physical0 and3 are reserved")
source=source.replace("'User authorized evaluation on GPU1/2/3; GPU0 remains reserved'", "'User chose to preserve Exp2 and move Exp4 to other GPUs; physical0 and3 remain reserved'")
source=source.replace("parser.add_argument('--gpus', default='1,2,3')", "parser.add_argument('--gpus', default='5,6,7')")
source=source.replace("    allowed = [int(value) for value in args.gpus.split(',')]", "    assert args.stage in PREVIOUS and not args.adopt_full, 'This version only starts the remaining stages'\n    allowed = [int(value) for value in args.gpus.split(',')]")
target=OUT/'code/available_gpu_evaluate_sequence.py'
assert 'SAFE_GPU_SET' in source and "default='5,6,7'" in source
compile(source,str(target),'exec')
if target.exists(): assert target.read_text()==source
else:
    with target.open('x') as stream: stream.write(source)
policy=OUT/'meta/evaluation-available-gpu-policy.json'
document={'recorded_at_utc':datetime.now(timezone.utc).isoformat(),'allowed_gpus':[5,6,7],
    'reserved_gpus':[0,3],'user_instruction':'保留Exp2，Exp4再使用其他GPU',
    'previous_policy_retained':'meta/evaluation-gpu-policy.json',
    'min_free_mib':34816,'max_existing_compute_mib':10240,
    'scope':'CPU admission only; no frozen navigation, mapping, model, seed, episode or calibration change',
    'rationale':'Cards5/6/7 have about35GiB free and8.2GiB pre-existing compute. Prior OOM recorded about25.8GiB in the two Exp4 processes. This is a shared-resource admission choice, not a proven peak-memory bound; monitor actual headroom and preserve any infrastructure failures.'}
assert not policy.exists()
policy.write_text(json.dumps(document,indent=2,ensure_ascii=False)+'\n')
spec=importlib.util.spec_from_file_location('exp4_available_gpu_scheduler',target)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class Admission(unittest.TestCase):
    def inventory(self,free=34816,occupied=0,indices=(5,)):
        return {'gpus':[{'index':str(i),'uuid':str(i),'memory.free':str(free)} for i in indices],
            'compute_apps':[{'gpu_uuid':str(i),'used_gpu_memory':str(occupied)} for i in indices]}
    def test_free_boundary(self):
        self.assertEqual(module.eligible_gpus(self.inventory(),[5]),[5])
        self.assertEqual(module.eligible_gpus(self.inventory(free=34815),[5]),[])
    def test_compute_boundary(self):
        self.assertEqual(module.eligible_gpus(self.inventory(occupied=10240),[5]),[5])
        self.assertEqual(module.eligible_gpus(self.inventory(occupied=10241),[5]),[])
    def test_reserved_zero_and_three(self):
        self.assertEqual(module.eligible_gpus(self.inventory(indices=(0,3,5,6,7)),[0,3,5,6,7]),[5,6,7])
    def test_unrequested_card(self):
        self.assertEqual(module.eligible_gpus(self.inventory(indices=(4,5,6,7)),[5]),[5])
    def test_unknown_memory_fails_closed(self):
        inv=self.inventory();inv['gpus'][0]['memory.free']='N/A'
        self.assertEqual(module.eligible_gpus(inv,[5]),[])
        self.assertEqual(module.eligible_gpus(self.inventory(occupied=-1),[5]),[])
result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Admission))
assert result.wasSuccessful()
readiness,_=module.readiness('nearest',[5,6,7],False)
assert readiness['ready'],readiness['reasons']
assert not readiness['native_workers_started']
assert source_fingerprint()==fingerprint and digest(OUT/'meta/calibration.json')==parameter
assert digest(OUT/'code/shared_gpu_evaluate_sequence.py')=='e4f9b37823c4ccfc9a884ae496bf044133f6d12a4582f7bf30d20ff5ca779eb4'
stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
proof={'checked_at_utc':datetime.now(timezone.utc).isoformat(),'source':'code/available_gpu_evaluate_sequence.py',
    'source_sha256':digest(target),'base_controller_sha256':digest(OUT/'code/evaluate_sequence.py'),
    'reference_E4_unchanged_sha256':digest(OUT/'code/shared_gpu_evaluate_sequence.py'),
    'actual_admission_tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
    'actual_readiness':readiness,'policy':document,'parameter_sha256':parameter,
    'native_source_entries':len(fingerprint),'source_and_parameters_unchanged':True,
    'native_workers_started':False,'old_gpu3_guard_stopped':True}
path=OUT/'diagnostics/available-gpu-scheduling-preflight'/f'{stamp}.json'
path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(proof,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({'proof':path.relative_to(OUT).as_posix(),'proof_sha256':digest(path),
    'controller_sha256':digest(target),'tests':result.testsRun,'ready':readiness['ready'],
    'eligible_gpus':readiness['eligible_gpus_at_check'],'allowed_gpus':[5,6,7],
    'native_workers_started':False,'source_and_parameters_unchanged':True}))
