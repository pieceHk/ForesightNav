"""Check current remaining-stage denials without authorizing or launching work."""
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import subprocess
import sys
from evaluate_sequence import OUT, digest, source_fingerprint, write


def main():
    stages=('nearest','myopic','suffix_only')
    files=[f'meta/evaluation-{s}-runs.json' for s in stages]
    files += [f'meta/{s}-start-a3.json' for s in stages]
    files += [f'meta/{s}-shared-gpu-launch-a3.json' for s in stages]
    files += [f'logs/evaluation-{s}-controller-a3.log' for s in stages]
    before={p:(OUT/p).exists() for p in files}
    assert not any(before.values())
    native=source_fingerprint()
    parameter=digest(OUT/'meta/calibration.json')
    controller=OUT/'code/shared_gpu_evaluate_sequence.py'
    controller_sha=digest(controller)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    log=OUT/'logs'/f'shared-remaining-stage-preflight-{stamp}.jsonl'
    checks=[]
    environment=dict(os.environ,CUDA_VISIBLE_DEVICES='')
    with log.open('x') as handle:
        for stage in stages:
            result=subprocess.run([sys.executable,str(controller),'--stage',stage,'--check','--gpus','1,2,3'],
                cwd=OUT.parents[1],env=environment,capture_output=True,text=True)
            assert result.returncode==0, result.stderr[-1000:]
            readiness=json.loads(result.stdout)
            assert readiness['ready'] is False and readiness['native_workers_started'] is False
            assert readiness['episodes']==400 and readiness['scenes']==20
            assert readiness['source_sha256']==controller_sha and readiness['parameter_sha256']==parameter
            first=readiness['reasons'][0]
            if stage=='nearest':
                assert first=='The preceding evaluation stage is not complete'
            else:
                predecessor='nearest' if stage=='myopic' else 'myopic'
                assert 'Missing checkpoint artifact:' in first and f'evaluation-{predecessor}-runs.json' in first
            row={'stage':stage,'exit_code':result.returncode,'actual_readiness':readiness,
                 'predecessor_gate_denial':first}
            checks.append(row)
            handle.write(json.dumps(row)+'\n')
    after={p:(OUT/p).exists() for p in files}
    assert after==before and source_fingerprint()==native
    assert digest(OUT/'meta/calibration.json')==parameter and digest(controller)==controller_sha
    proof={'checked_at_utc':datetime.now(timezone.utc).isoformat(),'checks':checks,
        'artifacts_before':before,'artifacts_after':after,'native_workers_launched':False,
        'source_and_parameters_unchanged':True,'controller_sha256':controller_sha,
        'parameter_sha256':parameter,'source_sha256':digest(Path(__file__)),
        'log':str(log.relative_to(OUT)),'log_sha256':digest(log),
        'scope':'Actual read-only CLI checks: Full incomplete and later predecessor artifacts absent; CP4-reported acceptance after Full400 is not certified by this preflight'}
    target=OUT/'diagnostics/shared-remaining-stage-preflight'/f'{stamp}.json'
    target.parent.mkdir(exist_ok=True)
    write(target,proof)
    print(json.dumps({'proof':str(target.relative_to(OUT)),**proof}))


if __name__=='__main__':
    main()
