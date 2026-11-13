"""Replay immutable online snapshots, including recursion and complete orders.

This CPU diagnostic verifies recorded online decisions; it never substitutes
counterfactual scores for the four actual online ablations.
"""
from collections import Counter
from dataclasses import asdict
from datetime import datetime,timezone
from pathlib import Path
import json,math

EXACT_FIELDS=('variant','k','ordered_raw_ranks','chosen_raw_rank','unconstrained_raw_rank',
    'nearest_raw_rank','hysteresis_held','switched','fallback','fallback_counts','memo_states')
FLOAT_FIELDS=('E','S','q','B','d','V_succ','V_fail','Q')

def require(condition,message):
    if not condition:raise ValueError(message)

def key(row):return (str(row['scene_id']),str(row['episode_id']),row['t'],row['k'])

def verify_row(row,observed,variant,parameters,parameter_binding,solver,factory):
    require(key(row)==key(observed),'Snapshot and decision identities differ')
    require(row['shadow'] is False,'Shadow snapshot cannot certify an online variant')
    require(row['parameters']==parameters and row['parameter_binding']==parameter_binding,'Frozen parameter or held-out fold binding differs')
    snapshot=row['snapshot'];k=row['k']
    require(type(k) is int and k in range(3) and snapshot['k']==k,'Changed target index')
    layers=snapshot['layers'];require(len(layers)==3,'Changed three-goal snapshot')
    require(all(len(layer)<=5 for layer in layers),'Snapshot candidate cap changed')
    require(all([c['raw_rank'] for c in layer]==sorted(c['raw_rank'] for c in layer) for layer in layers),'Snapshot layers are not raw-rank sorted')
    if variant=='myopic':require(not any(layers[k+1:]),'Myopic queried future targets')
    ranks=snapshot['all_raw_ranks']
    require(len(ranks)==len(set(ranks)),'Repeated input candidate raw rank')
    order=observed['ordered_raw_ranks']
    require(len(order)==len(ranks) and len(order)==len(set(order)) and set(order)==set(ranks),'Output is not the complete input permutation')
    replay=solver(snapshot,factory(**parameters),variant,row['active_raw_rank'])
    for field in EXACT_FIELDS:require(observed.get(field)==replay[field],'Replay differs in '+field)
    require(len(observed['candidates'])==len(replay['candidates']),'Replay candidate cardinality differs')
    max_error=0.
    for actual,expected in zip(observed['candidates'],replay['candidates']):
        for field in ('raw_rank','N_views','location','position'):
            require(actual.get(field)==expected[field],'Replay candidate differs in '+field)
        for field in FLOAT_FIELDS:
            a,b=actual[field],expected[field]
            require(type(a) in (int,float) and math.isfinite(a),'Non-finite recorded candidate value')
            require(math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-8),'Replay recursive value differs in '+field)
            max_error=max(max_error,abs(a-b))
    return {'candidate_count':len(replay['candidates']),'complete_order_count':len(order),
        'max_float_abs_error':max_error,'empty_root':not bool(replay['candidates'])}

def read_rows(paths):
    return [json.loads(line) for path in sorted(paths) for line in path.read_text().splitlines() if line]

def audit_scope(require_all=False):
    import evaluate_sequence as base
    from pbsp.navigation.blp import BeliefParameters,solve_snapshot
    from report_variants import CalibrationDocument
    root=base.OUT/'scopes/first8';scope=base.load(root/'meta/scope.json')
    require(base.source_fingerprint()==scope['source_fingerprint'],'Scientific sources changed')
    require(base.digest(base.OUT/'meta/calibration.json')==scope['parameter_sha256'],'Frozen calibration changed')
    document=CalibrationDocument.from_mapping(base.load(base.OUT/'meta/calibration.json'))
    helper_sha=base.digest(Path(__file__));results=[];pending=[]
    for variant in ('full','nearest','myopic','suffix_only'):
        ledger=root/f'meta/evaluation-{variant}-runs.json'
        if not ledger.exists():pending.append(variant);continue
        jobs=base.load(ledger)['jobs']
        if len(jobs)!=8 or any(j['status']!='completed' or j.get('returncode')!=0 for j in jobs):pending.append(variant)
        for job in jobs:
            if job['status']!='completed' or job.get('returncode')!=0:continue
            run=Path(job['run_directory']);base.audit_scene(run,job)
            expected=[r for r in scope['identities'] if r['scene_id']==job['scene']]
            require(job['identities']==expected and len(expected)==20,'Changed fixed scene assignment')
            paths=sorted([*run.rglob('blp_snapshots.jsonl'),*run.rglob('blp_decisions.jsonl')])
            inputs={str(p.relative_to(run)):base.digest(p) for p in paths}
            binding={'helper_sha256':helper_sha,'scene':job['scene'],'variant':variant,'native_run':str(run),
                'launch_sha256':base.digest(run/'launch.json'),'status_sha256':base.digest(run/'status.json'),
                'input_sha256':inputs,'source_fingerprint':scope['source_fingerprint'],'parameter_sha256':scope['parameter_sha256']}
            path=root/f'diagnostics/snapshot-replay/{variant}/{job["scene"]}.json'
            if path.exists():
                proof=base.load(path)
                require(proof['binding']==binding and proof['passed'] is True,'Existing snapshot audit binding changed')
                results.append({'proof':str(path.relative_to(base.OUT)),'proof_sha256':base.digest(path),**{k:proof[k] for k in ('variant','scene','n_snapshots','n_candidates','max_float_abs_error')}})
                continue
            snapshots=read_rows(run.rglob('blp_snapshots.jsonl'));observed=read_rows(run.rglob('blp_decisions.jsonl'))
            require(len(snapshots)==len(observed)>0,'Snapshot and decision counts differ')
            require(len({key(r) for r in snapshots})==len(snapshots),'Repeated snapshot identity requires review')
            require(len({key(r) for r in observed})==len(observed),'Repeated decision identity requires review')
            by_key={key(r):r for r in observed};require(set(by_key)=={key(r) for r in snapshots},'Missing online snapshot or decision')
            wanted={(r['scene_id'],str(r['runtime_episode_id'])) for r in expected}
            require({key(r)[:2] for r in snapshots}==wanted,'Snapshots do not cover all fixed20 episodes')
            values,parameter_binding=document.resolve(job['scene']);parameters=asdict(values)
            parameter_binding=dict(parameter_binding,calibration_sha256=scope['parameter_sha256'])
            counts=Counter();max_error=0.
            for row in snapshots:
                checked=verify_row(row,by_key[key(row)],variant,parameters,parameter_binding,solve_snapshot,BeliefParameters)
                counts['snapshots']+=1;counts['candidates']+=checked['candidate_count'];counts['empty_roots']+=checked['empty_root']
                max_error=max(max_error,checked['max_float_abs_error'])
            proof={'at_utc':datetime.now(timezone.utc).isoformat(),'passed':True,'variant':variant,'scene':job['scene'],
                'binding':binding,'n_episodes':20,'n_snapshots':counts['snapshots'],'n_candidates':counts['candidates'],
                'empty_roots':counts['empty_roots'],'max_float_abs_error':max_error,
                'all_complete_input_permutations_verified':True,'actual_variant_recursion_values_replayed':True,
                'myopic_future_layers_empty_verified':variant=='myopic','Native_or_GPU_workers_started':False,
                'original_online_planning_times_preserved':True}
            path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(proof,indent=2)+'\n')
            results.append({'proof':str(path.relative_to(base.OUT)),'proof_sha256':base.digest(path),**{k:proof[k] for k in ('variant','scene','n_snapshots','n_candidates','max_float_abs_error')}})
    require(not require_all or not pending,'Snapshot audit still needs complete variants: '+', '.join(pending))
    value={'at_utc':datetime.now(timezone.utc).isoformat(),'helper_sha256':helper_sha,'audits':results,'pending_whole_variants':pending,
        'all_four_complete':not pending,'episodes_verified':20*len(results),'snapshots_verified':sum(r['n_snapshots'] for r in results),
        'source_fingerprint':scope['source_fingerprint'],'parameter_sha256':scope['parameter_sha256'],
        'definition':'Replay of actual online variant inputs, not counterfactual online ablation scores; measured plan_ms unchanged',
        'Native_or_GPU_workers_started':False}
    target=root/'diagnostics/snapshot-replay/current.json';target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(value,indent=2)+'\n');return value

if __name__=='__main__':
    result=audit_scope();print(json.dumps({k:v for k,v in result.items() if k not in ('audits','source_fingerprint')}),flush=True)
