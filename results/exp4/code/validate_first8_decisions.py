"""Check the required decision fields and formula identities without simulation."""
import json,math,statistics
from collections import Counter

FIELDS={'episode_id','scene_id','t','k','variant','candidates','chosen_raw_rank','nearest_raw_rank',
    'unconstrained_raw_rank','ordered_raw_ranks','switched','plan_ms','fallback','fallback_counts'}
CANDIDATE_FIELDS={'raw_rank','E','N_views','S','q','B','d','V_succ','V_fail','Q'}

def require(condition,message):
    if not condition:raise ValueError(message)

def numeric(value):return type(value) in (int,float) and math.isfinite(value)

def close(left,right):return math.isclose(left,right,rel_tol=1e-8,abs_tol=1e-9)

def validate_decision_rows(rows,variant,identities,parameters_by_scene):
    wanted={(str(r['scene_id']),str(r['runtime_episode_id'])) for r in identities}
    require(len(wanted)==len(identities),'Repeated fixed episode identities')
    latencies=[];episodes=set();candidates=eligible=proposed=held=0;fallbacks=Counter();memo=Counter();incidence=Counter()
    max_errors={'S':0.,'q':0.,'B':0.,'Q':0.}
    for index,row in enumerate(rows):
        context=f'{variant} decision{index}: '
        require(FIELDS<=row.keys(),context+'Missing required decision fields')
        key=(str(row['scene_id']),str(row['episode_id']));require(key in wanted,context+'Unknown fixed identity')
        episodes.add(key);require(row['variant']==variant,context+'Changed variant')
        require(type(row['k']) is int and row['k'] in (0,1,2),context+'Runtime target index must be0-based')
        require(type(row['t']) is int and row['t']>=0,context+'Invalid runtime frame')
        require(type(row['switched']) is bool,context+'Switch marker must be boolean')
        require(numeric(row['plan_ms']) and row['plan_ms']>=0,context+'Invalid measured planning time')
        latencies.append(row['plan_ms'])
        require(row['fallback'] is None or isinstance(row['fallback'],str),context+'Invalid fallback marker')
        if row['fallback']:fallbacks[row['fallback']]+=1
        require(isinstance(row['fallback_counts'],dict) and all(type(v) is int and v>=0 for v in row['fallback_counts'].values()),context+'Invalid fallback-state counts')
        memo.update(row['fallback_counts']);incidence.update(k for k,v in row['fallback_counts'].items() if v)
        current=row['candidates'];require(isinstance(current,list) and len(current)<=5,context+'Candidate cap changed')
        order=row['ordered_raw_ranks'];require(isinstance(order,list) and all(type(v) is int and v>=0 for v in order),context+'Invalid complete order')
        require(len(order)==len(set(order)),context+'Repeated raw rank in complete order')
        ranks=[];params=parameters_by_scene[key[0]]
        for candidate in current:
            require(CANDIDATE_FIELDS<=candidate.keys(),context+'Missing candidate fields')
            rank=candidate['raw_rank'];require(type(rank) is int and rank>=0,context+'Invalid raw rank');ranks.append(rank)
            require(all(numeric(candidate[name]) for name in ('E','S','q','B','d','V_succ','V_fail','Q')),context+'Non-finite candidate value')
            views=candidate['N_views'];require(type(views) is int and 0<=views<=12,context+'Invalid12-bin coverage count')
            expected_s=1-math.exp(-views/3.)
            z=params['a']*candidate['E']+params['b0']
            expected_q=1/(1+math.exp(-z)) if z>=0 else math.exp(z)/(1+math.exp(z))
            expected_b=1. if variant=='suffix_only' else expected_s*expected_q+(1-expected_s)*params['pi']
            expected_qcost=candidate['d']+candidate['B']*candidate['V_succ']+(1-candidate['B'])*candidate['V_fail']
            for name,expected in (('S',expected_s),('q',expected_q),('B',expected_b),('Q',expected_qcost)):
                error=abs(candidate[name]-expected);max_errors[name]=max(max_errors[name],error)
                require(close(candidate[name],expected),context+name+' differs from frozen formula')
            require(0<=candidate['q']<=1 and 0<=candidate['B']<=1 and -1e-12<=candidate['S']<=1,context+'Invalid probability/coverage')
            require(all(candidate[name]>=0 for name in ('d','V_succ','V_fail','Q')),context+'Negative metre cost')
            if variant=='nearest':require(close(candidate['Q'],candidate['d']) and candidate['V_succ']==candidate['V_fail']==0,context+'Nearest has extra cost')
            if variant=='myopic':require(candidate['V_succ']==0,context+'Myopic includes a future success value')
            if variant=='suffix_only':require(candidate['V_fail']==0,context+'Suffix-only has a failure branch')
            if row['k']==2:require(candidate['V_succ']==0,context+'Last goal has a nonzero future value')
            candidates+=1
        require(len(ranks)==len(set(ranks)) and set(ranks)<=set(order),context+'Current candidates do not belong to complete order')
        by_rank={c['raw_rank']:c for c in current}
        if current:
            for field in ('chosen_raw_rank','nearest_raw_rank','unconstrained_raw_rank'):require(row[field] in by_rank,context+'Selected rank outside reachable top5')
            require(close(by_rank[row['nearest_raw_rank']]['d'],min(c['d'] for c in current)),context+'Nearest does not minimize geodesic distance')
            require(close(by_rank[row['unconstrained_raw_rank']]['Q'],min(c['Q'] for c in current)),context+'Proposal does not minimize Q')
            eligible+=1;proposed+=row['unconstrained_raw_rank']!=row['nearest_raw_rank'];held+=row['chosen_raw_rank']!=row['nearest_raw_rank']
            if variant=='nearest':require(row['unconstrained_raw_rank']==row['nearest_raw_rank'],context+'Nearest proposal deviation is nonzero')
        else:
            require(all(row[field] is None for field in ('chosen_raw_rank','nearest_raw_rank','unconstrained_raw_rank')),context+'Empty reachable set selected a candidate')
    require(latencies,'A formal variant has no recorded planning decisions')
    ordered=sorted(latencies);position=.95*(len(ordered)-1);lo=math.floor(position);hi=math.ceil(position)
    p95=ordered[lo]*(hi-position)+ordered[hi]*(position-lo) if lo!=hi else ordered[lo]
    return {'n_decisions':len(latencies),'decision_episodes':len(episodes),'n_candidates':candidates,'eligible_proposals':eligible,
        'proposal_deviation_count':proposed,'after_hysteresis_deviation_count':held,
        'proposal_deviation_pct':100*proposed/eligible if eligible else None,
        'after_hysteresis_deviation_pct':100*held/eligible if eligible else None,
        'planning_ms':{'n':len(latencies),'median':statistics.median(latencies),'p95':p95},
        'fallback_root_decisions':dict(fallbacks),'fallback_memo_state_counts':dict(memo),'fallback_decision_incidence':dict(incidence),
        'max_formula_abs_errors':max_errors,'runtime_target_index_base':0,'normalized_episode_target_index_base':1}

def validate_summary_process(audit,process):
    require(audit['n_decisions']==process['n_computed_snapshots'],'Decision count differs from process summary')
    require(audit['eligible_proposals']==process['eligible_proposals'],'Eligible proposal count differs')
    for source,target in (('proposal_deviation_pct','deviation_from_nearest_proposal_pct'),('after_hysteresis_deviation_pct','deviation_from_nearest_after_hysteresis_pct')):
        require(audit[source]==process[target] or (audit[source] is not None and process[target] is not None and close(audit[source],process[target])),'Decision deviation summary differs')
    for key in ('fallback_root_decisions','fallback_memo_state_counts','fallback_decision_incidence'):require(audit[key]==process[key],'Fallback counts differ')
    require(audit['planning_ms']['n']==process['planning_ms']['n'],'Planning sample count differs')
    for key in ('median','p95'):require(close(audit['planning_ms'][key],process['planning_ms'][key]),'Planning quantile differs')

def audit_scope(scope_root,require_all=True):
    from datetime import datetime,timezone
    from pathlib import Path
    from evaluate_sequence import OUT,load,digest,source_fingerprint
    from first8_evaluate import completed
    root=Path(scope_root);scope=load(root/'meta/scope.json')
    require(source_fingerprint()==scope['source_fingerprint'],'Scientific runtime changed')
    require(digest(OUT/'meta/calibration.json')==scope['parameter_sha256'],'Frozen calibration changed')
    audits={};pending=[]
    for stage in ('full','nearest','myopic','suffix_only'):
        ledger=root/f'meta/evaluation-{stage}-runs.json'
        if not ledger.exists() or load(ledger)['status']!='completed':pending.append(stage);continue
        completed(stage,scope)
        summary=load(root/f'summary/{stage}.json');path=root/f'logs/{stage}_decisions.jsonl'
        with path.open() as stream:audit=validate_decision_rows((json.loads(line) for line in stream),stage,scope['identities'],summary['calibration_binding']['parameters_by_scene'])
        validate_summary_process(audit,summary['process'])
        audits[stage]={'decision_log':str(path.relative_to(OUT)),'decision_log_sha256':digest(path),
            'summary':str((root/f'summary/{stage}.json').relative_to(OUT)),'summary_sha256':digest(root/f'summary/{stage}.json'),**audit}
    require(not require_all or not pending,'Complete decision audit still requires: '+', '.join(pending))
    return {'at_utc':datetime.now(timezone.utc).isoformat(),'scope':'first8','audits':audits,'pending':pending,'all_four_complete':not pending,
        'source_fingerprint':scope['source_fingerprint'],'parameter_sha256':scope['parameter_sha256'],
        'validator_sha256':digest(Path(__file__)),'Native_workers_modified':False}

if __name__=='__main__':
    from evaluate_sequence import OUT
    result=audit_scope(OUT/'scopes/first8',require_all=False)
    print(json.dumps({k:v for k,v in result.items() if k!='source_fingerprint'}))
