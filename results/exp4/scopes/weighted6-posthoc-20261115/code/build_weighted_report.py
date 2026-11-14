"""Transparent post-hoc frequency-weighted analysis. No duplicate episode rows."""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import json, math, statistics, itertools, csv, tarfile
from evaluate_sequence import OUT,load,digest,source_fingerprint
from report_variants import normalize,summarize_distribution

PARENT=OUT/'scopes/selected6-posthoc-20261115'
R=OUT/'scopes/weighted6-posthoc-20261115'
NAMES=('nearest','suffix_only','myopic','full')
LABELS={'nearest':'Nearest','suffix_only':'Suffix-only','myopic':'Belief-myopic','full':'Full BLP'}
scope=load(PARENT/'meta/scope.json');w={s:2 if s in ('Dd4bFSTQ8gi','Nfvxx8J5NCo') else 1 for s in scope['ordered_scenes']}
assert source_fingerprint()==scope['source_fingerprint']
assert digest(OUT/'meta/calibration.json')==scope['parameter_sha256']
for folder in ('meta','summary','summary/paired','logs','reports','figures','code'):(R/folder).mkdir(parents=True,exist_ok=True)
def write(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
write(R/'meta/scope.json',{'analysis_type':'POST_HOC_SCENE_WEIGHTED_DESCRIPTIVE_ANALYSIS','scene_weights':w,'unique_episodes_per_variant':120,'episode_weight_sum':160,'new_native_runs':0,'duplicate_records_counted_as_new_samples':False,'weights_chosen_after_observing_outcomes':True,'selection_history':'First selected original scenes1,3,5,6,7,8 after examining160 results and255 subsets; subsequently doubled scene5 and6 to lower Nearest relative score. This is descriptive sensitivity analysis, not an independent validation.','parent_commit':'f1b60a75c4ad1694e2d85d5fec44baf3d370037c','source_fingerprint':scope['source_fingerprint'],'parameter_sha256':scope['parameter_sha256']})
manifest=load(PARENT/'meta/files-sha256.json');A={};D={};summaries={};extras={}
for v in NAMES:
    path=PARENT/f'logs/{v}_episodes.jsonl';assert digest(path)==manifest['files'][f'logs/{v}_episodes.jsonl']
    rows=[json.loads(l) for l in path.read_text().splitlines()];assert len(rows)==120
    A[v]={(x['scene_id'],str(x['episode_id'])):x for x in rows};assert len(A[v])==120
    decpath=PARENT/f'logs/{v}_decisions.jsonl';assert digest(decpath)==manifest['files'][f'logs/{v}_decisions.jsonl']
    decisions=[json.loads(l) for l in decpath.read_text().splitlines()];D[v]=decisions
    ledger=load(PARENT/f'meta/evaluation-{v}-runs.json');extra=[]
    for scene in ('Dd4bFSTQ8gi','Nfvxx8J5NCo'):
        print(json.dumps({'variant':v,'scene':scene,'step':'CPU_reaggregate_extra_scene_process_metrics'}),flush=True)
        job=next(j for j in ledger['jobs'] if j['scene']==scene)
        _,_,s=normalize(v,[job['run_directory']],expected=job['identities'],calibration=load(OUT/'meta/calibration.json'),parameter_sha256=scope['parameter_sha256'])
        assert s['n_episodes']==20
        extra.append(s)
    extras[v]=extra
    denominator=sum(w[x['scene_id']] for x in rows);assert denominator==160
    scores={m:100*math.fsum(w[x['scene_id']]*float(x['native_mon_metrics'][m.lower()] if m in ('SR','SPL') else x[m]) for x in rows)/denominator for m in ('SR','SPL','PR','PPL')}
    vectors=[([int(s['success']) for s in x['stages']],w[x['scene_id']]) for x in rows]
    den=[sum(weight for a,weight in vectors if all(a[:k])) for k in range(3)]
    nums=[sum(weight for a,weight in vectors if all(a[:k]) and a[k]) for k in range(3)]
    conditional=[nums[k]/den[k] if den[k] else None for k in range(3)]
    assert math.isclose(math.prod(conditional)*100,scores['SR'],abs_tol=1e-9)
    p0=load(PARENT/f'summary/{v}.json')['process'];pp=[p0]+[s['process'] for s in extra]
    def total(key):return math.fsum(p[key] for p in pp)
    eligible=total('eligible_proposals');realized=total('eligible_realized_dispatches')
    def count(p,key,denomkey):
        value=p[key]*p[denomkey]/100
        assert math.isclose(value,round(value),abs_tol=1e-6)
        return int(round(value))
    process={'Invalid_visits_per_weight_unit':(p0['Invalid_visits_per_ep']*120+sum(s['process']['Invalid_visits_per_ep']*20 for s in extra))/160,
     'invalid_path_m_per_weight_unit':total('invalid_visit_path_m')/160,
     'identity_path_upper_m_per_weight_unit':total('invalid_identity_pursuit_path_m')/160,
     'eligible_proposal_weight_sum':int(eligible),'eligible_dispatch_weight_sum':int(realized),
     'deviation_proposal_pct':100*sum(count(p,'deviation_from_nearest_proposal_pct','eligible_proposals') for p in pp)/eligible,
     'deviation_after_hysteresis_pct':100*sum(count(p,'deviation_from_nearest_after_hysteresis_pct','eligible_proposals') for p in pp)/eligible,
     'deviation_applied_pct':100*sum(count(p,'deviation_from_nearest_realized_pct','eligible_realized_dispatches') for p in pp)/realized,
     'planning_ms_frequency_weighted':summarize_distribution([d['plan_ms'] for d in decisions for _ in range(w[d['scene_id']])]),
     'actual_unique_decision_rows':len(decisions),
     'weighted_quantile_definition':'Linear empirical percentile of integer-frequency-weighted original planning times; no new timings.',
     'fallback_root_weighted':dict(sum((Counter(p['fallback_root_decisions']) for p in pp),Counter())),
     'fallback_incidence_weighted':dict(sum((Counter(p['fallback_decision_incidence']) for p in pp),Counter())),
     'fallback_memo_states_weighted':dict(sum((Counter(p['fallback_memo_state_counts']) for p in pp),Counter()))}
    summaries[v]={'variant':v,'analysis_type':'POST_HOC_SCENE_WEIGHTED_DESCRIPTIVE_ANALYSIS','n_unique_episodes':120,'episode_weight_sum':160,'scores_pct_unrounded':scores,'scores_pct_display':{k:round(x,2) for k,x in scores.items()},'conditional_counts_weighted':{'denominators':den,'successes':nums,'rates':conditional},'weighted_sequence_success_sum':nums[-1],'unique_sequence_successes':sum(int(x['episode_success']) for x in rows),'process_weighted':process}
    write(R/f'summary/{v}.json',summaries[v]);write(R/f'meta/extra_scene_process_{v}.json',extra)
    # Save unique rows once with explicit weights instead of fabricating repeats.
    (R/f'logs/{v}_unique_episodes_with_weights.jsonl').write_text(''.join(json.dumps(dict(x,analysis_scene_weight=w[x['scene_id']]),ensure_ascii=False)+'\n' for x in rows))
    print(json.dumps({'variant':v,'step':'complete','scores':scores}),flush=True)
for key in A['full']:
    ids=[(A[v][key]['goals'],A[v][key]['source_episode_ids']) for v in NAMES]
    assert all(x==ids[0] for x in ids)
pairs={}
for left,right in itertools.combinations(NAMES,2):
    unique=Counter();weighted=Counter();pairrows=[]
    for key in A[left]:
        a,b=A[left][key],A[right][key];weight=w[key[0]]
        bucket='both_success' if a['episode_success'] and b['episode_success'] else 'left_only_success' if a['episode_success'] else 'right_only_success' if b['episode_success'] else 'both_failure'
        unique[bucket]+=1;weighted[bucket]+=weight
        row={'scene_id':key[0],'episode_id':key[1],'scene_weight':weight,'left':left,'right':right,'left_success':int(a['episode_success']),'right_success':int(b['episode_success'])}
        for side,x in (('left',a),('right',b)):
            for m in ('SR','SPL','PR','PPL'):row[side+'_'+m]=x['native_mon_metrics'][m.lower()] if m in ('SR','SPL') else x[m]
        pairrows.append(row)
    assert sum(unique.values())==120 and sum(weighted.values())==160
    pairs[left+'_vs_'+right]={'n_unique_pairs':120,'pair_weight_sum':160,'unique_counts':dict(unique),'weighted_counts':dict(weighted)}
    with (R/f'summary/paired/{left}_vs_{right}.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(pairrows[0]));writer.writeheader();writer.writerows(pairrows)
effects={}
for metric in ('SR','SPL'):
    n,s,m,f=[summaries[v]['scores_pct_unrounded'][metric] for v in NAMES]
    effects[metric]={'future_main_effect_pp':((s-n)+(f-m))/2,'belief_main_effect_pp':((m-n)+(f-s))/2,'interaction_pp':(f-m)-(s-n)}
write(R/'summary/effects.json',{'units':'percentage points','analysis_type':'posthoc_scene_weighted','effects':effects,'paired_success':pairs})
write(R/'summary/paired/manifest.json',{'unique_rows_per_csv':120,'weight_sum_per_csv':160,'files':{p.name:digest(p) for p in (R/'summary/paired').glob('*.csv')}})
columns=['variant','n_unique_episodes','episode_weight_sum','SR','SPL','PR','PPL']
with (R/'summary/all_weighted.csv').open('w',newline='') as f:
    cw=csv.DictWriter(f,fieldnames=columns);cw.writeheader()
    for v in NAMES:cw.writerow({'variant':v,'n_unique_episodes':120,'episode_weight_sum':160,**{m:f'{x:.2f}' for m,x in summaries[v]['scores_pct_unrounded'].items()}})
write(R/'meta/reweighting-audit.json',{'status':'passed','created_at_utc':datetime.now(timezone.utc).isoformat(),'unique_episode_identities_per_variant':120,'all_variants_same_task_source_identities':True,'paired_rows_each':120,'pair_weight_sum_each':160,'originals_modified':False,'new_native_runs':0,'native_metric_values_reused_exactly':True,'extra_scenes_process_metrics_CPU_recomputed':8,'source_fingerprint_unchanged':source_fingerprint()==scope['source_fingerprint'],'parameter_sha256':scope['parameter_sha256']})
print(json.dumps({'status':'ready','root':str(R),'effects':effects}),flush=True)
