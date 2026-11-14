"""Reaggregate six whole scenes using the frozen native Exp4 analysis functions.
This is an explicitly post-hoc presentation subset, not an independent test.
"""
from pathlib import Path
from datetime import datetime, timezone
import csv, hashlib, itertools, json, math, shutil, tarfile
from evaluate_sequence import OUT, load, digest, source_fingerprint
from report_variants import normalize, format_summary, export_comparison, effects_and_pairs
from validate_first8_pairs import validate_pair_rows

SOURCE=OUT/'scopes/first8'
DEST=OUT/'scopes/selected6-posthoc-20261115'
SCENES=['4ok3usBNeis','6s7QHgap2fW','Dd4bFSTQ8gi','Nfvxx8J5NCo','QaLdnwvtxbs','TEEsavR23oF']
NAMES=('nearest','suffix_only','myopic','full')
def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
scope=load(SOURCE/'meta/scope.json')
assert source_fingerprint()==scope['source_fingerprint']
assert digest(OUT/'meta/calibration.json')==scope['parameter_sha256']
selected=[r for r in scope['identities'] if r['scene_id'] in SCENES]
assert len(selected)==120
posthoc={'scope':'selected6-posthoc-20261115','analysis_status':'POST_HOC_OUTCOME_SELECTED_PRESENTATION_SUBSET',
 'selected_after_outcomes_observed':True,'selection_search_space':255,
 'selection_description':'User selected original scene indices 1,3,5,6,7,8 after exhaustive scene-subset comparisons; SR ranking was considered. This is not a preregistered or independent confirmatory test.',
 'ordered_scenes':SCENES,'excluded_scenes':[s for s in scope['ordered_scenes'] if s not in SCENES],
 'n_scenes':6,'n_episodes_per_variant':120,'identities':selected,
 'parent_scope':'first8','parent_n_episodes_per_variant':160,'parent_publication_commit':'391d7e25b12171144a77cfbc41f220efe11294fb',
 'source_fingerprint':scope['source_fingerprint'],'parameter_sha256':scope['parameter_sha256'],
 'new_native_runs':0,'algorithm_or_parameter_changes':False}
write(DEST/'meta/scope.json',posthoc)
for directory in ('logs','summary','summary/paired','reports','figures','code','meta'):(DEST/directory).mkdir(exist_ok=True)
variants={}
for stage in NAMES:
    ledger=load(SOURCE/f'meta/evaluation-{stage}-runs.json')
    jobs=[j for j in ledger['jobs'] if j['scene'] in SCENES]
    assert len(jobs)==6 and all(j['status']=='completed' and j['returncode']==0 for j in jobs)
    print(json.dumps({'stage':stage,'status':'recomputing_native_metrics_and_CPU_GT','runs':len(jobs)}),flush=True)
    rows,decisions,summary=normalize(stage,[j['run_directory'] for j in jobs],expected=selected,
        calibration=load(OUT/'meta/calibration.json'),parameter_sha256=scope['parameter_sha256'])
    original=[json.loads(l) for l in (SOURCE/f'logs/{stage}_episodes.jsonl').read_text().splitlines()]
    a={(r['scene_id'],str(r['episode_id'])):r for r in rows}
    b={(r['scene_id'],str(r['episode_id'])):r for r in original if r['scene_id'] in SCENES}
    assert a==b,'Recomputed episode metrics differ from archived original results'
    assert len(rows)==120 and summary['process']['native_stage_path_check_count']==360
    summary.update(scope=posthoc['scope'],n_scenes=6,analysis_status=posthoc['analysis_status'],selected_after_outcomes_observed=True,parent_n_episodes=160,new_native_runs=0)
    formatted=format_summary(summary)
    write(DEST/f'summary/{stage}.json',formatted)
    for kind,values in (('episodes',rows),('decisions',decisions)):
        (DEST/f'logs/{stage}_{kind}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in values))
    write(DEST/f'meta/evaluation-{stage}-runs.json',{'scope':posthoc['scope'],'analysis_status':posthoc['analysis_status'],'jobs':jobs,'n_episodes':120,'source_parent':f'../first8/meta/evaluation-{stage}-runs.json'})
    variants[stage]=(rows,decisions,formatted)
    print(json.dumps({'stage':stage,'status':'complete','n':120,'SR':formatted['SR'],'SPL':formatted['SPL']}),flush=True)
export_comparison(DEST,variants)
comparison=effects_and_pairs(variants);pairs=[]
for left,right in itertools.combinations(NAMES,2):
    name=f'{left}_vs_{right}.csv'; source=SOURCE/'summary/paired'/name
    with source.open() as stream:
        reader=csv.DictReader(stream); fields=reader.fieldnames; rows=[r for r in reader if r['scene_id'] in SCENES]
    counts,means=validate_pair_rows(rows,variants[left][0],variants[right][0],left,right,expected_n=120)
    assert counts==comparison['paired_episode_success'][f'{left}_vs_{right}']
    for side,v in (('left',left),('right',right)):
        for m in ('SR','SPL','PR','PPL'):assert math.isclose(means[side][m],variants[v][2]['rates_pct_unrounded'][m],abs_tol=1e-9)
    target=DEST/'summary/paired'/name
    with target.open('w',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=fields);w.writeheader();w.writerows(rows)
    pairs.append({'path':'summary/paired/'+name,'n_paired':120,'sha256':digest(target)})
write(DEST/'summary/paired/manifest.json',{'scope':posthoc['scope'],'analysis_status':posthoc['analysis_status'],'n_episodes_each_pair':120,'pairs':pairs})
raw=load(SOURCE/'meta/raw-archives/manifest.json');kept=[x for x in raw['runs'] if x['scene_id'] in SCENES]
assert len(kept)==24
for x in kept:assert digest(OUT/x['archive'])==x['archive_sha256']
write(DEST/'meta/raw-archives-references.json',{'scope':posthoc['scope'],'archive_base':'results/exp4','parent_commit':posthoc['parent_publication_commit'],'archived_runs':24,'runs':kept})
for name in ('folds.json','calibration.json','calibration_per_fold.json','env.txt'):
    shutil.copyfile(OUT/'meta'/name,DEST/'meta'/name)
for path in (OUT/'logs').glob('calibration_fold*_*s.jsonl'):shutil.copyfile(path,DEST/'logs'/path.name)
for cp in ('CP1','CP2a','CP2b','CP3','CP4'):
    if (OUT/f'reports/{cp}.md').exists():shutil.copyfile(OUT/f'reports/{cp}.md',DEST/f'reports/{cp}.md')
shutil.copyfile(SOURCE/'reports/CP5.md',DEST/'reports/CP5_original160.md')
for name in ('report_variants.py','action_path_metrics.py','native_candidate_gt.py','candidate_identities.py','validate_first8_pairs.py'):
    shutil.copyfile(OUT/'code'/name,DEST/'code'/name)
write(DEST/'meta/reaggregation-audit.json',{'status':'passed','completed_at_utc':datetime.now(timezone.utc).isoformat(),
 'analysis_status':posthoc['analysis_status'],'n_by_variant':{v:120 for v in NAMES},'identities_and_episode_values_match_original':True,
 'native_GT_rechecked':True,'action_stage_paths_checked':1440,'paired_tables':6,'rows_each':120,
 'original_source_fingerprint_unchanged':source_fingerprint()==scope['source_fingerprint'],'parameter_sha256':scope['parameter_sha256'],
 'original160_artifacts_modified':False,'new_native_runs':0})
archive=Path('/tmp/exp4-selected120-posthoc-analysis.tar.gz')
with tarfile.open(archive,'w:gz') as t:
    for p in sorted(DEST.rglob('*')):
        if p.is_file():t.add(p,arcname=p.relative_to(DEST),recursive=False)
print(json.dumps({'status':'ready_for_document_generation','destination':str(DEST),'archive':str(archive),'bytes':archive.stat().st_size,'sha256':digest(archive)}),flush=True)
