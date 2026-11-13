from datetime import datetime,timezone
from pathlib import Path
import json,shutil
from evaluate_sequence import OUT,load,digest,source_fingerprint

root=OUT/'scopes/first8';scope=load(root/'meta/scope.json');rows=[]
assert source_fingerprint()==scope['source_fingerprint']
for stage in ('full','nearest','myopic','suffix_only'):
    path=root/f'meta/evaluation-{stage}-runs.json'
    if not path.exists():continue
    for job in load(path)['jobs']:
        if not job.get('run_directory'):continue
        run=Path(job['run_directory']);files=[]
        for source in [run/'launch.json',*run.glob('*environment*.json')]:
            if not source.is_file():continue
            target=root/'diagnostics/native-role-environments'/stage/job['scene']/source.name
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
            assert digest(source)==digest(target)
            files.append({'file':str(target.relative_to(OUT)),'sha256':digest(target)})
        mapping=any(Path(f['file']).name=='mapping_environment.json' for f in files)
        navigation=any(Path(f['file']).name.startswith('navigation_environment') for f in files)
        rows.append({'stage':stage,'scene':job['scene'],'gpu':job['gpu'],
                     'mapping_recorded':mapping,'navigation_recorded':navigation,'files':files})
stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
value={'at_utc':datetime.now(timezone.utc).isoformat(),'scope':'first8','role_pairs':rows,
    'recorded_pairs_by_stage':{s:sum(r['stage']==s and r['mapping_recorded'] and r['navigation_recorded'] for r in rows)
                               for s in ('full','nearest','myopic','suffix_only')},
    'source_and_parameters_unchanged':source_fingerprint()==scope['source_fingerprint'],'native_states_modified':False}
target=root/'diagnostics'/('environment-index-'+stamp+'.json');target.write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps({'proof':str(target.relative_to(OUT)),'recorded_pairs_by_stage':value['recorded_pairs_by_stage'],
                  'source_and_parameters_unchanged':value['source_and_parameters_unchanged']}))
