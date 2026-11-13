"""Collect the accepted named calibration/CP1–CP4 evidence for publication."""
from datetime import datetime,timezone
from pathlib import Path
import hashlib,json

def collect(out):
    out=Path(out)
    def load(relative):return json.loads((out/relative).read_text())
    def digest(relative):return hashlib.sha256((out/relative).read_bytes()).hexdigest()
    def safe(relative):
        path=Path(relative)
        assert not path.is_absolute() and '..' not in path.parts and '.private' not in relative
        assert (out/path).is_file() and not (out/path).is_symlink()
        return relative
    required=['meta/folds.json','meta/calibration.json','meta/calibration_per_fold.json','meta/env.txt',
        *['reports/'+cp+'.md' for cp in ('CP1','CP2a','CP2b','CP3','CP4')],
        *['logs/calibration_fold'+str(f)+'_'+kind+'.jsonl' for f in range(3) for kind in ('events','snapshots')],
        'figures/calibration_evidence.png','figures/calibration_evidence_common_bins.png',
        'figures/calibration_reliability.png','figures/calibration_bootstrap_intervals.png',
        'figures/CP3/4ok3usBNeis_0.png','figures/CP3/5cdEh9F2hJL_0.png','figures/CP3/planning_latency.png',
        'reports/eq5_wording.md','reports/dataset_scope.md']
    for name in ('CP1_acceptance.json','CP2a_review.json','CP2b_acceptance.json','CP3_acceptance.json','CP4_reported.json'):
        relative='meta/'+name;document=load(relative)
        assert document.get('accepted',document.get('reported')) is True
        required.append(relative)
        for evidence,sha in document.get('evidence_sha256',{}).items():
            safe(evidence);assert digest(evidence)==sha
            required.append(evidence)
    archives=[]
    def references(value):
        if isinstance(value,dict):
            for key,item in value.items():
                if key=='archive' and isinstance(item,str) and item.endswith(('.tar.gz','.tar')):
                    safe(item);sha=value.get('archive_sha256',value.get('sha256'))
                    assert isinstance(sha,str) and len(sha)==64,('Missing archive SHA',item,list(value))
                    assert digest(item)==sha,('Referenced archive differs',item)
                    archives.append({'archive':item,'sha256':sha})
                    yield item
                else:yield from references(item)
        elif isinstance(value,list):
            for item in value:yield from references(item)
    for relative in ('meta/calibration-log-archive.json','meta/evaluation-smoke-log-archive.json'):
        required.append(relative);required.extend(references(load(relative)))
    files=sorted(set(map(safe,required)))
    sizes={relative:(out/relative).stat().st_size for relative in files}
    assert all(0<size<95*1024**2 for size in sizes.values()),{p:n for p,n in sizes.items() if not 0<n<95*1024**2}
    return {'at_utc':datetime.now(timezone.utc).isoformat(),'files':files,
        'sha256':{name:digest(name) for name in files},'bytes':sizes,'referenced_archive_sha256_verified':archives,
        'scope':'Accepted CP1–CP4 and360 calibration evidence; current first8 CP5 completion is a separate requirement',
        'collector_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'current_four_variant_completion_claimed':False,'Native_states_modified':False}

if __name__=='__main__':
    from evaluate_sequence import OUT
    result=collect(OUT)
    print(json.dumps({'files':len(result['files']),'referenced_archives':len(result['referenced_archive_sha256_verified']),
        'bytes':sum(result['bytes'].values()),'current_four_variant_completion_claimed':False}))
