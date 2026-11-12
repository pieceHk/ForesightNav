"""Bind whole-attempt infrastructure exclusions to the completed CP5 report."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from evaluate_sequence import GateError
from report_full_evaluation import verify

ROOT=Path(__file__).resolve().parents[1]
SELF=Path(__file__).resolve()
MANIFEST='diagnostics/infrastructure-restarts/nearest-20261111T172644/manifest.json'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def load(path):
    return json.loads(path.read_text())
def write(path,data):
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

def inputs():
    _,_,binding=verify('CP5')
    execution=load(ROOT/'diagnostics/CP5/complete-evidence-execution.json')
    prepared=load(ROOT/'meta/CP5_report_prepared.json')
    retired=load(ROOT/MANIFEST)
    assert retired['canonical_retired'] and retired['archive_round_trip_verified']
    assert retired['exported_records_total']==57 and retired['original_native_runs_retained']
    for relative,expected in retired['evidence_sha256'].items():
        assert digest(ROOT/relative)==expected,relative
    assert execution['checkpoint']==prepared['checkpoint']=='CP5'
    assert execution['input_sha256']==prepared['input_sha256']==binding
    assert execution['source_and_parameters_unchanged'] and all(row['exit_code']==0 for row in execution['steps'])
    assert digest(ROOT/'reports/CP5.md')==execution['report_sha256']==prepared['output_sha256']['reports/CP5.md']
    if 'infrastructure_attempt_augmentation' in execution:
        old=execution['infrastructure_attempt_augmentation']
        assert old['source_sha256']==digest(SELF) and old['manifest_sha256']==digest(ROOT/MANIFEST)
        assert old['report_after_sha256']==digest(ROOT/'reports/CP5.md')
    return execution,prepared

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true');args=parser.parse_args()
    try:
        execution,prepared=inputs()
    except (GateError,FileNotFoundError,ValueError,KeyError) as error:
        if not args.check: raise
        print(json.dumps({'checkpoint':'CP5','ready':False,'reason':str(error),
            'writes_formal_outputs':False,'launches_workers':False,'native_states_modified':False}))
        return
    if args.check:
        print(json.dumps({'checkpoint':'CP5','ready':True,'writes_formal_outputs':False,
            'launches_workers':False,'native_states_modified':False}));return
    if 'infrastructure_attempt_augmentation' in execution:
        print(json.dumps({'checkpoint':'CP5','infrastructure_augmentation_reused':True,
            'report_sha256':execution['report_sha256'],'native_states_modified':False}));return
    report=ROOT/'reports/CP5.md';before=report.read_bytes()
    snapshot='diagnostics/CP5/report-before-infrastructure-augmentation.md'
    with (ROOT/snapshot).open('xb') as stream: stream.write(before)
    text=before.decode('utf-8').rstrip()+'\n\n'
    text+=('首次Nearest尝试因物理GPU3的建图CUDA显存不足中断。该次导出的57条已整体归档，'
        '全部排除于正式Nearest统计；正式Nearest按相同固定400条身份、seed与冻结参数重新完整运行。'
        '本表中各次正式400条运行的导航失败回合均保留。'
        '重跑由基础设施异常触发，未按导航成功率挑选保留回合。'
        '事故、整组排除规则及原始日志SHA见[恢复记录](Nearest_infrastructure_restart.md)和'
        f'[归档清单](../{MANIFEST})。\n')
    report.write_text(text)
    augmentation={'recorded_at_utc':datetime.now(timezone.utc).isoformat(),
        'source':SELF.relative_to(ROOT).as_posix(),'source_sha256':digest(SELF),
        'manifest':MANIFEST,'manifest_sha256':digest(ROOT/MANIFEST),
        'excluded_entire_infrastructure_attempt_records':57,'formal_nearest_episodes':400,
        'selection_by_navigation_outcome':False,'report_before_snapshot':snapshot,
        'report_before_sha256':hashlib.sha256(before).hexdigest(),'report_after_sha256':digest(report)}
    prepared['infrastructure_attempt_augmentation']=augmentation
    prepared['output_sha256']['reports/CP5.md']=digest(report)
    prepared['output_sha256'][snapshot]=digest(ROOT/snapshot)
    write(ROOT/'meta/CP5_report_prepared.json',prepared)
    execution['infrastructure_attempt_augmentation']=augmentation
    execution['report_sha256']=digest(report)
    execution['output_sha256']['reports/CP5.md']=digest(report)
    execution['output_sha256'][snapshot]=digest(ROOT/snapshot)
    execution['output_sha256'][augmentation['source']]=digest(SELF)
    execution['output_sha256']['meta/CP5_report_prepared.json']=digest(ROOT/'meta/CP5_report_prepared.json')
    write(ROOT/'diagnostics/CP5/complete-evidence-execution.json',execution)
    print(json.dumps({'checkpoint':'CP5','infrastructure_augmentation_applied':True,
        'report_sha256':digest(report),'retired_infrastructure_records':57,
        'native_states_modified':False,'launches_workers':False,
        'actual_figure_review_must_bind_this_report_sha256':True}))

if __name__=='__main__': main()
