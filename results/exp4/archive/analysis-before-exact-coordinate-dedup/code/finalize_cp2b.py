"""Bind an accepted corrected CP2 proposal to tested runtime inputs; no launch."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess

from evaluate_sequence import validate_calibration_coverage
from experiments.exp4_collect import source_fingerprint, write
from pbsp.navigation.blp_parameters import CalibrationDocument
from report_calibration import selection_rule

REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / 'results/exp4'


def read(relative):
    return json.loads((ROOT / relative).read_text())


def sha(relative):
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify():
    review = read('meta/CP2a_review.json')
    require(review.get('accepted') is True and review.get('collection_attempt') == 'a3',
            'A current accepted a3 proposal is required')
    require(not list((ROOT / 'meta').glob('evaluation-*-runs.json')), 'Do not rebind an evaluation in progress')
    require(not (ROOT / 'meta/CP2b_acceptance.json').exists(), 'Inspect the existing CP2b binding before recovery')
    ledger = read('meta/calibration_runs.json')
    validate_calibration_coverage(ROOT, ledger)
    fit = read('meta/calibration_per_fold.json')
    selection, _ = selection_rule(fit)
    parameters = read('meta/calibration.json')
    document = CalibrationDocument.from_mapping(parameters)
    require(document.selection == selection, 'Frozen selection differs from the actual stability rule')
    expected_choice = 'three_fold_cross_fit' if selection == 'cross_fit' else 'three_fold_pooled'
    require(review.get('chosen_choice') == expected_choice, 'Review and frozen selection differ')
    proposal_path = ('meta/calibration.crossfit.proposed.json' if selection == 'cross_fit' else
                     'meta/calibration.pooled.review-proposed.json')
    require(parameters.get('approved_proposal_sha256') == sha(proposal_path), 'Frozen proposal SHA differs')
    for relative, expected in review['evidence_sha256'].items():
        require(sha(relative) == expected, f'Reviewed evidence changed: {relative}')
    binding = read('meta/calibration-fit-binding.json')
    require(binding['collection_attempt'] == 'a3', 'Fit binding belongs to an older collection')
    for relative, expected in {**binding['input_sha256'], **binding['output_sha256']}.items():
        require(sha(relative) == expected, f'Fit evidence changed: {relative}')
    fingerprint = source_fingerprint()
    cp1 = read('meta/CP1_acceptance.json')
    require(cp1.get('accepted') is True and cp1.get('collection_attempt') == 'a3' and
            cp1['source_fingerprint'] == ledger['source_fingerprint'] == fingerprint,
            'Corrected CP1, collection, and runtime inputs differ')
    require(cp1['unit_tests'] == 51 and cp1['disabled_regression_calls'] == 219,
            'The corrected runtime validation is incomplete')
    for relative, expected in cp1['evidence_sha256'].items():
        require(sha(relative) == expected, f'Corrected runtime validation changed: {relative}')
    test_log = (ROOT / 'code/depth-normalization-tests.txt').read_text()
    require(bool(re.search(r'Ran 51 tests in ', test_log)) and test_log.rstrip().endswith('OK'),
            'The corrected 51-test result is required')
    regression = read('diagnostics/depth-normalization-regression.json')
    require(regression['all_decisions_identical'] and regression['coverage_complete'] and
            sum(regression['calls_per_episode'].values()) == 219, 'Disabled-mode replay is incomplete')
    folds = read('meta/folds.json')['folds']
    resolved = {}
    if selection == 'cross_fit':
        require(document.collected_folds == (0, 1, 2), 'Expected the three collected folds')
        require(dict(document.scene_to_fold) == {s: f for f, scenes in enumerate(folds) for s in scenes},
                'Frozen scene mapping differs from the fixed folds')
    for f, scenes in enumerate(folds):
        for scene in scenes:
            value, origin = document.resolve(scene)
            expected = fit['cross_fit_proposals'][str(f)] if selection == 'cross_fit' and f < 3 else fit['theta_pool']
            require(asdict(value) == expected, f'Frozen numeric parameters differ for {scene}')
            if selection == 'cross_fit':
                require(origin['training_folds'] == [x for x in range(3) if x != f],
                        f'Cross-fit training groups differ for {scene}')
            resolved[scene] = dict(origin, parameters=asdict(value))
    return review, document, fingerprint, resolved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    try:
        review, document, fingerprint, resolved = verify()
    except (ValueError, FileNotFoundError, KeyError) as error:
        if args.check:
            print(json.dumps({'ready': False, 'reason': str(error), 'launches_workers': False}))
            return
        raise
    if args.check:
        print(json.dumps({'ready': True, 'selection': document.selection, 'launches_workers': False}))
        return
    evidence = ('meta/CP1_acceptance.json', 'meta/calibration-fit-binding.json',
                'code/depth-normalization-tests.txt', 'diagnostics/depth-normalization-regression.json',
                'code/finalize_cp2b.py')
    value = {'accepted': True, 'collection_attempt': 'a3', 'selection': document.selection,
             'label_source': 'system_verification', 'accepted_at_utc': datetime.now(timezone.utc).isoformat(),
             'approval': review['confirmation'], 'source_fingerprint': fingerprint,
             'source_commit': subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip(),
             'parameter_sha256': sha('meta/calibration.json'),
             'calibration_ledger_sha256': sha('meta/calibration_runs.json'),
             'cp2a_review_sha256': sha('meta/CP2a_review.json'),
             'unit_tests': 51, 'disabled_regression_calls': 219,
             'evidence_sha256': {p: sha(p) for p in evidence}, 'resolved_parameters_by_scene': resolved}
    write(ROOT / 'meta/CP2b_acceptance.json', value)
    lines = ['# CP2b：修正版本参数冻结与源码绑定', '',
             f'已按本次CP2a接受记录绑定{document.selection}参数。参数只来自a3的2-MON系统验证标签，数值与审阅提案逐字段一致。', '',
             '| 评估组 | 拟合组 | a | b0 | pi | C_exp (m) |',
             '| --- | --- | --- | --- | --- | --- |']
    for f, scenes in enumerate(read('meta/folds.json')['folds']):
        record = resolved[scenes[0]]
        training = record['training_folds'] if document.selection == 'cross_fit' else [0, 1, 2]
        lines.append('| ' + ' | '.join([str(f), ', '.join(map(str, training)),
            *[f'{record["parameters"][k]:.9f}' for k in ('a', 'b0', 'pi', 'c_exp')]]) + ' |')
    lines += ['', '当前源码通过51项运行实现检查和219次关闭模式匹配输入回放；原生深度覆盖管线的小试证明与CP1记录绑定。报告辅助脚本的检查单独保留，不计入这51项。', '',
              f'参数文件SHA256：`{value["parameter_sha256"]}`。', '',
              f'运行提交：`{value["source_commit"]}`。', '',
              '[冻结参数](../meta/calibration.json)、[本次接受记录](../meta/CP2a_review.json)、'
              '[源码与参数绑定](../meta/CP2b_acceptance.json)、[拟合证据](../meta/calibration-fit-binding.json)。', '',
              '下一步运行新的固定20条Full冒烟，核查完整轨迹、逐场景参数、过程指标和两张俯视图。此报告不包含Full成绩。', '']
    (ROOT / 'reports/CP2b.md').write_text('\n'.join(lines))
    print(json.dumps({'accepted': True, 'collection_attempt': 'a3', 'selection': document.selection,
                      'source_files': len(fingerprint), 'parameter_sha256': value['parameter_sha256'],
                      'launches_workers': False}, indent=2))


if __name__ == '__main__':
    main()
