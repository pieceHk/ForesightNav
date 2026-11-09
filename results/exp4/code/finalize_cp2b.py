"""Bind the user-approved cross-fit parameters to tested runtime sources.

This does not fit parameters, choose a mode, or launch evaluation workers.
"""
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess

from experiments.exp4_collect import source_fingerprint, write
from pbsp.navigation.blp_parameters import CalibrationDocument

REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / 'results/exp4'


def read(relative):
    return json.loads((ROOT / relative).read_text())


def sha(relative):
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def main():
    review = read('meta/CP2a_review.json')
    assert review.get('accepted') is True and review.get('chosen_choice') == 'three_fold_cross_fit'
    assert not list((ROOT / 'meta').glob('evaluation-*-runs.json')), 'Do not rebind an evaluation in progress'
    assert not (ROOT / 'meta/CP2b_acceptance.json').exists(), 'Inspect an existing binding before recovery'
    parameters = read('meta/calibration.json')
    document = CalibrationDocument.from_mapping(parameters)
    assert document.selection == 'cross_fit' and document.collected_folds == (0, 1, 2)
    assert parameters['approved_proposal_sha256'] == sha('meta/calibration.crossfit.proposed.json')
    for relative, expected_hash in review['evidence_sha256'].items():
        assert sha(relative) == expected_hash, relative
    fit = read('meta/calibration_per_fold.json')
    folds = read('meta/folds.json')['folds']
    resolved = {}
    for f, scenes in enumerate(folds):
        for scene in scenes:
            value, binding = document.resolve(scene)
            expected = fit['cross_fit_proposals'][str(f)] if f < 3 else fit['theta_pool']
            assert asdict(value) == expected
            assert binding['training_folds'] == [x for x in range(3) if x != f]
            resolved[scene] = dict(binding, parameters=asdict(value))
    preparation = read('diagnostics/crossfit-preparation.json')
    for relative, wanted in preparation['source_sha256'].items():
        assert hashlib.sha256((REPO / relative).read_bytes()).hexdigest() == wanted, relative
    log = (ROOT / 'code/crossfit-adoption-tests.txt').read_text()
    assert re.search(r'Ran 49 tests in ', log) and log.rstrip().endswith('OK')
    regression = read('diagnostics/crossfit-adoption-regression.json')
    assert regression['all_decisions_identical'] and regression['coverage_complete']
    assert sum(regression['calls_per_episode'].values()) == 219
    value = {'accepted': True, 'selection': document.selection, 'label_source': 'system_verification',
             'accepted_at_utc': datetime.now(timezone.utc).isoformat(),
             'approval': review['confirmation'], 'source_fingerprint': source_fingerprint(),
             'source_commit': subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip(),
             'parameter_sha256': sha('meta/calibration.json'),
             'calibration_ledger_sha256': sha('meta/calibration_runs.json'),
             'cp2a_review_sha256': sha('meta/CP2a_review.json'),
             'unit_tests': 49, 'disabled_regression_calls': 219,
             'evidence_sha256': {p: sha(p) for p in ('code/crossfit-adoption-tests.txt',
                                                    'diagnostics/crossfit-adoption-regression.json')},
             'resolved_parameters_by_scene': resolved}
    write(ROOT / 'meta/CP2b_acceptance.json', value)
    lines = ['# CP2b：三组交叉拟合参数冻结', '',
             '用户已确认采用CP2a报告列出的三组交叉拟合参数，完成加载与冻结后进入固定20条Full冒烟。本次未重拟合参数，没有使用GT诊断参数；冻结文件保留原始JSON精度。', '',
             '| 评估组 | 训练组 | a | b0 | pi | C_exp (m) |',
             '| --- | --- | --- | --- | --- | --- |']
    for f in range(5):
        scene = folds[f][0]; record = resolved[scene]; theta = record['parameters']
        lines.append('| ' + ' | '.join([str(f), ', '.join(map(str, record['training_folds'])),
            *[f'{theta[k]:.9f}' for k in ('a', 'b0', 'pi', 'c_exp')]]) + ' |')
    lines.extend(['', '已采纳通用scene加载、逐快照参数来源校验及被动逐动作目标记录。主工作区49项CPU检查通过，三个原生基线回合的219次匹配输入决策全部一致；这些不代替真实GPU冒烟。', '',
                  '原360条标定ledger和源码指纹作为历史证据保留，评估单独绑定新源码指纹及参数文件。四个变体使用相同scene分配，未知scene和错组记录均拒绝。', '',
                  f'参数文件SHA256：`{value["parameter_sha256"]}`。', '',
                  f'标定ledger SHA256：`{value["calibration_ledger_sha256"]}`。', '',
                  '[冻结参数](../meta/calibration.json)、[源码与参数绑定](../meta/CP2b_acceptance.json)、[用户选择记录](../meta/CP2a_review.json)。', '',
                  '下一步为固定20条Full冒烟，并核查完整轨迹、参数来源、过程指标及两张俯视图。此报告不包含Full成绩。', ''])
    (ROOT / 'reports/CP2b.md').write_text('\n'.join(lines))
    print(json.dumps({'accepted': True, 'selection': document.selection,
                      'source_files': len(value['source_fingerprint']),
                      'parameter_sha256': value['parameter_sha256']}, indent=2))


if __name__ == '__main__':
    main()
