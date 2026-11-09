"""Render a complete, source-bound CP3 report; never accepts or launches Full."""
import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

from evaluate_sequence import audit_scene, expected_identities
from experiments.exp4_collect import source_fingerprint
from pbsp.navigation.blp_parameters import CalibrationDocument

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / 'diagnostics/CP3'


def load(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def verify():
    ledger = load(ROOT / 'meta/evaluation-smoke-runs.json')
    completed = sum(job['status'] == 'completed' for job in ledger['jobs'])
    if ledger['status'] not in ('completed', 'completed_requires_score_review') or completed != 20:
        raise ValueError(f'Fixed smoke is incomplete: {completed}/20; status={ledger["status"]}')
    cp2b = load(ROOT / 'meta/CP2b_acceptance.json')
    assert source_fingerprint() == ledger['source_fingerprint'] == cp2b['source_fingerprint']
    assert digest(ROOT / 'meta/calibration.json') == ledger['parameter_sha256'] == cp2b['parameter_sha256']
    assert ledger['n_episodes'] == 20 and ledger['n_error'] == 0
    for job in ledger['jobs']:
        audit_scene(Path(job['run_directory']), job)
    for relative in ('summary/full.json', 'logs/full_episodes.jsonl', 'logs/full_decisions.jsonl'):
        assert digest(DESTINATION / relative) == ledger['artifact_sha256'][relative], relative
    summary = load(DESTINATION / 'summary/full.json')
    episodes = jsonl(DESTINATION / 'logs/full_episodes.jsonl')
    decisions = jsonl(DESTINATION / 'logs/full_decisions.jsonl')
    wanted = {(r['scene_id'], str(r['runtime_episode_id'])): r for r in expected_identities('smoke', ROOT)}
    assert len(episodes) == 20 and {(r['scene_id'], str(r['episode_id'])) for r in episodes} == set(wanted)
    for row in episodes:
        identity = wanted[row['scene_id'], str(row['episode_id'])]
        assert len(row['stages']) == 3 and row['goals'] == identity['ordered_categories']
        assert row['source_episode_ids'] == identity['source_episode_ids']
    document = CalibrationDocument.from_mapping(load(ROOT / 'meta/calibration.json'))
    assert summary['calibration_binding'] == {
        'selection': document.selection, 'parameter_sha256': ledger['parameter_sha256'],
        'parameters_by_scene': {s: asdict(document.resolve(s)[0]) for s in sorted({s for s, e in wanted})}}
    assert summary['n_episodes'] == 20 and summary['n_error'] == 0 and not summary['diagnostic_only']
    process = summary['process']
    assert process['invalid_path_complete'] and not process['invalid_path_association_gaps']
    assert process['invalid_path_source'] == 'blp_action_target_v1_native_gt'
    assert process['native_stage_path_check_count'] == 60
    assert process['n_computed_snapshots'] == len(decisions)
    precise = summary['rates_pct_unrounded']
    assert math.isclose(precise['Joint'], precise['SR'], abs_tol=1e-10)
    start = load(ROOT / 'meta/smoke_start.json')
    assert summary['source_commits'] == [start['head']]
    environment = load(ROOT / 'meta/environment-smoke.json')
    assert environment['completed_cases_checked'] == 20, 'Refresh the complete twenty-case environment inventory'
    assert environment['runtime_commit'] == start['head'] and environment['boot_id'] == ledger['boot_id']
    assert all(environment['uniform_python_lock_and_package_inventory_by_role'].values())
    coverage = load(DESTINATION / 'evidence-coverage-audit.json')
    assert coverage['completed_scenes'] == 20 and coverage['snapshots'] == len(decisions), 'Refresh complete observation-coverage diagnostics'
    figures = load(DESTINATION / 'figure-review.json')
    assert figures['visual_qa_passed'] and len(figures['figures']) == 2
    assert [r['scene_id'] for r in figures['figures']] == [j['scene'] for j in ledger['jobs'][:2]]
    assert figures['renderer_sha256'] == digest(ROOT / 'code/plot_blp_episode.py')
    for figure in figures['figures']:
        base = ROOT / 'figures/CP3' / (figure['scene_id'] + '_' + figure['episode_id'])
        assert digest(base.with_suffix('.png')) == figure['png_sha256']
        assert digest(base.with_suffix('.json')) == figure['numeric_metadata_sha256']
    return ledger, summary, episodes, decisions, figures


def number(value, digits=2):
    return '未定义（分母为0）' if value is None else f'{value:.{digits}f}'


def render(ledger, summary, episodes, decisions, figures):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    for script in ('audit_completed_smoke.py', 'audit_smoke_mechanisms.py'):
        subprocess.run([sys.executable, str(ROOT / 'code' / script)], check=True)
    behavior = load(DESTINATION / 'completed-behavior-review.json')
    mechanism = load(DESTINATION / 'mechanism-audit.json')
    assert len(behavior['episodes']) == 20 and not mechanism['violations']
    assert set(mechanism['snapshots_by_scene']) == {row['scene_id'] for row in episodes}
    times = np.sort(np.asarray([row['plan_ms'] for row in decisions]))
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    axes[0].hist(times, bins=min(60, max(10, int(math.sqrt(len(times))))), color='#376f9e')
    axes[0].set(xlabel='BLP planning time (ms)', ylabel='Snapshot count')
    axes[1].plot(times, np.arange(1, len(times) + 1) / len(times), color='#376f9e')
    if times.min() > 0:
        axes[1].set_xscale('log')
    axes[1].set(xlabel='BLP planning time (ms)', ylabel='Empirical cumulative fraction')
    figure.suptitle('Full BLP planning latency | fixed 20-episode 3-MON smoke')
    latency = ROOT / 'figures/CP3/planning_latency.png'
    figure.savefig(latency, dpi=180)
    plt.close(figure)
    process, conditional = summary['process'], summary['conditional_counts']
    lines = ['# Exp4 CP3：固定20条Full冒烟', '', '## 完成了什么', '',
        '固定20个场景各1条3-MON全部完成。场景、source episode和目标顺序与冻结清单一致，三组交叉拟合参数按场景加载；未使用GT调参。四变体全量评估尚未启动。', '',
        f'运行提交：`{summary["source_commits"][0]}`。源码指纹、参数SHA、原生完成状态、60阶段路径及全部规范化日志已核对。', '',
        '[本阶段环境与GPU配置](../meta/environment-smoke.json)覆盖全部20条，角色Python、锁文件及包清单一致；每条原生run还保留独立环境文件。[观测覆盖检查](../diagnostics/CP3/evidence-coverage-audit.json)分别统计当前与未来目标的有效视角，不使用GT。', '',
        '## 关键数字', '', '| 指标 | 结果 |', '| --- | --- |']
    for name in ('SR', 'SPL', 'PR', 'PPL', 'P_C1', 'P_C2_given_C1', 'P_C3_given_C1C2', 'Joint'):
        lines.append(f'| {name} (%) | {number(summary[name])} |')
    lines += ['', 'PPL为新增补充实现：全部阶段PR乘以成功前缀的路径效率。原生PBSP在阶段失败后继续执行后续阶段；条件率只在前序全成功的子集计算，分母为0时留为未定义。', '',
        '| 阶段 | 前序全成功episode数 | 本阶段成功数 | 条件率 (%) |', '| --- | --- | --- | --- |']
    for k, (denominator, numerator, rate) in enumerate(zip(conditional['conditioned_episodes'], conditional['successes'], conditional['rates'])):
        lines.append(f'| C{k+1} | {denominator} | {numerator} | {number(100 * rate if rate is not None else None)} |')
    lines += ['', '| 过程指标 | 结果 |', '| --- | --- |']
    metrics = [('GT无效不同候选访问数/episode', process['Invalid_visits_per_ep']),
        ('GT无效候选追逐路径/episode (m)', process['invalid_visit_path_m_per_ep']),
        ('无效身份全部追逐路径上界/episode (m)', process['invalid_identity_pursuit_path_m_per_ep']),
        ('无约束排序偏离最近 (%)', process['deviation_from_nearest_proposal_pct']),
        ('滞回后偏离最近 (%)', process['deviation_from_nearest_after_hysteresis_pct']),
        ('实际下发偏离最近 (%)', process['deviation_from_nearest_realized_pct']),
        ('规划耗时中位数 (ms)', process['planning_ms']['median']),
        ('规划耗时P95 (ms)', process['planning_ms']['p95']),
        ('逐动作/原生阶段路径最大误差 (m)', process['native_stage_path_max_abs_error_m'])]
    for name, value in metrics:
        lines.append(f'| {name} | {number(value, 8 if "最大误差" in name else 2)} |')
    lines += ['', f'规划共{len(decisions)}个快照；实际非空下发分母为{process["eligible_realized_dispatches"]}。plan_ms从BLP快照构建计时至排序返回。无效访问路径按实际动作目标的原生GT标签分配，同一位移不重复计数；身份全部追逐路径另列上界。', '',
        '![规划耗时分布](../figures/CP3/planning_latency.png)', '',
        '| 回退类型 | 根决策次数 | 涉及该边界的决策数 | 递归状态命中数 |', '| --- | --- | --- | --- |']
    groups = [process[k] for k in ('fallback_root_decisions', 'fallback_decision_incidence', 'fallback_memo_state_counts')]
    for key in sorted(set().union(*groups)):
        lines.append('| ' + ' | '.join([key, *[str(group.get(key, 0)) for group in groups]]) + ' |')
    lines += ['', '## 行为与俯视图', '',
        '例图固定采用清单最先运行的两条。每阶段显示最早实际下发且至少有两个候选的快照，否则显示最早非空已下发快照；黑线为整阶段实际轨迹。候选编号与belief表对应，橙星为滞回后BLP选择，红圈为实际下发。', '']
    for item in figures['figures']:
        name = item['scene_id'] + '_' + item['episode_id']
        lines += [f'![{name}](../figures/CP3/{name}.png)', '']
    outcomes = Counter(stage['outcome_description'] for row in behavior['episodes'] for stage in row['stages'])
    lines += ['| 观察到的阶段结局 | 数量 |', '| --- | --- |']
    lines += [f'| {name} | {count} |' for name, count in sorted(outcomes.items())]
    lines += ['', '上述结局按原生成功及STOP原因分类，不是因果归因。候选坐标GT与agent最终成功分别保留；系统确认后原生失败的阶段会单列审查。', '',
        f'完整Full快照重放无违例，覆盖{mechanism["counts"]["snapshots"]}个快照；末阶段Full/Myopic及Suffix-only/Nearest的Q等价核对通过。相同Full输入上的其他变体提议仅作反事实诊断，不代替在线消融。', '',
        '## 全部固定回合', '', '| 场景/episode | 三阶段成功 | 步数 | SPL (%) | PR (%) | PPL (%) |', '| --- | --- | --- | --- | --- | --- |']
    for row in sorted(episodes, key=lambda r: r['scene_id']):
        values = [row['scene_id'] + '/' + str(row['episode_id']),
                  '/'.join(str(int(s['success'])) for s in row['stages']), str(row['total_steps']),
                  number(100 * row['native_mon_metrics']['spl']), number(100 * row['PR']), number(100 * row['PPL'])]
        lines.append('| ' + ' | '.join(values) + ' |')
    lines += ['', '## 问题、差异与下一步', '',
        'E沿用已确认的候选高斯分数，S为候选级方位覆盖；共享滞回和类别冷却按三个层级分别报告偏离率。标定与评估存在共享场景及source目标，交叉拟合及其泛化限制见CP2a/CP2b与deviations.md。', '',
        '20条用于检查管线、行为与计算开销；当前尚无四变体完整在线比较，不能据此给出Full的机制增益结论。按设计CP3的【需确认】检查点提交审阅，通过后运行Full400，再依次运行Nearest、Myopic和Suffix-only各400条。', '',
        '[完整汇总](../diagnostics/CP3/summary/full.json)、[逐episode日志](../diagnostics/CP3/logs/full_episodes.jsonl)、[逐决策日志](../diagnostics/CP3/logs/full_decisions.jsonl)、[行为审查](../diagnostics/CP3/completed-behavior-review.json)。', '']
    (ROOT / 'reports/CP3.md').write_text('\n'.join(lines))
    return {'generated_at_utc': datetime.now(timezone.utc).isoformat(), 'n_episodes': 20,
            'report_sha256': digest(ROOT / 'reports/CP3.md'), 'latency_figure_sha256': digest(latency),
            'stage_accepted': False, 'launches_workers': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    try:
        evidence = verify()
    except (AssertionError, ValueError, FileNotFoundError) as error:
        if args.check:
            print(json.dumps({'ready': False, 'reason': str(error), 'launches_workers': False}), flush=True)
            return
        raise
    result = {'ready': True, 'n_episodes': 20, 'launches_workers': False} if args.check else render(*evidence)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
