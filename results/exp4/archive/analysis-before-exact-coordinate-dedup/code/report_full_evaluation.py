"""Render CP4/CP5 from complete frozen evaluations; never authorize or launch."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

from evaluate_sequence import validate_stage_proof, GateError
from experiments.exp4_collect import source_fingerprint
import report_variants as analysis

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ('nearest', 'suffix_only', 'myopic', 'full')
LABELS = {'nearest': 'Nearest', 'suffix_only': 'Suffix-only',
          'myopic': 'Belief-myopic', 'full': 'Full BLP'}
RATES = ('SR', 'SPL', 'PR', 'PPL', 'P_C1', 'P_C2_given_C1', 'P_C3_given_C1C2', 'Joint')


def load(relative):
    return json.loads((ROOT / relative).read_text())


def digest(relative):
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def rows(relative):
    return [json.loads(line) for line in (ROOT / relative).read_text().splitlines() if line]


def number(value, digits=2):
    return '未定义' if value is None else f'{value:.{digits}f}'


def table(headers, values):
    return ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join('---' for _ in headers) + ' |'] + [
        '| ' + ' | '.join(map(str, row)) + ' |' for row in values]


def verify(stage):
    names = ('full',) if stage == 'CP4' else VARIANTS
    fingerprint = source_fingerprint()
    cp2 = load('meta/CP2b_acceptance.json')
    parameter_sha = digest('meta/calibration.json')
    require(cp2.get('accepted') is True and cp2['source_fingerprint'] == fingerprint and
            cp2['parameter_sha256'] == parameter_sha, 'The current source-bound CP2b is required')
    versions = {'report_variants.py': analysis.SOURCE_SHA256,
                'action_path_metrics.py': analysis.ACTION_SOURCE_SHA256,
                'native_candidate_gt.py': analysis.GT_SOURCE_SHA256,
                'blp_parameters.py': analysis.PARAMETER_SOURCE_SHA256}
    variants, evidence = {}, {}
    for name in names:
        # An unreviewed anomalous score blocks normal publication. The raw
        # summary remains available for the required stop-and-review report.
        ledger = validate_stage_proof(name, ROOT)
        require(ledger['source_fingerprint'] == fingerprint and ledger['parameter_sha256'] == parameter_sha,
                f'{name}: runtime source or frozen parameter assignment differs')
        summary = load(f'summary/{name}.json')
        episodes = rows(f'logs/{name}_episodes.jsonl')
        require(summary['analysis_sources_sha256'] == versions, f'{name}: reanalyze with the current common analysis sources')
        require(summary['n_error'] == 0 and summary['n_episodes'] == len(episodes) == 400,
                f'{name}: four hundred complete error-free episodes are required')
        process = summary['process']
        require(process['invalid_path_complete'] and not process['invalid_path_association_gaps'] and
                process['invalid_path_source'] == 'blp_action_target_v1_native_gt' and
                process['native_stage_path_check_count'] == 1200, f'{name}: incomplete accepted-action path audit')
        precise = summary['rates_pct_unrounded']
        require(math.isclose(precise['Joint'], precise['SR'], rel_tol=0., abs_tol=1e-10), f'{name}: Joint and SR differ')
        if name == 'nearest':
            require(process['deviation_from_nearest_proposal_pct'] == 0., 'Nearest unconstrained proposals deviate')
        variants[name] = (episodes, [], summary)
        for relative in (f'meta/evaluation-{name}-runs.json', f'summary/{name}.json',
                         f'logs/{name}_episodes.jsonl', f'logs/{name}_decisions.jsonl'):
            evidence[relative] = digest(relative)
    evidence.update({relative: digest(relative) for relative in
                    ('meta/CP2b_acceptance.json', 'meta/calibration.json', 'meta/3mon-identities.json',
                     'code/report_full_evaluation.py')})
    comparison = None
    if stage == 'CP5':
        comparison = analysis.effects_and_pairs(variants)
        require(load('summary/effects.json') == comparison, 'Canonical effects or paired counts differ from the normalized inputs')
        with (ROOT / 'summary/all.csv').open(newline='') as handle:
            csv_rows = list(csv.DictReader(handle))
        require(len(csv_rows) == 4 and {row['variant'] for row in csv_rows} == set(VARIANTS), 'Canonical comparison CSV coverage differs')
        for row in csv_rows:
            name = row['variant']
            summary = variants[name][2]
            require(row['n_episodes'] == '400' and row['n_error'] == '0', 'Comparison CSV counts differ')
            require(row['method'] == summary['method'] and row['task'] == summary['task'] and
                    row['future_suffix'] == str(name in ('suffix_only', 'full')) and
                    row['validity_belief'] == str(name in ('myopic', 'full')), 'Comparison CSV variant definition differs')
            for key in (*RATES, 'P_C4_given_C1C2C3'):
                expected = '' if summary[key] is None else f'{summary[key]:.2f}'
                require(row[key] == expected, f'Comparison CSV has stale {key} values')
            for key in ('Invalid_visits_per_ep', 'deviation_from_nearest_proposal_pct',
                        'deviation_from_nearest_after_hysteresis_pct', 'deviation_from_nearest_realized_pct'):
                expected = '' if summary['process'][key] is None else f'{summary["process"][key]:.2f}'
                require(row[key] == expected, f'Comparison CSV has stale {key} values')
        evidence.update({p: digest(p) for p in ('summary/all.csv', 'summary/effects.json')})
    return variants, comparison, evidence


def process_tables(variants):
    names = [name for name in VARIANTS if name in variants]
    lines = table(['变体', '无效访问/ep', '无效访问路径/ep (m)', '无效身份路径上界/ep (m)',
                   'proposal偏离 (%)', '滞回后偏离 (%)', '实际下发偏离 (%)', '规划中位数 (ms)', '规划P95 (ms)'], [
        [LABELS[name], *[number(variants[name][2]['process'][key]) for key in
         ('Invalid_visits_per_ep', 'invalid_visit_path_m_per_ep', 'invalid_identity_pursuit_path_m_per_ep',
          'deviation_from_nearest_proposal_pct', 'deviation_from_nearest_after_hysteresis_pct',
          'deviation_from_nearest_realized_pct')],
         number(variants[name][2]['process']['planning_ms']['median']),
         number(variants[name][2]['process']['planning_ms']['p95'])] for name in names])
    lines += ['', '无效访问以检查时的候选坐标GT标注；路径按每个已接受动作的实际目标坐标分配，同一位移只计一次。身份全部追逐路径另列上界。三个偏离率使用各自适用分母；共享1米滞回和类别冷却继续生效。', '']
    for name in names:
        process = variants[name][2]['process']
        groups = [process[k] for k in ('fallback_root_decisions', 'fallback_decision_incidence', 'fallback_memo_state_counts')]
        lines += [f'{LABELS[name]}回退记录：', '']
        lines += table(['类型', '根决策数', '涉及该边界的决策数', '递归状态命中数'],
                       [[key, *[g.get(key, 0) for g in groups]] for key in sorted(set().union(*groups))])
        lines += ['', f'非空实际下发分母：{process["eligible_realized_dispatches"]}；'
                  f'逐动作/原生阶段路径核对{process["native_stage_path_check_count"]}项，'
                  f'最大绝对误差{process["native_stage_path_max_abs_error_m"]:.9g}米。', '']
    return lines


def comparison_figure(variants, comparison):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    destination = ROOT / 'figures/CP5'
    destination.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), constrained_layout=True)
    for ax, metric in zip(axes, ('SR', 'SPL')):
        values = [variants[name][2]['rates_pct_unrounded'][metric] for name in VARIANTS]
        bars = ax.bar(range(4), values, color=['#64748b', '#6f88b8', '#78a6a3', '#d98b54'])
        ax.set(xticks=range(4), xticklabels=['Nearest', 'Suffix-only', 'Myopic', 'Full'],
               ylabel=f'{metric} (%)', ylim=(0, 100))
        ax.bar_label(bars, labels=[number(value) for value in values], padding=3, fontsize=9)
    fig.suptitle('Frozen 3-MON evaluation | 400 paired episodes per variant')
    fig.savefig(destination / 'SR_SPL.png', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), constrained_layout=True)
    keys = ('future_main_effect_pp', 'belief_main_effect_pp', 'interaction_pp')
    for ax, metric in zip(axes, ('SR', 'SPL')):
        values = [comparison['effects'][metric][key] for key in keys]
        bars = ax.bar(range(3), values, color=['#6f88b8', '#78a6a3', '#d98b54'])
        ax.axhline(0, color='#555555', linewidth=.8)
        ax.set(xticks=range(3), xticklabels=['Future', 'Belief', 'Interaction'],
               ylabel=f'{metric} effect (percentage points)')
        ax.bar_label(bars, labels=[number(value) for value in values], padding=3, fontsize=9)
        ax.margins(y=.2)
    fig.suptitle('2 x 2 effects computed from unrounded scores')
    fig.savefig(destination / 'effects.png', dpi=180)
    plt.close(fig)
    return ['figures/CP5/SR_SPL.png', 'figures/CP5/effects.png']


def render(stage, variants, comparison, evidence):
    names = [name for name in VARIANTS if name in variants]
    lines = [f'# Exp4 {stage}：' + ('Full完整评估' if stage == 'CP4' else '四变体机制消融'), '',
             '以下数字来自固定20场景、每场景20条3-MON的完整在线评估。原生完成状态、400条有序身份、冻结参数与实际动作路径均已核对。', '',
             'SR、SPL、PR调用原PBSP评估结果；PPL为新增补充实现，等于全部阶段PR乘以成功前缀的路径效率。原生失败后继续执行后续阶段，条件成功率仅在前序全成功子集计算，分母为0时留为未定义。', '']
    lines += table(['变体', 'Future', 'Belief', *[key + ' (%)' for key in RATES]], [
        [LABELS[name], name in ('suffix_only', 'full'), name in ('myopic', 'full'),
         *[number(variants[name][2][key]) for key in RATES]] for name in names])
    lines += ['', '所有比率以百分数表示，展示保留两位；计算主效应使用未舍入值。', '']
    for name in names:
        summary = variants[name][2]
        counts = summary['conditional_counts']
        lines += [f'{LABELS[name]}条件成功分母：', '']
        lines += table(['阶段', '前序全成功episode数', '本阶段成功数', '条件成功率 (%)'],
            [[k + 1, denominator, numerator, number(100 * rate if rate is not None else None)]
             for k, (denominator, numerator, rate) in enumerate(zip(
                 counts['conditioned_episodes'], counts['successes'], counts['rates']))])
        lines += ['', '运行提交：' + ', '.join('`' + commit + '`' for commit in summary['source_commits']) + '。', '']
    lines += process_tables(variants)
    figures = []
    if stage == 'CP5':
        lines += table(['指标', '前瞻主效应 (pp)', 'belief主效应 (pp)', '交互项 (pp)'], [
            [metric, *[number(comparison['effects'][metric][key], 6) for key in
                       ('future_main_effect_pp', 'belief_main_effect_pp', 'interaction_pp')]] for metric in ('SR', 'SPL')])
        lines += ['', '前瞻主效应=[(Suffix-only−Nearest)+(Full−Myopic)]/2；belief主效应=[(Myopic−Nearest)+(Full−Suffix-only)]/2；交互项=(Full−Myopic)−(Suffix-only−Nearest)。', '']
        lines += table(['配对', '均成功', '左成功右失败', '左失败右成功', '均失败', '配对数'], [
            [name, *[record[key] for key in ('both_success', 'left_only_success', 'right_only_success', 'both_failure', 'n_paired')]]
            for name, record in comparison['paired_episode_success'].items()])
        figures = comparison_figure(variants, comparison)
        for relative in figures:
            lines += ['', f'![{Path(relative).stem}](../{relative})']
        lines += ['', '这组比较评估冻结协议下的两个机制及其交互，主效应和配对计数本身不提供显著性检验结论。标定与评估共享场景及部分source目标，泛化边界见CP2和差异记录。', '',
                  '[全部变体CSV](../summary/all.csv)、[未舍入效应与六组配对计数](../summary/effects.json)。', '']
    else:
        lines += ['Full的400条完整评估已汇总，其他三个变体的在线对比尚待完成。按设计，汇报CP4后依次运行Nearest、Belief-myopic、Suffix-only各400条。此工具没有将CP4标为已汇报，也没有启动下一阶段。', '']
    for name in names:
        lines += [f'[{LABELS[name]}汇总](../summary/{name}.json)、[逐episode](../logs/{name}_episodes.jsonl)、[逐决策](../logs/{name}_decisions.jsonl)。', '']
    lines += ['[冻结参数](../meta/calibration.json)、[差异与处理](../deviations.md)。', '']
    report = f'reports/{stage}.md'
    (ROOT / report).write_text('\n'.join(lines))
    proof = {'prepared_at_utc': datetime.now(timezone.utc).isoformat(), 'checkpoint': stage,
             'n_episodes_by_variant': {name: 400 for name in names}, 'input_sha256': evidence,
             'output_sha256': {p: digest(p) for p in [report, *figures]},
             'reported_in_chat': False, 'authorizes_next_stage': False, 'launches_workers': False,
             'figures_require_visual_review': bool(figures)}
    analysis.write_json(ROOT / f'meta/{stage}_report_prepared.json', proof)
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', choices=('CP4', 'CP5'), required=True)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    try:
        evidence = verify(args.checkpoint)
    except (ValueError, GateError, FileNotFoundError, KeyError) as error:
        if args.check:
            print(json.dumps({'ready': False, 'reason': str(error), 'launches_workers': False}))
            return
        raise
    result = {'ready': True, 'checkpoint': args.checkpoint, 'launches_workers': False} if args.check else render(args.checkpoint, *evidence)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
