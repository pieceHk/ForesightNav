"""Assemble CP2a from completed artifacts; never fit or freeze parameters."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read(relative):
    return json.loads((ROOT / relative).read_text())


def sha(relative):
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def write(relative, value):
    (ROOT / relative).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join('---' for _ in headers) + ' |'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def pct(value):
    return f'{100 * value:.2f}%'


def main():
    fit = read('meta/calibration_per_fold.json')
    ledger = read('meta/calibration_runs.json')
    folds = read('meta/folds.json')['folds']
    audit = read('meta/completed-calibration-scene-audits.json')
    assert ledger['status'] == 'completed_three_folds_pending_CP2a'
    assert audit['complete_episodes'] == 360 and len(audit['completed_scenes']) == 12
    assert audit['frozen_runtime_unchanged']
    assert all(s['native_exit_code'] == 0 and not s['shadow_errors'] for s in audit['completed_scenes'])
    assert not (ROOT / 'meta/calibration.json').exists(), 'Do not overwrite a frozen review'
    assert not fit['stable_under_protocol_rule'] and not fit['all_fold_ci_overlap']['c_exp']
    assert all(g['bootstrap']['n_success'] == 200 for g in fit['per_fold'].values())

    cohorts, hashes, all_checks = [], {}, []
    for f in range(3):
        path = f'logs/calibration_fold{f}_events.jsonl'
        events = [json.loads(line) for line in (ROOT / path).read_text().splitlines() if line]
        checks = [r for r in events if r['event'] == 'verification_final']
        all_checks.extend(checks)
        key = lambda r: (r['scene_id'], str(r['episode_id']), r['k'])
        starts = {key(r): r for r in events if r['event'] == 'empty_activation_started'}
        ends = {key(r): r for r in events if r['event'] == 'empty_activation'}
        assert len(starts) == sum(r['event'] == 'empty_activation_started' for r in events)
        assert len(ends) == sum(r['event'] == 'empty_activation' for r in events)
        assert set(ends) <= set(starts)
        unique_v, unique_y = {}, {}
        for r in checks:
            identity = (*key(r), r['candidate_id'])
            unique_v[identity] = max(unique_v.get(identity, 0), r['v'])
            unique_y[identity] = max(unique_y.get(identity, 0), r['y'])
        theta = fit['per_fold'][str(f)]['theta']
        assert abs(theta['pi'] - sum(unique_v.values()) / len(unique_v)) < 1e-12
        assert abs(theta['c_exp'] - np.mean([r['path_to_first_candidate_m'] for r in ends.values()])) < 1e-12
        cohorts.append({'fold': f, 'episodes': 120, 'scenes': folds[f],
                        'event_counts': dict(Counter(r['event'] for r in events)),
                        'verification_events': len(checks), 'distinct_checked': len(unique_v),
                        'distinct_confirmed': sum(unique_v.values()), 'distinct_GT_valid': sum(unique_y.values()),
                        'v_counts': dict(Counter(r['v'] for r in checks)),
                        'y_counts': dict(Counter(r['y'] for r in checks)),
                        'uncompleted_empty_activations': [starts[k] for k in sorted(set(starts) - set(ends))]})
        hashes[path] = sha(path)
        snapshots = f'logs/calibration_fold{f}_snapshots.jsonl'
        hashes[snapshots] = sha(snapshots)
    for path in ('meta/calibration_per_fold.json', 'meta/calibration_runs.json', 'meta/folds.json',
                 'meta/completed-calibration-scene-audits.json', 'meta/calibration-overlap-audit.json',
                 'meta/protocol_decisions.json', 'diagnostics/native-offline-gt-check.json',
                 'meta/calibration-log-archive.json', 'code/archive_calibration_logs.py',
                 'code/fit_calibration_reviewed.py', 'code/report_calibration.py'):
        hashes[path] = sha(path)
    summary = {'generated_at_utc': datetime.now(timezone.utc).isoformat(),
               'scope': 'completed three-fold CP2a; no evaluation outcomes and no parameter freeze',
               'cohorts': cohorts, 'evidence_sha256': hashes,
               'recommendation': 'three_fold_cross_fit_pending_user_confirmation'}
    write('meta/CP2a_data_summary.json', summary)
    proposal = {'schema_version': 'exp4_crossfit_proposal_v1', 'frozen': False,
                'label_source': 'system_verification', 'selection': 'cross_fit',
                'collected_folds': [0, 1, 2],
                'parameters_excluding_fold': fit['cross_fit_proposals'],
                'parameters_for_uncollected_folds': fit['theta_pool'],
                'scene_to_fold': {scene: f for f, scenes in enumerate(folds) for scene in scenes},
                'runtime_support': 'pending implementation; existing runtime loader accepts pooled only',
                'evidence_sha256': hashes,
                'selection_reason': 'All decision flips <10%; exploration-cost confidence intervals do not overlap.'}
    write('meta/calibration.crossfit.proposed.json', proposal)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.3), constrained_layout=True)
    keys = ('a', 'b0', 'pi', 'c_exp')
    colors = ('#2468a2', '#db7c26', '#2b9070')
    for ax, parameter in zip(axes, keys):
        for f in range(3):
            g = fit['per_fold'][str(f)]
            lo, hi = g['bootstrap']['ci95'][parameter]
            center = g['theta'][parameter]
            ax.errorbar(center, f, xerr=[[center-lo], [hi-center]], fmt='o', capsize=4,
                        color=colors[f], label='95% episode bootstrap CI' if f == 0 else None)
        ax.axvline(fit['theta_pool'][parameter], color='#666666', linestyle='--',
                   linewidth=1, label='Pooled self-label estimate')
        ax.set(yticks=range(3), yticklabels=['Fold 0', 'Fold 1', 'Fold 2'],
               ylim=(2.5, -.5), xlabel={'b0': 'b0', 'pi': 'pi', 'c_exp': 'C_exp (m)'}.get(parameter, parameter))
        ax.grid(axis='x', alpha=.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, 1.09), ncol=2, frameon=False)
    fig.savefig(ROOT / 'figures/calibration_bootstrap_intervals.png', dpi=180, bbox_inches='tight')
    plt.close(fig)

    # Both labels use exactly the same E bin edges, with their event counts
    # shown explicitly; this changes only presentation of the raw evidence.
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), constrained_layout=True)
    shared_bins = np.linspace(min(r['E'] for r in all_checks), max(r['E'] for r in all_checks), 21)
    for label, color in ((0, 'tab:red'), (1, 'tab:blue')):
        values = [r['E'] for r in all_checks if r['v'] == label]
        axes[0].hist(values, bins=shared_bins, alpha=.5, color=color, label=f'v={label} (n={len(values)})')
    axes[0].set(xlabel='Semantic score E', ylabel='Verification events')
    axes[0].legend()
    axes[1].hist([r['N_views'] for r in all_checks], bins=np.arange(14) - .5)
    axes[1].set(xlabel='Distinct visible bearing bins', ylabel='Verification events', xticks=range(13))
    fig.savefig(ROOT / 'figures/calibration_evidence_common_bins.png', dpi=180)
    plt.close(fig)

    out = ['# CP2a：前三组标定审查', '',
           '360条2-MON已完整结束。三组各200次episode自助重采样全部成功；各组相对合并参数的决策翻转率均低于0.7%。但C_exp的组间95%区间不重叠，按设计第2.5节，**不能直接冻结pooled参数，应改用交叉拟合并等待确认**。', '',
           '建议使用下文已列出的三组交叉拟合参数继续。暂不补第4、5组：已有3849次检查，决策对参数变化的敏感性较低；新增组不会改变现有两组已不重叠的区间。若优先覆盖全部20个场景，可另补240条2-MON，再重新审查，但这不是恢复pooled资格的办法。', '',
           '当前未生成冻结的calibration.json，未运行Full 3-MON。交叉拟合的运行时加载尚未实现；本文件及JSON均为可审阅提案。', '',
           '数据按固定seed=20261107分组。运行源码为d94ca7d所绑定的同一指纹，收集期间未修改；12个原生launcher均正常退出。完整身份、source episode、目标顺序、轨迹和STOP核查通过。', '',
           table(['Fold', '场景', 'Episode', '检查事件', '被检查的不同候选', '确认的不同候选', '完成的探索事件'],
                 [[c['fold'], ', '.join(c['scenes']), c['episodes'], c['verification_events'], c['distinct_checked'], c['distinct_confirmed'], c['event_counts']['empty_activation']] for c in cohorts]), '',
           '逻辑回归使用每次最终验证事件的v；π按持久空间身份去重，统计是否曾被确认。两者分母不同。两项均只使用被检查的候选，存在选择偏差。E沿用用户确认的单个候选高斯codebook加权文本分数；B表示检查时被系统确认的概率。', '',
           '自标定参数如下。a严格为正；C_exp单位为米。', '',
           table(['参数集', *keys], [[f'Fold {f}', *[f'{fit["per_fold"][str(f)]["theta"][k]:.6f}' for k in keys]] for f in range(3)] +
                 [['Pooled', *[f'{fit["theta_pool"][k]:.6f}' for k in keys]]]), '',
           '每组按episode完整重采样200次，随机种子为20261107+fold，95%区间为：', '',
           table(['Fold', *keys], [[f, *[f'[{fit["per_fold"][str(f)]["bootstrap"]["ci95"][k][0]:.6f}, {fit["per_fold"][str(f)]["bootstrap"]["ci95"][k][1]:.6f}]' for k in keys]] for f in range(3)]), '',
           '![各组自标签参数的95%区间](../figures/calibration_bootstrap_intervals.png)', '',
           'a、b0、π均有共同区间；C_exp没有。具体为Fold 1的[1.099441, 2.572049]与Fold 2的[0.643488, 1.024613]米不相交，间隔约0.0748米。即使翻转率很低，设计也同时要求区间重叠，不能放宽该规则。', '',
           table(['参数', '组均值', '总体标准差', '最小', '最大', '极差'], [[k, *[f'{fit["between_fold_statistics"][k][s]:.6f}' for s in ('mean','std_population','min','max','range')]] for k in keys]), '',
           '下面以y代替v拟合的参数仅用于诊断，不进入提案或导航。', '',
           table(['GT诊断参数集', *keys], [[f'Fold {f}', *[f'{fit["per_fold"][str(f)]["theta_gt"][k]:.6f}' for k in keys]] for f in range(3)] +
                 [['Pooled GT', *[f'{fit["theta_pool_gt_diagnostic_only"][k]:.6f}' for k in keys]]]), '',
           '混淆矩阵以系统确认v为预测、原生GOAT候选坐标判据y为真值。FPR=FP/(FP+TN)，FNR=FN/(FN+TP)；不是只看确认样本的错误比例。', '',
           table(['数据', 'TP', 'TN', 'FP', 'FN', '一致率', 'FPR', 'FNR', 'ECE vs GT'],
                 [[name, d['tp'], d['tn'], d['fp'], d['fn'], pct(d['agreement']), pct(d['false_positive_rate']), pct(d['false_negative_rate']), f'{d["ece_vs_gt"]:.6f}']
                  for name, d in [(f'Fold {f}', fit['per_fold'][str(f)]['diagnostics']) for f in range(3)] + [('Pooled', fit['pooled_diagnostics'])]]), '',
           'GT诊断显示系统确认会漏掉部分GT有效候选；不能将q或B直接解释为GT成功概率。可靠性图使用q=sigmoid(aE+b0)，分10档，以下为各档实际样本量。该图是标定数据内诊断，不是独立验证集表现。', '',
           table(['q档位', 'n', '平均q', 'GT有效比例'], [[f'[{b["lower"]:.1f}, {b["upper"]:.1f}{"]" if b["upper"]==1 else ")"}', b['n'], f'{b["predicted_q"]:.6f}' if b['n'] else '—', f'{b["gt_valid_fraction"]:.6f}' if b['n'] else '—'] for b in fit['pooled_diagnostics']['reliability_bins']]), '',
           '![相对GT的可靠性图](../figures/calibration_reliability.png)', '',
           '离线Full BLP重放使用全部19453个影子快照，分母包含3787个空根快照（None与None不翻转）；另列15666个非空根快照下的条件比例。以下比较不代表独立轨迹或3-MON成绩。', '',
           table(['参数 vs Pooled', '翻转数 / 全快照', '主翻转率', '非空条件翻转率'],
                 [[name, f'{d["flips"]}/{d["all_snapshots"]}', pct(d['rate']), pct(d['eligible_rate'])]
                  for name, d in [(f'Fold {f}', fit['per_fold'][str(f)]['flip_vs_pool']) for f in range(3)] + [('Pooled GT（仅诊断）', fit['pool_gt_decision_flip'])]]), '',
           '![E按系统确认标签分布及N_views分布](../figures/calibration_evidence_common_bins.png)', '',
           '交叉拟合提案按待评估场景所属组排除整组数据。未收集的Fold 3、4按设计使用三组合并参数。以下四套参数全部来自v，没有GT参数。', '',
           table(['评估场景所在组', '拟合所用组', *keys],
                 [[f'Fold {f}', ', '.join(str(x) for x in range(3) if x != f), *[f'{fit["cross_fit_proposals"][str(f)][k]:.6f}' for k in keys]] for f in range(3)] +
                 [['Fold 3、4', '0, 1, 2', *[f'{fit["theta_pool"][k]:.6f}' for k in keys]]]), '',
           table(['Fold', '对应场景'], [[f, ', '.join(scenes)] for f, scenes in enumerate(folds)]), '',
           '有两条激活事件没有对应的首候选完成记录：DYehNKdT76V/5/k0和wcojb4TFT35/17/k1。C_exp使用361条完成记录的条件均值，没有为未完成事件虚构0米；报告保留这项删失限制。', '',
           '2-MON和3-MON是不同的有序episode文件，但共享12个场景、720个底层source episode；12条评估episode的前两目标顺序与标定一致。因此不能声称独立场景或目标泛化。此分析没有读取3-MON评估结果。', '',
           '固定原生GOAT离线核查覆盖全部360条标定和两次排除拟合的隔离管线：3880条候选标签、7246条恢复点标签及测地距离全部匹配，最大距离误差0。标定保留13处旧式无效路径归属缺口，不影响E、S、v/y或C_exp；正式CP3前仍须采纳已在隔离工作区验证的逐动作记录补丁。', '',
           '数值求解有2次v标签优化器状态失败，均使用同一正斜率MLE目标重试并通过约束KKT检查；600次bootstrap全部成功，无一类标签或分离失败被伪造为有限值。详见元数据中的numerical_optimizer_retry。', '',
           '可复查材料：[完整拟合JSON](../meta/calibration_per_fold.json)、[交叉拟合提案（未冻结）](../meta/calibration.crossfit.proposed.json)、[数据与SHA清单](../meta/CP2a_data_summary.json)、[场景分组](../meta/folds.json)、[完整场景审计](../meta/completed-calibration-scene-audits.json)、[原生GT核查](../diagnostics/native-offline-gt-check.json)、[数据重叠审计](../meta/calibration-overlap-audit.json)。', '',
           '三个组的完整事件与快照另以gzip随仓库保存，[压缩清单](../meta/calibration-log-archive.json)列出6份文件的原始及压缩SHA256，解压后逐字节一致。原始未压缩日志仍保留在服务器results/exp4/logs/；可用于复现拟合和19453快照重放。', '',
           '下一步待确认：采用上述三组交叉拟合参数，完成加载实现、记录补丁及源码绑定后冻结，并进入固定20条Full冒烟；或先补齐第4、5组再重审。当前没有把提案作为可用的冻结参数。', '']
    (ROOT / 'reports/CP2a.md').write_text('\n'.join(out))
    print(json.dumps({'report': 'reports/CP2a.md', 'proposal_frozen': False,
                      'n_verifications': sum(c['verification_events'] for c in cohorts),
                      'n_completed_explorations': sum(c['event_counts']['empty_activation'] for c in cohorts),
                      'bootstrap_success': 600, 'stable_under_protocol_rule': False}, indent=2))


if __name__ == '__main__':
    main()
