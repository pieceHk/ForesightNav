"""Build CP2a from the corrected collection and its bound fit; never freeze."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from experiments.exp4_collect import source_fingerprint
from candidate_identities import canonicalize, SOURCE_SHA256 as IDENTITY_SHA256

ROOT = Path(__file__).resolve().parents[1]
KEYS = ('a', 'b0', 'pi', 'c_exp')


def read(relative):
    return json.loads((ROOT / relative).read_text())


def sha(relative):
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def write(relative, value):
    (ROOT / relative).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                           allow_nan=False) + '\n')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join('---' for _ in headers) + ' |'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def pct(value):
    return '未定义' if value is None else f'{100 * value:.4f}%'


def selection_rule(fit):
    groups = fit['per_fold']
    require(set(groups) == {'0', '1', '2'}, 'Exactly three fitted groups are required')
    overlap = {k: max(g['bootstrap']['ci95'][k][0] for g in groups.values()) <=
               min(g['bootstrap']['ci95'][k][1] for g in groups.values()) for k in KEYS}
    stable = all(g['flip_vs_pool']['rate'] <= .1 for g in groups.values()) and all(overlap.values())
    require(fit['all_fold_ci_overlap'] == overlap, 'Reported interval overlap differs from the actual intervals')
    require(fit['stable_under_protocol_rule'] == stable, 'Selection differs from the frozen protocol rule')
    return 'pooled' if stable else 'cross_fit', overlap


def existing_approval_comparison(fit, selection):
    archive = ROOT / 'archive/calibration-a2-inherited-before-depth-fix/meta'
    old_review = json.loads((archive / 'CP2a_review.json').read_text())
    old = json.loads((archive / 'calibration.crossfit.proposed.json').read_text())
    approved = old_review.get('accepted') is True and old_review.get('chosen_choice') == 'three_fold_cross_fit'
    identical = (selection == 'cross_fit' and
                 old['parameters_excluding_fold'] == fit.get('cross_fit_proposals') and
                 old['parameters_for_uncollected_folds'] == fit['theta_pool'])
    return {'old_choice_was_accepted': approved, 'new_four_numeric_sets_identical': identical,
            'existing_numeric_approval_can_be_retained': approved and identical,
            'selection_or_values_changed': not identical,
            'accepted_by_this_script': False}


def parameter_proposal(fit, folds, hashes, selection):
    proposal = {'frozen': False, 'label_source': 'system_verification', 'selection': selection,
                'collected_folds': [0, 1, 2], 'evidence_sha256': hashes,
                'runtime_support': 'implemented and tested; parameter file not yet frozen'}
    if selection == 'cross_fit':
        proposal.update(schema_version='exp4_crossfit_proposal_v1',
                        scene_to_fold={scene: f for f, scenes in enumerate(folds) for scene in scenes},
                        parameters_excluding_fold=fit['cross_fit_proposals'],
                        parameters_for_uncollected_folds=fit['theta_pool'])
    else:
        proposal.update(parameters=fit['theta_pool'])
    return proposal


def verify():
    ledger = read('meta/calibration_runs.json')
    require(ledger['status'] == 'completed_three_folds_pending_CP2a', 'The corrected 360-episode collection is incomplete')
    require(not (ROOT / 'meta/calibration.json').exists(), 'Do not overwrite a frozen review')
    start = read('meta/calibration-start-a3.json')
    binding = read('meta/calibration-fit-binding.json')
    require(start['attempt'] == binding['collection_attempt'] == 'a3', 'A corrected a3 fit binding is required')
    require(start['source_fingerprint'] == ledger['source_fingerprint'] == source_fingerprint(), 'Frozen runtime inputs differ')
    require(start['source_commit'] == binding['collection_source_commit'], 'Fit and collection source commits differ')
    for relative, expected in {**binding['input_sha256'], **binding['output_sha256']}.items():
        require(sha(relative) == expected, f'Fit input or output changed: {relative}')
    fit, folds = read('meta/calibration_per_fold.json'), read('meta/folds.json')['folds']
    selection, overlap = selection_rule(fit)
    require(all(g['bootstrap']['n_success'] == 200 for g in fit['per_fold'].values()), 'Every group needs 200 successful episode bootstrap samples')
    audit = read('meta/completed-calibration-scene-audits.json')
    jobs = ledger['jobs']
    runs = {str(Path(j['run_directory']).resolve()) for j in jobs}
    audited = {str(Path(j['run_directory']).resolve()) for j in audit['completed_scenes']}
    require(len(jobs) == 12 and runs == audited and audit['complete_episodes'] == 360,
            'Current collection and native audit coverage differ')
    require(audit['frozen_runtime_unchanged'] and all(j['native_exit_code'] == 0 and not j['shadow_errors']
            for j in audit['completed_scenes']), 'Native collection integrity failed')
    gt = read('diagnostics/native-offline-gt-check.json')
    checks = {str(Path(j['run_directory']).resolve()): j for j in gt['checks']}
    require(runs <= set(checks), 'Native GT audit refers to an older collection')
    require(all(checks[r]['max_distance_error_m'] <= 1e-5 for r in runs), 'Native GT label/distance replay requires review')
    cohorts, hashes, all_checks = [], {}, []
    for f in range(3):
        relative = f'logs/calibration_fold{f}_events.jsonl'
        events = [json.loads(line) for line in (ROOT / relative).read_text().splitlines() if line]
        events, _, identity_audit = canonicalize(events)
        require(fit['checked_candidate_identity']['source_sha256'] == IDENTITY_SHA256 and
                fit['checked_candidate_identity']['per_fold'][str(f)] == identity_audit,
                'Fitted candidate identity rule or counts differ')
        require(all(r['scene_id'] in folds[f] for r in events), 'An event is assigned to the wrong fold')
        checks_f = [r for r in events if r['event'] == 'verification_final']
        all_checks.extend(checks_f)
        identity = lambda r: (r['scene_id'], str(r['episode_id']), r['k'])
        starts = {identity(r): r for r in events if r['event'] == 'empty_activation_started'}
        ends = {identity(r): r for r in events if r['event'] == 'empty_activation'}
        require(len(starts) == sum(r['event'] == 'empty_activation_started' for r in events), 'Duplicate exploration start')
        require(len(ends) == sum(r['event'] == 'empty_activation' for r in events) and set(ends) <= set(starts), 'Exploration event mismatch')
        unique_v, unique_y = {}, {}
        for r in checks_f:
            key = (*identity(r), r['candidate_id'])
            unique_v[key] = max(unique_v.get(key, 0), r['v'])
            unique_y[key] = max(unique_y.get(key, 0), r['y'])
            require(0 <= r['N_views'] <= 12 and int(r['N_views']) == r['N_views'] and
                    math.isclose(r['S'], 1 - math.exp(-r['N_views'] / 3), abs_tol=1e-12), 'Invalid candidate coverage statistics')
        theta = fit['per_fold'][str(f)]['theta']
        require(math.isclose(theta['pi'], sum(unique_v.values()) / len(unique_v), abs_tol=1e-12), 'Prior differs from checked-candidate confirmations')
        require(math.isclose(theta['c_exp'], float(np.mean([r['path_to_first_candidate_m'] for r in ends.values()])), abs_tol=1e-12), 'Exploration cost differs from completed exploration events')
        cohorts.append({'fold': f, 'episodes': 120, 'scenes': folds[f],
                        'event_counts': dict(Counter(r['event'] for r in events)),
                        'verification_events': len(checks_f), 'distinct_checked': len(unique_v),
                        'distinct_confirmed': sum(unique_v.values()), 'distinct_GT_valid': sum(unique_y.values()),
                        'candidate_identity_audit': identity_audit,
                        'uncompleted_empty_activations': [starts[k] for k in sorted(set(starts) - set(ends))]})
        hashes[relative] = sha(relative)
        snapshots = f'logs/calibration_fold{f}_snapshots.jsonl'
        hashes[snapshots] = sha(snapshots)
    require(sum(c['verification_events'] for c in cohorts) == sum(j['final_labels'] for j in audit['completed_scenes']),
            'Aggregated checks differ from the native run audits')
    archive = read('meta/calibration-log-archive.json')
    for item in archive['files']:
        require(item['round_trip_sha256_matches'] and sha(item['raw']) == item['raw_sha256'] and
                sha(item['gzip']) == item['gzip_sha256'], 'Compressed logs do not match current raw logs')
    paths = ('meta/calibration_per_fold.json', 'meta/calibration_runs.json', 'meta/folds.json',
             'meta/completed-calibration-scene-audits.json', 'meta/calibration-overlap-audit.json',
             'meta/protocol_decisions.json', 'diagnostics/native-offline-gt-check.json',
             'meta/calibration-log-archive.json', 'meta/calibration-fit-binding.json',
             'code/fit_calibration_reviewed.py', 'code/report_calibration.py', 'code/candidate_identities.py')
    hashes.update({p: sha(p) for p in paths})
    return {'fit': fit, 'folds': folds, 'start': start, 'cohorts': cohorts, 'checks': all_checks,
            'hashes': hashes, 'selection': selection, 'overlap': overlap, 'audit': audit,
            'approval_comparison': existing_approval_comparison(fit, selection),
            'native_gt': [checks[r] for r in sorted(runs)],
            'dataset_overlap': read('meta/calibration-overlap-audit.json')}


def render(evidence):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fit, cohorts, checks = (evidence[k] for k in ('fit', 'cohorts', 'checks'))
    folds, selection = evidence['folds'], evidence['selection']
    n_checks = sum(c['verification_events'] for c in cohorts)
    n_explore = sum(c['event_counts'].get('empty_activation', 0) for c in cohorts)
    censored = [r for c in cohorts for r in c['uncompleted_empty_activations']]
    comparison = evidence['approval_comparison']
    summary = {'generated_at_utc': datetime.now(timezone.utc).isoformat(),
               'scope': 'corrected a3 three-fold CP2a; no evaluation outcomes or parameter freeze',
               'collection_attempt': 'a3', 'cohorts': cohorts, 'evidence_sha256': evidence['hashes'],
               'recommended_selection': selection, 'approval_comparison': comparison}
    write('meta/CP2a_data_summary.json', summary)
    proposal = parameter_proposal(fit, folds, evidence['hashes'], selection)
    if selection == 'cross_fit':
        write('meta/calibration.crossfit.proposed.json', proposal)
    else:
        write('meta/calibration.pooled.review-proposed.json', proposal)
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.3), constrained_layout=True)
    for ax, parameter in zip(axes, KEYS):
        for f, color in enumerate(('#2468a2', '#db7c26', '#2b9070')):
            g = fit['per_fold'][str(f)]
            lo, hi = g['bootstrap']['ci95'][parameter]
            ax.plot([lo, hi], [f, f], color=color, linewidth=2)
            ax.plot(g['theta'][parameter], f, 'o', color=color)
        ax.axvline(fit['theta_pool'][parameter], color='#666666', linestyle='--')
        ax.set(yticks=range(3), yticklabels=['Fold 0', 'Fold 1', 'Fold 2'],
               ylim=(2.5, -.5), xlabel='C_exp (m)' if parameter == 'c_exp' else parameter)
        ax.grid(axis='x', alpha=.2)
    fig.suptitle('95% episode-bootstrap intervals; dashed line: pooled estimate')
    fig.savefig(ROOT / 'figures/calibration_bootstrap_intervals.png', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), constrained_layout=True)
    lo, hi = min(r['E'] for r in checks), max(r['E'] for r in checks)
    bins = np.linspace(lo - 1e-8 if lo == hi else lo, hi + 1e-8 if lo == hi else hi, 21)
    for label, color in ((0, 'tab:red'), (1, 'tab:blue')):
        values = [r['E'] for r in checks if r['v'] == label]
        axes[0].hist(values, bins=bins, alpha=.5, color=color, label=f'v={label} (n={len(values)})')
    axes[0].set(xlabel='Semantic score E', ylabel='Verification events')
    axes[0].legend()
    axes[1].hist([r['N_views'] for r in checks], bins=np.arange(14) - .5)
    axes[1].set(xlabel='Distinct visible bearing bins', ylabel='Verification events', xticks=range(13))
    fig.savefig(ROOT / 'figures/calibration_evidence_common_bins.png', dpi=180)
    plt.close(fig)
    flips = [(f'Fold {f}', fit['per_fold'][str(f)]['flip_vs_pool']) for f in range(3)]
    flips += [('Pooled GT（仅诊断）', fit['pool_gt_decision_flip'])]
    units = lambda theta: [f'{theta[k]:.9f}' for k in KEYS]
    rule_text = ('三组决策翻转率均不超过10%，四个参数的组间95%区间均有共同交集，按设计采用pooled提案。'
                 if selection == 'pooled' else
                 '组间稳定性未同时满足翻转率不超过10%和参数区间重叠两个条件，按设计采用交叉拟合提案。')
    out = ['# CP2a：修正深度单位后的前三组标定', '',
           f'固定12场景、360条2-MON已完成。实际检查事件{n_checks}次，完成的空候选探索事件{n_explore}次。三组各200次episode自助重采样成功。{rule_text}', '',
           f'运行提交：`{evidence["start"]["source_commit"]}`。本报告仅使用a3收集、审计及绑定的拟合结果。当前参数尚未冻结，新的Full冒烟和四变体全量评估尚未开始。', '',
           table(['Fold', '场景', 'Episode', '检查事件', '不同候选', '确认候选', '完成探索'],
                 [[c['fold'], ', '.join(c['scenes']), c['episodes'], c['verification_events'], c['distinct_checked'],
                   c['distinct_confirmed'], c['event_counts'].get('empty_activation', 0)] for c in cohorts]), '',
           '逻辑回归保留每次最终检查事件拟合v；π按持久登记身份去重，同一scene/episode/task内最终检查坐标完全相同的不同ID再作合并，统计是否曾被系统确认。坐标合并不用距离阈值、v或GT，原始日志保留。两者分母不同，且只观察被检查的候选，存在选择偏差。E为用户确认保留的单个候选高斯codebook加权文本分数。S采用候选级12档方位覆盖，归一化深度在覆盖观察器中转换为米。B表示检查时被系统确认的概率。', '',
           table(['自标签参数集', *KEYS], [[f'Fold {f}', *units(fit['per_fold'][str(f)]['theta'])] for f in range(3)] +
                 [['Pooled', *units(fit['theta_pool'])]]), '',
           'C_exp单位为米。每组按完整episode重采样200次，种子为20261107+fold；95%区间如下。', '',
           table(['Fold', *KEYS], [[f, *[f'[{fit["per_fold"][str(f)]["bootstrap"]["ci95"][k][0]:.9f}, {fit["per_fold"][str(f)]["bootstrap"]["ci95"][k][1]:.9f}]' for k in KEYS]] for f in range(3)]), '',
           table(['参数', '组间共同交集', '均值', '总体标准差', '最小', '最大', '极差'],
                 [[k, evidence['overlap'][k], *[f'{fit["between_fold_statistics"][k][s]:.9f}'
                   for s in ('mean', 'std_population', 'min', 'max', 'range')]] for k in KEYS]), '',
           '![组内参数区间](../figures/calibration_bootstrap_intervals.png)', '',
           '以下y拟合只用于偏差诊断，不进入提案或导航。', '',
           table(['GT诊断参数集', *KEYS], [[f'Fold {f}', *units(fit['per_fold'][str(f)]['theta_gt'])] for f in range(3)] +
                 [['Pooled GT', *units(fit['theta_pool_gt_diagnostic_only'])]]), '',
           '混淆矩阵以系统确认v为预测、原生GOAT候选坐标判据y为真值。FPR=FP/(FP+TN)，FNR=FN/(FN+TP)。可靠性按q分10档，是标定数据内诊断。', '',
           table(['数据', 'TP', 'TN', 'FP', 'FN', '一致率', 'FPR', 'FNR', 'ECE vs GT'],
                 [[name, d['tp'], d['tn'], d['fp'], d['fn'], pct(d['agreement']), pct(d['false_positive_rate']),
                   pct(d['false_negative_rate']), f'{d["ece_vs_gt"]:.9f}'] for name, d in
                  [(f'Fold {f}', fit['per_fold'][str(f)]['diagnostics']) for f in range(3)] + [('Pooled', fit['pooled_diagnostics'])]]), '',
           table(['q区间', 'n', '平均q', 'GT有效比例'],
                 [[f'{b["lower"]:.1f}至{b["upper"]:.1f}', b['n'],
                   f'{b["predicted_q"]:.9f}' if b['n'] else '未定义',
                   f'{b["gt_valid_fraction"]:.9f}' if b['n'] else '未定义']
                  for b in fit['pooled_diagnostics']['reliability_bins']]), '',
           '![相对GT的可靠性图](../figures/calibration_reliability.png)', '',
           '决策敏感性重放使用全部影子快照；主分母包含空根快照，两边均不选候选时不算翻转。另报非空根条件比例。它不代表其他变体的独立轨迹或3-MON成绩。', '',
           table(['参数 vs Pooled', '翻转数', '全部快照', '非空根', '主翻转率', '非空条件翻转率'],
                 [[name, d['flips'], d['all_snapshots'], d['eligible_snapshots'], pct(d['rate']), pct(d['eligible_rate'])]
                  for name, d in flips]), '',
           '![E与N_views分布](../figures/calibration_evidence_common_bins.png)', '']
    if selection == 'cross_fit':
        out += ['评估已收集组的场景时排除该组，Fold3、4使用三组合并参数。四套数值均来自v。', '',
                table(['评估组', '拟合组', *KEYS],
                      [[f'Fold {f}', ', '.join(str(x) for x in range(3) if x != f),
                        *units(fit['cross_fit_proposals'][str(f)])] for f in range(3)] +
                      [['Fold 3、4', '0, 1, 2', *units(fit['theta_pool'])]]), '']
    else:
        out += ['所有20个场景使用同一套pooled参数提案。', '']
    out += [table(['Fold', '场景'], [[f, ', '.join(scenes)] for f, scenes in enumerate(folds)]), '',
            f'尚有{len(censored)}条空候选激活事件没有首候选完成记录。C_exp仅用{n_explore}条完成事件的条件均值，未完成事件没有填成0米。', '']
    if censored:
        out += [table(['场景', 'episode', 'k（从0开始）'],
                      [[r['scene_id'], r['episode_id'], r['k']] for r in censored]), '']
    overlap = evidence['dataset_overlap']
    out += [f'标定与评估共享{overlap["shared_scenes"]}个场景、{overlap["shared_source_episodes"]}个底层source episode；'
            f'{overlap["evaluation_ordered_first_two_targets_equal_calibration_sequence"]}条评估episode前两目标顺序与标定一致。因此这些数据不支持独立场景或目标泛化的结论。', '',
            f'原生GT离线复核明确覆盖本次全部12个run，候选检查{sum(r["candidate_verifications"] for r in evidence["native_gt"])}条，'
            f'恢复点检查{sum(r["recovery_verifications"] for r in evidence["native_gt"])}条。'
            f'当前标定路径归属缺口总数为{sum(len(r["path_association_gaps"]) for r in evidence["audit"]["completed_scenes"])}。具体口径及原始证据保留在审计JSON中。', '',
            '数值求解的重试次数及触发原因见calibration_per_fold.json的numerical_optimizer_retry；重试使用相同正斜率MLE目标并检查约束KKT残差。', '',
            ('本次四套参数与此前用户已批准数值逐字段一致，三组交叉拟合规则也一致，可沿用既有数值选择；仍须完成本次证据与源码绑定。'
             if comparison['existing_numeric_approval_can_be_retained'] else
             '本次参数或选择规则与此前已批准方案不同，需要审阅本报告的具体提案后确认。'), '',
            '本脚本只生成报告与未冻结提案。第4、5组未收集，后续仍按用户已选择的三组方案推进。下一步完成CP2审查与绑定，再运行新的固定20条Full冒烟。', '',
            '[拟合结果](../meta/calibration_per_fold.json)、[拟合输入绑定](../meta/calibration-fit-binding.json)、'
            '[数据与SHA清单](../meta/CP2a_data_summary.json)、[场景分组](../meta/folds.json)、'
            '[场景审计](../meta/completed-calibration-scene-audits.json)、[原生GT复核](../diagnostics/native-offline-gt-check.json)、'
            '[压缩日志清单](../meta/calibration-log-archive.json)。', '']
    (ROOT / 'reports/CP2a.md').write_text('\n'.join(out))
    return {'report': 'reports/CP2a.md', 'collection_attempt': 'a3', 'proposal_frozen': False,
            'n_verifications': n_checks, 'n_completed_explorations': n_explore,
            'bootstrap_success': 600, 'recommended_selection': selection,
            'approval_comparison': comparison}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    try:
        evidence = verify()
    except (ValueError, FileNotFoundError, KeyError) as error:
        if args.check:
            print(json.dumps({'ready': False, 'reason': str(error), 'freezes_parameters': False}))
            return
        raise
    result = {'ready': True, 'freezes_parameters': False} if args.check else render(evidence)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
