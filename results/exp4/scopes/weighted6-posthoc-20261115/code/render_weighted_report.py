"""Create an explicitly weighted, outcome-informed Exp4 report and plots."""
from pathlib import Path
from datetime import datetime, timezone
import json,math,hashlib
from evaluate_sequence import OUT,load,digest
R=OUT/'scopes/weighted6-posthoc-20261115';P=OUT/'scopes/selected6-posthoc-20261115';B=OUT/'scopes/first8'
assert load(R/'meta/reweighting-audit.json')['status']=='passed'
N=('nearest','suffix_only','myopic','full');L={'nearest':'Nearest','suffix_only':'Suffix-only','myopic':'Belief-myopic','full':'Full BLP'}
S={v:load(R/f'summary/{v}.json') for v in N};U={v:load(P/f'summary/{v}.json') for v in N};O={v:load(B/f'summary/{v}.json') for v in N};C=load(R/'summary/effects.json');scope=load(R/'meta/scope.json');w=scope['scene_weights']
def f(x):return '未定义' if x is None else f'{x:.2f}'
def t(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join('---' for _ in headers)+' |']+['| '+' | '.join(map(str,row))+' |' for row in rows])
base='https://github.com/pieceHk/ForesightNav/blob/f1b60a75c4ad1694e2d85d5fec44baf3d370037c/results/exp4/scopes/selected6-posthoc-20261115/'
rows=[]
for v in N:
 s=S[v];p=s['process_weighted'];rows.append([L[v],'✓' if v in ('suffix_only','full') else '✗','✓' if v in ('myopic','full') else '✗',*[f(s['scores_pct_unrounded'][m]) for m in ('SR','SPL','PR','PPL')],*[f(x*100) if x is not None else '未定义' for x in s['conditional_counts_weighted']['rates']],f(p['Invalid_visits_per_weight_unit']),f(p['deviation_proposal_pct'])])
lines=['# Exp4 场景加权消融分析报告','',
 '生成日期：2026年11月15日。','',
 '**分析类型：事后场景加权的描述性分析。** 用户在查看原160条结果、255种场景子集以及选定120条的结果后，指定scene 5、6各计两次。本报告将其实现为权重2，其余所选场景权重1。每个变体仍只有120条唯一任务、6个唯一场景，episode权重和为160；没有新增仿真或独立样本。','',
 '## 完成了什么','',
 '本报告比较Nearest、Suffix-only、Belief-myopic和Full BLP。四组使用相同的120条3-MON任务、目标顺序、原生指标、冻结算法及交叉拟合参数，统一应用同一组场景权重。每条episode在日志中只出现一次，并附analysis_scene_weight字段；配对CSV使用scene_weight字段。','',
 t(['原场景编号','Scene','唯一episode数/变体','权重','episode权重贡献'],[[i,s,20,w[s],20*w[s]] for i,s in zip((1,3,5,6,7,8),w)]),'',
 '组合写法为1＋3＋5＋5＋6＋6＋7＋8。scene 5和6各占加权汇总的25%，其余各占12.5%。480条真实方法运行（4×120）没有变为640条新运行。','',
 '## 计算方法','',
 '对任一变体v和逐episode指标m，加权指标=100×Σ[w(scene_e)×m(v,e)]/Σw(scene_e)，其中分母为160。SR、SPL和PR直接读取已核查原生逐episode结果；PPL沿用原报告的补充实现（全部阶段PR乘成功前缀路径效率）。所有排序与效应使用未舍入值，表格保留两位小数并沿用原报告舍入口径。','',
 '条件成功率使用带相同场景权重的成功分子和前序成功分母，不能对已算好的场景条件率简单平均。无效访问和路径按场景加权累计后除以160；偏离率按加权决策/派发分母计算；规划中位数和P95按原始决策耗时的整数频数权重计算。重复权重不增加真实决策计时数量。','',
 'Nearest只使用当前测地距离；Suffix-only启用未来代价、令belief为1；Belief-myopic启用belief与当前目标失败分支，关闭未来目标价值；Full使用完整递归。参数继承三组交叉拟合方案，不在此子集重新拟合。E沿用候选高斯的加权文本分数，S为候选轨迹12档有效观测方位覆盖。','',
 '## 关键数字','',
 '以下导航指标与过程指标均为当前权重下的汇总；比率单位为%，无效访问为每单位episode权重的均值。','',
 t(['Variant','Future','Belief','Weighted SR','Weighted SPL','Weighted PR','Weighted PPL','Weighted P(C1)','Weighted P(C2∣C1)','Weighted P(C3∣C1,C2)','Weighted invalid visits','Proposal deviation (%)'],rows),'',
 t(['Variant','唯一任务中的全成功数','成功指示值的加权和','唯一任务数','episode权重和'],[[L[v],S[v]['unique_sequence_successes'],S[v]['weighted_sequence_success_sum'],120,160] for v in N]),'',
 '例如Full的38是成功指示值的加权和；真实唯一成功episode数仍为28。不得把38写成38条不同任务成功。','',
 '### 前瞻、belief主效应与交互项','',
 '前瞻主效应=[(Suffix-only−Nearest)+(Full−Myopic)]/2；belief主效应=[(Myopic−Nearest)+(Full−Suffix-only)]/2；交互项=(Full−Myopic)−(Suffix-only−Nearest)。单位为百分点。','',
 t(['Metric','Future effect','Belief effect','Interaction'],[[m,*[f(C['effects'][m][k]) for k in ('future_main_effect_pp','belief_main_effect_pp','interaction_pp')]] for m in ('SR','SPL')]),'',
 '这些效应描述当前权重对应的任务分布。belief单独的主效应仍为负值，交互项为正值，不能把该表解释为两个模块均有独立正收益。','',
 '### 条件成功率分子与分母','',
 t(['Variant','阶段','加权前序成功分母','加权本阶段成功分子','条件成功率 (%)'],[[L[v],k+1,S[v]['conditional_counts_weighted']['denominators'][k],S[v]['conditional_counts_weighted']['successes'][k],f(100*S[v]['conditional_counts_weighted']['rates'][k])] for v in N for k in range(3)]),'',
 '未舍入的三项条件成功率乘积与加权SR一致。分母为权重和，不是新增的唯一任务数量。原生系统在前一目标失败后仍可执行后续目标；条件成功率只统计前序全成功者。','',
 '### 六组配对：唯一任务计数','',
 t(['Pair','都成功','仅左成功','仅右成功','都失败','唯一配对数'],[[pair,*[p['unique_counts'].get(k,0) for k in ('both_success','left_only_success','right_only_success','both_failure')],p['n_unique_pairs']] for pair,p in C['paired_success'].items()]),'',
 '### 相同配对的加权成功指示值和','',
 t(['Pair','都成功权重和','仅左权重和','仅右权重和','都失败权重和','总权重'],[[pair,*[p['weighted_counts'].get(k,0) for k in ('both_success','left_only_success','right_only_success','both_failure')],p['pair_weight_sum']] for pair,p in C['paired_success'].items()]),'',
 '每份配对CSV只保存120行唯一配对，显式记录scene_weight。加权计数只能用于上表的描述性汇总，不能作为160个独立配对用于显著性检验。','',
 '### 加权过程指标','',
 t(['Variant','Invalid path/weight unit (m)','Identity path upper/weight unit (m)','After hysteresis deviation (%)','Applied deviation (%)','Weighted planning median (ms)','Weighted planning P95 (ms)','真实决策行数','决策权重和'],[[L[v],*[f(S[v]['process_weighted'][k]) for k in ('invalid_path_m_per_weight_unit','identity_path_upper_m_per_weight_unit','deviation_after_hysteresis_pct','deviation_applied_pct')],f(S[v]['process_weighted']['planning_ms_frequency_weighted']['median']),f(S[v]['process_weighted']['planning_ms_frequency_weighted']['p95']),S[v]['process_weighted']['actual_unique_decision_rows'],S[v]['process_weighted']['planning_ms_frequency_weighted']['n']] for v in N]),'',
 'Nearest的proposal偏离率为0；共享1米滞回和类别冷却可改变最终实际派发，所以滞回后与实际下发偏离率单列。无效路径使用原来的接受动作目标归因和原生GT口径；scene 5、6的过程汇总已用冻结分析函数重新在CPU上核查。不同变体运行时共享机器资源有变化，加权耗时不代表新计时实验或资源隔离下的速度收益。','',
 t(['Variant','Boundary','加权Root decisions','加权Decision incidence','加权Memo-state hits'],[[L[v],b,S[v]['process_weighted']['fallback_root_weighted'].get(b,0),S[v]['process_weighted']['fallback_incidence_weighted'].get(b,0),S[v]['process_weighted']['fallback_memo_states_weighted'].get(b,0)] for v in N for b in ('empty_candidates','frontier_coordinates_unavailable','no_reachable_frontier')]),'',
 '回退的根决策数、涉及该回退的决策数和递归状态命中数含义不同，不能互相替代。上表同样是权重和。','',
 '## 与预期是否一致；不一致的地方','',
 f"在所指定权重下，Full的加权SR、SPL、PR、PPL均为四组最高。Full相对Nearest的SR差值为{f(S['full']['scores_pct_unrounded']['SR']-S['nearest']['scores_pct_unrounded']['SR'])}个百分点，SPL差值为{f(S['full']['scores_pct_unrounded']['SPL']-S['nearest']['scores_pct_unrounded']['SPL'])}个百分点。Myopic的加权SR和SPL最低；Nearest不是最低。",'',
 '### 与未加权结果对照','',
 t(['Variant','原8场景160唯一任务 SR/SPL','选定6场景120唯一任务 SR/SPL','相同120任务加权 SR/SPL'],[[L[v],f(O[v]['SR'])+' / '+f(O[v]['SPL']),f(U[v]['SR'])+' / '+f(U[v]['SPL']),f(S[v]['scores_pct_unrounded']['SR'])+' / '+f(S[v]['scores_pct_unrounded']['SPL'])] for v in N]),'',
 '原8场景中Nearest的SR和SPL最高；选定6场景未加权时Full的SR最高、Nearest的SPL最高；增加scene 5、6的权重后，当前加权表中Full的SR与SPL均最高。这个变化来自评估场景及其权重的事后选择。原任务池还经过历史1-MON结果筛选，因此不据此声称独立盲测或未见任务泛化。','',
 '## 遇到的问题与采用的处理','',
 '用户指定scene 5、6重复计入，本报告将“重复”落实为显式权重，原始任务身份和指标值不变，每条记录只存一次。所有方法使用相同权重，未按单个episode的成功或失败删选。代码、参数、预算、原160/400条归档和此前120条未加权报告均保留。','',
 '三组交叉拟合参数、两阶段验证口径、候选级观测覆盖及共享滞回机制均继承既有版本。Myopic的两次基础设施失败尝试仍按父报告整体排除；当前表使用已核查的正式运行。','',
 '## 结论及适用范围','',
 '本次报告给出用户指定场景权重下的描述性消融结果。Full在该加权分布上的四项导航指标最高；该结果不能作为在完整评估分布上稳定优于Nearest的证据，也不提供独立统计显著性结论。唯一任务数始终为120，权重和160只用于计算加权均值。','',
 '## 交付物与可复现性','',
 '[权重与范围](../meta/scope.json)、[四组加权汇总CSV](../summary/all_weighted.csv)、[效应与双口径配对计数](../summary/effects.json)、[配对CSV索引](../summary/paired/manifest.json)、[重加权审计](../meta/reweighting-audit.json)。逐episode日志保存120条唯一记录及analysis_scene_weight字段。','',
 '[原120条报告]('+base+'reports/CP5.md)、[冻结参数]('+base+'meta/calibration.json)、[原8场景160条报告](https://github.com/pieceHk/ForesightNav/blob/391d7e25b12171144a77cfbc41f220efe11294fb/results/exp4/scopes/first8/reports/CP5.md)。','',
 '![场景加权SR与SPL](../figures/weighted_SR_SPL.png)','',
 '![场景加权主效应与交互项](../figures/weighted_effects.png)']
(R/'reports/Exp4_weighted_analysis.md').write_text('\n'.join(lines)+'\n')
(R/'README.md').write_text('# Exp4 事后场景加权报告\n\n每组120条唯一任务，scene 5、6权重为2，其余所选场景权重为1；episode权重和160，不增加真实样本。\n\n入口：[实验报告](reports/Exp4_weighted_analysis.md)。所有汇总、权重、120行配对CSV及原始指标引用一并提供。原始未加权120条和160条报告由固定提交链接引用，未被覆盖。\n')
(R/'deviations.md').write_text('# 偏差说明\n\n场景与权重均在观察结果后确定，属于结果相关的事后加权分析。当前权重用于描述性均值，不作为新的预注册评估范围。相同任务只记录一次，加权和160不等于160个独立样本；不使用重复权重扩大显著性检验样本量。原实验的算法、标定及协议偏差继承父报告。\n')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
colors=['#64748b','#0d9488','#d97706','#7c3aed'];labels=['Nearest','Suffix-only','Myopic','Full BLP']
fig,axes=plt.subplots(1,2,figsize=(10,4.4),layout='constrained')
for ax,m in zip(axes,('SR','SPL')):
 vals=[S[v]['scores_pct_unrounded'][m] for v in N];bars=ax.bar(labels,vals,color=colors,width=.65)
 ax.bar_label(bars,labels=[f(x) for x in vals],padding=3,fontsize=10);ax.set_ylabel('Weighted '+m+' (%)');ax.set_ylim(0,max(vals)*1.25);ax.spines[['top','right']].set_visible(False)
fig.suptitle('Exp4 | post-hoc scene weighting | 120 unique episodes, total weight 160',fontsize=12);fig.savefig(R/'figures/weighted_SR_SPL.png',dpi=180);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(10,4.4),layout='constrained')
for ax,m in zip(axes,('SR','SPL')):
 vals=[C['effects'][m][k] for k in ('future_main_effect_pp','belief_main_effect_pp','interaction_pp')];bars=ax.bar(['Future','Belief','Interaction'],vals,color=colors[1:])
 ax.bar_label(bars,labels=[f'{x:+.2f}' for x in vals],padding=4,fontsize=10);ax.axhline(0,color='#475569',linewidth=1);ax.set_ylabel('Weighted '+m+' contrast (pp)');ax.set_ylim(min(0,min(vals))-max(.75,max(abs(x) for x in vals)*.15),max(0,max(vals))*1.3+.15);ax.spines[['top','right']].set_visible(False)
fig.suptitle('Exp4 | post-hoc scene weighting | descriptive factorial contrasts',fontsize=12);fig.savefig(R/'figures/weighted_effects.png',dpi=180);plt.close(fig)
print(json.dumps({'report':str(R/'reports/Exp4_weighted_analysis.md'),'report_sha256':digest(R/'reports/Exp4_weighted_analysis.md'),'figures':{p.name:digest(p) for p in (R/'figures').glob('*.png')}}))
