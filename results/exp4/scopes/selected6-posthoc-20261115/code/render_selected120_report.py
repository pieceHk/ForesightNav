"""Document the user-selected, outcome-informed six-scene Exp4 subset."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil,csv,statistics
from evaluate_sequence import OUT,load,digest
ROOT=OUT/'scopes/selected6-posthoc-20261115';BASE=OUT/'scopes/first8'
NAMES=('nearest','suffix_only','myopic','full')
LABELS={'nearest':'Nearest','suffix_only':'Suffix-only','myopic':'Belief-myopic','full':'Full BLP'}
scope=load(ROOT/'meta/scope.json');audit=load(ROOT/'meta/reaggregation-audit.json');assert audit['status']=='passed'
S={v:load(ROOT/f'summary/{v}.json') for v in NAMES};comparison=load(ROOT/'summary/effects.json')
F={v:load(BASE/f'summary/{v}.json') for v in NAMES}
def table(headers,rows):
 return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join('---' for _ in headers)+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def num(x):return '未定义' if x is None else f'{x:.2f}'
def write(p,text):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text+'\n')
def jwrite(p,obj):write(p,json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False))
for path in (OUT/'figures').glob('calibration*.png'):shutil.copyfile(path,ROOT/'figures'/path.name)
(ROOT/'figures/CP3').mkdir(exist_ok=True)
for path in (OUT/'figures/CP3').glob('*.png'):shutil.copyfile(path,ROOT/'figures/CP3'/path.name)
for file in ('dataset_scope.md','eq5_wording.md','action_path_accounting.md'):
 if (OUT/'reports'/file).exists():shutil.copyfile(OUT/'reports'/file,ROOT/'reports'/file)
# The historical checkpoint reports keep their original scope, filenames and content.
originals=[]
for cp in ('CP1','CP2a','CP2b','CP3','CP4'):
 p=ROOT/f'reports/{cp}.md'
 if p.exists():originals.append({'checkpoint':cp,'path':str(p.relative_to(ROOT)),'sha256':digest(p),'status':'inherited original evidence; not a new120 run'})
rows=[]
for v in NAMES:
 s=S[v];p=s['process'];rows.append([LABELS[v],'✓' if v in ('suffix_only','full') else '✗','✓' if v in ('myopic','full') else '✗',*[num(s[m]) for m in ('SR','SPL','PR','PPL','P_C1','P_C2_given_C1','P_C3_given_C1C2')],num(p['Invalid_visits_per_ep']),num(p['deviation_from_nearest_proposal_pct'])])
lines=['# Exp4 CP5：用户指定6场景、每组120条的事后子集报告','',
 '**分析属性：事后按结果选定的展示子集（post hoc）。** 本报告对应用户在查看原160条结果及255种场景组合分析后选定的原编号1、3、5、6、7、8。场景选择考虑了SR排序，不能作为预先固定范围的独立验证。本次重汇总不新增原生实验，四种方法各保留所选场景的全部120条相同任务。','',
 '## 完成了什么','',
 '采用原始冻结代码、原生评估函数和已冻结的三组交叉拟合参数，从四组160条结果中提取6个完整场景；每个场景20条3-MON任务，合计480条方法运行、每组360段阶段路径。未按单个episode的成功或失败删选。候选真值、接受动作路径与原生阶段路径重新在CPU上核对。','',
 table(['原编号','Scene','每变体episode数'],[[i,s,20] for i,s in zip((1,3,5,6,7,8),scope['ordered_scenes'])]),'',
 '排除的两个场景为原编号2（5cdEh9F2hJL）和4（DYehNKdT76V）。完整160条结果仍保留为本报告的参照，原Full400结果也保留。','',
 '## 关键数字','',
 '下表比率单位均为百分数；Invalid visits/ep为候选数。SR、SPL、PR沿用原生评估器，PPL为既有补充实现：全部阶段PR乘成功前缀路径效率。条件成功率仅以前序目标全部成功的episode为分母。','',
 table(['Variant','Future','Belief','SR','SPL','PR','PPL','P(C1)','P(C2∣C1)','P(C3∣C1,C2)','Invalid visits/ep','Proposal deviation (%)'],rows),'',
 table(['Variant','三目标全成功数 / episode数'],[[LABELS[v],f"{S[v]['native_aggregate']['counts']['successful_episodes']} / 120"] for v in NAMES]),'',
 '### 两个主效应与交互项','',
 '前瞻主效应=[(Suffix-only−Nearest)+(Full−Myopic)]/2；belief主效应=[(Myopic−Nearest)+(Full−Suffix-only)]/2；交互项=(Full−Myopic)−(Suffix-only−Nearest)。由未舍入值计算，单位为百分点。','',
 table(['Metric','Future effect (pp)','Belief effect (pp)','Interaction (pp)'],[[m,*[num(comparison['effects'][m][k]) for k in ('future_main_effect_pp','belief_main_effect_pp','interaction_pp')]] for m in ('SR','SPL')]),'',
 '### 条件成功率的实际分母','',
 table(['Variant','阶段','前序全成功episode数','本阶段成功数','条件成功率 (%)'],[[LABELS[v],i+1,S[v]['conditional_counts']['conditioned_episodes'][i],S[v]['conditional_counts']['successes'][i],num(100*S[v]['conditional_counts']['rates'][i]) if S[v]['conditional_counts']['rates'][i] is not None else '未定义'] for v in NAMES for i in range(3)]),'',
 '三项条件成功率的乘积与各组SR一致。原生运行在某个目标失败后仍可执行后续目标，条件分母不等同于“执行过该阶段”的数量。','',
 '### 六组逐episode配对','',
 table(['Pair','Both success','Left only','Right only','Both failure','Paired N'],[[pair,*[p[k] for k in ('both_success','left_only_success','right_only_success','both_failure','n_paired')]] for pair,p in comparison['paired_episode_success'].items()]),'',
 '六个CSV各120条，保留scene、episode及两侧SR/SPL/PR/PPL；source目标身份由配套episode日志和scope记录核对。配对成功计数与四组汇总、原始逐回合结果一致。文件索引：[paired/manifest.json](../summary/paired/manifest.json)。','',
 '### 过程指标','',
 table(['Variant','Invalid path/ep (m)','Identity path upper/ep (m)','After hysteresis deviation (%)','Applied deviation (%)','Planning median (ms)','Planning P95 (ms)','Planning N'],[[LABELS[v],*[num(S[v]['process'][k]) for k in ('invalid_visit_path_m_per_ep','invalid_identity_pursuit_path_m_per_ep','deviation_from_nearest_after_hysteresis_pct','deviation_from_nearest_realized_pct')],num(S[v]['process']['planning_ms']['median']),num(S[v]['process']['planning_ms']['p95']),S[v]['process']['planning_ms']['n']] for v in NAMES]),'',
 'Nearest的无约束proposal偏离率为0；共享的1米切换滞回和类别冷却可改变实际下发，因此三个口径分别列出。无效路径仅统计追逐“曾检查为真值无效的候选身份”且当时目标坐标仍真值无效时的接受动作位移；同一身份全部追逐路径作为上界。不同变体运行期间的共享资源状态有变化，规划耗时不作为排除了机器资源影响的因果比较。','',
 table(['Variant','Boundary','Root decisions','Decision incidence','Memo-state hits'],[[LABELS[v],b,S[v]['process']['fallback_root_decisions'].get(b,0),S[v]['process']['fallback_decision_incidence'].get(b,0),S[v]['process']['fallback_memo_state_counts'].get(b,0)] for v in NAMES for b in ('empty_candidates','frontier_coordinates_unavailable','no_reachable_frontier')]),'',
 'Root、incidence与递归状态命中是不同计数，不能相互替代。','',
 '## 与预期是否一致；不一致的地方','',
 f"本子集中Full的SR为{num(S['full']['SR'])}%，Nearest为{num(S['nearest']['SR'])}%，差值为{num(S['full']['rates_pct_unrounded']['SR']-S['nearest']['rates_pct_unrounded']['SR'])}个百分点，对应多成功1条episode。Myopic的SR最低。",'',
 f"SPL最高的是Nearest（{num(S['nearest']['SPL'])}%），Full为{num(S['full']['SPL'])}%；在SR和SPL两项中，仅SR的排序符合Full最高的预期。Full的PR和PPL也最高，但不能据此称Full在全部指标上最好。belief的SR主效应仍为负值。上述差值为描述性结果，未据此声称统计显著。",'',
 '### 与完整160条结果对照','',
 table(['Variant','160条SR (%)','120条SR (%)','160条SPL (%)','120条SPL (%)'],[[LABELS[v],num(F[v]['SR']),num(S[v]['SR']),num(F[v]['SPL']),num(S[v]['SPL'])] for v in NAMES]),'',
 '完整160条中Nearest的SR与SPL最高；当前120条是看过结果后选择的场景组合，不能用子集排序覆盖这一事实。任务池本身还经过历史1-MON结果筛选，本报告不主张独立盲测或未见任务泛化。原报告：[CP5_original160.md](CP5_original160.md)。','',
 '## 遇到的问题与采用的处理','',
 '范围从原设计20场景400条，先缩为8场景160条，再由用户在观察结果后选为当前6场景120条。每次范围变化均在scope与deviations中记录。本次只做派生汇总，不改原生结果、算法、预算或标定参数。','',
 'Myopic的DY和6s各有一次显存故障导致的整场景废弃尝试；原160条正式结果仅使用相同固定身份和seed的完整替代运行。当前子集保留6s、排除DY；两个失败尝试及29条导出仍在父级归档中整体保留并排除。不能将两次基础设施故障记为120条中的导航失败或新增实验。','',
 'E沿用候选高斯的加权文本分数，非邻近支撑集平均值；S采用空间候选轨迹的12档有效观测方位覆盖；参数以系统两阶段验证标签拟合，真值仅作诊断与路径审计。已确认的三组交叉拟合参数沿用，不在当前120条上重拟合。','',
 '## 需要用户确认的问题','',
 '无新增算法或参数确认项。用户已指定该120条组合；该组合在本报告中以事后选择属性呈现。','',
 '## 下一步','',
 '本次派生数据、过程指标和配对表已完成汇总。CP1至CP4是原先实施阶段的历史证据，其原始范围不重写为120条；当前范围见meta/scope.json和本CP5报告。若后续需验证当前子集上观察到的排序，应在预先规定范围的新任务上验证。','',
 '![SR and SPL：事后选定6场景](../figures/SR_SPL.png)','',
 '![2×2 effects：事后选定6场景](../figures/effects.png)','',
 '[四组汇总](../summary/all.csv) · [效应与配对计数](../summary/effects.json) · [重汇总审计](../meta/reaggregation-audit.json) · [冻结参数](../meta/calibration.json) · [原始运行归档引用](../meta/raw-archives-references.json)']
write(ROOT/'reports/CP5.md','\n'.join(lines))
write(ROOT/'deviations.md','\n'.join(['# Exp4 用户选定120条的偏差记录','',table(['位置','原规定 / 父范围','实际做法','原因及解释'],[
 ['评估范围','原设计20场景400条；已完成父范围8场景160条','原编号1、3、5、6、7、8，各20条，总120条/变体','用户在看到结果及255组合搜索后选定；明确为事后结果相关选择'],
 ['最终文档','完整评估范围CP5','当前CP5为选定120条，附原160条报告和对照表','不把派生子集写成预先固定或独立验证'],
 ['统计解释','检验机制收益','报告实际全部指标，不假定Full均最优','Full的SR最高，Nearest的SPL仍最高；不声明显著性'],
 ['原始记录','保存运行配置及日志','引用父提交24个正式场景归档及全部原始文件','本次不新增GPU运行；原160/400及故障尝试原样保留'],
 ['E / S','按论文约定','E为现有候选高斯加权文本分数；S为候选轨迹观测覆盖','继承已确认的CP1口径'],
 ['Deviation','Nearest恒为0','proposal为0，滞回后及实际下发另报','继承用户已确认的三口径报告方式'],
 ['条件成功率','前序成功条件','仅以前序全部成功episode计数','原生运行失败后可继续后续阶段，执行数量不作条件分母'],
 ['计时','规划耗时分布','保留实测时间及资源约束说明','不同运行阶段共享资源不同，不能解释为资源隔离的因果效应']])]))
write(ROOT/'README.md','\n'.join(['# Exp4 用户选定120条交付包','',
 '**事后选择声明：** 用户在观察完整160条结果与255种场景组合后选择当前6个场景。该包是用户指定的最终展示子集，不能称为预注册评测或独立确认性结果。','',
 '入口：[reports/CP5.md](reports/CP5.md)。当前四组各120条，所有指标均从对应原始运行重汇总；没有新增仿真或调参。','',
 '文档采用Exp4设计约定的Markdown格式。summary/all.csv、四组summary JSON、effects JSON、六组逐回合配对CSV、四组episode/decision JSONL、冻结标定及原始环境信息同时提供。','',
 '历史CP1、CP2a、CP2b、CP3、CP4保留原记录，分别对应实现、标定、冒烟和原Full400阶段。CP5_original160.md保留8场景160条的父报告。父归档内部的相对链接以原仓库results/exp4为根，原始完整证据见固定父提交。','',
 '源提交：https://github.com/pieceHk/ForesightNav/tree/391d7e25b12171144a77cfbc41f220efe11294fb/results/exp4。当前24个正式场景归档见meta/raw-archives-references.json；这些大档案仍在原仓库，不在该派生文档包内重复复制。','',
 '新选定范围见meta/scope.json，CPU重汇总证据见meta/reaggregation-audit.json。所有原160条和原Full400均保留。']))
# Standard scientific plotting; labels explicitly disclose post-hoc selection.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
colors=['#64748b','#0d9488','#d97706','#7c3aed'];labels=['Nearest','Suffix-only','Myopic','Full BLP']
fig,axes=plt.subplots(1,2,figsize=(10,4.2),layout='constrained')
for ax,metric in zip(axes,('SR','SPL')):
 vals=[S[v]['rates_pct_unrounded'][metric] for v in NAMES]
 bars=ax.bar(labels,vals,color=colors,width=.65);ax.bar_label(bars,labels=[num(v) for v in vals],padding=3,fontsize=10)
 ax.set_ylabel(metric+' (%)');ax.set_ylim(0,max(vals)*1.25);ax.spines[['top','right']].set_visible(False)
fig.suptitle('Exp4 | post-hoc selected 6 scenes | 120 paired episodes per variant',fontsize=12)
fig.savefig(ROOT/'figures/SR_SPL.png',dpi=180);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(10,4.2),layout='constrained')
for ax,metric in zip(axes,('SR','SPL')):
 vals=[comparison['effects'][metric][k] for k in ('future_main_effect_pp','belief_main_effect_pp','interaction_pp')]
 bars=ax.bar(['Future','Belief','Interaction'],vals,color=colors[1:]);ax.bar_label(bars,labels=[f'{x:+.2f}' for x in vals],padding=4,fontsize=10)
 ax.axhline(0,color='#475569',linewidth=1);ax.set_ylabel(metric+' contrast (percentage points)');ax.set_ylim(min(0,min(vals))*1.35-.15,max(0,max(vals))*1.30+.15);ax.spines[['top','right']].set_visible(False)
fig.suptitle('Exp4 | post-hoc selected subset | descriptive factorial contrasts',fontsize=12)
fig.savefig(ROOT/'figures/effects.png',dpi=180);plt.close(fig)
jwrite(ROOT/'meta/document-manifest.json',{'analysis_status':scope['analysis_status'],'generated_at_utc':datetime.now(timezone.utc).isoformat(),'inherited_checkpoints':originals,'report_sha256':digest(ROOT/'reports/CP5.md'),'figures':{p.name:digest(p) for p in (ROOT/'figures').glob('*.png')},'actual_visual_QA_pending':True})
print(json.dumps({'report':str(ROOT/'reports/CP5.md'),'summary':{v:{m:S[v][m] for m in ('SR','SPL','PR','PPL')} for v in NAMES},'effects':comparison['effects']}),flush=True)
