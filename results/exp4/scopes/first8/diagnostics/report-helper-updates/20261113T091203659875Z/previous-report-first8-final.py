"""Generate the final report only from four complete, audited first8 variants."""
from datetime import datetime,timezone
from pathlib import Path
import argparse,csv,itertools,json,math
import evaluate_sequence as base
import first8_evaluate as scoped
from report_variants import effects_and_pairs
from validate_first8_pairs import validate_pair_rows

ROOT=base.OUT;SCOPE=ROOT/'scopes/first8'
NAMES=('nearest','suffix_only','myopic','full')
LABELS={'nearest':'Nearest','suffix_only':'Suffix-only','myopic':'Belief-myopic','full':'Full BLP'}

def data():
    scope=base.load(SCOPE/'meta/scope.json')
    assert base.source_fingerprint()==scope['source_fingerprint']
    assert base.digest(ROOT/'meta/calibration.json')==scope['parameter_sha256']
    assert base.load(SCOPE/'meta/progress.json')['status']=='completed_pending_delivery'
    variants={}
    for stage in NAMES:
        scoped.completed(stage,scope)
        rows=[json.loads(line) for line in (SCOPE/f'logs/{stage}_episodes.jsonl').read_text().splitlines()]
        summary=base.load(SCOPE/f'summary/{stage}.json')
        assert len(rows)==summary['n_episodes']==160 and summary['n_error']==0
        assert summary['process']['native_stage_path_check_count']==480
        assert summary['process']['invalid_path_complete']
        variants[stage]=(rows,[],summary)
    comparison=effects_and_pairs(variants)
    assert comparison==base.load(SCOPE/'summary/effects.json')
    manifest=base.load(SCOPE/'summary/paired/manifest.json')
    assert manifest['n_episodes_each_pair']==160 and len(manifest['pairs'])==6
    expected_pairs={f'summary/paired/{left}_vs_{right}.csv':(left,right) for left,right in itertools.combinations(NAMES,2)}
    assert {row['path'] for row in manifest['pairs']}==set(expected_pairs)
    for pair in manifest['pairs']:
        path=SCOPE/pair['path'];assert base.digest(path)==pair['sha256']
        with path.open() as stream:rows=list(csv.DictReader(stream))
        assert pair['n_paired']==160
        left,right=expected_pairs[pair['path']]
        actual,means=validate_pair_rows(rows,variants[left][0],variants[right][0],left,right)
        expected=comparison['paired_episode_success'][left+'_vs_'+right]
        assert actual==expected
        for side,name in (('left',left),('right',right)):
            for metric in ('SR','SPL','PR','PPL'):
                observed=means[side][metric]
                assert math.isclose(observed,variants[name][2]['rates_pct_unrounded'][metric],abs_tol=1e-9)
    archive=base.load(SCOPE/'meta/raw-archives/manifest.json')
    assert archive['complete'] and archive['archived_runs']==32
    assert archive['by_stage']=={name:8 for name in NAMES}
    for row in archive['runs']:
        target=ROOT/row['archive']
        assert target.stat().st_size==row['archive_bytes'] and base.digest(target)==row['archive_sha256']
        assert row['n_episodes']==20 and row['all_round_trip_sha256_match']
        assert row['parameter_sha256']==scope['parameter_sha256']
        assert row['source_fingerprint']==scope['source_fingerprint']
    environment=max((SCOPE/'diagnostics').glob('environment-index-*.json'),key=lambda p:p.name)
    env=base.load(environment)
    assert env['recorded_pairs_by_stage']=={name:8 for name in NAMES}
    for row in env['role_pairs']:
        for file in row['files']:assert base.digest(ROOT/file['file'])==file['sha256']
    return scope,variants,comparison,archive,environment

def number(x):return '' if x is None else f'{x:.2f}'

def table(headers,rows):
    return ['| '+' | '.join(headers)+' |','| '+' | '.join('---' for _ in headers)+' |',
            *['| '+' | '.join(map(str,row))+' |' for row in rows]]

def figures(variants,comparison):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=SCOPE/'figures';folder.mkdir(exist_ok=True)
    colors=['#64748b','#0d9488','#d97706','#7c3aed']
    labels=[LABELS[n].replace('Belief-myopic','Myopic') for n in NAMES]
    fig,axes=plt.subplots(1,2,figsize=(9,4),layout='constrained')
    for axis,metric in zip(axes,('SR','SPL')):
        values=[variants[n][2]['rates_pct_unrounded'][metric] for n in NAMES]
        bars=axis.bar(labels,values,color=colors,width=.65)
        axis.bar_label(bars,labels=[f'{v:.2f}' for v in values],padding=3,fontsize=9)
        axis.set_ylabel(metric+' (%)');axis.set_ylim(0,max(1,max(values))*1.25)
        axis.spines[['top','right']].set_visible(False);axis.grid(axis='y',alpha=.18);axis.set_axisbelow(True)
    fig.suptitle('Exp4 | first 8 fixed scenes | 160 paired episodes per variant',fontsize=12)
    fig.savefig(folder/'SR_SPL.png',dpi=200);plt.close(fig)
    keys=['future_main_effect_pp','belief_main_effect_pp','interaction_pp']
    fig,axes=plt.subplots(1,2,figsize=(9,4),layout='constrained')
    for axis,metric in zip(axes,('SR','SPL')):
        values=[comparison['effects'][metric][key] for key in keys]
        bars=axis.bar(['Future','Belief','Interaction'],values,color=['#0d9488','#d97706','#7c3aed'],width=.6)
        axis.bar_label(bars,labels=[f'{v:+.2f}' for v in values],padding=4,fontsize=9)
        limit=max(1,max(abs(v) for v in values))*1.4
        axis.set_ylim(-limit,limit);axis.axhline(0,color='#334155',linewidth=.8)
        axis.set_ylabel(metric+' contrast (percentage points)')
        axis.spines[['top','right']].set_visible(False);axis.grid(axis='y',alpha=.18);axis.set_axisbelow(True)
    fig.suptitle('Exp4 | 2 x 2 contrasts within the selected first-8 pool',fontsize=12)
    fig.savefig(folder/'effects.png',dpi=200);plt.close(fig)
    return ['figures/SR_SPL.png','figures/effects.png']

def prepare():
    scope,variants,comparison,archive,environment=data()
    plots=figures(variants,comparison)
    rows=[]
    for stage in NAMES:
        s=variants[stage][2];p=s['process']
        rows.append([LABELS[stage],'✓' if stage in ('full','suffix_only') else '✗',
            '✓' if stage in ('full','myopic') else '✗',*[number(s[k]) for k in
            ('SR','SPL','PR','PPL','P_C1','P_C2_given_C1','P_C3_given_C1C2')],
            number(p['Invalid_visits_per_ep']),number(p['deviation_from_nearest_proposal_pct'])])
    lines=['# Exp4 CP5：固定前8场景的2×2消融','',
        '四个变体各完成同样160条3-MON任务，覆盖原固定清单前8个场景、每场景20条，共640条。所有场景都保留成功和失败回合；每组480段实际动作路径已经核对。', '',
        '用户在已获得部分结果后将范围从20场景缩至固定前8场景。场景沿用原清单顺序，没有根据本次成绩重排。原任务池本身经过历史1-MON结果筛选，因此本报告描述该固定池内的消融，不作为独立盲测或未见任务泛化证据。原Full400及范围外原始记录保留。', '',
        '算法、三组交叉拟合参数和共享导航机制沿用已确认版本。实验后半段按用户授权扩至8张GPU，保留此前3路原生进程，CPU调度交接记录了真实进程退出状态。不同阶段的共享资源条件有变化，规划耗时不能作为已隔离资源影响的独立因果比较。','']
    lines+=table(['Variant','Future','Belief','SR (%)','SPL (%)','PR (%)','PPL (%)','P(C1) (%)',
        'P(C2∣C1) (%)','P(C3∣C1,C2) (%)','Invalid visits/ep','Proposal deviation (%)'],rows)
    lines+=['','SR、SPL、PR来自原生评估器；PPL为补充指标，按全部阶段PR与成功前缀路径效率计算。比率展示保留两位，主效应及交互项使用未舍入值。条件成功率只在前序全部成功的回合中计算。','']
    lines+=table(['Metric','Future effect (pp)','Belief effect (pp)','Interaction (pp)'],
        [[metric,*[number(comparison['effects'][metric][key]) for key in
          ('future_main_effect_pp','belief_main_effect_pp','interaction_pp')]] for metric in ('SR','SPL')])
    lines+=['','前瞻主效应=[(Suffix−Nearest)+(Full−Myopic)]/2；belief主效应=[(Myopic−Nearest)+(Full−Suffix)]/2；交互项=(Full−Myopic)−(Suffix−Nearest)。','']
    lines+=table(['Pair','Both success','Left only','Right only','Both failure','Paired N'],
        [[name,*[row[key] for key in ('both_success','left_only_success','right_only_success','both_failure','n_paired')]]
         for name,row in comparison['paired_episode_success'].items()])
    lines+=['','六组逐回合配对CSV各160条，成功不一致计数与上表、原始分数均已回读核对。','']
    lines+=table(['Variant','Invalid path/ep (m)','Identity path upper/ep (m)','After hysteresis deviation (%)',
        'Applied deviation (%)','Planning median (ms)','Planning P95 (ms)'],
        [[LABELS[n],*[number(variants[n][2]['process'][key]) for key in ('invalid_visit_path_m_per_ep',
          'invalid_identity_pursuit_path_m_per_ep','deviation_from_nearest_after_hysteresis_pct','deviation_from_nearest_realized_pct')],
          number(variants[n][2]['process']['planning_ms']['median']),number(variants[n][2]['process']['planning_ms']['p95'])] for n in NAMES])
    lines+=['','Nearest的无约束proposal偏离率应为0。四组共享的1米切换滞回、类别冷却等机制仍可改变最终实际下发，三个偏离口径分别保留。无效访问路径按接受动作与检查时坐标GT关联；身份追逐路径单列为上界。','']
    lines+=table(['Variant','Boundary','Root decisions','Decision incidence','Memo-state hits'],
        [[LABELS[n],key,p['fallback_root_decisions'].get(key,0),p['fallback_decision_incidence'].get(key,0),
          p['fallback_memo_state_counts'].get(key,0)] for n in NAMES for p in [variants[n][2]['process']]
         for key in ('empty_candidates','frontier_coordinates_unavailable','no_reachable_frontier')])
    lines+=['','![SR and SPL](../figures/SR_SPL.png)','![Effects](../figures/effects.png)','',
        '32个场景级原始配置/日志归档包含launch、真实完成状态、源码快照、三目标身份、原生指标、轨迹、BLP/SCSP记录和mapping/navigation环境。归档创建时逐文件回读SHA核对，最终索引与档案外层SHA再次核对。图片和模型/地图二进制不在这份原始日志归档内；原始目录继续保留。','',
        '[固定身份与范围](../meta/scope.json)、[冻结参数](../../../meta/calibration.json)、[数据适用范围](../../../reports/dataset_scope.md)、[六组配对CSV](../summary/paired/manifest.json)、[原始归档清单](../meta/raw-archives/manifest.json)。',
        '[CP1](../../../reports/CP1.md)、[CP2a](../../../reports/CP2a.md)、[CP3](../../../reports/CP3.md)、[原Full400 CP4](../../../reports/CP4.md)、[范围内四组汇总](../summary/all.csv)。','']
    path=SCOPE/'reports/CP5.md';path.write_text('\n'.join(lines))
    # Keep the named CP5 entry at the original results root, with working links.
    root_text=path.read_text().replace('../../../','@EXP4_ROOT@/')
    root_text=root_text.replace('../','../scopes/first8/')
    root_text=root_text.replace('@EXP4_ROOT@/','../')
    root_report=ROOT/'reports/CP5.md';root_report.write_text(root_text)
    proof={'scope':'first8','prepared_at_utc':datetime.now(timezone.utc).isoformat(),
        'n_episodes_by_variant':{name:160 for name in NAMES},'raw_archives':32,'paired_csvs':6,
        'report_sha256':base.digest(path),'figure_sha256':{name:base.digest(SCOPE/name) for name in plots},
        'root_report':'reports/CP5.md','root_report_sha256':base.digest(root_report),
        'visual_review_required':True,'actual_visual_review_performed':False,
        'environment_index':str(environment.relative_to(ROOT)),'environment_index_sha256':base.digest(environment),
        'source_fingerprint':scope['source_fingerprint'],'parameter_sha256':scope['parameter_sha256'],
        'reporter_source_sha256':base.digest(Path(__file__)),
        'native_states_modified':False,'launches_workers':False}
    save=SCOPE/'meta/CP5-report-prepared.json';save.write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps({'report':str(path.relative_to(ROOT)),**proof}))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');args=parser.parse_args()
    if args.check:
        try:data()
        except (AssertionError,base.GateError,FileNotFoundError) as error:
            print(json.dumps({'ready':False,'reason':f'{type(error).__name__}: {error}','native_states_modified':False}))
        else:print(json.dumps({'ready':True}))
    else:prepare()
