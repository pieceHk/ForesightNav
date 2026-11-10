# Full400已完成回合的过程复核

此快照核查已导出完整原生指标的21/400条回合，覆盖4ok3usBNeis：8条、5cdEh9F2hJL：13条。场景任务仍在运行，尚无完整Full400汇总；不把当前两个场景的前缀当作全场景成绩。

已核对63个阶段的原生步数、终止STOP、实际轨迹及候选路径归属，全部完整，路径长度最大绝对差1.18808499e-06米。所有已完成回合均按记录顺序纳入，包括失败回合。

候选事件GT检查、逐动作路径及候选身份合并复用原有固定分析组件；逐快照核对场景参数、交叉拟合分组及文件SHA。没有修改正在运行场景的完成状态，也没有生成正式400条结果。

| 场景/episode | 三阶段成功 | 步数 | 3阶段路径完整 |
| --- | --- | --- | --- |
| 4ok3usBNeis/0 | 1/1/1 | 387 | true |
| 4ok3usBNeis/1 | 0/1/1 | 333 | true |
| 4ok3usBNeis/2 | 0/0/1 | 1020 | true |
| 4ok3usBNeis/3 | 1/0/0 | 1208 | true |
| 4ok3usBNeis/4 | 0/1/1 | 553 | true |
| 4ok3usBNeis/5 | 1/1/0 | 755 | true |
| 4ok3usBNeis/6 | 0/0/0 | 1500 | true |
| 4ok3usBNeis/7 | 1/1/1 | 776 | true |
| 5cdEh9F2hJL/0 | 1/0/1 | 568 | true |
| 5cdEh9F2hJL/1 | 0/1/1 | 418 | true |
| 5cdEh9F2hJL/10 | 0/0/0 | 1026 | true |
| 5cdEh9F2hJL/11 | 1/1/1 | 559 | true |
| 5cdEh9F2hJL/12 | 1/1/0 | 730 | true |
| 5cdEh9F2hJL/2 | 1/1/1 | 331 | true |
| 5cdEh9F2hJL/3 | 1/0/1 | 406 | true |
| 5cdEh9F2hJL/4 | 1/0/1 | 656 | true |
| 5cdEh9F2hJL/5 | 0/1/0 | 43 | true |
| 5cdEh9F2hJL/6 | 1/0/1 | 631 | true |
| 5cdEh9F2hJL/7 | 0/0/1 | 411 | true |
| 5cdEh9F2hJL/8 | 1/1/1 | 734 | true |
| 5cdEh9F2hJL/9 | 0/1/1 | 586 | true |

[逐回合诊断与绑定](../diagnostics/completed-prefix/full/summary.json)。后续新增完整回合可增量复核；完成400条后仍需由原控制器做完整归一化及1200阶段核查。

CP4/CP5报告生成器已补充固定任务池的历史筛选和泛化范围说明；原有4项合成报告校验测试通过。这部分只修改报告措辞，未改变导航、固定参数或五个共用指标分析源码。[测试及源码绑定](../diagnostics/reporter-scope-validation.json)、[数据来源](dataset_scope.md)。
