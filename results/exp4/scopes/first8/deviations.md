
## 2026-11-13：用户调整评估范围与GPU分配

<!-- exp4-first8-user-authorized-scope-and-eight-gpu-20261113 -->

| 位置 | 原规定或先前安排 | 实际做法 | 原因与影响 |
| --- | --- | --- | --- |
| CP4/CP5评估条数 | 20场景，每变体400条3-MON | 用户要求仅运行原固定清单前8场景，每场景20条、每变体160条，四变体共640条。Full和Nearest复用对应8场景的全部原生记录，并重新审计；Myopic、Suffix使用完全相同160条身份。原Full400和范围外记录保留。 | 用户在已有部分结果之后明确缩减范围；没有根据本次成绩重排场景或筛除回合。原任务池本身曾由历史1-MON结果筛选，本范围不能作为独立盲测或未见任务泛化证据。 |
| 执行资源 | 曾使用GPU5、6、7，保留GPU0及Exp2资源 | 用户暂停Exp2并释放8卡后，当前允许物理GPU0–7，每卡一路Native，仍执行32GiB空闲/既有compute≤11GiB准入。已运行的三路Native保留，五路新场景加入；仅CPU调度器交接，不重启原生。 | 用户最新明确授权覆盖旧GPU0/3保留安排；保留Exp2数据，不清理其他用户任务。不同阶段的共享资源条件变化，耗时不能视为隔离资源因素后的因果比较。 |

CP5当前继续按Nearest完成后运行Belief-myopic，再运行Suffix-only的全局阶段顺序。评估期间算法、交叉拟合参数、种子、任务及source ID、步数预算、精度和共享导航机制均保持冻结。

证据：[固定范围](meta/scope.json)、[范围切换与保留记录](../../meta/first8-scope-cutoff.json)、[当前8卡授权](meta/gpu-allocation.json)、[CPU交接与原生保留](meta/eight-gpu-handoff.json)。
