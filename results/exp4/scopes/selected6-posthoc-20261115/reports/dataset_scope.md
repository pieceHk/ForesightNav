# Exp4固定任务池的来源与适用范围

> 历史阶段报告：正文保留原阶段范围与结果；文内证据链接固定到原始提交391d7e2，不指向当前120条派生数据。

实际数据集为 `HM3Dv1 3-MON balanced common-1200 paired-calibrated v1`，包含20个场景、400条3-MON、1200个不重复的原始ObjectNav任务。原始manifest明确记录 `not_independent_held_out=true`、`not_outcome_blind=true` 和 `selection_type=historical paired-outcome-calibrated`。

任务池经过历史1-MON基线与候选方法的配对结果筛选。这些历史标签不代表本次Full、Nearest、Belief-myopic或Suffix-only的成绩。当前实验继续采用已冻结清单及参数，不根据本次3-MON结果删选、重排回合或拟合参数。

因此，结果适用于这个固定筛选池内的受控消融；不能据此声称独立盲测、未见场景或未见底层任务的泛化性能。交叉拟合排除了对应标定组，但不能消除任务池既有的选择偏差。标定/评估共享来源的审计另见[既有记录](https://github.com/pieceHk/ForesightNav/blob/391d7e25b12171144a77cfbc41f220efe11294fb/results/exp4/meta/calibration-overlap-audit.json)。

| 历史源任务配对结局 | 数量 |
| --- | --- |
| both_failure | 663 |
| both_success | 327 |
| gain | 116 |
| regression | 94 |

这些计数仅用于说明数据来源，没有用于本次belief拟合或在线决策。

## 起点、高度与navmesh静态检查

一条MON只执行第一个源任务的起始姿态；后续源任务的起点只作来源记录，不会重置agent。后续阶段应从真实转换位置计算路径和SPL。目标按类别匹配；高度检查覆盖该类别的全部记录viewpoint。

| 检查 | 固定20条 | 完整400条 |
| --- | --- | --- |
| 三个目标都存在距初始高度≤0.5m的viewpoint | 13 / 20 | 239 / 400 |
| 三个目标从初始点到viewpoint均有有限navmesh测地距离 | 18 / 20 | 375 / 400 |
| 三个目标均有高度相近且测地距离有限的viewpoint | 12 / 20 | 237 / 400 |

高度相近只是代理条件，不能证明同楼层或策略可达；navmesh连通也不能证明当前2D导航策略能走通楼梯。以上距离均从回合初始位置计算，不能替代后续阶段真实起点的指标，也不足以解释某一回合的成败。未找到有限路径的项目保留原始记录，未剔除或修补。

[高度逐条检查](https://github.com/pieceHk/ForesightNav/blob/391d7e25b12171144a77cfbc41f220efe11294fb/results/exp4/diagnostics/CP3/dataset-height-availability.json)、[CPU原生navmesh逐条检查](https://github.com/pieceHk/ForesightNav/blob/391d7e25b12171144a77cfbc41f220efe11294fb/results/exp4/diagnostics/CP3/dataset-navmesh-availability.json)。后者复用固定版本的HabitatSim测地距离与PathFinder，不创建renderer、模型或策略，不把GT反馈给规划或拟合。

## 可核对的原始文件

- [原始manifest副本](https://github.com/pieceHk/ForesightNav/blob/391d7e25b12171144a77cfbc41f220efe11294fb/results/exp4/diagnostics/dataset/3mon-manifest.json)，SHA256 `c3245c4bc0322d618b12e2845f138842c5a2553c2b0f4bfbdfffba641c415165`。
- [原始source trace副本](https://github.com/pieceHk/ForesightNav/blob/391d7e25b12171144a77cfbc41f220efe11294fb/results/exp4/diagnostics/dataset/3mon-source-trace.json)，SHA256 `34931f42e88feb55d87f2c9c3caf4063e984f32efc4972fd82a40d210d32f04a`。
- [来源汇总](https://github.com/pieceHk/ForesightNav/blob/391d7e25b12171144a77cfbc41f220efe11294fb/results/exp4/meta/dataset-scientific-scope.json)和[固定400条身份](https://github.com/pieceHk/ForesightNav/blob/391d7e25b12171144a77cfbc41f220efe11294fb/results/exp4/meta/3mon-identities.json)。

本次新增审计与说明没有修改数据、原生执行源码或冻结参数。CP3报告将同时提交这些限制，进入主评测仍需按既定CP3检查点审阅。
