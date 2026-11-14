# CP3审阅入口：固定20条Full

> 历史阶段报告：正文保留原阶段范围与结果；文内证据链接固定到原始提交391d7e2，不指向当前120条派生数据。

[完整CP3指标、20条逐例结果与两张俯视图](https://github.com/pieceHk/ForesightNav/blob/391d7e25b12171144a77cfbc41f220efe11294fb/results/exp4/reports/CP3.md)。本次3/20条全部目标成功，21/60个阶段成功；当前尚无四变体在线对照，无法从这些结果判断机制增益。

| 指标 | 结果 |
| --- | --- |
| SR / SPL | 15.00% / 6.67% |
| PR / PPL | 35.00% / 10.88% |
| 实际下发偏离最近 | 23.29% |
| 规划耗时中位数 / P95 | 8.75 / 22.41 ms |

已核对全部60阶段路径、1999个快照及场景/episode/目标顺序；源码与冻结参数未变。20份原始日志归档共4.39 MiB，逐文件与压缩往返SHA均匹配。[原始日志及SHA清单](https://github.com/pieceHk/ForesightNav/blob/391d7e25b12171144a77cfbc41f220efe11294fb/results/exp4/meta/evaluation-smoke-log-archive.json)。

## 全量前需要看到的限制

规划耗时有长尾：1999次中8次超过100ms，1次超过1秒，最大9.258秒。所有记录均保留在统计和图中，CDF采用对数横轴；目前未确定长尾原因。[具体快照](https://github.com/pieceHk/ForesightNav/blob/391d7e25b12171144a77cfbc41f220efe11294fb/results/exp4/diagnostics/CP3/latency-tail-review.json)。

60个阶段中21个原生成功、29个预算耗尽后失败、10个在系统确认后停止但原生失败。后10个阶段的最后一次v=1检查，其候选坐标GT为0的有10个、GT为1的有0个、未记录的有0个。候选检查坐标与最终agent位置不同，不能据此直接归因为导航器或递归排序错误。[逐阶段复核](https://github.com/pieceHk/ForesightNav/blob/391d7e25b12171144a77cfbc41f220efe11294fb/results/exp4/diagnostics/CP3/confirmation-failure-review.json)。

固定任务池经过历史配对结果筛选，结论限于该池内受控消融；此外有部分目标存在高度或navmesh连通性疑点。所有回合保留，未据本次3-MON结果改参数或选样。[原始来源与可达性静态审计](https://github.com/pieceHk/ForesightNav/blob/391d7e25b12171144a77cfbc41f220efe11294fb/results/exp4/reports/dataset_scope.md)。

建议按已冻结协议完成四变体对照，以检验belief、suffix及交互项。若需要独立盲测或跨数据泛化结论，应另建独立任务池并统一重新标定、评估。按实验设计CP3【需确认】，当前未创建CP3接受记录，也未启动Full400。
