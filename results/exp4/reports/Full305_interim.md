# Full 305条中期结果

统计时点：2026年11月11日20:43（北京时间）。当时原生完成308条，下表只统计已完成GT、身份及路径审计的305条。正式CP4仍须等待完整400条。

| 已审计回合 | SR (%) | SPL (%) | PR (%) | PPL (%) |
| --- | --- | --- | --- | --- |
| 305 | 14.75 | 4.87 | 47.54 | 12.08 |

三目标全部成功45/305条；单目标成功435/915个。P(C1)=135/305=44.26%，P(C2|C1)=79/135=58.52%，P(C3|C1,C2)=45/79=56.96%。

SR、SPL和PR复用原PBSP聚合；PPL复用既有补充实现。逐项核对原生记录SHA与已审计缓存，冻结参数及五个共用分析组件保持一致。本次只读汇总没有重算GT、拟合参数、修改原生状态或生成正式400条汇总。

Nearest、Belief-myopic和Suffix-only在此时尚未运行，尚无四变体对比结论。任务池的历史筛选及适用范围见[数据来源](dataset_scope.md)。

[未舍入值、305条成员和原生记录SHA](../diagnostics/interim/full305-20261111T124331Z.json)；[对应的305条审计快照](https://github.com/pieceHk/ForesightNav/blob/39fd3250d0a77b5cafe2edeb4051f16876fd66f9/results/exp4/reports/Full400_completed_prefix.md)。
