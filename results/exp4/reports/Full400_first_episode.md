# Full400首条完整回合复核

固定核查执行清单首场景的首条回合（4ok3usBNeis/0），不按成败选择。Full400控制器和两个场景任务仍在运行，本文件不替代400条正式汇总。

该回合原生387步，三阶段成功向量[1, 1, 1]，原生SPL为45.83%。身份、类别、source episode、原生步数和三个终止STOP已核对。

复用已冻结的GT及路径分析组件，三个阶段逐动作路径归属完整，与原生阶段长度最大绝对误差为5.22112995e-07米。未修改运行中场景的完成状态，也未绕过完整400条的正式归一化门禁。

复核时共1/400条已导出完整原生指标，累计1147动作、120规划快照。源码和参数保持冻结，五类日志错误0。后续记录继续由同一原生控制器写入。

[单回合诊断及分析源码SHA](../diagnostics/Full400/first-episode-audit.json)、[首批40条实际launch身份校验](../diagnostics/Full400/first-launch-validation.json)、[此刻进度快照](../diagnostics/Full400/first-episode-progress.json)、[Full400启动记录](Full400_start.md)。
