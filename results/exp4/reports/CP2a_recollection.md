# 修正深度单位后的三组标定重收

2026-11-09已启动calibration a3，固定Fold0/1/2共12场景、360条2-MON，仅使用GPU2、3。代码提交为eaa29e5f8f46782554d31226a8d619ec2534c34a，控制器PID31036；本报告对应的只读核查时间为2026-11-09T04:58:22.356322+00:00，已完成0/360，控制器、boot及160项固定输入均匹配。

新增S观察器现按传感器配置把归一化深度转换为米。51项测试和219次旧模式匹配输入回放通过，原生小试48动作及两个阶段已完成且排除拟合，路径核对通过。原导航收到的深度输入保持原样。

旧calibration a2的S/敏感性诊断与smoke a1均已标为被替代版本，原始数据留在旧工作区，继承元数据和压缩日志另归档于本分支archive目录。新标定在/nvme/khy/ForesightNav_exp4_depthfix/results/exp4单独收集，不与旧轮次混合，也不按成绩挑样本。

当前由原ordered_suffix_mvp驱动并影子计算Full BLP。完成360条后再拟合与审查参数，重跑固定20条Full冒烟；1600条四变体全量尚未启动。已有E定义及共享滞回/冷却选择继续沿用。

[单位修正与验证](depth_unit_fix.md)、[启动记录](../meta/calibration-start-a3.json)、[本次只读核查](../meta/current-live-audit.json)、[归档与新目录绑定](../meta/depth-unit-restart.json)。
