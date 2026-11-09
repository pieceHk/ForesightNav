# 旧轮次已停止：S深度单位不一致

2026-11-09核实GOAT输出归一化深度，原BLP覆盖观察器将其直接按米使用。例如0.5实际表示2.75米。此前计数、递归重放及路径核对未覆盖这个单位接口，不能证明S正确。

smoke a1已在10条完成、2条运行时主动停止，GPU2、3上的本任务进程已退出。该轮全部S/B及排序结果不进入修正版本的正式CP3结论，原始数据和旧报告完整保留。

单位转换已修正，51项CPU检查及219次关闭模式匹配输入回放通过。修正只为覆盖观察器转换深度，原导航输入保持原样。原三组标定的S和敏感性诊断也需重做；修正版本将按原固定三组重收360条2-MON，再重跑固定20条Full冒烟。Full400尚未启动。

[修正代码与说明](https://github.com/pieceHk/ForesightNav/blob/6b681ebf3877d386380b755bd221e81008e33542/results/exp4/reports/depth_unit_fix.md)、[实际适配器反例与停止证明](../diagnostics/depth-unit-fix/smoke-depth-unit-stop.json)、[旧报告（已作废）](archive/CP3_progress.smoke-a1-depth-unit-invalid.md)。
