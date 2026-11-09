# 只读监督日志补充CPU上下文

已在supervise_calibration和watch_evaluation的观察输出中加入逻辑核数、1/5/15分钟load及CPU PSI。它们用于解释实际运行耗时，不修正规划耗时，也不替代CP3正式时延统计。

本次只改results目录内两个只读工具，未改160项运行输入或CPU等待器绑定的11份脚本。原生controller、worker和参数均保持原状。旧监督窗口先正常结束并保存20次证明，再采纳新观察工具，历史日志的源码SHA保持原值。

两份脚本语法检查通过；新calibration观察器实际执行一次，读到116/360、原controller/worker/CPU等待器live、boot及运行输入匹配，CPU PSI约76%。这次只读检查没有启动新native，也不是Full冒烟。正式evaluation观察器要等实际评估阶段再使用。

calibration观察器默认20次、每次45秒，可用--max-polls指定1至160次；遇到新场景完成或状态变化提前结束。例如--max-polls 80可持续跟踪至cv结束，期间仍逐次核查原生任务。每次观察在metadata中保存本脚本实际SHA，避免把后续工具修改误作旧日志来源。

[采纳与实际核查](../diagnostics/host-observer-update.json)、[原始新工具](../code/supervise_calibration.py)、[CPU竞争诊断](resource_contention.md)。
