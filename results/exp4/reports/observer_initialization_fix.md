# 观察端初始化状态修复

控制器先记录新场景为running，随后发现run_directory。旧watcher在该间隙返回进程信息，supervisor却直接读取计数，触发KeyError。保存的实际失败快照包含这一状态；原实验继续运行。

watcher现在明确标记telemetry_status为awaiting_run_directory。supervisor将未采集的计数记null，同时保留launcher存活状态。已采集数据保留原值和非零错误；缺少计数但没有初始化标记时仍报错。

6项CPU回归读取实际失败快照，并在临时目录调用watcher，检查未知计数、已测量的0、非零错误、进程存活标记和矛盾数据。真实任务已恢复连续观察。此项是只读观察工具修复，160项运行输入和11份CPU后处理脚本的SHA均未变；运行算法51项和后续logger60项检查计数不变。

[失败快照](../diagnostics/observer-initialization-failure-audit.json)、[失败与原生存活核查](../diagnostics/observer-initialization-failure.json)、[测试及真实运行绑定](../diagnostics/observer-initialization-fix-validation.json)、[6项检查日志](../code/observer-telemetry-tests.txt)。
