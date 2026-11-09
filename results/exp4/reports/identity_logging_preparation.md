# 候选身份日志与去重修复准备

a3标定运行代码保持eaa29e5，160项固定输入未变。隔离分支exp4/identity-logfix的03242fd已提供被动日志补丁，当前未向运行中的标定应用。

补丁仅在原有几何绑定失败时查询精确坐标的已登记ID。原登记分组、验证事件和排序不变。60项运行检查通过，219次关闭模式匹配输入回放一致；1500次配对函数操作产生的839条记录及原有context字段一致。Qa/18前三次真实查询、12条首次出现事件和29帧旧绑定可精确重现，补丁只改变第10至19帧的缺失身份。完整57动作派生轨迹通过两个原生阶段路径核对，最大误差1.7605547597554505e-7米。派生轨迹仅作日志验证，原始数据保留。

90条候选去重审查发现TEEs/25/k1在同一精确XYZ坐标下有两个已检查ID，两个最终标签均为系统否决。分析修复按原登记身份分组，并合并同一scene/episode/task中共享精确完整XYZ检查坐标的不同ID。没有新增距离阈值，也不读取v/y决定分组。逻辑回归保留每条验证事件，π按合并身份计数；episode bootstrap的不同抽样实例继续分别统计。

90条实际原生数据的核查中，已检查身份由864降为863。全部归一化episode、决策、SR/SPL/PR/PPL及条件成功率均一致；接受动作总路径及GT无效坐标路径也一致。1033次候选和1378次恢复点GT复核全部匹配，距离误差0。Qa原来的10帧身份缺口仍保留invalid_path_complete=false。该去重修复不会填补原始缺失身份，也不会使正式汇总绕过完整性检查。

9项去重专项检查、40项分析/日志检查及11项报告检查通过。两个临时合成统计流程各完成600次bootstrap，分别验证pooled/cross-fit分支；未拟合真实未完成数据。第一次报告检查因隔离目录缺本地配置未能执行，补齐原固定配置后通过，失败日志保留。

已按PID、start_ticks、boot和空子进程核对后停止旧CPU等待器115226，未向native发信号。旧状态与旧分析脚本已归档。当前已采用分析去重模块，并重建独立CPU等待器122348/start_ticks2658437，绑定11份脚本；只有360条完成后才审计、拟合并生成待审提案，不自动冻结或启动评估。原生控制器仍为31036。

评估前还须采纳并绑定已测的被动日志补丁，完成新的固定20条Full冒烟及CP3审查。当前证据不能替代真实冒烟，也不代表3-MON效果提升。

[源码差异与检查](../code/pending_identity_binding/README.md)、[被动日志重放](../diagnostics/identity-binding-validation.json)、[90条原生统计对比](../diagnostics/identity-canonicalization-native-validation.json)、[分析采纳记录](../diagnostics/identity-analysis-adoption.json)、[CPU重新排队](../diagnostics/postprocess-identity-analysis-requeue.json)。

评估源码迁移的严格校验与12项检查已准备，当前预检因标定未完成而拒绝执行。[分别绑定标定与评估源码](evaluation_runtime_binding.md)。
