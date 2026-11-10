# Exp4 Full400启动记录

用户明确回复“通过 CP3，继续全量”，已绑定[CP3接受记录](../meta/CP3_acceptance.json)。原生Full400于北京时间2026-11-10 18:12:17在物理GPU2/3启动，20场景各20条。

运行源码仍为`03242fda76fd6a631596a7269c8e7990c657ea30`，161项指纹与冒烟/CP2b一致；冻结参数SHA `e1d75f3c9503c7439467748fc97be6ae2ad450c429f40dc7235b1564ff7852ae`。原标定和冒烟证据保留，未改参数、清单或评估设置。

原生控制器PID 468645、start_ticks 11385483；独立只读观察器PID 468659、start_ticks 11385509。两者已实际核对存活及身份，观察窗口7天；每30秒记录日志。SSH断开后先核查原PID/启动时刻与boot，不重复启动。

本次启动快照观测到50个动作和6个规划快照，五类日志错误均为0。此为带UTC时刻的启动快照，Full400尚未完成，逐帧检查随原生日志继续。

主机384逻辑核，负载1/5/15分钟为[2171.8, 2163.4, 1966.45]；CPU竞争较强，可能影响吞吐和耗时。实际耗时记录保留。

Full完成后核对400条和1200阶段路径，生成并汇报CP4；之后依次Nearest、Belief-myopic、Suffix-only各400条，最后计算SR/SPL主效应、交互和六组配对结果。

[启动元数据](../meta/full-start-a3.json)、[观察器元数据](../meta/full-observer-a3.json)、[启动审计](../diagnostics/Full400/start-audit.json)、[固定400条身份](../meta/3mon-identities.json)。
