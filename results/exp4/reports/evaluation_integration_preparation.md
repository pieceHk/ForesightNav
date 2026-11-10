# a3评估代码整合准备

本分支将完整a3标定报告与已审查的被动logger032合并到一个隔离工作区，便于复核后续评估源码。它尚未采纳到实际运行目录；服务器权威目录仍为`/nvme/khy/ForesightNav_exp4_depthfix`，原生HEAD仍EAA。

原生源码仅变化`experiments/blp_observer.py`并新增`experiments/test_blp_identity.py`。实际调用既有`validate_source_change`已核对160→161项指纹：159项相同，观察器和新增测试恰为已审查SHA，没有其他输入变化。完整证据见[整合准备JSON](../diagnostics/evaluation-integration-preparation.json)。此项是源码一致性检查，不能计作实际60项测试或219次回放。

首次检查发现报告父提交f87a5e7没有收录三份冻结的episode身份清单，因FileNotFoundError结束。现直接补入服务器原件：2-MON清单600条、3-MON清单400条、固定冒烟清单20条；每份SHA均与原160项绑定一致，没有重新生成或改变顺序。600条2-MON覆盖全部五组，本次只按已确认的前三组收集360条。

六份压缩标定日志已在隔离工作区还原，并逐份核对原始SHA；新四套参数提案SHA保持3c7b3c7bfa0bd38210f7fdd378dace278f5e3aee8e1f5cdccc7a4bd27782abbe。运行机私有runtime.exp4.local.yaml只按原字节复制到忽略路径，没有提交配置内容。

新具体参数仍待用户确认。当前没有创建新CP2a接受记录、calibration.json或CP2b绑定，没有运行新的Full冒烟或主评测。批准后仍需在选定的实际评估源执行60项检查及219次匹配输入回放，并通过严格CP2b绑定；本次隔离合并不代替那些步骤。

[新CP2a](CP2a.md)及[具体参数与下一步](CP2a_next_step.md)继续为本次审阅依据。全量前还需固定20条Full、过程指标、两张指定俯视图及CP3确认。
