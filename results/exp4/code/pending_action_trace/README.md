# 正式工作区尚未应用的逐动作记录补丁

当前标定源码保持冻结。本补丁已在隔离工作区验证，尚未参与正式运行；不应在正式收集器运行时执行 `git apply`。

补丁把目标记录作为原生 `trajectory.jsonl` 每行动作的可选 `blp_action_target` 字段。轨迹写入层只在环境接受动作后保存该字段，并核对帧号和环境的STOP前阶段。查询候选、navmesh恢复点和frontier分开标注；未解析的查询身份明确记为unresolved。

视觉确认后，原Navigator会继续向SEEM视觉目标移动，即使新的查询刷新已经改变了GS候选。本补丁在确认时保存父候选，只用于记录，在视觉目标有效期间延续其身份。Runner实际替换的恢复转向另行记录；候选注册表只读，不增加候选、观测、验证或排序调用。

在服务器仓库根目录运行：

```sh
PYTHONPATH=. /nvme/khy/conda_envs/pbsp/navigation/bin/python results/exp4/code/prepare_action_trace_patch.py
git apply --check results/exp4/code/pending_action_trace/action-trace-after-calibration.patch
PYTHONPATH=.:results/exp4/code /nvme/khy/conda_envs/pbsp/navigation/bin/python -m unittest discover -s results/exp4/code -p test_pending_action_trace.py -v
```

生成器只写本目录下的副本和diff，校验4份基准源码SHA256，必要时从已记录的 `d94ca7d` 提取基准。它不修改运行文件。当前4份提案均可编译，补丁适配检查通过，8项测试使用提案中的真实轨迹适配器及目标绑定函数，在临时模拟环境验证已接受动作、STOP、恢复覆盖和身份延续。

上述8项测试使用模拟环境。另已在隔离工作区完成两次GPU试跑，共1048条动作，以及219次旧模式输入回归；证据见 `reports/action_trace_isolated_validation.md`。正式采纳时须核对与已测文件SHA256一致，在CP3实际检查目标与每帧轨迹。若后续交叉拟合修改了同一文件，应合并后重新验证，不能强行套补丁。

分析端已实现逐动作消费及CPU原生GT核查：1048条动作、4个阶段的路径与原生指标一致，GT标签和测地距离复现见 `reports/action_path_accounting.md`。正式汇总现在要求完整的新记录并拒绝影子快照。主收集源尚未应用本补丁，旧数据的缺口仍保留；标定结束后采纳记录补丁并更新源码指纹和CP2b绑定，再进行CP3实际验证，四个变体使用同一记录方式。
