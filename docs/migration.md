# PBSP 重构进度

工作分支为 `refactor/full-pbsp`，方法来源为 `full-pbsp`。重构目标和模块划分见[架构设计](architecture.md)。每个阶段经过验证后提交并推送到同一分支。

项目自身代码采用 [MIT 许可证](../LICENSE)，第三方来源与原有许可证保留在[第三方清单](../THIRD_PARTY_NOTICES.md)中。

## 当前代码

候选证据、候选排序、类别冷却、有序后缀规划及建图的小型策略位于 `pbsp/`，所有调用方已使用这些实现。指标计算和决策记录分别位于 `experiments/metrics.py`、`experiments/telemetry.py`。

`configs/experiments/hm3d_2mon.yaml`、`hm3d_3mon.yaml`、`hm3d_4mon.yaml` 引用同一个完整方法配置。`runtime/config.py` 读取配置并检查字段、数值、资源版本和引用路径。

`scripts/prepare_data.py` 验证三组固定版本数据的 manifest、内容和 source trace 校验值，在 `data/` 中建立资源链接，并为全部 episode 生成统一身份索引。清单使用 `assets/datasets.json`，以便数据准备只依赖 Python 标准库。

在线模型、Mapper 工作线程和导航动作循环仍使用原运行目录；接下来的两个阶段将迁移其实际实现。当前可运行的复现入口为 `Exp/scripts/run_full_pbsp.py`。

## 阶段交付

| 阶段 | 交付内容 | 状态 |
| --- | --- | --- |
| 固定基准 | 原始数据身份、复现工具、设计文档及完整测试 | 已提交 `422cb8e` |
| 迁移纯算法 | 浅层包、共享配置、完整数据身份索引和原有调用方 | 已验证，253 项测试通过 |
| 拆分建图 | 实际在线 Mapper、编码、融合、检索和快照发布 | 待实施 |
| 拆分导航 | Agent、候选规划、导航执行、验证、恢复及评测循环 | 待实施 |
| 接入入口与安装 | `run.py`、工作进程、模型资源、环境脚本和离线流程 | 待实施 |
| 验证完整流程 | 记录重放、2/3/4-MON 集成实验、全新 Linux 安装和完整选定 split | 待实施 |

## 验证要求

- 原有数值逻辑迁移时，对比函数定义与基准，并运行现有行为测试。
- 配置验证覆盖全部原 profile 显式参数，2/3/4-MON 共享方法和评测配置。
- 身份索引覆盖全部 scene、episode 和顺序任务，历史结果标签不进入决策输入。
- 建图对比固定关键帧输入下的结果；导航重放固定观测、查询结果、版本及消息顺序。
- 同一入口运行完整子任务序列，episode 内保留地图，episode 切换时清理状态。
- 安装验证使用独立 Linux 环境，依赖、权重与源码均脱离旧工作目录。
- README 按数据集、环境、启动排列，给出完整实验命令和结果位置。
- GitHub 工作分支、本机与服务器使用一致的已验证提交；数据和权重保留在资源目录。

## 已固定的基准

`422cb8e` 保存初始复现工具和设计。服务器再次运行原测试：231 项通过。单 episode 的原始运行和指标见[基准记录](../Exp/release/REPRODUCTION_VALIDATION.json)。这些记录用于后续行为比较。

纯算法迁移对比了 17 个模块，除导入语句外，其可执行定义和常量与基准一致。服务器完整测试为 253 项通过。三组数据共 1,300 个 episode 的全部有序身份与原记录一致，数据准备重复运行的结果相同。详见[源码对比](validation/phase1_source_parity.json)及[阶段结果](validation/phase1_results.json)。
