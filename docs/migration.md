# PBSP 重构进度

工作分支为 `refactor/full-pbsp`，方法来源为 `full-pbsp`。重构目标和模块划分见[架构设计](architecture.md)。每个阶段经过验证后提交并推送到同一分支。

项目自身代码采用 [MIT 许可证](../LICENSE)，第三方来源与原有许可证保留在[第三方清单](../THIRD_PARTY_NOTICES.md)中。

## 当前代码

候选证据、候选排序、类别冷却、有序后缀规划及建图策略位于 `pbsp/`，所有调用方已使用这些实现。实际语义检索也已迁入 `pbsp/mapping/query.py`，包含特征加权、支持度重排、空间 NMS、单类别及批量查询。指标计算和决策格式分别位于 `experiments/metrics.py`、`experiments/telemetry.py`。Provider 快照记录、规划回退和分派结果的构造、距离摘要及耗时 CSV 位于 `experiments/recorder.py`。规划决策 JSONL 的写入及写入失败处理也由 Recorder 完成；实验主循环仍在旧入口。

`configs/experiments/hm3d_2mon.yaml`、`hm3d_3mon.yaml`、`hm3d_4mon.yaml` 引用同一个完整方法配置。`runtime/config.py` 读取配置并检查字段、数值、资源版本和引用路径。

`scripts/prepare_data.py` 验证三组固定版本数据的 manifest、内容和 source trace 校验值，在 `data/` 中建立资源链接，并为全部 episode 生成统一身份索引。清单使用 `assets/datasets.json`，以便数据准备只依赖 Python 标准库。

在线 Mapper 的实际状态管理、关键帧接纳、后台融合及地图发布已迁入 `pbsp/mapping/mapper.py`，模型输入和推理调用位于 `model.py`。`encoder.py` 执行首轮编码及增量更新，`features.py` 提取区域特征并建立语义码本，`gaussians.py` 保存可扩容的高斯，`fusion.py` 执行投影关联、几何和特征融合及遮挡细化。模型组装与权重加载由 `adapters/embodied_splat.py`、`embodied_splat_model.py` 完成，实验图像由 `experiments/visualization.py` 写入。底层网络层依赖的目录整理仍待完成。当前复现入口为 `Exp/scripts/run_full_pbsp.py`。

阅读建图代码可从 `mapper.py` 的 `add_live_frame()`、`_rebuild_worker()` 和 `query_batch_with_records()` 开始。`reset_episode()` 清理整张地图；同一 episode 的任务切换保留地图。原运行目录中的 Mapper 文件只保留兼容入口。

导航的候选集合刷新与版本校验位于 `pbsp/navigation/candidates.py`，视觉确认历史位于 `verification.py`，受困诊断和恢复目标状态位于 `recovery.py`。实际动作循环、episode 重置和任务切换位于 `navigator.py`，局部地图更新及 RGB-D 预处理位于 `local_map.py`，坐标变换位于 `geometry.py`。`verification.py` 同时执行各目标模态的确认逻辑。模型构造和感知调用分别由 `adapters/lagmemo_agent.py`、`adapters/perception.py` 组装，图像写入交给实验观察器。原 `glue_agent.py` 仅保留兼容导入。候选投影绑定、候选层排序、版本化测地距离缓存和当前任务的顺序分派位于 `planning.py`。Habitat 导航网格投影、严格测地距离和恢复采样位于 `adapters/geometry.py`，参数由调用方显式传入。后缀计算和刷新事务也已迁入 `planning.py`，分别返回 `CandidatePlan` 和 `CandidateRefresh`。核心通过回调访问外部查询、版本检查和诊断，实验桥接层保留来源信息及记录格式。`pbsp/agent.py` 持有显式的候选、验证与恢复状态，并执行首次及周期候选分派；冷却控制、刷新后的实际目标和原始候选序号在这里统一处理。模型适配器接收 `NavigationSettings`，旧入口使用兼容构造器读取环境变量。观测、规划与 episode 的完整组装及评测循环仍待迁移。

## 阶段交付

| 阶段 | 交付内容 | 状态 |
| --- | --- | --- |
| 固定基准 | 原始数据身份、复现工具、设计文档及完整测试 | 已提交 `422cb8e` |
| 迁移纯算法 | 浅层包、共享配置、完整数据身份索引和原有调用方 | 已验证，253 项测试通过 |
| 拆分建图 | 实际在线 Mapper、编码、融合、检索和快照发布 | 在线流程与模型构造已迁移，底层网络依赖待整理 |
| 拆分导航 | Agent、候选规划、导航执行、验证、恢复及评测循环 | 候选刷新、验证、恢复、动作执行和候选分派已迁移，完整组装及评测循环待拆分 |
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

语义检索迁移使用 8 组直接从基准代码记录的输入与输出，验证阈值边界、支持度重排、NMS、第二候选池和分数相同的情况。另验证批量查询去重、地图版本记录及显式参数隔离。服务器完整测试为 264 项通过；详见[检索迁移结果](validation/phase2_query_results.json)。

Mapper 迁移后，服务器完整测试为 271 项通过。固定 RGB-D 序列与基准实现的状态、查询结果和全部模型输入张量逐项相同；测试同时覆盖关键帧上限、高斯上限、待执行重建、重置和关闭。真实 Habitat 场景的 12 帧观测在 GPU 上生成 287,475 个高斯点，并完成语义查询和 episode 重置。此项验证覆盖建图调用，不包含导航 SR/SPL；详见[Mapper 迁移结果](validation/phase3_mapper_results.json)。

相机参数中保留了一个需要注意的基准行为：原 Mapper 总是使用 640×480、79° HFOV 对应的固定内参。兼容工厂显式传入同一矩阵，保持当前实验一致；核心预处理函数使用调用方提供的内参。

高斯存储和融合迁移后，服务器完整测试为 282 项通过。CPU 对比覆盖存储扩容与替换、语义码本排序、循环特征融合、加权融合及遮挡细化。GPU 上，基准编码器和新实现使用完全相同的权重及 12 帧真实观测，26 个地图张量逐项相同，查询候选也相同；1,745 个 checkpoint 参数键严格匹配。详见[融合迁移结果](validation/phase4_fusion_results.json)。

在线编码器迁移后，服务器完整测试为 292 项通过。GPU 对比增加了增量恢复状态，共 50 个张量与基准逐项相同，候选及公开状态也相同。构造函数和在线路径的可执行定义对比通过，区域特征提取方法保留原有实现。计时通过可选回调记录，在线编码器不写离线渲染或导出文件。详见[编码器迁移结果](validation/phase5_encoder_results.json)。

编码器迁移后的完整 4-MON 单 episode 验证完成了四个有序子任务，均成功，共 414 个动作。44 条规划记录均包含完整身份，未执行过期后缀计划；此记录验证完整连接，与初始单次运行不构成受控的性能比较；完整 split 评测仍待完成。详见[集成结果](validation/phase5_integration_results.json)。

导航状态迁移后，服务器完整测试为 299 项通过。13 个函数及方法的可执行定义与基准一致；重放覆盖候选刷新、过期元数据拒绝、多帧确认、历史过期、受困检测及恢复重置。实际 Agent 的对应方法已绑定核心包，详见[导航状态结果](validation/phase6_agent_state_results.json)。

导航状态迁移后的 2-MON 和 3-MON 单 episode 实验均正常退出。2-MON 完成 2 个任务，成功 1 个，两个任务都耗尽 500 步预算；3-MON 的 3 个任务均以目标确认结束并成功。两次运行共 135 条规划记录均保留完整任务身份，未执行过期后缀计划。详见[导航状态集成结果](validation/phase6_integration_results.json)。

动作循环迁移增加了 19 组固定输入重放，直接调用基准提交中的原方法进行比较，覆盖动作、停止条件、多帧确认、三种目标模态、位姿和局部地图更新；服务器上全部通过。另有 31 个方法及辅助函数通过可执行定义对比。局部地图现在始终提供算法所需的 explored/frontier 输入，其值与原来开启可视化时相同，图像写入可独立关闭。本机 CPU 上的 20 项导航检查也全部通过；完整测试为 316 项通过、3 项失败，失败涉及本机缺少服务器数据及 macOS OpenCV 的线程数行为。该阶段的 Linux 和 Habitat 补验见后续集成记录；源码与重放结果详见[导航执行迁移结果](validation/phase7_navigator_results.json)。

候选规划与几何迁移增加了 19 项基准重放，覆盖距离状态、缓存失效、重复类别任务身份、分派顺序及恢复采样的楼层和岛屿约束；11 个函数通过可执行定义对比。本机完整测试为 335 项通过、3 项环境相关失败，与上一阶段一致。该阶段旧入口已调用上述实现；完整刷新事务随后迁入 `planning.py`。详见[规划迁移结果](validation/phase8_planning_results.json)。

规划事务迁移增加了 24 项基准重放，逐项比较完整规划事件、确定性计时、缓存和排序器状态，以及重查与异常路径的全部调用顺序；规划重放合计 43 项通过。后缀计算、刷新控制流和最终版本检查的源码对比也通过。本机完整测试为 359 项通过、3 项环境相关失败；此阶段的服务器补验结果见下文。详见[规划事务结果](validation/phase9_planning_transaction_results.json)。

内网恢复后，本机、服务器和 GitHub 同步到 `a10e7e0`。服务器 Linux 环境完整测试为 362 项通过；此前本机的 3 项环境限制已补验通过。该提交的真实 4-MON 单 episode 集成已完成，四项任务全部成功，共 414 个动作；44 条规划记录身份完整，没有过期后缀计划被执行。详见[导航与规划集成结果](validation/phase9_integration_results.json)。

实验记录模块已迁出 6 个实际函数及耗时阈值常量，其可执行定义与基准一致。101 项相关检查通过，包含输出失败时保留有效版本语义、异常记录和过期分派拒绝。详见[记录模块结果](validation/phase10_recorder_results.json)。

记录模块在服务器上验证后，完整测试为 362 项通过。核对真实运行的最终 BEV 时发现，旧重建日志在 CUDA 回传完成前读取了点数：日志为 3,081,037，完成后的图像为 3,088,415。嵌套状态迁移现在在返回 CPU 状态前等待各源 CUDA 流完成；三个 GPU 对照用例均复现旧读数错误，修复后全部正确。真实 12 帧 RGB-D 重放中，10 帧及 12 帧的公开计数与张量计数分别一致，最终为 287,475 个点，检索、重置和关闭也通过。修复后的 Linux 完整测试为 364 项通过。详见[CPU 状态发布修复](validation/phase11_host_copy_results.json)。

Agent 状态及候选分派迁移后，服务器完整测试为 390 项通过。5 组构造状态对照和 19 组原入口代码块重放通过，覆盖冷却、首步、候选缺失、过期版本与刷新失败，并比较实际目标、候选状态和刷新调用顺序。模型与局部地图构造的 69 条语句保持一致；首次和周期分派的实际计算也通过源码对比。详见 [Agent 状态与分派结果](validation/phase12_agent_results.json)。

实际 Agent 分派路径的完整 4-MON 单 episode 验证完成，四项任务全部成功，共 414 个动作。44 条决策保留完整身份，未执行过期后缀计划；31 次地图重建均完成。最终日志点数与 BEV 图像中的 3,088,415 一致，图像标题逐像素核对通过。此项结果验证该 episode 的完整连接，完整 split 评测仍待完成。详见 [Agent 集成结果](validation/phase12_integration_results.json)。

规划决策的 JSONL 写入和降级诊断已迁入 `experiments/recorder.py`，旧桥接层保留计算与异常事件的组装。写入部分与基准代码一致；75 项相关检查通过，随后服务器完整测试为 390 项通过。序列化失败时，记录模块保留已经发生的分派结果，并尝试写入诊断。详见 [决策持久化结果](validation/phase13_recorder_persistence_results.json)。
