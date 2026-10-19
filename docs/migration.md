# PBSP 重构进度

工作分支为 `refactor/full-pbsp`，方法来源为 `full-pbsp`。重构目标和模块划分见[架构设计](architecture.md)。每个阶段经过验证后提交并推送到同一分支。

项目自身代码采用 [MIT 许可证](../LICENSE)，第三方来源与原有许可证保留在[第三方清单](../THIRD_PARTY_NOTICES.md)中。

## 当前代码

公开统一命令已在另一个空目录中离线安装完成，耗时 617 秒。Mapping 的 129 个、navigation 的 173 个已安装分发包均与锁定版本逐项一致，依赖检查、原生扩展及 CUDA 检查通过。12 帧建图重放的 50 个张量和导航模型及 Habitat reset 的 21 个张量均与基线完全相同。21 个公开源码归档也与固定 Git 导出逐项一致；macOS 上的两套离线缓存检查分别通过 166 和 236 项。这里验证的是公开安装器及受控重放，模型资源仍使用已准备好的权重。详见[公开安装结果](validation/phase23_public_install_results.json)。

辅助权重和 tokenizer 已通过资源对象接入默认模型构造，原生流程不再要求源码目录内的权重链接。自定义 semantic injector 在构造时传入，可跳过默认语义模型。新增 `assets/models.json` 固定默认流程的 22 个文件，`scripts/prepare_models.py` 支持联网缓存、离线导入和只读校验；公开下载与迁移后的真实推理验证仍在进行。本机完整测试为 602 项通过、2 项跳过。三个旧实验测试改用仓库内的示例机器配置，OpenCV 线程测试同时核对实际设置调用和本机后端行为，不再依赖服务器配置文件或固定线程池实现。权重准备见[模型资源说明](models.md)。

公开安装流程整理为 `scripts/setup_envs.py`，分别读取 mapping 与 navigation 的锁定清单；两套缓存通过后才创建环境。导航清单包含 50 个 Conda 包、163 个 Python 输入及 23 个源码归档，覆盖 Habitat-Sim 的全部递归子模块、完整 Detectron2 和离线 Imath。共享校验代码保持 mapping 安装主体不变，新增 15 项检查覆盖递归归档、符号链接、路径重叠、缓存变更及非法清单。该提交的 Linux 完整测试为 593 项通过。环境入口和系统依赖见[安装说明](installation.md)及[导航环境指南](navigation_installation.md)。

两个独立环境已共同完成完整 4-MON 单 episode：四项任务全部成功，共 414 个动作，44 条规划记录身份完整，三个工作进程均以 0 退出。动作数、指标和最终地图哈希与上一轮一致。新导航环境在空目录中创建并重新编译 Habitat-Sim、Magnum 与 Detectron2；21 个模型及 Habitat 初始观测张量的数值、形状和类型均与原环境一致，完整测试为 578 项通过。实际工作进程的动态库映射也未引用旧环境或旧工作目录。安装验证修复了 Imath 临时联网、源码符号链接处理、Detectron2 导入遮蔽及其旧副本缺少 `data` 模块的问题。当前使用完整固定版本的 Detectron2 包；统一导航安装器及另一空环境中的最终流程验证仍待完成。详见[导航环境结果](validation/phase22_navigation_install_results.json)和[双环境集成结果](validation/phase22_integration_results.json)。

独立 mapping 环境的完整 4-MON 单 episode 验证已通过：四个目标均成功，共 414 个动作；44 条规划记录身份完整，没有过期后缀分派；31 次地图重建均完成，最终为 301 个关键帧和 3,088,415 个高斯点。建图、导航及资源监控进程退出码均为 0，动作数、指标和最终地图哈希与此前记录一致。首次运行虽完成全部目标，但监控读取已退出进程的 procfs 信息时发生竞态，根入口按失败记录；修复 `ProcessLookupError` 处理后另行运行并通过。新增五项监控检查后，Linux 完整测试为 577 项通过。此实验的 navigation 仍使用原环境。详见[集成结果及首次失败记录](validation/phase21_integration_results.json)。

Mapping 的独立离线安装已通过验证。安装器校验 42 个 Conda 包、122 个 Python 包输入和固定扩展源码，在新目录中创建环境并构建四个 CUDA 扩展；安装后依赖、扩展导入和 CUDA 检查均通过。真实推理补齐了 FastSAM 延迟导入的 pandas 依赖，最终锁定文件又在第二个空环境中完整执行。12 帧 RGB-D 重放的 50 个地图及增量状态张量与原环境逐项相同，候选和公开状态也一致。新增 13 项安装检查后，Linux 完整测试为 572 项通过。导航环境独立安装及完整 split 评测仍待完成。详见[安装指南](mapping_installation.md)和[安装验证](validation/phase21_mapping_install_results.json)。

默认建图路径的可选依赖已按后端加载。FastSAM/MaskAdapter 构造时不再导入 TensorFlow、MinkowskiEngine 或 SAM；OpenSeg 的设备设置在首次使用时执行。配置转换函数已独立，推理读取配置时不再连带导入训练模型封装。真实模型在禁止导入上述可选包的进程中成功加载；同一组 12 帧 RGB-D 的冻结对照中，50 个地图及增量状态张量逐项相同，候选和公开状态也一致。Linux 完整测试为 559 项通过。两个独立 Conda 基础环境已创建，GPU 依赖下载和源码构建仍在进行，尚未完成独立环境的完整实验验证。详见[依赖隔离结果](validation/phase20_dependencies_results.json)。

候选证据、候选排序、类别冷却、有序后缀规划及建图策略位于 `pbsp/`，所有调用方已使用这些实现。语义检索位于 `pbsp/query/engine.py`，包含特征加权、支持度重排、空间 NMS、单类别及批量查询；`service.py` 通过独立的文本编码器和索引读取器组装查询。完整 episode 循环位于 `experiments/runner.py`，规划事件的计算、重查和记录桥接位于 `experiments/planning.py`。指标计算和决策格式分别位于 `experiments/metrics.py`、`experiments/telemetry.py`。Provider 快照记录、规划回退和分派结果的构造、距离摘要及耗时 CSV 位于 `experiments/recorder.py`。规划决策 JSONL 的写入及写入失败处理也由 Recorder 完成。

`configs/experiments/hm3d_2mon.yaml`、`hm3d_3mon.yaml`、`hm3d_4mon.yaml` 引用同一个完整方法配置。`runtime/config.py` 读取配置并检查字段、数值、资源版本和引用路径。 `runtime/plan.py` 先核对完整数据身份，再按 scene 和真实 episode ID 选择完整任务序列；`runtime/launch.py` 保存本次配置并直接管理建图、导航和资源监测进程。原生工厂位于 `runtime/factories.py`，默认运行不经过旧启动脚本或环境变量参数转换。

`scripts/prepare_data.py` 验证三组固定版本数据的 manifest、内容和 source trace 校验值，在 `data/` 中建立资源链接，并为全部 episode 生成统一身份索引。清单使用 `assets/datasets.json`，以便数据准备只依赖 Python 标准库。

在线 Mapper 的实际状态管理、关键帧接纳、后台融合及地图发布已迁入 `pbsp/mapping/mapper.py`，模型输入和推理调用位于 `model.py`。`encoder.py` 执行首轮编码及增量更新，通过 `SemanticInjector` 调用 `pbsp/semantics/features.py` 中的区域特征提取及语义码本操作。`gaussians.py` 保存可扩容的高斯，`fusion.py` 执行投影关联、几何和特征融合及遮挡细化。模型组装与权重加载由 `adapters/embodied_splat.py`、`embodied_splat_model.py` 完成，语义资源和高斯索引分别由 `embodied_splat_semantics.py`、`embodied_splat_query.py` 适配，实验图像由 `experiments/visualization.py` 写入。底层网络层依赖的目录整理仍待完成。当前复现入口为根目录 `run.py`；`Exp/scripts/` 中的旧入口保留兼容。

阅读建图代码可从 `mapper.py` 的 `add_live_frame()`、`_rebuild_worker()` 和 `query_batch_with_records()` 开始。`reset_episode()` 清理整张地图；同一 episode 的任务切换保留地图。原运行目录中的 Mapper 文件只保留兼容入口。

四个组件的接口见[组件说明](components.md)。Mapping 接收独立的 `QueryBackend`；query 只消费 `SemanticIndex` 和匹配的 `TextEncoder`，检索参数位于独立的 `QuerySettings`。导航动作循环通过 `NavigationBackend` 执行局部规划，支持用外部控制器跟随 PBSP 选出的世界坐标 waypoint，并单独保留探索回退。默认 Gaussian 存储仍共同承载几何及语义融合；完全替换地图表示需要配套的索引读取和导出适配。VLFM 和 ApexNav 尚未接入或评测。

Habitat 的传感器、任务查询和动作转换位于 `adapters/habitat.py`，语义源、类别表、分割模型构造和动作编号由调用方提供。Detic 和禁用语义模式不读取语义传感器或场景标注。成功标签、路径状态和测量重置由 `experiments/evaluator.py` 持有；实际入口在动作之后、任务重置之前读取评测状态。真值俯视图由实验可视化层单独读取，不进入导航观测。导航器按 GPS 更新上一位置，不再依赖可视化数据。Habitat 组装位于 `adapters/goat.py`，配置合并位于 `adapters/habitat_config.py`；旧文件保留兼容入口。

Runner 接收 `RunSettings`、`PlanningRunSettings`、环境、Agent、Mapper provider 和规划服务，维护跨任务及跨 episode 的运行状态。刷新间隔来自显式配置；默认完整方法仍为 10 步。`RunArtifacts` 持有指标、耗时、Mapper 状态记录和距离缓存。`PlanningBridge` 接收投影、严格测地距离和时钟，不读取环境变量。`runtime/navigation_worker.py` 接收显式运行请求，组装环境、Agent 和规划服务，再调用同一个 Runner；根目录启动器直接读取 YAML，并从保存的运行清单启动服务；完整资源准备仍待完成。

建图客户端、命令处理和消息帧位于 `runtime/mapping_client.py`、`mapping_worker.py` 与 `protocol.py`。实际启动器直接运行原生建图工作进程，导航入口显式注入数据读取器。27 个客户端状态及数值方法、7 个服务端处理方法保留已验证逻辑；协议新增版本号和正常退出流程。工作进程在连接处理结束后关闭 Mapper，启动器记录并校验退出码。原客户端和服务端文件保留兼容导入，线程设置统一由 `runtime/bootstrap.py` 提供。协议和资源所有权见[运行时说明](runtime.md)；根目录启动器通过 `--run-plan` 直接组装两个原生工作进程。

导航的候选集合刷新与版本校验位于 `pbsp/navigation/candidates.py`，视觉确认历史位于 `verification.py`，受困诊断和恢复目标状态位于 `recovery.py`。实际动作循环、episode 重置和任务切换位于 `navigator.py`，局部地图更新及 RGB-D 预处理位于 `local_map.py`，坐标变换位于 `geometry.py`。`verification.py` 同时执行各目标模态的确认逻辑。模型构造和感知调用分别由 `adapters/lagmemo_agent.py`、`adapters/perception.py` 组装，图像写入交给实验观察器。原 `glue_agent.py` 仅保留兼容导入。候选投影绑定、候选层排序、版本化测地距离缓存和当前任务的顺序分派位于 `planning.py`。Habitat 导航网格投影、严格测地距离和恢复采样位于 `adapters/geometry.py`，参数由调用方显式传入。后缀计算和刷新事务也已迁入 `planning.py`，分别返回 `CandidatePlan` 和 `CandidateRefresh`。核心通过回调访问外部查询、版本检查和诊断，实验桥接层保留来源信息及记录格式。`pbsp/agent.py` 持有显式的候选、验证与恢复状态，并执行首次及周期候选分派；冷却控制、刷新后的实际目标和原始候选序号在这里统一处理。模型适配器接收 `NavigationSettings`，旧入口使用兼容构造器读取环境变量。运行组装与资源生命周期位于 `runtime/navigation_worker.py`，旧参数和环境变量仅在 `runtime/legacy_navigation_run.py` 边界解析。启动器直接运行原生工作进程，并记录各场景的命令、PID 和退出码。

## 阶段交付

| 阶段 | 交付内容 | 状态 |
| --- | --- | --- |
| 固定基准 | 原始数据身份、复现工具、设计文档及完整测试 | 已提交 `422cb8e` |
| 迁移纯算法 | 浅层包、共享配置、完整数据身份索引和原有调用方 | 已验证，253 项测试通过 |
| 拆分建图 | 实际在线 Mapper、编码、融合、检索和快照发布 | 在线流程与模型构造已迁移，底层网络依赖待整理 |
| 拆分导航 | Agent、候选规划、导航执行、验证、恢复及评测循环 | 候选刷新、验证、恢复、动作执行及实验循环已迁移，运行组装待完善 |
| 四组件接口 | Mapping、semantic injection、query 和 navigation 的运行路径及替换契约 | 已接入默认路径并验证，外部导航后端待适配 |
| 环境与评测边界 | 传感器和动作适配、真值指标、任务测量重置、可视化 | 已迁入浅层模块并接入实际入口 |
| 实验 Runner | 完整 episode 循环、任务边界、规划与记录桥接 | 已迁入实际运行路径，采用显式设置与服务对象 |
| 建图 IPC 与工作进程 | 实际客户端、协议、服务处理、连接及模型清理 | 已接入默认启动路径，退出码写入运行记录 |
| 导航工作进程 | 实际服务组装、显式请求、身份读取和资源清理 | 已接入默认启动路径，旧入口转发到原生工作进程 |
| 根目录启动 | 共享 YAML、完整身份选择、运行清单、原生服务与进程管理 | 已实现并接入真实运行；35 项新增检查和 550 项 Linux 完整测试通过 |
| 发布资源与安装 | 模型资源、依赖固定、环境脚本和离线流程 | 独立环境与依赖构建验证进行中 |
| 验证完整流程 | 记录重放、2/3/4-MON 集成实验、全新 Linux 安装和完整选定 split | 新入口已分别完成 2/3/4-MON 单 episode；全新环境和完整 split 待验证 |

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

四组件拆分后，服务器完整测试为 407 项通过。两个语义提取方法的可执行定义与冻结代码一致；受控 GPU 对照使用相同权重和 12 帧真实 RGB-D 输入，50 个地图及增量状态张量逐项相同，候选结果和公开状态也相同。首次与历史缓存的非受控比较出现差异，因此另行执行了固定随机种子、线程和 cuDNN 设置的独立对照。README 已按数据、环境和启动重排，并补充组件替换说明；参考项目及 skill 见[开源项目参考](open_source_references.md)。详见[组件拆分结果](validation/phase14_components_results.json)。

默认四组件路径的完整 4-MON 单 episode 验证正常结束，四项任务全部成功，共 414 个动作。44 条规划记录身份完整，没有过期后缀分派；31 次地图重建均完成。动作数、指标和最终 BEV 文件哈希与上一轮 Agent 集成记录一致。此项验证覆盖一个完整有序 episode，完整 split 评测仍待完成。详见[四组件集成结果](validation/phase14_integration_results.json)。

环境与评测拆分后，服务器完整测试为 441 项通过。新增 34 项检查覆盖冻结代码的观测转换、三种目标模态、动作编号、路径状态、测量重置顺序及无评测地图时的导航状态；评测路径读取的可执行定义与基准一致。实际 4-MON 单 episode 的四个目标全部成功，共 414 个动作，任务身份和版本记录完整。动作数、指标及最终地图文件哈希与拆分前的上一轮记录一致。详见[环境与评测结果](validation/phase15_habitat_results.json)及[集成结果](validation/phase15_integration_results.json)。

实际 episode 循环及三段规划桥接逻辑迁入实验模块后，12 组完整循环重放逐项比较了动作、调用顺序、候选状态和输出文件，覆盖 2/3/4 个目标、连续两个 episode、恢复、不可达及异常路径。循环的可执行定义与迁移前一致，仅将预算、计时选项和默认 10 步的刷新间隔改为显式参数。后续补齐了回退日志的时钟注入，服务器完整测试为 460 项通过。新 Runner 的真实 4-MON 单 episode 四项任务全部成功，共 414 个动作，指标及最终地图哈希与上一轮一致。GPU 运行提交和计时修复的验证范围分别记录在[Runner 结果](validation/phase16_runner_results.json)与[集成结果](validation/phase16_integration_results.json)中。

建图客户端、七个服务处理方法及实际 Unix socket 通信迁入 `runtime` 后，服务器完整测试为 491 项通过。三组冻结帧载荷的 JPEG、深度、位姿及内参完全一致；31 项新增检查覆盖真实连接、分片消息、查询回退、版本拒绝和异常清理。原生建图进程的完整 4-MON 单 episode 四项任务全部成功，共 414 个动作，退出码为 0。动作数、指标及最终地图文件哈希与上一轮一致。详见[IPC 结果](validation/phase17_ipc_results.json)、[集成结果](validation/phase17_integration_results.json)及[运行协议](runtime.md)。

导航启动与服务组装已迁入原生工作进程。三组冻结启动重放比较了实际组装参数、Episode 选择、规划配置和资源路径；补齐非默认投影距离的兼容绑定后全部一致。新增检查也覆盖构造失败、非法身份和信号退出；后续补丁将 Agent 的导航后端纳入清理，即使关闭报错也继续释放其余服务。服务器完整测试为 515 项通过。实际 3-MON 单 episode 的三个目标全部成功，共 307 个动作，33 条规划记录身份完整，没有执行过期后缀计划；两个工作进程均以 0 退出。GPU 实验采用组装迁移提交，后续清理补丁的验证单独列出，详见[导航工作进程结果](validation/phase18_worker_results.json)及[集成结果](validation/phase18_integration_results.json)。

根目录 `run.py` 已直接读取共享 YAML，保存本次配置、模型输入和完整任务身份，再启动原生建图、导航及资源监测进程。35 项新增检查覆盖真实子进程的失败与终止、身份选择、清单恢复、参数传入和记录开关，Linux 完整测试为 550 项通过。三组数据的 1,300 个 episode 也已通过新入口的完整预检查。新入口分别运行了完整 2/3/4-MON 单 episode：2-MON 完成两项任务，成功一项，两项均耗尽 500 步预算；3-MON 和 4-MON 的目标全部成功。三次运行共 179 条规划记录身份完整，没有执行过期后缀计划，九个工作进程均正常退出。4-MON 的指标和最终地图哈希与上一轮一致；3-MON 动作数相同，但路径指标和地图略有差异，结果原样保留。详见[根入口结果](validation/phase19_launch_results.json)与[三组集成结果](validation/phase19_integration_results.json)。
