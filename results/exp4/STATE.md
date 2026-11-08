# Exp4 接续状态

目标仍在进行。CP1及前三组360条标定、真实CP2a拟合均已完成。C_exp区间未通过组间重叠规则，当前提议三组交叉拟合，待用户选择；尚未冻结参数或启动 Full 3-MON。交叉拟合加载已在独立分支准备并通过CPU检查，尚未采纳到主工作区。不要重复启动已正常结束的收集器。

## 工作区与连接

- 本机材料：`/Users/hikong/Documents/ChatGPT/30服务器/ForesightNav-exp4-20261107`。
- 服务器：`/nvme/khy/ForesightNav_exp4`；结果：`results/exp4`。
- 权威基线来自原 `/nvme/khy/PBSP-open-source` 实际工作树，含未提交改动，保存为 `7c7e9f4`。原工作区留给其他实验。
- GitHub 主实验分支 `exp4/blp-eq13`，CP2a及完整日志已推送 `4994fcc`。标定运行源码仍为 `d94ca7d` 的同一指纹。隔离准备分支 `exp4/crossfit-prep` 已推送 `72be3e6`，尚未合入主工作区。
- SSH：`ssh -S /tmp/codex-onemap30-20261107.sock 10.29.150.30`。普通直连被重置，现有复用连接可用。
- GitHub 可写 deploy key 已获用户授权并实际推送成功。仓库 core.sshCommand 使用 `/home/htz/.ssh/foresightnav_exp4`、`/tmp/exp4_github_proxy.py` 和代理 `127.0.0.1:19181`；不再索要凭据，不输出私钥。
- NAV Python：`/nvme/khy/conda_envs/pbsp/navigation/bin/python`；MAP Python：`/nvme/khy/conda_envs/pbsp/mapping/bin/python3.10`。

## 已确认的协议

用户确认：E 保留现有 CandidateRecord.semantic_score，论文改为所选高斯的 codebook 加权文本分数，不再写成邻近支撑集均值。四变体共享 1 m 滞回及原生类别冷却；分别报告无约束排序、滞回后和最终下发的偏离率，Nearest 仅无约束排序保证 0。见 `meta/protocol_decisions.json`。

参数只用固定 2-MON 系统最终验证标签拟合。GT 仅用于诊断及过程统计，不进入参数拟合、候选排序或评估调参。标定与评估共享场景和部分目标，见 `meta/calibration-overlap-audit.json`，CP2 必须披露。

## 当前正式收集与监督

2026-11-08 17:50 CST 已确认 **360/360**：12个场景各30条均正常结束，控制器1979824及全部launcher已退出，ledger状态为 `completed_three_folds_pending_CP2a`。18:06再核查时boot_id一致、原157项指纹未变、错误0，GPU2、3无本任务计算进程。没有正在运行或排队的评估任务；本任务空闲是收集正常完成。恢复前应重新核查GPU2、3资源，18:06查询时另有两个其他任务的新进程，不要将它们当作本实验残留。

本轮持续监督至三组全部结束，随后完成最终审计、拟合和CPU原生GT复核。观察会话43594、审计78519、拟合89206、GT复核54283、较早审计20877均自然结束，无待恢复工具会话。没有重跑正常数据、修改标定源码或提前运行3-MON。

Ledger：`meta/calibration_runs.json`；日志：`logs/calibration-controller-a2.log`。已结束的控制器原先每10秒检查157项源码/配置/固定清单及错误。每卡一个建图与导航共置worker，GPU2/2、3/3；后续实验仍只用GPU2、3，不碰其他GPU。

实时只读核查：

```bash
cd /nvme/khy/ForesightNav_exp4
PYTHONPATH=. /nvme/khy/conda_envs/pbsp/navigation/bin/python results/exp4/code/watch_calibration.py
```

结果写 `meta/current-live-audit.json`。当前controller_live=false、所有launcher_live=false是正常终态，不能据此重启。若以后正式采用新的运行代码，该工具对旧标定指纹会显示变化；保留旧ledger，不要篡改过去指纹来匹配新代码。

若用户选择补第4、5组，尽量复用相同运行源、保留现有360条及原ledger，在results目录补充明确的240条调度/汇总工具，不重跑挑结果。若选择三组交叉拟合，则实现新加载和记录补丁、验证后再为评估绑定新源码；过去的标定指纹仍保持原值。

## 已完成的关键证据

- CP1 核心18项测试；原生固定4ok3usBNeis/{0,1,2}三个完整3-MON共219次规划调用，关闭模式匹配输入回放全部一致。原输入 `code/regression_inputs/*.pkl.gz`，入口 `python -m experiments.exp4_regression --require-complete`。不能据此声称独立轨迹逐帧相同。
- 首次标定在TEEs/0第108步把非查询脱困waypoint误当查询候选，已整体归档排除；d94ca7d按candidate_source区分、保留历史查询E和非候选诊断。a2实际重放前107步相同，第108步正确记录并越过。见 `meta/recovery-waypoint-fix-audit.json`、`meta/calibration_runs.attempt1.json`、归档SHA证明。不要再重收这项修复。
- 全部12个场景360条均独立核对actual identity、manifest/source IDs/目标顺序、轨迹/STOP/native计数，3849次候选检查与最终标签逐项匹配，19453个影子快照、错误0。见 `meta/completed-calibration-scene-audits.json`。
- 旧式过程路径归属现共13处缺口：6s7的6/k0、26/k1，TEEs的8/k1，Qa的7/k1，cv的0/k1、1/k0、9/k0，Nfv的11/k1，zt的18/k0，XB的25/k0、28/k0，DY的4/k0、25/k0。全部继续保留invalid_path_complete=false，不影响标定E/S/v/y/C_exp；不能清空缺口放行正式成绩。

## 已测但主工作区尚未采纳的记录补丁

准备材料 `code/prepare_action_trace_patch.py`、`code/pending_action_trace/`：4个源文件绑定d94ca7d，77行新增；4份可编译，git apply --check通过，8项辅助/真实适配器检查通过。主工作区尚未应用，等CP2a确定是否继续补标定后再采纳。

补丁在环境接受动作后，将实际目标及来源写入 trajectory.blp_action_target。捕获STOP前任务、确认视觉目标的父候选和Runner恢复覆盖；Observer只读身份注册表，不改变排序、注册或策略。

独立worktree `/nvme/khy/ForesightNav_exp4_tracecheck`（detached1041d41加补丁）已完成两次排除于拟合的固定2-MON GPU3验证及219次关闭模式匹配输入回放，所有临时进程已退出：

- 6s7/0：48动作，11条确认视觉父目标；除新增字段外，完整动作位姿与原plumbing-v2相同。
- TEEs/8：1000动作，29次检查，覆盖确认后改查其他候选及恢复覆盖、两个预算STOP。身份及初始位姿一致，独立长轨迹不同，不能声称长轨迹等价。
- 首次预检缺依赖、尚未启动GPU的失败保存在 diagnostics/action-trace-plumbing/attempt1-preflight-missing-dependency；已链接原固定依赖，不删除失败记录。

证据 `diagnostics/action-trace-validation.json`、`diagnostics/action-trace-regression.json`、两次controller/audit及 `reports/action_trace_isolated_validation.md`。正式采纳时核对4份已测SHA；若合并cross-fit代码，须重新核对/测试，不盲目覆盖全文件。

## 逐动作分析消费已完成（本次新增）

`code/action_path_metrics.py`、`code/native_candidate_gt.py` 已接入 report_variants：逐动作、逐坐标调用固定GOAT原生GT判据。主 invalid_visit_path_m 只计曾被检查无效身份中、当前实际目标坐标也GT无效的位移；invalid_identity_pursuit_path_m 另列全身份追逐上界。恢复/frontier/未检查目标分开，确认视觉父目标延续，同一位移不重复。

CPU原生复核已扩至全部360条标定与两次隔离数据：3880个候选标签、7246个恢复标签，11126次y/测地距离全部匹配，距离误差0。两次隔离数据共1048动作、4阶段的物理路径与原生指标一致，最大误差6.45e-7m。短/长无效路径分别1.000000458m、39.513657511m；长样本旧间隔法不完整，新逐动作归属完整。旧诊断嵌套保留，不修改旧标定缺口。

分析测试10+9+8项通过；复现入口 `code/verify_action_path_pipeline.py`，证据 `diagnostics/native-offline-gt-check.json`、`diagnostics/action-target-path-check.json`，说明 `reports/action_path_accounting.md`。这些都是诊断，不是CP3/Full成绩。

正式汇总拒绝旧轨迹、影子快照、参数不一致或路径不完整。汇总记录实际加载的三个分析模块SHA，四变体比较要求同一分析版本；若修正分析代码，使用同版重新汇总原始数据，无需重跑GPU。

## CP2a真实结果及待确认选择

`fit_calibration_reviewed.py`已实际执行完成，600/600次episode bootstrap成功；2次数值优化状态失败按同一MLE目标重试并通过KKT，未改运行源。原始结果 `meta/calibration_per_fold.json`、日志 `logs/calibration-fit-three-folds.json`，不要无新原因重复拟合或重跑收集。

- θ_pool=(45.2470468461, -12.2367842831, 0.1528218136, 1.1948893037)，依次为a、b0、π、C_exp，仅为未冻结提案。
- 三组翻转率0.6734%、0.4215%、0.6169%，分母19453；非空分母15666另报。self与GT诊断池翻转率1.4856%。
- a、b0、π的组区间重叠；C_exp不重叠：Fold1=[1.099441,2.572049]、Fold2=[0.643488,1.024613]米，故pooled规则不通过。
- 建议三组交叉拟合；具体四套参数及20场景映射在 `meta/calibration.crossfit.proposed.json`，frozen=false。Fold0/1/2分别排除自身组，未收集的Fold3/4使用pool；GT参数只作诊断。主工作区当前仍只支持pooled；通用三/五组交叉拟合加载、汇总和调度已在隔离分支实现，采用后才能评估。
- 报告 `reports/CP2a.md` 已含各组/GT参数、600次bootstrap区间、组间统计、混淆矩阵、可靠性、翻转表及三张已视觉检查的图。`code/report_calibration.py`可复现，不重新拟合。
- 探索事件363次启动、361次完成；两条未完成记录未虚构0米，报告披露删失限制。
- 六份完整事件/快照日志压缩为约11.9MB，原始77.4MB，gzip解压SHA全部一致；`meta/calibration-log-archive.json`列出原始及压缩哈希，原文件仍在服务器。`code/archive_calibration_logs.py`可复现。

设计第2.5节对不稳定时改交叉拟合明确要求“停下汇报，等待确认”，CP2a/CP2b/CP3也标注【需确认】。已准备可审阅材料，`meta/CP2a_review.json`状态为awaiting_user_choice、accepted=false，并绑定报告/参数/ledger/分组/日志压缩清单SHA。下一步请用户选择：采用上述三组交叉拟合参数，完成实现和冻结后进入固定20条Full冒烟；或先补240条2-MON到五组再重审。新增组不会消除原三组已有的C_exp区间不重叠。

## 等待选择期间完成的隔离准备

服务器 `/nvme/khy/ForesightNav_exp4_crossfit`，分支 `exp4/crossfit-prep`，提交 `72be3e62a99d208da1ee362e15013ae24da139f7`。它从4994fcc创建，包含已测77行逐动作补丁及新的scene参数加载，不要在采用后再次强行套同一补丁。

新增 `pbsp/navigation/blp_parameters.py`：严格识别冻结v1参数文件、固定20场景映射、3/5个完整标定组；已收集组排除自身，未收集组用pool，未知场景拒绝。`request.scene`原样传入初始化；快照记录参数、scene/fold、训练组和文件SHA。正式汇总和调度使用同一解析器，拒绝错组/错SHA，跨变体必须同一场景分配。

CPU检查49项通过（原18+参数9+汇总11+调度11），关闭模式219次匹配输入回放全部一致。无GPU仿真或实际成绩；检查仍因缺冻结文件返回ready=false。证据在该分支 `results/exp4/diagnostics/crossfit-preparation*.json`、`code/crossfit-preparation-tests.txt`，说明 `reports/crossfit_preparation.md`。主工作区原157项指纹再次核对未变、没有calibration.json或评估ledger。

CP2a选择请求已发出一次，尚未收到回复；不要重复提问，也不要把工具的accepted=true当成用户已选择。该通用准备不选择最终参数，三组/五组选项均可复用。

## 确认后的剩余工作

1. 不重复提出已确认的E、共享滞回或GitHub权限问题；已有CP2a待答请求时不要重复索要。没有用户选择前，不冻结参数或启动3-MON。
2. 若选三组交叉拟合，审阅并采纳隔离提交72be3e6，按已确认的四套数值形成冻结v1文件和CP2b源码/参数绑定。主分支仅新增状态记录时可cherry-pick该提交；它已含记录补丁。保留原标定ledger，不改过去源哈希。若选五组，主工作区保持原标定源，保留360条，加收八场景240条并扩展拟合到五组，之后再采用通用加载支持。
3. 采纳记录补丁时核对已测4份SHA；若与cross-fit修改相交，合并后重审、重做有针对性的回归，不盲目覆盖全文件。
4. `code/evaluate_sequence.py`每次一阶段：smoke固定20条→Full400→Nearest400→Myopic400→Suffix-only400。必须保持固定身份、参数绑定、原生完整状态及前阶段审阅证明；首次真实GPU阶段持续监督。
5. CP3要有真实20条、两张俯视图、过程统计及行为审阅；之后Full400写CP4，再其余变体写CP5四行指标、2×2主效应/交互项和6对配对比较，持续同步GitHub。全部真实实验及报告完成前，goal不能标记complete。
