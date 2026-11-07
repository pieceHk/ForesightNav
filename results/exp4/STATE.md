# Exp4 接续状态

目标仍在进行。CP1 已完成；修复记录器后的正式标定 attempt a2 正在运行。尚未拟合真实参数、冻结参数或启动 Full 3-MON。本文历史采样不能代替实时进程核查。

## 工作区与连接

- 本机材料：`/Users/hikong/Documents/ChatGPT/30服务器/ForesightNav-exp4-20261107`。
- 服务器：`/nvme/khy/ForesightNav_exp4`；结果：`results/exp4`。
- 权威基线来自原 `/nvme/khy/PBSP-open-source` 实际工作树，含未提交改动，保存为 `7c7e9f4`。原工作区留给其他实验。
- GitHub 分支 `exp4/blp-eq13`。实际运行源码冻结在 `d94ca7d`；之后的提交仅涉及分析和结果材料。逐动作分析消费与验证证据已推送 `98727ee`。
- SSH：`ssh -S /tmp/codex-onemap30-20261107.sock 10.29.150.30`。普通直连被重置，现有复用连接可用。
- GitHub 可写 deploy key 已获用户授权并实际推送成功。仓库 core.sshCommand 使用 `/home/htz/.ssh/foresightnav_exp4`、`/tmp/exp4_github_proxy.py` 和代理 `127.0.0.1:19181`；不再索要凭据，不输出私钥。
- NAV Python：`/nvme/khy/conda_envs/pbsp/navigation/bin/python`；MAP Python：`/nvme/khy/conda_envs/pbsp/mapping/bin/python3.10`。

## 已确认的协议

用户确认：E 保留现有 CandidateRecord.semantic_score，论文改为所选高斯的 codebook 加权文本分数，不再写成邻近支撑集均值。四变体共享 1 m 滞回及原生类别冷却；分别报告无约束排序、滞回后和最终下发的偏离率，Nearest 仅无约束排序保证 0。见 `meta/protocol_decisions.json`。

参数只用固定 2-MON 系统最终验证标签拟合。GT 仅用于诊断及过程统计，不进入参数拟合、候选排序或评估调参。标定与评估共享场景和部分目标，见 `meta/calibration-overlap-audit.json`，CP2 必须披露。

## 当前正式收集与监督

2026-11-08 06:36 CST 实测完成 **120/360**：fold0 四场景均30条正常结束，旧launcher均退出。fold1 已自动启动：GPU2 的 Nfvxx8J5NCo launcher2023735，GPU3 的 p53SfW6mjZe launcher2023736，已分别接受96/89条动作。控制器1979824和两项新launcher的真实 /proc 身份匹配，boot_id一致，冻结文件未变，错误0，最近各10快照可重放。不要用已退出 launcher 判断任务，也不要重启健康进程。本轮临时只读观察会话4237已自然退出，没有仍待恢复的工具会话。

Ledger：`meta/calibration_runs.json`；日志：`logs/calibration-controller-a2.log`。控制器 `python -m experiments.exp4_collect` 每10秒检查157项源码/配置/固定清单及错误；先收前三组12场景，每场景30条2-MON，共360条。每卡一个建图与导航共置 worker，GPU2/2、3/3；组间有屏障。不要另加常规 worker 或碰其他GPU。

实时只读核查：

```bash
cd /nvme/khy/ForesightNav_exp4
PYTHONPATH=. /nvme/khy/conda_envs/pbsp/navigation/bin/python results/exp4/code/watch_calibration.py
```

结果写 `meta/current-live-audit.json`，含 /proc、boot_id、指纹、回合/动作增长、最近快照重放和错误字节。忽略尚未写完的JSONL尾行，不因瞬时读失败重启任务。主代理约4–5分钟复核，异常及时处理。

**收集期间不可修改** `pbsp/`、`experiments/`、`configs/`、运行local配置、固定folds/身份及协议文件。允许在 `results/exp4/code/`、reports、diagnostics 下做分析。三组结束控制器停在 `completed_three_folds_pending_CP2a`，不会自动启动评估。

## 已完成的关键证据

- CP1 核心18项测试；原生固定4ok3usBNeis/{0,1,2}三个完整3-MON共219次规划调用，关闭模式匹配输入回放全部一致。原输入 `code/regression_inputs/*.pkl.gz`，入口 `python -m experiments.exp4_regression --require-complete`。不能据此声称独立轨迹逐帧相同。
- 首次标定在TEEs/0第108步把非查询脱困waypoint误当查询候选，已整体归档排除；d94ca7d按candidate_source区分、保留历史查询E和非候选诊断。a2实际重放前107步相同，第108步正确记录并越过。见 `meta/recovery-waypoint-fix-audit.json`、`meta/calibration_runs.attempt1.json`、归档SHA证明。不要再重收这项修复。
- 首组四个完整场景120条已独立核对 actual identity、manifest/source IDs/目标顺序、轨迹/STOP/native计数，1446次候选检查与最终标签逐项匹配，6105个影子快照、错误0。见 `meta/completed-calibration-scene-audits.json`。
- 旧式过程路径归属现共7处缺口：6s7的6/k0、26/k1，TEEs的8/k1，Qa的7/k1，cv的0/k1、1/k0、9/k0。全部继续保留 invalid_path_complete=false，不影响标定E/S/v/y/C_exp。不能清空缺口来放行正式成绩。

## 已测但主工作区尚未采纳的记录补丁

准备材料 `code/prepare_action_trace_patch.py`、`code/pending_action_trace/`：4个源文件绑定d94ca7d，77行新增；4份可编译，git apply --check通过，8项辅助/真实适配器检查通过。主工作区尚未应用，须等收集结束。

补丁在环境接受动作后，将实际目标及来源写入 trajectory.blp_action_target。捕获STOP前任务、确认视觉目标的父候选和Runner恢复覆盖；Observer只读身份注册表，不改变排序、注册或策略。

独立worktree `/nvme/khy/ForesightNav_exp4_tracecheck`（detached1041d41加补丁）已完成两次排除于拟合的固定2-MON GPU3验证及219次关闭模式匹配输入回放，所有临时进程已退出：

- 6s7/0：48动作，11条确认视觉父目标；除新增字段外，完整动作位姿与原plumbing-v2相同。
- TEEs/8：1000动作，29次检查，覆盖确认后改查其他候选及恢复覆盖、两个预算STOP。身份及初始位姿一致，独立长轨迹不同，不能声称长轨迹等价。
- 首次预检缺依赖、尚未启动GPU的失败保存在 diagnostics/action-trace-plumbing/attempt1-preflight-missing-dependency；已链接原固定依赖，不删除失败记录。

证据 `diagnostics/action-trace-validation.json`、`diagnostics/action-trace-regression.json`、两次controller/audit及 `reports/action_trace_isolated_validation.md`。正式采纳时核对4份已测SHA；若合并cross-fit代码，须重新核对/测试，不盲目覆盖全文件。

## 逐动作分析消费已完成（本次新增）

`code/action_path_metrics.py`、`code/native_candidate_gt.py` 已接入 report_variants：逐动作、逐坐标调用固定GOAT原生GT判据。主 invalid_visit_path_m 只计曾被检查无效身份中、当前实际目标坐标也GT无效的位移；invalid_identity_pursuit_path_m 另列全身份追逐上界。恢复/frontier/未检查目标分开，确认视觉父目标延续，同一位移不重复。

CPU原生复核覆盖90条完整标定与两次隔离数据：1064个候选标签、1404个恢复标签，2468次y/测地距离全部匹配，距离误差0。两次隔离数据共1048动作、4阶段的物理路径与原生指标一致，最大误差6.45e-7m。短/长无效路径分别1.000000458m、39.513657511m；长样本旧间隔法不完整，新逐动作归属完整。旧诊断嵌套保留，不修改旧标定缺口。

分析测试10+9+8项通过；复现入口 `code/verify_action_path_pipeline.py`，证据 `diagnostics/native-offline-gt-check.json`、`diagnostics/action-target-path-check.json`，说明 `reports/action_path_accounting.md`。这些都是诊断，不是CP3/Full成绩。

正式汇总拒绝旧轨迹、影子快照、参数不一致或路径不完整。汇总记录实际加载的三个分析模块SHA，四变体比较要求同一分析版本；若修正分析代码，使用同版重新汇总原始数据，无需重跑GPU。

## 下一步

1. 持续监督固定360条，不改运行源。当前必要的分析工具已具备，不再因等待而无必要扩展。
2. 三组完整结束后，运行 `PYTHONPATH=.:results/exp4/code NAVPY results/exp4/code/fit_calibration_reviewed.py`。实际执行时将NAVPY换成上述完整路径。该入口保留原正斜率MLE，仅在优化器失败时按同目标SLSQP和KKT复核重试；两条合成分支各3组×200 bootstrap已通过，真实数据尚未拟合。主翻转率以所有快照为分母，另报非空eligible率。
3. 写CP2a，审查bootstrap、混淆/可靠性图、跨组漂移及翻转率，决定是否加第4/5组。收集器目前只支持前三组；若加组，应保留现有360并明确新增8场景，不重收挑结果。若不稳定，按设计汇报cross-fit选项；运行/分析当前只支持pooled，必须先实现并测试scene-specific参数再评估。
4. 收集结束后采纳已测逐动作补丁，核对哈希、回归证据，更新运行指纹和CP2b真实参数冻结绑定。
5. `code/evaluate_sequence.py` 一次只跑一个阶段：smoke固定20条→Full400→Nearest400→Myopic400→Suffix-only400。门禁要求CP2b、固定身份、前阶段native/normalized证明及CP3行为审阅。8项门禁测试已通过，尚无实际3-MON阶段ledger。首次实际GPU阶段须持续监督。
6. CP3实际20条、两张可视化及行为审阅完成后才能Full400；CP4之后才能Nearest，全部完成写CP5四行指标、2×2效果和6对配对比较，持续同步GitHub。只有这些真实实验和全部报告完成后才能完成goal。
