# Exp4 接续状态

目标仍在进行。CP1 已完成，首轮标定因记录器错误停止并归档，修复后的 attempt a2 已启动，尚未进入 3-MON Full 冒烟或全量评估。不要重复启动控制器，也不要把本页里的 PID 当作当前存活证据。

## 工作区与连接

- 本机材料目录：`/Users/hikong/Documents/ChatGPT/30服务器/ForesightNav-exp4-20261107`。
- 实验服务器目录：`/nvme/khy/ForesightNav_exp4`；结果：`results/exp4`。
- 服务器最新代码来自原 `/nvme/khy/PBSP-open-source` 的实际工作树，保存为 `7c7e9f4`，原工作区没有为本实验修改。
- 已推送 GitHub 分支 `exp4/blp-eq13`；实际运行代码为 `d94ca7d`（含脱困 waypoint 记录器修复），随后分析工具提交 `0e1ed67` 不修改冻结运行代码。
- SSH 复用：`ssh -S /tmp/codex-onemap30-20261107.sock 10.29.150.30`。普通直连本轮被重置，复用连接可用。不要输出密码或私钥。
- GitHub 用户已添加服务器专用可写 deploy key。仓库 `core.sshCommand` 已设置，使用 `/home/htz/.ssh/foresightnav_exp4` 及 `/tmp/exp4_github_proxy.py`；代理复用服务器 `127.0.0.1:19181`。推送已实际验证，不需再次向用户索要许可或凭据。

## 已确认的协议

用户在本聊天确认：保留现有 `semantic_score` 作为 E，论文按所选高斯的 codebook 加权文本分数描述；四个变体保留共享 1 m 滞回和类别冷却，分别报告无约束排序、滞回后排序和最终下发的偏离率。原文在 `meta/protocol_decisions.json`，不需要再问。

参数只使用固定 2-MON 的系统最终验证标签拟合。GT 只用于诊断，调用原 GOAT measures 的副本在候选坐标计算，另保存检查时 agent 的标签。不得用 3-MON 成绩选超参数。

## CP1 证据

- 初版 16/16 单元测试通过；记录器修复后 18/18 通过。
- 原版固定 `4ok3usBNeis/{0,1,2}` 三个完整 3-MON episode 已结束。219 次规划调用，分别 33/85/101，修改后关闭模式逐次一致，覆盖已完整。
- 基线运行：`/nvme/khy/PBSP-open-source/experiments/runs/20261107_233046_569437_hm3d_3mon_full_pbsp_exp4-regression-before`，独立源码 `/nvme/khy/ForesightNav_exp4_baseline`。
- 原输入在 `results/exp4/code/regression_inputs/*.pkl.gz`；验证入口 `python -m experiments.exp4_regression --require-complete`。
- 两次固定 `6s7QHgap2fW/0` 的 2-MON 管线试跑均结束、排除于正式拟合。v2 正常结束，48 步、6 快照、2 最终验证；中位规划 7.94 ms，P95 9.42 ms；诊断修正前后的完整轨迹完全一致。
- v2 原始运行：`results/exp4/runs/20261107_234611_495499_exp4_calibration_plumbing-v2`。
- 一个 plant 候选被系统确认但 GT 无效，距目标 6.809 m；couch 候选有效。这是两条管线样本，不是整体偏差统计。
- CP1 报告和差异见 `reports/CP1.md`、`deviations.md`，已同步 GitHub。

## 当前正式标定

控制器 `python -m experiments.exp4_collect`，attempt a2 初始 PID `1979824`，boot_id `4c5dbf86-de88-4a87-956c-06ccf6973cbb`。先看真实 `/proc`、JSON 状态和轨迹更新时间，再判断是否仍存活。

状态文件：`results/exp4/meta/calibration_runs.json`。该文件包含 157 个源文件、配置和固定清单的 SHA256，以及所有场景子任务。不要修改被绑定的任何源文件、配置、folds、episode 清单或协议决策；控制器会立即停止。

首轮（已停止、不可恢复使用旧 PID）的两项如下，新的 a2 目录及 PID 必须从当前 ledger 读取：

- fold0 `6s7QHgap2fW`，GPU2，原生 launcher PID `1977675`，运行目录 `results/exp4/runs/20261107_235932_493509_exp4_calibration_calibration-fold0-6s7QHgap2fW`。
- fold0 `TEEsavR23oF`，GPU3，原生 launcher PID `1977676`，运行目录 `results/exp4/runs/20261107_235932_578022_exp4_calibration_calibration-fold0-TEEsavR23oF`。

每卡把原建图和导航进程放在同一卡上（GPU2/2 或 3/3），每场景完整 30 条 2-MON。第 0、1、2 组依次收集，共 12 场景、360 条；组内两卡并行。每场景结束做覆盖、错误和诊断标签定义审计。完成三组后控制器停止于 `completed_three_folds_pending_CP2a`，不会启动评估。

首轮在 TEEsavR23oF/0 第108次决策因 `Verification candidate is not aligned with latest batch` 失败。该点是导航的临时脱困 waypoint，没有查询 E，记录器错误地按查询候选绑定。控制器按预期终止另一张卡。旧控制器/launcher 均已退出，整轮（含 1 个完整 episode 和两个未完成前缀）归档并排除拟合，两场景从头重收，不按结果筛选。

归档 ledger 为 `meta/calibration_runs.attempt1.json`，原接纳清单为 `meta/CP1_acceptance.initial.json`，不可更改的文件哈希在 `meta/calibration-attempt1-archive.json`。原日志和运行目录保留原路径。

修复在实际 verification_event 上记录 `candidate_source`。临时恢复点记为 `noncandidate_verification`，保留 v/GT、不进入候选 belief 拟合。仍受冷却保护的旧查询候选若离开最新 batch，从已记录 query 历史取 E，注明 `historical_query_record`；真正没有来源的查询候选写 `blp_errors.jsonl` 触发控制器停止。没有编造 E，没有修改导航动作或数值超参数。

a2 的接纳清单已绑定新代码和相同 folds/episode 配置；日志为 `logs/calibration-controller-a2.log` 及 `logs/calibration-a2-fold0-{scene}.controller.log`。必须先确认实际 a2 重放越过原第108步且正确记录非候选恢复点，再认为这次问题解决。

2026-11-08 00:11 已实际验证修复：TEEsavR23oF/0 前107步轨迹与首轮完全一致；第108步首次非查询脱困验证正确落盘；新运行已到212步，26条非查询验证，影子错误0，157项源码/配置/清单哈希未变。证据 `meta/recovery-waypoint-fix-audit.json`。该故障已经通过真实同轨迹前缀验证，不需重复跑这个修复检查。

a2 当前 launcher PID 为 `1979825`（GPU2）及 `1979826`（GPU3），控制器 `1979824`。运行目录分别为 `runs/20261108_000733_439246_exp4_calibration_calibration-a2-fold0-6s7QHgap2fW` 和 `runs/20261108_000733_533484_exp4_calibration_calibration-a2-fold0-TEEsavR23oF`。仍应从 ledger 和 /proc 核对实际存活、进度与错误后继续监督。

固定 folds 和 600/400 条原始身份清单在 `meta/`，20 条 Full 冒烟每场景一条，已预先固定。

## 后续操作

持续监督当前标定：检查真实 worker、逐 episode 指标/轨迹、`blp_events.jsonl`、`blp_snapshots.jsonl` 和 `blp_errors.jsonl`。先确认首批真实帧和事件正常，不因模型加载阶段暂时无轨迹误判停止。代码只允许使用 GPU2、3，不碰 GPU4 的 exp2 和 GPU5/6/7 的 exp1。

完成三组后运行 `python -m experiments.exp4_fit`，生成各组/合并拟合、200 次 episode bootstrap、混淆矩阵、可靠性图、决策翻转率和未冻结的参数提案。该脚本未在正式数据上执行过。若一类标签、完全分离或 bootstrap 不可辨识过多，会要求更多数据，不捏造参数。

按设计写 CP2a，判断是否补齐第 4、5 组。当前收集器只支持前三组；若需补齐，应另写明确的恢复/补充清单，保留原结果，不重跑已完成场景。若不稳定，应提供交叉拟合选择并汇报。

之后 CP2b 冻结 `meta/calibration.json`；再做 CP3 Full 固定20条；审阅实际行为和日志后再 Full400 → Nearest400 → Myopic400 → Suffix-only400。下文所列评测调度器已完成离线检查，仍未启动实际评估。

## 后续新增的分析准备（2026-11-08 00:45）

当前 a2 已完成12/360（首两场景各6），控制器和两项 launcher 的 /proc 命令身份真实匹配，哈希未变，影子错误0。最近每项10个快照的在线选择与离线重放一致。用新只读观察工具复核，不要从这个历史数字推断实时进度：

`cd /nvme/khy/ForesightNav_exp4 && PYTHONPATH=. /nvme/khy/conda_envs/pbsp/navigation/bin/python results/exp4/code/watch_calibration.py`

该工具输出 `/proc` 实际身份、boot_id、冻结文件检查、完整 episode 数、v 标签数、最近快照重放和错误字节数，并写入 `meta/current-live-audit.json`。读取 JSONL 时忽略未写完的尾行，不会因瞬时观察失败重启任务。

统计分析优先使用 `PYTHONPATH=.:results/exp4/code python results/exp4/code/fit_calibration_reviewed.py`（原导航 Python）。它调用原 CP2 driver，只在原优化器返回失败状态时，按同一个正斜率 MLE 用 SLSQP 重试，并做约束 KKT 检查；不改运行源。合成数据上两条分支各三组200/200 bootstrap均完成，真实参数未拟合、未冻结。输出记录重试次数；一类标签和完全分离不会被伪造为有限参数。

该分析入口将决策翻转率的主分母设为设计所写的所有快照：空根时 None 与 None 相同、不翻转；另给 `eligible_rate` 和非空分母，防止掩盖条件翻转率。这是在真实拟合前确定的口径。相关测试为 `code/test_fitter_statistics.py`。

`code/report_variants.py` 已具备原生指标复用、PPL、条件率、无效访问、三层偏离率、延迟分布、2×2 effects 和配对比较，5项黄金案例及真实三回合/管线样本检查通过。正式输出需要冻结参数和 episode 清单，拒绝错误变体、缺失参数证据及结构性错误。若无效访问路径归属有歧义，会拒绝正式发表汇总，需要逐动作审计或在 CP3 前补齐路径记录。不要把这个尚未用于正式评估的工具当作 CP4/CP5 已完成。

`code/plot_blp_episode.py` 可在CPU加载场景navmesh（已确认agent_radius=0.1m），画候选B、计划及实际下发和真实轨迹。管线诊断图已渲染检查，明确是影子建议与旧版轨迹，不是Full成绩。正式图要求active BLP，并仅选择实际下发快照。

`meta/calibration-overlap-audit.json` 表明：360条标定与400条评估共享12场景、720个底层source episode；其中12条3-MON的前两目标顺序与标定相同。没有读取评估结果或修改分组，CP2报告应明确这不是独立场景/目标的泛化测试。

2026-11-08 00:49 最近核查已完成14/360，首两场景各7条。两项launcher和控制器真实进程均存活，运行源指纹不变，影子错误0。所有实时数字都应通过 `watch_calibration.py` 重新获取。

2026-11-08 00:53 最新核查15/360（6s7QHgap2fW为7，TEEsavR23oF为8），控制器/launcher仍匹配真实进程，指纹未变，错误0。分析工具最新已推送提交 `6db5ebd`，实际运行代码仍为 `d94ca7d`。汇总工具现为6项黄金案例通过，补充覆盖了已确认无效目标后继续接近、最后因预算STOP的路径计数。CP2统计脚本最新合成测试已含空根快照，两分支仍为每组200/200。

可以在 `results/exp4/code/` 新增分析脚本和报告，不会改变运行源码指纹；当前正式标定期间不要新增或修改 `pbsp/`、`experiments/` 或 `configs/` 下文件。需要修复运行代码时先停止本控制器、保留尝试并记录差异，再明确重收哪些数据。

## 后续评测调度器（2026-11-08 01:13）

最新只读核查完成19/360（GPU2场景10条，GPU3场景9条），真实进程匹配，冻结文件未变，错误0；这是历史采样，不代替实时检查。

`code/evaluate_sequence.py` 已上传，8项检查通过，`report_variants.py` 的6项检查仍通过。当前执行 `PYTHONPATH=. python results/exp4/code/evaluate_sequence.py --stage smoke --check` 返回 `ready:false`，原因是标定未完成；没有启动评估任务或创建评估 ledger。

每次调用只运行一个阶段：`smoke`（Full固定20条）、`full`、`nearest`、`myopic`、`suffix_only`（各固定400条）。每卡一个原生launcher，GPU2/2、3/3。冒烟汇总保存在 `diagnostics/CP3/`，正式汇总保存在 `summary/` 与 `logs/`。阶段结束审计全部20个原生场景、固定身份、source episode及目标顺序，再生成报告；不会用部分完成数据代替完整结果。

启动冒烟前需要已结束的3组或5组标定、以自标签拟合的冻结参数，以及绑定源码指纹、参数SHA256和标定ledger SHA256的 `meta/CP2b_acceptance.json`。全量前还需完整原生及归一化冒烟证据，以及 `meta/CP3_acceptance.json` 中的实际行为审阅。Nearest前需CP4已汇报且绑定Full汇总哈希；其他变体依次等待前一完整400条。异常SR（0或超过90%）要求带汇总哈希的复核记录。

当前运行加载器与正式汇总只支持pool参数。若CP2选择交叉拟合，调度器会明确拒绝启动；必须先实现并验证按场景选取参数，不能把交叉拟合文件中的pool字段悄悄用于所有场景。不要为此修改尚在标定的运行源。

评测调度器自身、运行源与冻结参数在评测期间检查哈希。遇到错误只停止其拥有的launcher，保留原始数据和停止原因。已有阶段ledger时不会重复启动；`--finalize-only` 只用于完整原始运行的汇总重审，不恢复未完成实验。调度器尚未经过真实GPU阶段验证，首次冒烟仍需持续监督。

01:20 最近核查22/360（12+10），动作数6108/7113仍增长，错误0，指纹未变；已同步本机 `meta/current-live-audit.json`。调度工具已提交并推送 `d9aebc2`，运行代码仍为 `d94ca7d`。本机 `meta/current-data-quality-audit.json` 的已完成样本中，两类v均存在，E有变化、N_views为0–5，查询E均有current_batch来源；仅管线质量检查，没有提前拟合或选参数。后续主要工作是等待固定三组收齐并监督异常，不要继续无必要扩充分析工具，也不要因长时间运行而重启正常任务。

01:38 本轮为 verified wait：通过现存控制器1979824及launcher1979825/1979826的真实/proc命令反复核查，标定推进到27/360（15+12），动作数7590/8841，影子错误0，源指纹未变，最近快照仍可离线复现。两项长回合曾接近每目标500步预算，随后正常结束，不是卡死。本轮两段临时只读观察会话均已正常退出；正式控制器和GPU任务持续运行，没有新增或重启实验。尚无3-MON评估ledger、冻结参数或CP2报告。

01:59 又一轮 verified wait：原控制器及两项launcher的/proc身份仍匹配，完成33/360（18+15），动作数9391/10721，源指纹未变、错误0。本轮额外核对较早27条完整回合，共370次candidate_checked与370条verification_final一一完成，未发现完整回合漏最终标签。临时20分钟只读观察会话2904已正常退出，正式任务继续；没有源码变更、提前拟合或3-MON启动。最新实时核查与ledger已同步本机meta目录。

02:43 持续监督到50/360（两场景各25），原控制器与launcher均真实存活，源指纹未变，动作数13137/14821、错误0。按这两个场景截至目前的吞吐，三组标定总用时约20小时，仅粗略估计，不能据此判断其他场景或评估阶段用时。原生控制器仍每10秒核查源与错误，主代理间隔约4–5分钟检查真实进程和数据增长；当前无需重复启动或调整运行代码。

复核通用交付字段时，补齐分析汇总CSV原先遗漏的 `P_C4_given_C1C2C3` 列；3-MON该列为空。临时合成格式检查验证四行输出、空值与两位小数格式，输出已删除，只保留 `code/csv-format-check.json` 的检查结果；没有生成或冒充任何正式实验成绩。此次修改仅在results分析工具目录，未改变导航源。

02:58 本轮已持续监督约一小时，最新52/360（各26）。控制器1979824及两项launcher1979825/1979826仍在真实/proc中匹配，动作数14460/16181持续增长，源指纹未变、影子错误0。当前没有临时长期观察SSH会话，仅原控制器/实验本体在运行。CSV字段修正与当时的状态已推送 `6da1b5e`，运行代码仍为 `d94ca7d`；本机meta核查更新到02:58。继续等待首两个场景各完成30条及其原生覆盖审计，随后控制器会自动开始同fold其余两个场景，不要手动重复启动。尚未到CP2，禁止提前冻结或运行3-MON。

## 首批场景完成与待修复的过程记录（03:25）

首两场景均完成30条，共60/360。原launcher1979825/1979826已按正常流程退出，不能再用它们判断当前卡上任务。控制器1979824持续存活；当前GPU3场景QaLdnwvtxbs launcher2003961，GPU2场景cvZr5TUy5C5 launcher2004691，两者真实/proc身份匹配，动作已增长至273/81，错误0。具体新运行目录从当前ledger读取。

独立复核了全部60份实际identity.json，manifest SHA、source episode与目标顺序均匹配冻结清单；完整轨迹和终止STOP数与原生评估一致。802次候选检查对应802条最终标签，3547个影子快照，无记录错误，源码指纹仍未变。证据：`meta/completed-calibration-scene-audits.json`、`reports/calibration_first_pair_audit.md`。

**CP3前必须处理的真实缺口**：完整场景诊断发现3处“GT无效查询候选确认后又检查其他候选”的路径归属不明。当前分析工具正确保留invalid_path_complete=false，正式汇总拒绝不完整过程路径。不是标定E/S/v/y或C_exp错误，不能为此重启当前收集。收集结束后、CP3之前必须补齐被动逐动作实际导航目标及来源记录，保留STOP前任务索引并纳入Runner恢复动作替换，逐帧与原生轨迹对齐计算路径，不再依赖“最后一次确认到STOP”的推断。之后重新检查旧模式回归、更新源码指纹及CP2b绑定；四个变体同样记录。该源码修改尚未实现，不能仅清空分析器的gap列表来绕过。

已定位位置：Navigator的 `frontier_global_pose` 是本次动作实际使用的查询／override目标；`last_action_metadata`在STOP后可能已切换self.current_task_idx，因此须捕获STOP前的task和目标。Runner在agent.act之后还可能把动作改成TURN_RIGHT并设置恢复目标，应在最终env.apply_action前记录有效动作。当前 `trajectory.jsonl`只含action/pose，没有目标身份；仅靠刷新节奏的scsp_decisions不能精确重建每次候选切换。证据案例TEEsavR23oF/8/k1：确认t555、下一检查t601、STOPt997。

03:27 当前62/360：QaLdnwvtxbs已完成2条、动作534；cvZr5TUy5C5动作326，尚未完成首条。两个新launcher均真实存活，控制器存活、源指纹不变、错误0。可复用独立审计入口为 `PYTHONPATH=.:results/exp4/code python results/exp4/code/audit_completed_calibration.py`，已对两个完整场景执行成功；802次检查与最终标签按(scene,episode,k,candidate_id,t)多重集合严格匹配，不只是计数相等。审计脚本、报告和差异已推送 `5ae9991`；运行代码仍为d94ca7d。本轮为真实场景完成、独立完整审计与缺口发现的进展，未启动3-MON、未改变运行源码。

04:16 本轮为 verified wait，持续核查约50分钟后完成91/360，超过前三组收集目标的四分之一。QaLdnwvtxbs为23/30（launcher2003961，动作5171），cvZr5TUy5C5为8/30（launcher2004691，动作4897）；两项真实/proc身份匹配，控制器1979824存活，源指纹未变、影子错误0。源配置与数值口径未改，没有新启动的额外实验、没有提前拟合或3-MON运行。最新live audit与ledger已同步本机。仍等待固定三组完成，之前3处过程路径记录缺口必须在CP3前修复，不能因持续等待而遗忘。

## 尚未应用的逐动作记录补丁（04:23）

当前94/360（30+30+26+8），真实控制器和两项新launcher仍存活，源指纹未变、错误0。运行源码没有变化。新增准备材料在 `code/prepare_action_trace_patch.py`、`code/test_pending_action_trace.py` 和 `code/pending_action_trace/`。生成器只写源文件副本和diff，4个源文件SHA256绑定到d94ca7d；4份提案可编译，`git apply --check`通过，8项独立测试通过。没有执行git apply。

方案将可选 `blp_action_target` 直接附在原生trajectory每行，沿用其“环境接受后才记录”的保证。Navigator捕获STOP前任务和实际查询／override目标；确认视觉目标后保存父候选，避免GS刷新把旧视觉目标的路径误归给新候选。Runner如实标注实际恢复动作，Observer只读注册表绑定身份；没有E/S、注册、排序或验证算法调用的改变。关键新发现是stop_by_seem分支可继续追逐确认的SEEM目标，不一定追逐最新GS候选，因此仅记录frontier_flag是不够的。

8项测试仅对提案的辅助函数和真实TrajectoryEnvironment适配器在临时模拟环境验证：描述复制、来源分类、只读身份匹配、确认父身份延续、恢复覆盖、STOP原阶段、拒绝动作不落盘、帧任务匹配及旧模式行兼容。没有验证完整Navigator、GPU行为或正式路径统计；不能把准备补丁当成运行修复已完成。

仍须等标定收集结束后再审查和应用，重做219次旧模式回归并更新源指纹及CP2b绑定。正式汇总消费该字段、按原生GT判据处理同一候选重复检查／标签变化并计算逐动作路径的部分尚未实现。旧数据的3处gap继续保留，不得清空或宣称已修复。README明确这些限制；基准及提案的完整副本不必入Git，生成器可按SHA从当前源码或d94ca7d恢复，diff和测试需同步。

## 隔离GPU验证已完成（04:58）

正式标定104/360，前三个场景各30条已完整结束，cvZr5TUy5C5为14/30（launcher2004691，动作8718），控制器1979824真实存活，157项源码指纹未变、错误0。独立完整审计覆盖90条，1033次候选检查与最终标签逐项匹配。QaLdnwvtxbs新增1处旧式路径归属缺口，前三场景共4处；都保留，不影响标定E/S/v/y/C_exp。

利用GPU3组内等待，在独立worktree `/nvme/khy/ForesightNav_exp4_tracecheck`（detached1041d41）应用了77行被动记录补丁，正式主工作区没有应用。`code/run_tracecheck.py`只允许固定2-MON身份，在独立diagnostics输出，每2秒核查主收集器和GPU3是否被收回，预设900秒上限，只终止自己拥有的试跑。两个试跑及保护程序都已正常退出；没有仍需等待的临时exec会话。

短管线6s7QHgap2fW/0：48条动作、39查询/9frontier、11条确认视觉父目标记录，全部帧号、STOP前阶段和身份解析正确；除新增字段外的完整动作位姿与原plumbing-v2逐项完全相同。证据 `diagnostics/action-trace-plumbing/audit.json`，运行目录从其controller.json读取。首次预检因worktree缺少原Habitat链接失败，未启动GPU运行；已链接原固定依赖（habitat-lab、weights、superglue），失败记录和哈希保留在该目录attempt1-preflight-missing-dependency下。

长验证TEEsavR23oF/8：1000条动作、855查询/67恢复/78frontier、29次候选检查及最终标签；所有查询身份解析及验证身份匹配通过，1次验证后被Runner替换为恢复动作已正确区分。实际覆盖t547确认无效候选1:3、t585改查1:12，两个max_steps STOP在500/1000帧正确保存0/1阶段。新旧运行身份、初始位姿相同，整条轨迹不同；这是独立调用，不能声称长轨迹等价。证据 `diagnostics/action-trace-confirmation-switch/audit.json`。

隔离提案的关闭模式回归也已完成：219次匹配输入调用全部一致、完整覆盖3个原生基线回合。证据 `diagnostics/action-trace-regression.json`；原输入与regression_source只在worktree建立只读链接。当前已测4份源hash及主冻结指纹证明见 `diagnostics/action-trace-validation.json`。详见 `reports/action_trace_isolated_validation.md`。

下一步仍是继续收齐固定三组；可在不改主源码的前提下实现正式汇总对新轨迹字段的消费。需要区分查询/恢复/frontier、确认视觉父目标、重复检查及同一空间身份的GT标签变化，不能直接把整个ID的所有路径都当无效而忽略混合标签。现有正式汇总仍拒绝旧数据的不完整路径。正式采纳源码时核对与已测hash一致；若合并cross-fit等新代码，需要重新验证，然后更新CP2b绑定。两次GPU验证均排除拟合、不是Full BLP/CP3，参数仍未拟合或冻结。
