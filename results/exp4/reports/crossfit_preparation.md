# 交叉拟合加载准备：隔离CPU核查

分支 `exp4/crossfit-prep` 从CP2a提交 `4994fcc` 创建，位于服务器 `/nvme/khy/ForesightNav_exp4_crossfit`。原标定工作区仍保持原源码和未冻结状态；本分支没有运行3-MON、创建正式参数文件或改变CP2a的用户选择。

新增 `pbsp/navigation/blp_parameters.py` 为运行端和分析端共用的参数解析器。它支持固定三组或五组标定：已收集组的场景排除自身整组，未收集组的场景使用合并参数；未知场景、错误分组、缺失参数集、非自标签或未冻结文件均拒绝。运行端传递原始 `request.scene`，不根据模型路径猜测场景身份。

快照记录所用参数、精确场景、被留出的组、训练组及参数文件SHA256。正式汇总逐快照核对这些字段；调度器同时核对固定folds与真实完成的标定组数，四变体比较要求同一场景参数分配。旧的pooled文件仍使用相同数值，交叉拟合提案不能被误读为pooled。

已合并此前验证的77行逐动作记录补丁，保留STOP前阶段、确认视觉目标父候选及Runner恢复覆盖。没有修改Eq.(13)、四变体开关、共享滞回或类别冷却。

CPU核查共49项通过：原核心18项、参数解析与运行初始化9项、汇总11项、调度11项。新增检查覆盖20场景选择、三/五组训练排除、未知场景、未冻结和GT文件拒绝、快照错组/错SHA/错场景及不完整覆盖。调度测试只在TemporaryDirectory中构造checkpoint文件，没有启动仿真worker或产生实际成绩。

三个已完成原生基线回合的219次匹配输入回放全部一致，分别33、85、101次。该结果证明关闭模式的匹配输入决策保持一致，不证明独立GPU轨迹逐帧相同。

隔离工作区的 `--stage smoke --check` 明确返回ready=false，原因是不存在冻结参数文件，`launches_workers=false`。首次只读检查还发现工作树未带入被忽略的固定身份清单，已链接主工作区同一清单后复查；该检查没有启动GPU任务。

证据为 `code/crossfit-preparation-tests.txt`、`diagnostics/crossfit-preparation-regression.json` 和 `diagnostics/crossfit-preparation.json`。最后一项记录已测源码SHA、主工作区指纹未变，以及没有正式参数文件或评估ledger。

待用户选择后，才能采用相应参数并更新评估源码/CP2b绑定。若选择补齐五组，继续保留主工作区原标定源码；若选择三组交叉拟合，可审阅并采纳本分支，再在固定20条Full冒烟中验证真实GPU行为与日志。
