# 未检查候选别名与无效路径归属

XB场景新增18帧日志归属缺口：episode2第20至29帧，episode11第20至27帧。目标XYZ完整保留，原生身份顺序、物理轨迹及终止STOP核查通过。查询登记连续移动中心后，目标距最近登记中心分别0.9954及0.8384米，超过0.75米阈值，与先前Qa问题同类。

现有待采纳logger032在原始查询及动作回放中补齐全部18帧，登记/首次出现事件逐条相同。这是反事实日志检查，不是新的原生导航结果。episode11的旧ID随后以新ID被检查且坐标为GT无效；仅补动作ID仍会把早期接近标为未检查身份，漏计1.753523747米。

分析端现将未被检查ID的首次出现完整XYZ，与同scope已检查候选的final坐标精确关联。已检查分组保持原规则，无距离放宽，也不读取v/y决定分组。300条对比中，原生episode、决策、SR/SPL/PR/PPL及条件指标、物理路径均不变；3587条拟合关键记录和2705个已检查身份不变。原始数据未改，28帧缺口仍保留false。诊断中的无效候选路径从4544.87164补为4549.12164米，新增23条原生GT路径标注；这些不完整2-MON诊断不作正式3-MON成绩。

6项新检查及原51项分析/报告检查通过。XB两个派生回合的4阶段原生路径误差最大2.62e-7米；300条完整物理路径对应600阶段，最大误差2.54e-6米。原始null身份不由分析补写，未来仍须在完整收集及CP2后采纳logger032，并以新固定20条Full实际验证。

为在拟合前固定分析版本，原空闲CPU等待器122348经PID/start_ticks/cwd/UID核对后停止，旧state完整归档。只有旧等待器和观察结束后，才更新CPU11中的candidate_identities.py并重新排队；新PID281840继续等待360。原controller31036、GPU两工作进程和160项输入全程保持。supervisor只更新scope说明以反映动态CPU绑定，6项观察检查通过；原生51/待采纳60检查数量未变。

[XB回放](../diagnostics/unchecked-alias-XB-validation.json)、[300条对比](../diagnostics/unchecked-alias-native300-validation.json)、[交接证明](../diagnostics/unchecked-alias-adoption.json)、[实现及检查说明](../code/pending_unchecked_alias/README.md)。
