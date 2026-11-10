# Exp4 接续状态（2026-11-10）

修正深度单位后的a3标定360条、12场景完整审计及600次bootstrap已经完成。新CP2a建议三组交叉拟合，具体数值与旧批准方案不同，当前等待新数值确认。参数未冻结，新的20条Full冒烟与四变体主评测均未开始；任务尚未完成。

服务器权威运行目录：`/nvme/khy/ForesightNav_exp4_depthfix`，原生提交`eaa29e5f8f46782554d31226a8d619ec2534c34a`。收集期间160项输入及CPU11脚本绑定保持。报告工作区：`/nvme/khy/ForesightNav_exp4_reports`，GitHub分支`exp4/depth-reports`。本机镜像在`/Users/hikong/Documents/ChatGPT/30服务器/ForesightNav-exp4-20261107/depthfix/results/exp4`。

12个原生场景returncode均为0。原控制器31036已正常结束；原CPU281840按唯一流程依次完成audit、GT、archive、fit、report，终态为completed_pending_CP2a_review。观察CLI14229/60860均正常结束0，不要重复启动收集器或拟合。未来运行前重新检查GPU2、3资源。

360条身份、source目标顺序、轨迹和终止STOP检查通过；原生GT复核包含3849次候选检查和7220次恢复点检查，标签与距离全一致。原始28帧路径归属缺口保留在Qa及XB两场景，最终两场景没有新增；不能把这些标定轨迹当作完整正式过程指标。归档六份事件/影子快照约12MB，原文件保留，还原SHA全部通过。

三组翻转率0.4061%–1.0384%，a/b0/pi区间有共同交集，C_exp没有，因此按2.5规则交叉拟合。[完整CP2a](reports/CP2a.md)、[四套参数与下一步具体提案](reports/CP2a_next_step.md)、[未冻结JSON](meta/calibration.crossfit.proposed.json)及三张报告图已准备。2条未完成空候选激活不填0，C_exp取361条完成事件均值。MLE两次数值失败按相同目标重试，600次bootstrap成功，见拟合JSON。报告实际--check为ready=true。

用户已确认的三组继续、E保留单高斯codebook加权文本分数、四变体共享1m滞回/原生类别冷却及分层报告偏离率继续有效；GitHub写入已授权。不重问这些选择。新四套数值尚未批准，现无新的CP2a_review.json或calibration.json。

新分析candidate_identities.py为a311b37d45d59f379ce00b58d1350b5eb308da0b9a24d34ad2facdb495d9991d，按同scene/episode/task/resample的精确完整XYZ去重并关联unchecked旧ID，不依据GT或v/y选分组。原始日志不改。被动logger源码03242fda76fd6a631596a7269c8e7990c657ea30仍在exp4/identity-logfix分支，未采纳到实际运行源。

确认本次新数值后，冻结参数与证据，才采纳logger032、在实际评估源跑60项检查和219次关闭模式匹配输入回放并生成CP2b。此前隔离验证不能代替实际源验证；原CP1/标定160项指纹不得改写为新161项。随后GPU2、3固定20条Full冒烟，提交全指标/过程日志/两张指定俯视图，CP3确认后才进入Full400。

完整剩余目标：Full400→Nearest400→Belief-myopic400→Suffix-only400，完成CP4/CP5全部导航/条件/过程指标、SR/SPL因子效应、六配对成功计数、原日志、图和GitHub同步。不得把标定完成当作整个目标完成。

标定和评估共享12场景、720个底层source episode及12条相同有序前缀，不作独立场景或目标泛化主张，不使用3MON结果拟合参数。用户说会协调外部CPU线程；未据此认定协调已生效。旧a2标定和旧Full冒烟已因深度单位问题整批排除，保留在archive，不能混入本次a3结果。此前旧接续说明可由Git历史恢复。
