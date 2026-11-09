# 标定与评估源码的分别绑定

a3标定继续保留eaa29e5的160项固定输入。后续评估仅允许采用03242fd被动日志补丁：blp_observer的SHA必须与已测版本一致，并新增固定SHA的test_blp_identity；其他159项输入均须保持原值。评估161项指纹与原标定160项指纹分别记录，校验器拒绝通过改写旧ledger获得相同指纹。

当前已完成12项独立的合成绑定检查，包括正常映射、额外代码变更、错误观察器字节、缺失证明、过期参数哈希、少一项测试、回归总数相同但样本覆盖不同、证据路径越界等。合成文件全部在TemporaryDirectory，不能作为实际CP2接受证明。

新增validate_evaluation_runtime.py只有在360条完成、CPU后处理完成、当前CP2a已接受且参数已冻结后，才会在已采纳的源码上执行60项检查和219次匹配输入回归，并绑定参数、标定ledger、原CP1、当前review及实际日志SHA。它不会修改源码、拟合或冻结参数，也不会启动native。检查失败时保留日志，缺证明不能进入CP2b。

CP2b最终绑定器仍逐项检查原51项CP1、219次原回归、完整标定和已批准数值，并额外要求实际评估源码验证。新的源码证明尚未生成，当前原生运行仍用旧观察器。CPU等待器122348绑定的11份分析脚本均未修改。

2026-11-09的实际--check返回ready=false：运行源码验证因标定未完成被拒绝；CP2b因缺当前CP2a_review被拒绝。这是尚未达到阶段要求，当前未创建新的参数或接受记录。

后续顺序：完成360及CPU提案，审查当前CP2a数值并冻结；采纳03242fd；执行validate_evaluation_runtime；执行finalize_cp2b；启动固定20条Full冒烟。CP3审查通过后再开始四组400。

[源码绑定检查](../code/pending_identity_binding/runtime-binding-tests.txt)、[验证代码](../code/validate_evaluation_runtime.py)、[准备记录](../diagnostics/evaluation-runtime-binding-preparation.json)、[被动日志修复](identity_logging_preparation.md)。
