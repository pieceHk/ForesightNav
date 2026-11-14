# CP2b：修正版本参数冻结与源码绑定

> 历史阶段报告：正文保留原阶段范围与结果；文内证据链接固定到原始提交391d7e2，不指向当前120条派生数据。

已按本次CP2a接受记录绑定cross_fit参数。参数只来自a3的2-MON系统验证标签，数值与审阅提案逐字段一致。

| 评估组 | 拟合组 | a | b0 | pi | C_exp (m) |
| --- | --- | --- | --- | --- | --- |
| 0 | 1, 2 | 48.114599083 | -12.856439210 | 0.160102302 | 1.275942260 |
| 1 | 0, 2 | 44.269786779 | -12.012253606 | 0.148021495 | 0.933749452 |
| 2 | 0, 1 | 43.680260713 | -11.910765740 | 0.151041667 | 1.375726561 |
| 3 | 0, 1, 2 | 45.247046846 | -12.236784283 | 0.152870282 | 1.194889304 |
| 4 | 0, 1, 2 | 45.247046846 | -12.236784283 | 0.152870282 | 1.194889304 |

标定源码原有51项检查及深度覆盖小试保持原CP1绑定；评估源码仅新增精确坐标身份回查及9项测试，已通过60项检查和219次关闭模式匹配输入回放。标定160项指纹与评估161项指纹分别保留，参数数值及原始标定日志不改。

参数文件SHA256：`e1d75f3c9503c7439467748fc97be6ae2ad450c429f40dc7235b1564ff7852ae`。

运行提交：`03242fda76fd6a631596a7269c8e7990c657ea30`。

[冻结参数](https://github.com/pieceHk/ForesightNav/blob/391d7e25b12171144a77cfbc41f220efe11294fb/results/exp4/meta/calibration.json)、[本次接受记录](https://github.com/pieceHk/ForesightNav/blob/391d7e25b12171144a77cfbc41f220efe11294fb/results/exp4/meta/CP2a_review.json)、[源码与参数绑定](https://github.com/pieceHk/ForesightNav/blob/391d7e25b12171144a77cfbc41f220efe11294fb/results/exp4/meta/CP2b_acceptance.json)、[拟合证据](https://github.com/pieceHk/ForesightNav/blob/391d7e25b12171144a77cfbc41f220efe11294fb/results/exp4/meta/calibration-fit-binding.json)、[被动日志源码验证](https://github.com/pieceHk/ForesightNav/blob/391d7e25b12171144a77cfbc41f220efe11294fb/results/exp4/meta/evaluation-runtime-validation.json)。

下一步运行新的固定20条Full冒烟，核查完整轨迹、逐场景参数、过程指标和两张俯视图。此报告不包含Full成绩。
