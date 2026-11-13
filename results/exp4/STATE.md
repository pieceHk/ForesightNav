# Exp4：固定前8场景评估

<!-- exp4-current-first8-scope-entry-20261113 -->

用户在2026年11月13日将评估范围改为原固定清单前8个场景，每场景20条3-MON任务。Nearest、Suffix-only、Belief-myopic、Full各160条，共640条。四组使用同一批任务、source ID、种子和冻结参数。原Full400及范围外原生记录保留。

当前进度以[运行状态](scopes/first8/meta/progress.json)、[各组汇总目录](scopes/first8/summary/)和[当前报告目录](scopes/first8/reports/)为准。提交中的文件是带时间戳的发布快照，服务器原始记录在运行期间继续增长；不能把部分场景或已生成报告工具当作全部实验完成。

固定范围、身份及资源安排见[范围文件](scopes/first8/meta/scope.json)、[当前有效范围](meta/active-evaluation-scope.json)、[GPU分配](scopes/first8/meta/gpu-allocation.json)和[原始归档清单](scopes/first8/meta/raw-archives/manifest.json)。当前允许使用物理GPU0–7；完成场景的原始日志由CPU归档器逐文件回读核对。运行仍按Full、Nearest、Belief-myopic、Suffix-only的全局阶段顺序推进。

三组交叉拟合已由用户确认，评估期间冻结。参数SHA为`e1d75f3c9503c7439467748fc97be6ae2ad450c429f40dc7235b1564ff7852ae`，见[CP2b](reports/CP2b.md)、[冻结参数](meta/calibration.json)及[接受记录](meta/CP2b_acceptance.json)。[CP3](reports/CP3.md)已获确认；[CP4](reports/CP4.md)保留原Full400结果。完整CP5报告、两个2×2对比图和六组配对CSV，在当前四组160条均完成、路径与身份核对后生成，最终图表另做实际视觉检查。

实际运行目录为`/nvme/khy/ForesightNav_exp4_depthfix`，原生Git HEAD为`03242fda76fd6a631596a7269c8e7990c657ea30`，统一发布分支为`exp4/blp-2x2-a3`。科学源码161项指纹和冻结参数保持不变；本页只更新状态入口。每组运行的实际配置、环境和源码记录随原始日志归档。

此次范围缩减发生在已有部分结果之后，场景沿用原始清单顺序。原任务池本身经过历史1-MON结果筛选，本结果只描述固定池内消融，不作为独立盲测或未见任务泛化证据。见[数据适用范围](reports/dataset_scope.md)和[偏离记录](deviations.md)。

此前冒烟阶段的状态页已按原始字节备份至`scopes/first8/diagnostics/state-history/20261113T071343879335Z/STATE-before-first8-entry.md`；它是历史记录，当前范围以本页链接的文件为准。
