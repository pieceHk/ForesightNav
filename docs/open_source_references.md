# 开源项目与 README 参考

本轮查阅了 OpenAI curated skills 目录，并从 GitHub 官方 [awesome-copilot](https://github.com/github/awesome-copilot) 中找到 [readme-blueprint-generator](https://github.com/github/awesome-copilot/blob/main/skills/readme-blueprint-generator/SKILL.md)。本项目按该 skill 的流程核对代码、配置、目录与启动命令，使用现有架构和复现文档作为信息来源。公开展示的文字同时按本机 humanizer skill 整理。

## 同领域项目

| 项目 | README 与接口中可借鉴的做法 | PBSP 的落实方式 |
| --- | --- | --- |
| [VLFM](https://github.com/rai-opensource/vlfm) | 概览、数据目录和 Habitat 评测入口集中呈现；PointNav 调用可从策略中定位 | 首页给出数据与启动入口；导航请求显式携带传感器和目标点 |
| [ApexNav](https://github.com/Robotics-STAR-Lab/ApexNav) | 展示整体流程，分别说明模型服务、ROS 启动和数据集评测 | README 给数据流图，安装细节移至独立文档；外部导航按进程和消息接口适配 |
| [EmbodiedSplat](https://github.com/0nandon/EmbodiedSplat) | 说明重建用途、模型资源及推理方式 | 单独说明 mapping 与 semantic injection 的职责，保留实际模型来源和资源说明 |
| [ConceptGraphs](https://github.com/concept-graphs/concept-graphs) | 分别给出图像分割、建图和可视化步骤 | 为语义注入和 query 设置独立入口与数据契约，避免它们依赖整个 encoder |

## README 调整

首页改为项目简介、数据流图、数据集、环境、启动和组件入口。2/3/4-MON 的样本规模集中在一个表中，复现顺序保持“数据集 → 环境 → 启动”。历史版本表与阶段记录由文档链接承接。

展示内容只引用已有代码和验证记录。首页不添加尚不存在的论文链接或 CI 状态，也不把单 episode 结果作为完整 split 的性能结论。当前自定义 episode 的公开下载地址尚待补齐，环境文档保留独立安装验证的实际进度。

## 接口边界

新的 query 核心不读取 mapping encoder。语义索引由后端适配器生成，文本模型单独注入。semantic injection 的实际特征提取代码位于 `pbsp/semantics/`，通过统一调用接入 warmup 和增量建图。

navigation 的第一处替换边界是局部规划与 waypoint 执行。VLFM 的 PointNav 控制器可作为这一层的候选实现；其完整策略还包括探索、地图及目标选择。ApexNav 的评测入口通过 ROS 交换观测和动作，需要另写目标点、消息身份及生命周期适配。当前完成的是默认运行路径的接口拆分，VLFM/ApexNav 的完整集成尚未执行。

参考源码：[VLFM 策略](https://github.com/rai-opensource/vlfm/blob/main/vlfm/policy/base_objectnav_policy.py)、[ApexNav 评测入口](https://github.com/Robotics-STAR-Lab/ApexNav/blob/main/habitat_evaluation.py)。
