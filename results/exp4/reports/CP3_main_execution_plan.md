# Exp4四变体全量执行准备

此检查不创建CP3接受记录，不启动原生任务。当前Full门禁状态：未满足，Missing checkpoint artifact: /nvme/khy/ForesightNav_exp4_depthfix/results/exp4/meta/CP3_acceptance.json。

已在实际服务器核对四组均为固定20场景、每场景20条，共400条3-MON；数据、评估和seed=1一致，方法设置仅变体和名称不同。执行顺序为Full→Nearest→Belief-myopic→Suffix-only。Full完成后生成并汇报CP4，再自动继续其余三组。

当前GPU观测（卡号、型号、总显存MiB、空闲MiB）：

- `2, NVIDIA A40, 46068, 45368`
- `3, NVIDIA A40, 46068, 45245`

当前文件系统可用991.6 GiB。20条冒烟run目录占用110.8 MiB，按1600条线性外推约8.66 GiB；这只是当前保存策略下的规划估计，不保证后续回合或目录外输出体积。实际启动时重新读取资源状态。

原生控制器和只读观察器均应在服务器独立会话运行，观察窗口设为7天并在阶段结束时退出；以后阶段各自创建新观察会话。需要持续核对PID/start_ticks/boot_id、冻结源码和参数、候选绑定及动作目标记录；断开SSH后先重查原进程，不重复启动。

SSH CPU后处理通道设置了空CUDA_VISIBLE_DEVICES；启动原生控制器时必须移除此继承值，由控制器指定物理GPU2/3。四阶段固定源、参数与清单保持一致，不根据本次3-MON结果删选或调参。

[逐组指令和SHA](../diagnostics/CP3/main-preflight.json)、[CP3审阅材料](CP3_review.md)。
