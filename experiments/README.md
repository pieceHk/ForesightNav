<!-- pbsp-v5-progress:begin -->
2026-11-04 14:50（北京时间）：用户重启服务器后，PBSP v5 已按缺失 ID 续跑。当前共 507/927 个唯一完整回合，三目标完整成功 135/507（26.627219%）；已完成 CFR 为 6/142（4.225352%）。这些均为部分结果，尚未做全量最终核查。GPU 5、6 各运行一个原生 worker，建图和导航共卡，两路均已输出动作；本次观测主机 MemAvailable 为 680.3 GiB。

新 execution_id=`cfc27ac6-0011-4fdd-9522-e44a82eb754f`，控制器 PID `46240` / start_ticks `92637`，boot_id=`4c5dbf86-de88-4a87-956c-06ccf6973cbb`；experiment run_id 仍为 `f6b213c3-d187-4cd2-9c77-60564cc57c6a`。恢复起点保留 506 回合（原分区 156/156/162，加第一次续跑 32），剩余 421 个 ID 分配给 workers/6、7、8，分别 121/153/147 个；6/7 当前使用 GPU 5/6，8 等待空位，最多并行两个。旧 worker3 的 564 号中断轨迹单独保留，不计失败；旧 worker0/1/2 的中断记录也保持原样。

206 项输入、1009 项源码、2529 项原执行受保护文件及全部 2723 项既有 worker 文件均已核对；506 个已完成回合的合法前缀、预算、动作数和建图日志已检查。部分中断场景没有写出场景级 es_status.json，保留原日志、完整回合和后续 reset 证据，最终分段审计须区分这些证据，不能伪造状态文件。6 项恢复调度测试（含 Linux 真实子进程身份检查）通过，新 worker 原生源码快照通过，原生配置仅 GPU 分配不同。

第一次续跑的 206 项原文件（另附清单）已追加至既有 history.tar 并逐文件核对，前缀为 `hm3d_mfmon/f6b213c3-d187-4cd2-9c77-60564cc57c6a/executions/f2d1649f-13e3-4624-be31-eeafc59bf3f7/results/`；当前归档 SHA 为 `e59243fc90b42deb06c9b3783e3cde04db2ba73d9957dad8d363d82aba24bdb2`。此前未部署的运行中交接草稿已被本次重启恢复取代。最终仍须 927 回合覆盖、原始动作/连续性、指标重算、场景重载几何、所有实际 worker 源码与分段来源核查；ApexNav/LCN 已核查结果保持不变。
<!-- pbsp-v5-progress:end -->

<!-- pbsp-v5-interruption:begin -->
2026-11-03 11:21（北京时间）重连确认：PBSP v5 运行 `f6b213c3-d187-4cd2-9c77-60564cc57c6a` 已于当天约 02:23 因 GPU 0 显存不足停止，父子进程均已退出。已完成 474/927 个唯一回合，三目标完整成功 131/474（27.637131%）；已完成 CFR 为 6/124（4.838710%）。这些是中断时的部分结果，未通过全量最终核查。中断回合 468、469、488 不计失败。

本次执行的 2,536 个文件（约 2.65 GB）已收入本项目 `experiments/history.tar`，逐文件 SHA 核对通过；原目录未清理或改写。当前归档 SHA 为 `6cafedc3ec4e7cee344e28669807d40e8856a3433dc5f259dd6f9c45d152a727`。1,009 项运行源码 SHA 一致，474 个已完成回合的建图日志均未发现重建错误或最终融合缺口。剩余 453 个 ID 按原分区为 153/153/147，清单与恢复约束保存在 `PBSP/debug/recovery_plan.json`。

故障前最后一条 GPU 0 资源采样中，本实验约占 19,402 MiB，卡上总占用 42,662 MiB；差额包括其他任务和驱动。报错为 CUDA 显存不足，尚无证据表明当时主机 RAM 耗尽。重连后主机可用内存约 30 GiB，低于已观察的单 worker 匿名内存峰值 31.45 GiB，交换空间已满。目前等待资源余量后再做明确缺失 ID 的分段恢复；不得直接从零重跑，不得覆盖 474 份结果或把中断轨迹拼成完整回合。
<!-- pbsp-v5-interruption:end -->

# Experiments

## Author HM3D-MFMON comparison

The MFMON adapter uses all 927 ordered three-goal episodes from 36 scenes, including the author's 288 cross-floor-required episodes. Each episode shares a 1,500-action budget and ends at its first failed goal. Maps persist across targets and reset between episodes. Evaluation uses the author's posterior distance and SR/SPL/PR/PPL definitions; ground truth is used only by the evaluator.

For this comparison, the user selected **full goal-sequence access** for PBSP's existing `ordered_suffix_mvp` planner. ApexNav and LifelongCrossNav receive goals sequentially. Report this information advantage alongside results. The methods also retain their native perception models, sensor configuration and simulation versions.

The first run, `5fd26283-c71c-4cd3-a723-5c47698e95a2`, was stopped at 715 completed episodes on 2026-11-01 after adapter and control defects were confirmed. Its results and runtime sources were verified and archived in `experiments/history.tar`; they are diagnostic records and must not be used as the PBSP comparison score. The current repair state and active run identity are recorded in `experiments/mfmon_revision.json`.

Use the project-specific entry point on the server:

```bash
/nvme/khy/conda_envs/pbsp/navigation/bin/python3.9 /nvme/khy/PBSP-open-source/experiments/manage.py start hm3d_mfmon --gpu-pairs '0,0;3,3;6,6'
/nvme/khy/conda_envs/pbsp/navigation/bin/python3.9 /nvme/khy/PBSP-open-source/experiments/manage.py status hm3d_mfmon
/nvme/khy/conda_envs/pbsp/navigation/bin/python3.9 /nvme/khy/PBSP-open-source/experiments/manage.py verify hm3d_mfmon
```

Results use the fixed directory `experiments/results/hm3d_mfmon/`. Three workers each process 309 disjoint episode IDs (`global_id % 3`), with mapping and navigation colocated on GPUs 0, 3 and 6. This leaves GPUs used by other tasks untouched. The diagnostic check uses `hm3d_mfmon_smoke/` with explicit IDs 109, 115, 285, 288, 300 and 309; these records are excluded from the full results. Starting in a nonempty result directory is refused; earlier adapter trials are preserved in `experiments/history.tar`.

`metrics.json` contains full-set and CFR aggregates, including conditional and global stage metrics. Each worker retains raw action trajectories, model and environment records, and a source snapshot. Final verification checks all canonical IDs, legal terminal prefixes, continuous trajectories, recomputed metrics, terminal geometry, and frozen source/input hashes. Do not treat a running or unverified result as final.

The `mfmon_navigation_mapping_v5` adapter restores the installed native PBSP sensor's raw metric depth and default render frustum. The old MFMON renderer clipped distant surfaces at 5 m and then replaced zero values with 0.5 m, introducing false near obstacles. Detection-point export now retains finite raw measurements beyond 5 m; the 1.5 m horizontal and 1.1 m vertical success gates are unchanged.

The repair also separates physical arrival distance from goal-map dilation, bounds the local goal region, keeps stationary recovery active during scans, and supplies the actual optical pose to mapping with pitch-correct coordinate conversion. Historical encoder features remain losslessly on CPU; only the same selected views are transferred to the GPU. MFMON mapping errors propagate to the run rather than silently accepting an incomplete map. This repair does not add a floor-transition planner or claim a particular success rate. Native sensor differences, regression checks, archived runs and diagnostic outcomes are documented in the fixed comparison report.

The remaining sections describe the original PBSP datasets and their existing launch conventions.

Run all experiments through the root `run.py`. Select task count through `configs/experiments/hm3d_{2,3,4}mon.yaml`; the algorithm and evaluation loop are shared.

`runner.py` executes episodes. `dataset.py` and `identities.py` validate task identities. `evaluator.py` owns ground-truth measurements. `metrics.py` calculates SR, SPL, PSR and PeSPL. `recorder.py`, `trajectory.py` and `visualization.py` write evidence. Preparation utilities in this directory are exposed through `prepare.py`.

## Results

Each run receives its own directory under `experiments/runs/`, or the output path in `runtime.local.yaml`:

```text
<run>/
├── status.json               # launcher and worker completion
├── resolved_config.yaml     # effective configuration
├── launch.json               # selected scene/episode/task identities
├── provenance.json          # source and resource identity
├── model_manifest.json      # checked model files
├── logs/
├── mon_metrics.json         # aggregate metrics
├── results_<scene>/         # per-episode metrics, trajectories and decisions
└── artifacts/mapping_final/ # final maps and images
```

Summarize one or more completed runs:

```bash
python -m experiments.summarize experiments/runs/<run> --output experiments/runs/summary
```

The command validates completion, task identities, environment records, timing records and recomputed metrics before producing `summary.json`, `summary.csv` and `summary.md`. Interrupted runs retain their logs and are rejected as completed results. Use `python -m experiments.summarize --help` for protocol comparison and shard merging.

To check the two environments and run one complete ordered episode:

```bash
python prepare.py check --smoke
```

`--scene` and `--episode` select an exact episode. A single complete episode is sufficient for installation and output-format validation; full-set evaluation is a separate experiment.

## Native ObjectNav

Official HM3Dv1 and HM3Dv2 validation use the same PBSP models and runtime configuration. Link an existing official split and its scene assets, then run the generated experiment:

```bash
python prepare.py objectnav --dataset hm3dv1 --source /path/to/objectnav_hm3d_v1/val/val.json.gz --scenes /path/to/scene_datasets --scene-dataset hm3d/hm3d_annotated_basis.scene_dataset_config.json
python run.py --config data/episodes/objectnav/hm3dv1/experiment.yaml
```

Preparation checks all 2,000 episodes in 20 scenes for v1, or 1,000 episodes in 36 scenes for v2. For v2, use `--dataset hm3dv2` with its source split and scene configuration. Scene files and compressed episode content remain at their original locations. Each source identity uses the scene file and its episode index; the original `episode_id` is retained as `raw_episode_id` because those values can repeat.

Native evaluation reads Habitat's success and SPL with its ObjectNav viewpoint distance, 0.1 m success threshold and 500-step budget. The shared runner records PBSP decisions and complete action trajectories. Results are written to `objectnav_metrics.json`; `coverage.full_dataset_evaluation` distinguishes the complete split from a selection made with `--scene`, `--episode` or `--limit-episodes`. Recompute a completed native result with `python -m experiments.objectnav_results experiments/runs/<run>`.

An unreachable failed episode stays in the split with SPL=0. If Habitat returns NaN for its SPL, `native_spl_status` preserves that state and `undefined_native_spl_episodes` counts these cases.
