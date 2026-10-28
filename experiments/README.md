# Experiments

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
