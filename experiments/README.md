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
