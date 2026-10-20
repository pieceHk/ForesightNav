# Experiment protocol and results

The native entry point runs full PBSP with one shared method configuration. The three experiment presets select different ordered episode sets:

| Preset | Scenes | Episodes | Targets per episode |
| --- | ---: | ---: | ---: |
| `configs/experiments/hm3d_2mon.yaml` | 20 | 600 | 2 |
| `configs/experiments/hm3d_3mon.yaml` | 20 | 400 | 3 |
| `configs/experiments/hm3d_4mon.yaml` | 20 | 300 | 4 |

Each set contains 1,200 target instances from the same historically calibrated source pool. [Dataset preparation](datasets.md) preserves the fixed source indices, task order and scene correspondence. Installation and resource setup are in the [installation guide](installation.md); the [offline guide](offline.md) covers servers without internet access.

## Run a complete experiment

From the repository root, using the navigation interpreter:

```bash
.cache/envs/navigation/bin/python run.py --config configs/experiments/hm3d_4mon.yaml --check
.cache/envs/navigation/bin/python run.py --config configs/experiments/hm3d_4mon.yaml
```

The default seed is `1`, each task has a 500-step budget, and perception uses Detic without ground-truth semantic labels. The configuration files record the effective method and evaluation settings. Machine paths and GPU IDs live in `runtime.local.yaml`.

For a complete smoke episode, add `--limit-episodes 1`. To select an exact identity, use `--scene 4ok3usBNeis --episode 0`. A new run writes a separate output directory. Episode reset clears the map; subsequent tasks within that episode share it.

## Read the saved evidence

A completed run has `status.json` with `status: completed`. Mapping, monitor and scene-specific navigation process records contain successful exit codes. The launcher checks that the completed episode and task counts match its selected identities.

| File | Recorded content |
| --- | --- |
| `launch.json` | Effective configuration, model input specification and selected task identities |
| `provenance.json`, `source.patch` | Code revision, working-tree state, worker settings and checked resource identities |
| `dataset_manifest.json` | Fixed dataset version and scene-file hashes |
| `model_manifest.json` | Model manifest verified at launch |
| `environment_locks/`, `*_environment*.json` | Lock snapshots, actual interpreter, installed package versions and package/source paths |
| `results_*/.../episodes/<scene>/<episode>/identity.json` | Ordered source identities, pose convention and discrete action space |
| `results_*/.../episodes/<scene>/<episode>/trajectory.jsonl` | One pre-action world pose and issued action per accepted environment step |
| `results_*/.../per_episode_metrics.json` | Raw task measurements, stop reasons and derived episode metrics |
| `results_*/.../timing_log.csv` | Scene, episode, task, frame and per-step elapsed time |
| `mon_metrics.json` | Aggregate metrics for the selected complete episodes |

Current native runs declare environment and trajectory schemas in `provenance.json`. The summary checks copied lock hashes, installed package versions, interpreter/device bindings, and every trajectory frame against raw timing, task order and STOP reasons. Package versions and locks also enter the merge protocol; machine paths and GPU IDs may differ.

Earlier runs may lack these records or the model manifest. Their raw metrics can still be compared. Merging requires a saved checked model identity; runs without the newer recording schemas retain metric-only validation, shown by `environment_identity_checked: false` and `trajectory_checked: null` in the JSON report. Historical records are not backfilled.

## Metric definitions

For `M` episodes, each with `n` ordered tasks, let `S_ij` be target success, `P_ij` the recorded actual path length and `D_ij` the evaluator's recorded shortest-path length. The implementation is [experiments/metrics.py](../experiments/metrics.py).

- **SR** is the fraction of episodes whose tasks all succeed.
- **SPL** averages each episode's all-task success multiplied by `sum(D_ij) / max(sum(D_ij), sum(P_ij))`.
- **PSR** is `sum(S_ij) / (M * n)`.
- **PeSPL** is the sum of recorded target SPL contributions divided by `M * n`.

When actual and shortest path lengths are both zero, efficiency is `1`. Unreachable shortest paths retain an explicit status; affected path-efficiency totals remain null, and finite-path coverage is reported separately. All target contributions remain in the PSR and PeSPL denominators.

## Recompute and compare

The summary command checks completion, process exits, configuration and identity hashes, task order and raw timing coverage. It recomputes episode and run metrics and compares them with the saved values before producing a table:

```bash
python3 scripts/summarize.py outputs/<run> --output outputs/<summary>
```

It writes `summary.json`, `summary.csv` and `summary.md` in a new directory. Multiple run arguments produce separate comparison rows. Each row distinguishes a complete dataset from a selected subset; an asterisk marks a run launched with working-tree changes. CSV includes recording coverage and retains full numeric precision; unavailable path-efficiency values appear as empty CSV cells and `n/a` in Markdown.

For completed, disjoint shards of the same experiment:

```bash
python3 scripts/summarize.py outputs/<part-0> outputs/<part-1> \
  --merge --output outputs/<merged-summary>
```

Merging requires matching code, method, dataset, seed, model and evaluation identities, clean source records and nonoverlapping episodes. GPU IDs and machine paths may differ. The merged report retains its source runs and reports full-dataset coverage only when the combined selection covers the dataset manifest.
