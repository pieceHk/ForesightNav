# Dataset preparation

PBSP uses HM3Dv1 scenes and versioned ordered episode datasets. [Habitat's dataset guide](https://github.com/facebookresearch/habitat-lab/blob/main/DATASETS.md) explains access to the scene assets. The custom 2/3/4-MON episodes must match [the dataset catalog](../assets/datasets.json); their public download location is not yet listed in this repository.

## Source catalog

The current reproduction preparation expects this layout beneath `--datasets`:

```text
<dataset-catalog>/
├── HM3Dv1_objectnav_val_20scenes_3donav/
│   └── data/scene_datasets/
└── HM3Dv1_MON_balanced1200_v1_runtime_compat_20260829/
    ├── 2MON_600x2/
    ├── 3MON_400x3/
    └── 4MON_300x4/
```

Each MON directory contains:

```text
<split>/
├── val_seen.json.gz
├── content/<scene_id>.json.gz
├── dataset_manifest.json
└── source_trace.json
```

The manifest and source trace identify the released episode order. `source_trace.json` binds each runtime task to its original episode and category. Repeated categories remain distinct tasks. Keep the provided floor-compatible sequences when generating identities.

## Check and connect data

For the shallow data layout used by the component API:

```bash
python scripts/prepare_data.py --source /path/to/dataset-catalog
```

This creates `data/scenes/hm3dv1/`, `data/episodes/hm3dv1/{2mon,3mon,4mon}/` and an `identities.json` for each split. The script verifies manifest, source-trace and content hashes, scene counts and task identities before writing links.

The current experiment launcher uses `Exp/assets/datasets/`. The [installation preparation command](installation.md#connect-resources) connects the same catalog there and generates the launcher's identity sidecars.

## Task identity

The runtime key is `(scene_id, episode_id, task_idx)`. Candidate batches also carry the map revision and query serial. Episode reset clears the map; task switches retain it and reset task execution state.

Historical success labels from the source trace are excluded from the runtime identity index. The evaluator owns ground-truth goals and metrics.
