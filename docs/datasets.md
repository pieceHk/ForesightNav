# Dataset preparation

PBSP uses HM3D v0.1 scenes with the official HM3Dv1 ObjectNav annotations. The repository includes fixed task orders and provenance metadata for all three MON sets. The preparation script reconstructs their original episode files and checks their SHA-256 hashes against [the dataset catalog](../assets/datasets.json).

## Scene assets

Obtain these validation archives through [Matterport's HM3D v0.1 download instructions](https://github.com/matterport/habitat-matterport-3dresearch#downloading-hm3d-v01):

- `hm3d-val-habitat.tar`: basis meshes and navigation meshes.
- `hm3d-val-semantic-annots-v0.1.tar.gz`: semantic meshes and labels.
- `hm3d-val-semantic-configs-v0.1.tar`: annotated scene configuration.

Arrange the extracted files as follows. Pass the directory containing `hm3d/` to `--scenes`:

```text
scene_datasets/
└── hm3d/
    ├── hm3d_annotated_basis.scene_dataset_config.json
    └── val/
        └── 00843-DYehNKdT76V/          # one of the selected scenes
            ├── DYehNKdT76V.basis.glb
            ├── DYehNKdT76V.basis.navmesh
            ├── DYehNKdT76V.semantic.glb
            └── DYehNKdT76V.semantic.txt
```

The catalog lists the 20 selected scene IDs. Scene assets and derived annotations retain the [Matterport academic-use terms](https://matterport.com/legal/matterport-end-user-license-agreement-academic-use-model-data); PBSP's MIT license applies to its own code.

## Prepare all episode sets

Prebuilt episode bundle (2/3/4-MON): `<DATASET_DOWNLOAD_URL>`.

From the repository root, using Python 3.9 or newer:

```bash
python3 scripts/prepare_data.py --scenes /path/to/scene_datasets
```

This uses Python's standard library. It downloads the selected validation members from the [official ObjectNav archive](https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/hm3d/v1/objectnav_hm3d_v1.zip) linked by [Habitat's dataset guide](https://github.com/facebookresearch/habitat-lab/blob/main/DATASETS.md). Byte-range requests fetch about 2 MB of annotations. The archive version and every selected source file are pinned in the catalog.

The script combines those annotations with the fixed recipes in [assets/episodes](../assets/episodes), preserving source indices, episode starts and task order. It applies the recorded `sofa → couch` and `tv_monitor → tv` runtime aliases and verifies all 60 generated scene files before connecting them. It creates:

```text
data/
├── scenes/hm3dv1/                 # link to scene_datasets/
└── episodes/hm3dv1/
    ├── common/                   # fixed plans and source-pool provenance
    ├── 2mon/                     # 600 episodes, 1,200 targets
    ├── 3mon/                     # 400 episodes, 1,200 targets
    └── 4mon/                     # 300 episodes, 1,200 targets
        ├── val_seen.json.gz
        ├── content/<scene_id>.json.gz
        ├── dataset_manifest.json
        ├── source_trace.json
        ├── NOTICE.md
        └── identities.json       # generated runtime identity index
```

Source annotations and reconstructed files remain in the cache; `data/` connects them and adds the identity indices. Repeating the command checks and reuses the same files. Use `--destination /path/to/data` for another prepared directory, then pass that directory to [runtime preparation](installation.md#connect-resources).

## Offline preparation

On a computer with internet access, including macOS:

```bash
python3 scripts/prepare_data.py --download-only --cache ./offline-pbsp/datasets
```

Copy the checkout, cache and separately obtained scene assets to the server. Then run:

```bash
python3 scripts/prepare_data.py --scenes /path/to/scene_datasets \
  --cache /path/to/offline-pbsp/datasets --offline
```

An existing reproduction catalog is also accepted through `--source /path/to/dataset-catalog`. Its directory names, manifests and source traces must match the catalog; the script validates and connects its files without downloading annotations.

## Dataset scope and identity

The three sets use the same fixed pool of 1,200 source tasks to compare ordered sequence lengths. The source pool was selected using historical paired baseline/candidate outcomes. These sets are outcome-calibrated comparison data, rather than an independent held-out benchmark. The original selection records remain in `common/` and `source_trace.json`.

The runtime key is `(scene_id, episode_id, task_idx)`. Source records use the official file's sequence index because its `episode_id` values are not unique. Preserve the fixed floor-compatible task sequences and repeated-category identities when replacing a dataset. Historical success labels are excluded from `identities.json`; the evaluator owns ground-truth goals and metrics.

Candidate batches additionally carry the map revision and query serial. Episode reset clears the map; task switches retain it and reset task execution state.
