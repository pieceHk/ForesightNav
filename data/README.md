# Data

Obtain the HM3D v0.1 validation archives from the [official provider](https://github.com/matterport/habitat-matterport-3dresearch#downloading-hm3d-v01): `hm3d-val-habitat.tar`, `hm3d-val-semantic-annots-v0.1.tar.gz` and `hm3d-val-semantic-configs-v0.1.tar`.

Arrange the extracted files before preparing the episodes:

```text
scene_datasets/
└── hm3d/
    ├── hm3d_annotated_basis.scene_dataset_config.json
    └── val/<scene-directory>/
        ├── <scene-id>.basis.glb
        ├── <scene-id>.basis.navmesh
        ├── <scene-id>.semantic.glb
        └── <scene-id>.semantic.txt
```

From the repository root:

```bash
python3 prepare.py data --scenes /path/to/scene_datasets
```

The command reads the selected official ObjectNav annotations and the fixed recipes in [configs/resources/episodes](../configs/resources/episodes). It checks the generated hashes and creates:

```text
data/
├── scenes/hm3dv1/             # scene_datasets connection
└── episodes/hm3dv1/
    ├── common/               # source-pool and task-order provenance
    ├── 2mon/                 # 600 episodes
    ├── 3mon/                 # 400 episodes
    └── 4mon/                 # 300 episodes
        ├── val_seen.json.gz
        ├── content/<scene_id>.json.gz
        ├── dataset_manifest.json
        ├── source_trace.json
        └── identities.json
```

Prebuilt episode bundle: `<DATASET_DOWNLOAD_URL>`.

## Offline preparation

Fetch episode inputs on a connected computer, then copy the cache and scenes to the server:

```bash
python3 prepare.py data --download-only --cache /path/to/offline-pbsp/datasets
python3 prepare.py data --scenes /path/to/scene_datasets --cache /path/to/offline-pbsp/datasets --offline
```

Use `--source /path/to/dataset-catalog` to import an existing catalog. `--destination` selects another data directory; pass it to `prepare.py runtime --data`.

## Scope and identity

Each MON set uses the same fixed pool of 1,200 source tasks, selected using historical paired baseline/candidate outcomes. These sets support controlled sequence-length comparisons. They are not an independent held-out benchmark. Source selection records remain in `common/` and `source_trace.json`.

The runtime identity is `(scene_id, episode_id, task_idx)`. Fixed sequences preserve floor compatibility and repeated categories. Source records use the official file's sequence index because its `episode_id` values are not unique. Candidate batches also carry a map revision and query serial. Task switches keep the map; episode reset clears it.

Scene data and generated annotations retain their [dataset terms](../configs/resources/episodes/NOTICE.md).
