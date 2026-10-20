# Third-party resources

PBSP uses the inference components below. Model construction and component adaptation live in `pbsp/adapters/`; this directory contains the supporting libraries and local model files.

| Component | Used for |
| --- | --- |
| `embodied_splat/` | Neural reconstruction layers and tensor utilities |
| `fastsam/`, `ultralytics/`, `maskadapter/` | Image regions and semantic features |
| `langsplat-rasterization/`, `simple-knn/` | CUDA rendering and geometry extensions |
| `lagmemo/`, `frontier_exploration/` | Local maps, frontier planning and matching support |
| `detic/`, `seem/`, `mobileclip/` | Detection and goal verification |
| `habitat-lab/`, `superglue/` | Separately prepared, pinned runtime source |
| `weights/` | Model checkpoints, tokenizers and model caches |
| `licenses/` | License texts for external components |

`hiersam.py` and `open_clip_network.py` support optional mapping backends. Default semantic injection uses FastSAM and MaskAdapter. OpenSeg, SAM and sparse 3D backends require their optional dependencies when selected.

Sources retain their upstream license scope. See [third-party notices](../THIRD_PARTY_NOTICES.md). Public preparation pins source revisions and file hashes in `configs/environments/` and `configs/resources/`.

## Models

From the repository root:

```bash
python prepare.py models --zstd "$HOME/.local/share/pbsp/envs/mapping/bin/zstd"
```

The command prepares and verifies 22 model, tokenizer and metadata files in `third_party/weights/`. `--output` selects another directory. `--verify` verifies an already prepared destination.

## Offline preparation

An offline server can import existing weights without downloading the upstream archive:

```bash
python prepare.py models --source /path/to/prepared-weights --output third_party/weights --offline
python prepare.py runtime --mapping-python /path/to/map/bin/python --navigation-python /path/to/nav/bin/python --cache /path/to/offline-pbsp --offline
```

The second command prepares the pinned Habitat and SuperGlue source from cache and writes `runtime.local.yaml`. Prepare environment inputs with `prepare.py envs --download-only` on a connected computer. Dataset inputs have their own [offline command](../data/README.md#offline-preparation).

The online mapping weight source is an upstream pretrained archive. PBSP uses its FastSAM and reconstruction checkpoints. If those files are already available, use the prepared-weight import above. A whole model-cache export/restore has not been verified; the existing-weight route verifies each required file directly.

Environment and episode caches can be exported with `prepare.py offline --help`. HM3D scene assets are obtained separately from their provider.
