# Offline reproduction

Prepare Linux inputs on a connected computer, then transfer a verified export to the server. Cache preparation also runs on macOS. The target server needs the system libraries, CUDA 11.8 toolkit and NVIDIA driver described in the [mapping](mapping_installation.md) and [navigation](navigation_installation.md) guides.

## Prepare and export

From the committed PBSP checkout:

```bash
python3 scripts/setup_envs.py --download-only --cache .cache/pbsp-offline
python3 scripts/prepare_data.py --download-only --cache .cache/pbsp-offline/datasets
python3 scripts/prepare_models.py --download-only --cache .cache/pbsp-offline/models
python3 scripts/export_offline.py --output ../pbsp-offline \
  --micromamba /path/to/linux-x86_64/micromamba
```

The export contains the current branch and its history in `code.bundle`, the selected cache inputs, and a SHA-256 transfer manifest. Source versions, file modes and pinned cache identities are checked before exporting. The command creates a new directory and excludes downloaded partial files, installed environments, local machine settings and generated experiment outputs.

```text
pbsp-offline/
├── code.bundle
├── manifest.json
├── cache/
│   ├── conda/                    # locked Linux base packages
│   ├── wheels/                   # Python package artifacts
│   ├── sources/                  # fixed source archives and SuperGlue files
│   ├── models/                   # verified downloads, including the model archive
│   └── datasets/                 # official source annotations for fixed episodes
└── tools/micromamba              # optional Linux executable
```

Cache subdirectories follow their lockfiles. `--micromamba /path/to/linux-x86_64/micromamba` includes an existing Linux executable and checks its ELF architecture. A macOS executable cannot create the Linux environments. Otherwise, use the Conda or micromamba installation already available on the server.

Obtain HM3D scenes through the [official provider](datasets.md#scene-assets) and transfer them separately, preserving their directory layout.

## Restore on the server

After transferring `pbsp-offline/`:

```bash
git clone /path/to/pbsp-offline/code.bundle PBSP
cd PBSP
python3 scripts/export_offline.py --verify /path/to/pbsp-offline
```

Verification checks the complete file inventory, sizes, SHA-256 hashes, executable modes and bundled branch revision. Then create both environments:

```bash
python3 scripts/setup_envs.py --prefix .cache/envs \
  --micromamba /path/to/pbsp-offline/tools/micromamba \
  --cuda-home /usr/local/cuda-11.8 \
  --cache /path/to/pbsp-offline/cache --offline
```

Use `--conda /path/to/conda` instead when selecting Conda. Prepare data and models from the same export:

```bash
python3 scripts/prepare_data.py --scenes /path/to/scene_datasets \
  --cache /path/to/pbsp-offline/cache/datasets --offline
python3 scripts/prepare_models.py --output ./weights \
  --cache /path/to/pbsp-offline/cache/models --offline \
  --zstd .cache/envs/mapping/bin/zstd
```

Write the machine configuration and run a complete ordered episode:

```bash
.cache/envs/navigation/bin/python scripts/prepare_runtime.py \
  --mapping-python .cache/envs/mapping/bin/python \
  --navigation-python .cache/envs/navigation/bin/python \
  --data ./data --weights ./weights \
  --cache /path/to/pbsp-offline/cache --offline
.cache/envs/navigation/bin/python run.py --config configs/experiments/hm3d_4mon.yaml --limit-episodes 1
```

GPU defaults are mapping `0` and navigation `1`. Runtime preparation accepts explicit GPU IDs. Remove `--limit-episodes 1` to run the full selected split; choose the 2-MON or 3-MON preset for another experiment.

## Transfer selected resource groups

To transfer environments and episode inputs separately from model downloads:

```bash
python3 scripts/export_offline.py --group environments --group episodes \
  --output ../pbsp-offline-environments
```

The manifest records the selected groups. The default command exports all three groups; a selected-group export contains only those inputs and its code bundle. `--verify` checks either kind without downloading or modifying files.

## Transfer prepared weights

If the 22 model files are already prepared, export the environments and episodes with the selected-group command above and transfer the prepared weight directory separately. Import it on the server:

```bash
python3 scripts/prepare_models.py --source /path/to/prepared-weights \
  --output ./weights --offline
```

Continue with `prepare_runtime.py` using `--weights ./weights`. Import verifies all 22 file hashes and creates the pinned model references. The [public-model preparation](validation/phase46_public_model_and_record_results.json) and [CUDA episode](validation/phase47_public_cuda_smoke_results.json) records cover this prepared-weight route. The 6.98 GB upstream download contains model checkpoints, not episode or scene data; its complete cache export was not validated.
