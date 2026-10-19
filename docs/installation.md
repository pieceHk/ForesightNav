# Environment and model preparation

Mapping has a locked installation with source-built CUDA extensions, verified in an independent Linux environment. Navigation currently uses a provisioned environment; its independent installation is still being validated.

| Runtime | Python | PyTorch | torchvision | NumPy |
| --- | --- | --- | --- | --- |
| Mapping | 3.10 | 2.7.1+cu118 | 0.22.1+cu118 | 2.1.3 |
| Navigation | 3.9 | 2.1.2+cu118 | 0.16.2+cu118 | 1.23.5 |

Mapping needs the EmbodiedSplat CUDA extensions and region/CLIP models. Navigation needs Habitat-Sim, the pinned Habitat fork, Detectron2, Detic, SEEM, MobileCLIP and image-matching models. The retained sources and license scope are listed in [third-party notices](../THIRD_PARTY_NOTICES.md).

The default mapping configuration uses FastSAM and MaskAdapter. OpenSeg loads TensorFlow when selected; the sparse 3D backend loads MinkowskiEngine when selected. SAM is also optional. These backends are no longer imported during default model construction. Configuration conversion lives in a small inference adapter shared with the retained training configuration.

Install the project in each prepared environment:

```bash
/path/to/mapping-env/bin/python -m pip install -e .
/path/to/navigation-env/bin/python -m pip install -e .
```

To create the mapping environment from scratch, use the [mapping installer](mapping_installation.md). It supports online preparation and an offline cache, and installs PBSP as part of the build.

## Model assets

Use the EmbodiedSplat checkpoints and auxiliary weights described in its [official repository](https://github.com/0nandon/EmbodiedSplat#data-preparation). The current preparation command expects:

```text
<weights>/
├── freesplatpp_scannet_gtdepth.ckpt
├── FastSAM-x.pt
├── maskadpater_ckpt.pth
├── mobileclip_s0.pt
├── seem_focall_v0.pt
├── hf_home/
└── torch/hub/checkpoints/
    ├── depth-save.pth
    └── disk_lightglue_v0-1_arxiv-pth
```

The `maskadpater_ckpt.pth` spelling follows the existing asset name. The Detic asset source must also contain:

```text
<detic>/
├── models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth
└── datasets/metadata/lvis_v1_train_cat_info.json
```

Preserve the additional SuperGlue/matching and Hugging Face cache assets used by the provisioned navigation and mapping environments. See the retained [reproduction guide](../Exp/release/REPRODUCIBILITY.md) for their resource paths.

## Connect resources

Run from the repository root after [preparing the data catalog](datasets.md):

```bash
python Exp/scripts/prepare_reproduction.py \
  --mapping-python /path/to/mapping-env/bin/python \
  --navigation-python /path/to/navigation-env/bin/python \
  --datasets /path/to/dataset-catalog \
  --weights /path/to/weights \
  --detic /path/to/Detic
```

This writes the retained `Exp/runtime.local.toml`, connects scene and episode data, prepares identity sidecars, and installs the Habitat source copy at revision `8df2661e5b2203c0ddc47798d5da0132d5686bbe`. The native launcher loads Habitat from that copy. Complete task identities under `data/` are prepared by the [data guide](datasets.md).

For an offline server, add `--habitat-source /path/to/local/habitat-lab` to copy the committed dependency from an existing local checkout. Source code can be transferred with a Git bundle; data and weight caches remain separate resources.

Write the native launcher's machine configuration:

```bash
python scripts/prepare_runtime.py \
  --mapping-python /path/to/mapping-env/bin/python \
  --navigation-python /path/to/navigation-env/bin/python \
  --weights ./Exp/assets/weights \
  --torch-home-source /path/to/provisioned/.cache/torch
```

This writes `runtime.local.yaml` with absolute data, weight, output and interpreter paths. It copies the two required Kornia checkpoints into the selected weights directory. Omit `--torch-home-source` when they are already present there. GPU defaults are mapping `0`, navigation `1`; change them in this YAML or supply `--mapping-gpu` and `--navigation-gpu` during preparation.

The current model constructors still use several vendor-relative auxiliary paths. The native precheck verifies that those paths resolve to the selected weights. Consolidating that resource setup and validating the independent navigation installation remain unfinished.

## Verify

With the navigation environment activated:

```bash
python run.py --config configs/experiments/hm3d_4mon.yaml --check
python -m pytest tests -q
python Exp/scripts/validate.py
```

Then run a complete smoke episode using the [README command](../README.md#3-run). New component implementations should pass their interface checks and a full ordered episode before a selected-split evaluation.
