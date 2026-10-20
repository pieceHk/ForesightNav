# Environment and model preparation

The installers create separate mapping and navigation environments from locked inputs. The public combined command passed an offline installation from an empty directory, followed by exact mapping and navigation model replays. Independently built environments also passed a complete ordered episode. See the [installation record](validation/phase23_public_install_results.json) and [episode record](validation/phase22_integration_results.json).

| Runtime | Python | PyTorch | torchvision | NumPy |
| --- | --- | --- | --- | --- |
| Mapping | 3.10 | 2.7.1+cu118 | 0.22.1+cu118 | 2.1.3 |
| Navigation | 3.9 | 2.1.2+cu118 | 0.16.2+cu118 | 1.26.4 |

Mapping needs the EmbodiedSplat CUDA extensions and region/CLIP models. Navigation needs Habitat-Sim, the pinned Habitat fork, Detectron2, Detic, SEEM, MobileCLIP and image-matching models. The retained sources and license scope are listed in [third-party notices](../THIRD_PARTY_NOTICES.md).

The default mapping configuration uses FastSAM and MaskAdapter. OpenSeg loads TensorFlow when selected; the sparse 3D backend loads MinkowskiEngine when selected. SAM is also optional. These backends are no longer imported during default model construction. Configuration conversion lives in a small inference adapter shared with the retained training configuration.

Create both environments from the repository root with a current Conda or micromamba and the CUDA 11.8 toolkit available:

```bash
python scripts/setup_envs.py --prefix .cache/envs \
  --micromamba /path/to/micromamba --cuda-home /usr/local/cuda-11.8
```

This creates `.cache/envs/mapping` and `.cache/envs/navigation`, builds native extensions and installs PBSP in both. The [mapping guide](mapping_installation.md) and [navigation guide](navigation_installation.md) list system dependencies and individual installer options.

For an offline server, first prepare the Linux cache on a computer with internet access; macOS is supported:

```bash
python scripts/setup_envs.py --download-only --cache .cache/pbsp-offline
```

The [offline guide](offline.md) combines environment, model and episode inputs with a Git bundle, verifies the transfer, and gives the server installation commands. Both environment caches are verified before either environment is created.

## Model assets

Prepare the 22 selected model and tokenizer files:

```bash
python scripts/prepare_models.py --output /path/to/pbsp-weights \
  --zstd .cache/envs/mapping/bin/zstd
```

The [model guide](models.md) covers offline caching, importing existing weights and SHA-256 checks. All 22 publicly obtained files passed offline source preparation; whole-archive cache verification remains in progress. See the [model preparation record](validation/phase46_public_model_and_record_results.json). The model adapters receive the selected paths directly; weight links inside vendor source directories are unnecessary.

## Connect resources

After [preparing the data catalog](datasets.md), write the machine configuration:

```bash
python scripts/prepare_runtime.py \
  --mapping-python .cache/envs/mapping/bin/python \
  --navigation-python .cache/envs/navigation/bin/python \
  --data ./data --weights /path/to/pbsp-weights
```

This prepares the pinned Habitat source snapshot and six SuperGlue inference files from the environment cache, then writes `runtime.local.yaml` with absolute data, weight, output and interpreter paths. The native precheck verifies their source inventories and contents, plus all 22 model hashes. A clean Habitat checkout at the pinned revision is also accepted.

For an offline server, add `--cache /path/to/offline-pbsp --offline`. `scripts/prepare_dependencies.py` prepares or checks these source dependencies separately. Their versions and content hashes are pinned by the navigation lock and [SuperGlue source manifest](../assets/superglue.json). SuperGlue retains its [noncommercial research license](../third_party/licenses/superglue/LICENSE).

GPU defaults are mapping `0`, navigation `1`. Set `--mapping-gpu` and `--navigation-gpu` during preparation or edit `runtime.local.yaml`. The two Kornia checkpoints are part of the model manifest; `--torch-home-source` also supports importing them from an existing Torch cache.

## Verify

With the navigation environment activated:

```bash
python scripts/check_install.py
python -m pytest tests -q
python Exp/scripts/validate.py
```

The installation check verifies input hashes, both environments' locked package versions, module origins, native extension imports, tensor operations and Detectron2 rotated NMS. It writes a new evidence directory under `Exp/runs/check_install/`. `--cpu-only` skips CUDA initialization while retaining CPU and source checks.

Run `python scripts/check_install.py --smoke` to also load the default models and execute one complete ordered episode through the normal launcher. This checks Habitat reset, simulator actions and recorded episode metrics. `--scene <id> --episode <id>` selects an exact identity. New component implementations should pass their interface checks and a full ordered episode before a selected-split evaluation.

The [recorded public CUDA smoke](validation/phase47_public_cuda_smoke_results.json) completed one 4-MON episode with all four targets successful; trajectory, environment and source records passed validation.

Each run stores the checked model manifest and resource provenance beside its resolved experiment configuration.
