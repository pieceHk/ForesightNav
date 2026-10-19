# Mapping environment

The mapping runtime uses Linux x86_64, Python 3.10, PyTorch 2.7.1 with CUDA 11.8, and a CUDA-capable C++ compiler. Install [micromamba](https://mamba.readthedocs.io/en/stable/installation/micromamba-installation.html) or a current Conda, and make the CUDA 11.8 toolkit available.

Validation used Ubuntu 20.04, GCC 9.4, CUDA 11.8 and an NVIDIA A40. The system packages used for compilation and OpenCV are:

```bash
sudo apt-get install build-essential ninja-build libgl1 libglib2.0-0
```

From the PBSP repository root:

```bash
python scripts/setup_mapping.py --prefix .cache/envs/mapping \
  --micromamba /path/to/micromamba --cuda-home /usr/local/cuda-11.8
```

Use `--conda /path/to/conda` to create the environment with Conda. An existing prefix must match the locked Python and Conda base packages and have a working pip. The installer downloads verified artifacts, installs Python packages, builds the extension wheels, installs PBSP, and checks dependencies, extension imports and CUDA execution.

The default FastSAM/MaskAdapter configuration uses this environment. OpenSeg, SAM and sparse 3D backends have their own optional dependencies. Model weights and datasets are prepared separately through the [main installation guide](installation.md).

## Offline server

Prepare the Linux cache on a computer with internet access; this command also runs on macOS:

```bash
python scripts/setup_mapping.py --download-only --cache ./offline-mapping
```

Copy `offline-mapping/` and the PBSP checkout to the server. Then run:

```bash
python scripts/setup_mapping.py --prefix .cache/envs/mapping \
  --micromamba /path/to/micromamba --cache /path/to/offline-mapping --offline
```

The offline path verifies every artifact before creating or installing the environment. It uses local Conda archives and disables pip index access. CUDA builds infer architectures from visible GPUs; use `--cuda-arch 8.6` for an A40 build or supply the architecture list for the target hardware. `--jobs` controls compilation parallelism.

## Versions and build sources

| File | Contents |
| --- | --- |
| [linux-64.lock.json](../envs/mapping/linux-64.lock.json) | 42 Conda base packages, 122 Python package artifacts, source versions and SHA256 values |
| [conda-linux-64.lock](../envs/mapping/conda-linux-64.lock) | Explicit Conda base specification |
| [requirements.txt](../envs/mapping/requirements.txt) | Pinned Python distributions; extension sources are built by the installer |

CLIP and the depth rasterizer use fixed upstream commits. LangSplat rasterization, deformable aggregation and simple-knn are copied from the tracked PBSP source into a new build directory. Their hashes are checked before copying; existing build products are excluded. Source archives are checked by file content so a GitHub branch export and commit export with the same contents can share the cache. Original license texts are retained in [third-party notices](../THIRD_PARTY_NOTICES.md).

The MMCV source artifact is built with `MMCV_WITH_OPS=0`, producing the `mmcv-lite` distribution. PBSP's four CUDA extensions are built separately.

Each invocation writes evidence under `Exp/runs/setup_mapping/<timestamp>/`: `status.json`, `commands.json`, compiler and package logs, isolated source copies and built wheels. The status records the lock, installer and source revision, plus CUDA build controls. Full experiment outputs use the separate [runtime layout](runtime.md).

An independent offline installation passed dependency, native-extension and CUDA checks. With the same 12 real RGB-D frames and controlled inference settings, its 50 map/state tensors and chair/bed query results matched the provisioned environment exactly. See the [installation validation record](validation/phase21_mapping_install_results.json). Navigation installation is tracked separately.
