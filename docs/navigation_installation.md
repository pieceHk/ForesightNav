# Navigation environment

The navigation runtime uses Linux x86_64, Python 3.9, PyTorch 2.1.2 with CUDA 11.8, NumPy 1.26.4, Pillow 10.4.0 and timm 1.0.27. Validation used Ubuntu 20.04, GCC 9.4 and an NVIDIA A40. Install a current Conda or [micromamba](https://mamba.readthedocs.io/en/stable/installation/micromamba-installation.html), the CUDA 11.8 toolkit and these system build dependencies:

```bash
sudo apt-get install build-essential ninja-build libgl1-mesa-dev libegl1-mesa-dev libx11-dev zlib1g-dev libglib2.0-0
```

From the PBSP repository root:

```bash
python scripts/setup_navigation.py --prefix .cache/envs/navigation \
  --micromamba /path/to/micromamba --cuda-home /usr/local/cuda-11.8
```

Use `--conda /path/to/conda` to select Conda. `--jobs` controls compilation parallelism. CUDA architecture detection uses visible GPUs; `--cuda-arch 8.6` selects an A40 build. The combined [environment installer](installation.md) creates both PBSP environments with one command.

## Offline server

Prepare the Linux artifacts on a computer with internet access, including macOS:

```bash
python scripts/setup_navigation.py --download-only --cache ./offline-navigation
```

Copy this cache and the PBSP checkout to the server, then run:

```bash
python scripts/setup_navigation.py --prefix .cache/envs/navigation \
  --micromamba /path/to/micromamba --cache /path/to/offline-navigation --offline
```

The installer verifies all inputs before creating the environment. It uses local Conda archives, disables pip index access and supplies Imath directly to CMake with FetchContent networking disabled.

## Sources and checks

| File | Contents |
| --- | --- |
| [linux-64.lock.json](../envs/navigation/linux-64.lock.json) | 50 Conda packages, 163 Python inputs, 23 source archives and recursive source layout |
| [conda-linux-64.lock](../envs/navigation/conda-linux-64.lock) | Explicit Conda base specification |
| [requirements.txt](../envs/navigation/requirements.txt) | Python package versions, including the verification test runner |

Habitat-Sim and its 22 recursive submodules use fixed revisions. Repeated toolchain submodules share an archive in the cache. Source checks cover file contents, executable bits and relative symlink targets; assembly preserves links within the complete source group. Detectron2 uses the complete upstream revision `9604f5995cc628619f0e4fd913453b4d7d61db3f`.

The installer compiles headless Habitat-Sim without Bullet or CUDA features, installs its freshly built Magnum bindings, and builds Detectron2 with CUDA. Headless GPU rendering remains available. Habitat-Lab, Habitat-Baselines and the retained frontier package are also installed from their pinned sources. MobileCLIP and SEEM inference code is supplied through the worker's explicit source paths; MobileCLIP's training packages `clip-benchmark` and `datasets` are omitted.

Each run writes logs, commands, source copies, wheels and `status.json` under `Exp/runs/setup_navigation/<timestamp>/`. Final checks cover dependency consistency, native module origins, CUDA execution and an analytic CPU/CUDA rotated-NMS case. Model assets and datasets are prepared through the [main installation guide](installation.md).

The independent candidate environment passed 578 tests and matched 21 controlled model/Habitat tensors exactly. Together with the independent mapping environment it completed one ordered 4-MON episode. The final public recipe is undergoing a separate empty-prefix validation. See the [environment record](validation/phase22_navigation_install_results.json) and [episode record](validation/phase22_integration_results.json).
