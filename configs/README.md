# Configuration

`experiments/hm3d_{2,3,4}mon.yaml` selects the dataset and shared method. `methods/full_pbsp.yaml` contains algorithm settings; `evaluation/standard.yaml` controls budgets and recording. `models/network.yaml` is the resolved inference network configuration. Habitat sensors and the default local planner are configured in `habitat.yaml` and `navigation.yaml`.

Machine-specific data, weight, result, interpreter and GPU paths live in the ignored root `runtime.local.yaml`. See [runtime.example.yaml](runtime.example.yaml). Public presets contain no server paths.

## Environments

Validated platform: Ubuntu 20.04, GCC 9.4, NVIDIA A40 and CUDA toolkit 11.8. Install the system build dependencies:

```bash
sudo apt-get install build-essential ninja-build libgl1-mesa-dev libegl1-mesa-dev libx11-dev zlib1g-dev libglib2.0-0
```

Create both environments from the repository root:

```bash
python3 prepare.py envs --micromamba /path/to/micromamba --cuda-home /usr/local/cuda-11.8
```

Use `--conda /path/to/conda` to select Conda, `--environment mapping` or `navigation` for one environment, and `--prefix /path/to/envs` to select another location. `--jobs` controls build parallelism; `--cuda-arch 8.6` selects an A40 build.

[Mapping](environments/mapping/linux-64.lock.json) and [navigation](environments/navigation/linux-64.lock.json) locks pin packages, native source inputs, revisions and checksums. Mapping uses Python 3.10/PyTorch 2.7.1; navigation uses Python 3.9/PyTorch 2.1.2. Both use CUDA 11.8 builds.

For an offline server, prepare the cache on a connected computer and transfer it with the code:

```bash
python3 prepare.py envs --download-only --cache /path/to/offline-pbsp
python3 prepare.py envs --micromamba /path/to/micromamba --cache /path/to/offline-pbsp --offline
```

`resources/datasets.json`, `resources/models.json` and `resources/superglue.json` record data and model identities. Preparation validates these before use. Resource import commands are in [third_party](../third_party/README.md).
