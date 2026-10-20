"""Apply process resource controls before importing runtime dependencies."""

import os


def configure_runtime_threads() -> None:
    """Apply opt-in CPU thread caps before loading worker dependencies."""
    names = {
        "blas": "TDONAV_BLAS_NUM_THREADS",
        "torch": "TDONAV_TORCH_NUM_THREADS",
        "interop": "TDONAV_TORCH_NUM_INTEROP_THREADS",
        "opencv": "TDONAV_OPENCV_NUM_THREADS",
    }
    values = {key: int(os.environ.get(name, "0")) for key, name in names.items()}
    if not any(values.values()):
        return
    if any(value < 0 for value in values.values()):
        raise ValueError("TDONAV runtime thread limits must be non-negative")
    if values["blas"]:
        value = str(values["blas"])
        for name in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            os.environ.setdefault(name, value)
    if values["torch"] or values["interop"]:
        import torch

        if values["torch"]:
            torch.set_num_threads(values["torch"])
        if values["interop"]:
            torch.set_num_interop_threads(values["interop"])
    if values["opencv"]:
        import cv2

        cv2.setNumThreads(values["opencv"])
    print(f"[RUNTIME_THREADS] {values}", flush=True)


def worker_environment(plan, role, inherited=None):
    """Construct each child's source, cache and device environment explicitly."""
    if role not in {"mapping", "navigation"}:
        raise ValueError(f"Unknown worker role: {role}")
    inherited = os.environ if inherited is None else inherited
    environment = {
        name: value
        for name, value in inherited.items()
        if not name.startswith(
            ("ES_", "HM3D_EXP_", "LAGMEMO_", "TDONAV_", "THREEDONAV_")
        )
    }
    resources = plan.resources
    runtime = plan.experiment.runtime
    paths = [resources.root]
    if role == "mapping":
        paths.extend(
            [
                resources.mapping,
                resources.mapping / "src",
                resources.mapping / "src/third_party",
            ]
        )
        gpu = runtime.mapping_gpu
    else:
        gpu = runtime.navigation_gpu
        third_party = resources.navigation / "src/third_party"
        paths.extend(
            [
                resources.habitat / "habitat-lab",
                resources.habitat / "habitat-baselines",
                resources.navigation / "src/lagmemo",
                third_party / "seem/src",
                third_party / "ml-mobileclip",
                third_party / "frontier_exploration",
            ]
        )
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "PYTHONPATH": os.pathsep.join(map(str, paths)),
            "PYTHONUNBUFFERED": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONHASHSEED": str(plan.experiment.seed),
            "TDONAV_ROOT": str(resources.root),
            "TDONAV_LAGMEMO_ROOT": str(resources.navigation),
            "ES_TIMM_EFFV2_WEIGHTS": str(
                resources.weights / "tf_efficientnetv2_s_21ft1k-d7dafa41.pth"
            ),
            "HF_HOME": str(resources.hf_home),
            "HF_HUB_CACHE": str(resources.hf_home / "hub"),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TORCH_HOME": str(resources.torch_home),
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "TDONAV_TORCH_NUM_THREADS": "1",
            "TDONAV_TORCH_NUM_INTEROP_THREADS": "1",
            "MALLOC_ARENA_MAX": "2",
            "MALLOC_TRIM_THRESHOLD_": "131072",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "MAGNUM_LOG": "quiet",
            "HABITAT_SIM_LOG": "quiet",
            "MPLBACKEND": "Agg",
        }
    )
    return environment


def configure_run_process(plan, role):
    """Apply the saved process controls before importing model implementations."""
    import sys

    environment = worker_environment(plan, role)
    for name in tuple(os.environ):
        if name.startswith(("ES_", "HM3D_EXP_", "LAGMEMO_", "TDONAV_", "THREEDONAV_")):
            del os.environ[name]
    os.environ.update(environment)
    for path in reversed(environment["PYTHONPATH"].split(os.pathsep)):
        if path not in sys.path:
            sys.path.insert(0, path)
    configure_runtime_threads()
    import random
    import numpy as np
    import torch

    random.seed(plan.experiment.seed)
    np.random.seed(plan.experiment.seed)
    torch.manual_seed(plan.experiment.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(plan.experiment.seed)
    if role == "navigation":
        os.chdir(plan.resources.navigation)
        from runtime.detic_bootstrap import install_inference_import_shims

        install_inference_import_shims(plan.resources.navigation / "src/lagmemo")
