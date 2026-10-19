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
