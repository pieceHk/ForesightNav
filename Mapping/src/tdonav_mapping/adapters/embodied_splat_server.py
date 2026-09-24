"""Run the 3DONav-owned EmbodiedSplat mapper runtime."""

import argparse
import os
import runpy
import sys
from pathlib import Path


MAPPING_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = MAPPING_ROOT.parent
RUNTIME_ROOT = MAPPING_ROOT / "runtime"
SERVER_SOURCE = RUNTIME_ROOT / "server" / "es_mapper_server.py"
ENGINE_ROOT = RUNTIME_ROOT / "embodied_splat"


def configure_runtime_threads() -> None:
    """Apply opt-in CPU thread caps before loading the mapping runtime."""
    names = {
        "blas": "TDONAV_BLAS_NUM_THREADS",
        "torch": "TDONAV_TORCH_NUM_THREADS",
        "interop": "TDONAV_TORCH_NUM_INTEROP_THREADS",
        "opencv": "TDONAV_OPENCV_NUM_THREADS",
    }
    values = {
        key: int(os.environ.get(name, "0"))
        for key, name in names.items()
    }
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    args, passthrough = parser.parse_known_args()
    del args
    if not SERVER_SOURCE.is_file():
        raise FileNotFoundError(SERVER_SOURCE)
    if not ENGINE_ROOT.is_dir():
        raise FileNotFoundError(ENGINE_ROOT)
    os.environ.setdefault("TDONAV_ROOT", str(PROJECT_ROOT))
    os.environ.setdefault("ES_ROOT", str(ENGINE_ROOT))
    os.chdir(ENGINE_ROOT)
    for path in (SERVER_SOURCE.parent, ENGINE_ROOT, ENGINE_ROOT / "src", ENGINE_ROOT / "src" / "third_party"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    configure_runtime_threads()
    sys.argv = [str(SERVER_SOURCE), *passthrough]
    runpy.run_path(str(SERVER_SOURCE), run_name="__main__")


if __name__ == "__main__":
    main()
