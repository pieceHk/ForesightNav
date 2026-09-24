"""Run 3DONav-owned GOAT or HM3D evaluator code through Navigation."""

import argparse
import os
import runpy
import sys
from pathlib import Path


NAVIGATION_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = NAVIGATION_ROOT.parent
RUNTIME_ROOT = NAVIGATION_ROOT / "runtime"
LAGMEMO_ROOT = RUNTIME_ROOT / "lagmemo"
LAGMEMO_PACKAGE_ROOT = LAGMEMO_ROOT / "src" / "lagmemo"


def configure_runtime_threads() -> None:
    """Apply opt-in CPU thread caps before loading the evaluator runtime."""
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
    parser.add_argument("--entrypoint", choices=("goat", "hm3dv1"), required=True)
    args, passthrough = parser.parse_known_args()
    source = RUNTIME_ROOT / "evaluators" / args.entrypoint / "entrypoint.py"
    if not source.is_file():
        raise FileNotFoundError(source)
    if not LAGMEMO_ROOT.is_dir():
        raise FileNotFoundError(LAGMEMO_ROOT)
    if not LAGMEMO_PACKAGE_ROOT.is_dir():
        raise FileNotFoundError(LAGMEMO_PACKAGE_ROOT)
    os.environ.setdefault("TDONAV_ROOT", str(PROJECT_ROOT))
    os.environ.setdefault("TDONAV_LAGMEMO_ROOT", str(LAGMEMO_ROOT))
    third_party = LAGMEMO_ROOT / "src" / "third_party"
    runtime_paths = (
        source.parent,
        LAGMEMO_PACKAGE_ROOT,
        third_party / "seem" / "src",
        third_party / "detectron2",
        third_party / "ml-mobileclip",
        third_party / "frontier_exploration",
    )
    os.chdir(LAGMEMO_ROOT)
    for path in reversed(runtime_paths):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    configure_runtime_threads()
    sys.argv = [str(source), *passthrough]
    runpy.run_path(str(source), run_name="__main__")


if __name__ == "__main__":
    main()
