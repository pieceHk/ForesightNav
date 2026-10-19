import json
import os

import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
ADAPTERS = (
    ROOT / "Navigation/src/tdonav_navigation/adapters/lagmemo_evaluator.py",
    ROOT / "Mapping/src/tdonav_mapping/adapters/embodied_splat_server.py",
)


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_runtime_thread_limits_are_applied_before_runtime_load(adapter):
    script = f"""
import importlib.util
import json
import os

import cv2
cv2.setNumThreads(8)
expected_opencv = cv2.getNumThreads()
calls = []
set_threads = cv2.setNumThreads
def record_threads(value):
    calls.append(value)
    set_threads(value)
cv2.setNumThreads = record_threads

spec = importlib.util.spec_from_file_location("runtime_adapter", {str(adapter)!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.configure_runtime_threads()

import cv2
import torch
print(json.dumps({{
    "omp": os.environ["OMP_NUM_THREADS"],
    "mkl": os.environ["MKL_NUM_THREADS"],
    "openblas": os.environ["OPENBLAS_NUM_THREADS"],
    "numexpr": os.environ["NUMEXPR_NUM_THREADS"],
    "torch": torch.get_num_threads(),
    "interop": torch.get_num_interop_threads(),
    "opencv": cv2.getNumThreads(),
    "opencv_expected": expected_opencv,
    "opencv_calls": calls,
}}))
"""
    env = os.environ.copy()
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env.pop(name, None)
    env.update(
        {
            "TDONAV_BLAS_NUM_THREADS": "8",
            "TDONAV_TORCH_NUM_THREADS": "16",
            "TDONAV_TORCH_NUM_INTEROP_THREADS": "2",
            "TDONAV_OPENCV_NUM_THREADS": "8",
        }
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    values = json.loads(result.stdout.strip().splitlines()[-1])
    # OpenCV's backend may report a fixed pool size (e.g. GCD on macOS).
    assert values.pop("opencv_calls") == [8]
    expected_opencv = values.pop("opencv_expected")
    assert values == {
        "omp": "8",
        "mkl": "8",
        "openblas": "8",
        "numexpr": "8",
        "torch": 16,
        "interop": 2,
        "opencv": expected_opencv,
    }
