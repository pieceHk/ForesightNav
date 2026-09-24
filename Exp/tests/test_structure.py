import os
import subprocess
import sys
from pathlib import Path


def test_architecture_validator():
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, str(root / "Exp/scripts/validate.py")],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_socket_path_is_short_and_run_specific():
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "Exp" / "scripts"))
    from _runtime import socket_for_run

    first = root / "Exp" / "runs" / ("a" * 160)
    second = root / "Exp" / "runs" / ("b" * 160)
    first_socket = socket_for_run(first)
    second_socket = socket_for_run(second)

    assert first_socket != second_socket
