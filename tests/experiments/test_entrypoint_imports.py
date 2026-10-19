"""Entrypoints must resolve owned packages without an editable installation."""

import os
from pathlib import Path
import subprocess
import sys
import sysconfig

import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("filename", ("run_goat.py", "run_full_pbsp.py"))
def test_entrypoint_imports_outside_checkout_without_pythonpath(tmp_path, filename):
    path = ROOT / "Exp/scripts" / filename
    script = (
        "import sys, runpy\n"
        f"sys.path.append({sysconfig.get_path('purelib')!r})\n"
        f"sys.path.insert(0, {str(path.parent)!r})\n"
        f"sys.argv = [{str(path)!r}, '--help']\n"
        f"runpy.run_path({str(path)!r}, run_name='__main__')\n"
    )
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    result = subprocess.run([sys.executable, "-S", "-c", script], cwd=tmp_path,
                            env=environment, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
