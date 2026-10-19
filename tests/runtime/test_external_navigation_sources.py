"""Keep pinned installed packages visible through native and retained entries."""

import importlib
from importlib.machinery import PathFinder
import os
from pathlib import Path
import subprocess
import sys

import pytest

from runtime import legacy_navigation_run

ROOT = Path(__file__).resolve().parents[2]


def test_legacy_child_environment_excludes_an_ambient_detectron_source(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "Exp/scripts"))
    runtime = importlib.import_module("_runtime")
    stale = tmp_path / "old-project"
    (stale / "detectron2").mkdir(parents=True)
    (stale / "detectron2/__init__.py").write_text("origin = 'stale'\n")
    installed = tmp_path / "site-packages"
    (installed / "detectron2").mkdir(parents=True)
    expected = installed / "detectron2/__init__.py"
    expected.write_text("origin = 'installed'\n")
    monkeypatch.setenv("PYTHONPATH", str(stale))
    environment = runtime.navigation_environment(
        runtime.load_toml(ROOT / "Exp/runtime.local.example.toml"),
        runtime.load_toml(ROOT / "Exp/profiles/goat_val_seen.toml"),
        nav_gpu=1,
    )
    paths = environment["PYTHONPATH"].split(os.pathsep)
    assert str(stale) not in paths
    assert not any(Path(path).name == "detectron2" for path in paths)
    assert PathFinder.find_spec("detectron2", paths + [str(installed)]).origin == str(expected)


def test_retained_request_does_not_prepend_a_detectron_source(monkeypatch):
    monkeypatch.setenv("TDONAV_ROOT", str(ROOT))
    monkeypatch.setenv("ES_SCSP_RUNTIME_MODE", "off")
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(os, "chdir", lambda *args: None)
    monkeypatch.setattr("warnings.filterwarnings", lambda *args: None)
    monkeypatch.setattr("runtime.bootstrap.configure_runtime_threads", lambda: None)
    marker = object()
    monkeypatch.setattr(legacy_navigation_run, "resolve_request", lambda *args, **kwargs: marker)
    assert legacy_navigation_run.prepare_request([]) is marker
    assert str(ROOT / "Navigation/runtime/lagmemo/src/third_party/detectron2") not in sys.path


@pytest.mark.parametrize("name", ("detectron2", "habitat-lab"))
def test_external_license_text_preserves_the_frozen_bytes(name):
    path = f"Navigation/runtime/lagmemo/src/third_party/{name}/LICENSE"
    expected = subprocess.check_output(["git", "show", f"3665bae:{path}"], cwd=ROOT)
    actual = (ROOT / "third_party/licenses" / name.replace("-", "_") / "LICENSE").read_bytes()
    assert actual == expected
