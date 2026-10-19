"""Optional semantic runtimes load only when their backend is used."""

import ast
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

from adapters.openseg import tensorflow_runtime


ROOT = Path(__file__).resolve().parents[2]


def test_configuration_conversion_preserves_executable_definition():
    previous = ast.parse(subprocess.check_output(
        ["git", "show", "27cac50:Mapping/runtime/embodied_splat/src/config.py"], cwd=ROOT
    ))
    current = ast.parse((ROOT / "adapters/embodied_splat_config.py").read_text())
    def conversion(tree):
        return next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "load_typed_config")
    assert ast.dump(conversion(previous)) == ast.dump(conversion(current))


@pytest.mark.parametrize("name", ("__init__", "backbone_3d", "extract_openseg_img_feature"))
def test_backend_arithmetic_preserves_frozen_definition(name):
    path = "adapters/embodied_splat_model.py"
    previous = ast.parse(subprocess.check_output(["git", "show", "27cac50:" + path], cwd=ROOT))
    current = ast.parse((ROOT / path).read_text())

    class RemoveBackendImports(ast.NodeTransformer):
        def visit_Import(self, node):
            return None

        def visit_ImportFrom(self, node):
            return None

        def visit_Assign(self, node):
            if (isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Name)
                    and node.value.func.id == "tensorflow_runtime"):
                assert ast.unparse(node.targets[0]) in {"(tff2, tff)", "(_, tff)"}
                return None
            return node

    def function(tree):
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
        fn = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name)
        return RemoveBackendImports().visit(fn)
    assert ast.dump(function(previous)) == ast.dump(function(current))


def test_openseg_adapter_import_does_not_load_tensorflow():
    subprocess.run(
        [sys.executable, "-c", """
import importlib.abc
import sys
class BlockTensorFlow(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'tensorflow':
            raise AssertionError('TensorFlow was imported before backend selection')
sys.meta_path.insert(0, BlockTensorFlow())
import adapters.openseg
assert 'tensorflow' not in sys.modules
"""],
        check=True,
    )


@pytest.mark.parametrize("cpu", (False, True))
@pytest.mark.parametrize("already_initialized", (False, True))
def test_selected_openseg_applies_device_policy_once(monkeypatch, capsys, cpu, already_initialized):
    calls = []

    def configure(*args):
        calls.append(args)
        if already_initialized:
            raise RuntimeError("initialized")

    v2, compat, v1 = (ModuleType(name) for name in
                      ("tensorflow", "tensorflow.compat", "tensorflow.compat.v1"))
    v2.config = SimpleNamespace(
        set_visible_devices=configure,
        list_physical_devices=lambda kind: ["gpu0", "gpu1"],
        experimental=SimpleNamespace(set_memory_growth=configure),
    )
    v2.compat, compat.v1 = compat, v1
    for module in (v2, compat, v1):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setenv("EMBODIEDSPLAT_OPENSEG_CPU", "1" if cpu else "0")
    tensorflow_runtime.cache_clear()
    try:
        assert tensorflow_runtime() == (v2, v1)
        assert tensorflow_runtime() == (v2, v1)
        assert calls == ([([], "GPU")] if cpu else
                         [("gpu0", True)] if already_initialized else
                         [("gpu0", True), ("gpu1", True)])
        if already_initialized:
            assert "initialized" in capsys.readouterr().out
    finally:
        tensorflow_runtime.cache_clear()
