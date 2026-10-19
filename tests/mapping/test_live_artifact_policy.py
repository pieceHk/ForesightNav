import ast
from pathlib import Path


def test_live_mapper_never_persists_tensor_checkpoints():
    """Live navigation may emit PNG diagnostics, never per-episode tensors."""
    root = Path(__file__).resolve().parents[2]
    runtime = (root / "Exp" / "scripts" / "_runtime.py").read_text(encoding="utf-8")
    mapper = (root / "Mapping" / "runtime" / "server" / "online_es_mapper.py").read_text(
        encoding="utf-8"
    )
    assert '"ES_DISABLE_P3_SNAPSHOT": "1"' in runtime
    assert "ES_EXPORT_FINAL_MAP_DIR" not in mapper
    sources = list((root / "pbsp/mapping").glob("*.py")) + [root / "experiments/visualization.py"]
    for source in sources:
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert not (
                    isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "torch"
                    and node.func.attr == "save"
                ), source
                assert node.func.attr != "save_ply", source
