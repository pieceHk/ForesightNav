"""Keep replaceable retrieval/semantic code independent of concrete runtimes."""

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_retrieval_and_semantic_packages_do_not_import_mapping_or_experiment_implementations():
    forbidden = (
        "pbsp.mapping",
        "pbsp.navigation",
        "adapters",
        "runtime",
        "experiments",
        "Mapping",
        "Navigation",
    )
    for folder in ("query", "semantics"):
        for path in (ROOT / "pbsp" / folder).glob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                names = (
                    [alias.name for alias in node.names]
                    if isinstance(node, ast.Import)
                    else [node.module or ""] if isinstance(node, ast.ImportFrom) else []
                )
                assert not any(
                    name == prefix or name.startswith(prefix + ".")
                    for name in names
                    for prefix in forbidden
                ), path


def test_mapper_calls_query_port_and_encoder_calls_semantic_port():
    mapper = (ROOT / "pbsp/mapping/mapper.py").read_text()
    encoder = (ROOT / "pbsp/mapping/encoder.py").read_text()
    assert "query_category(" not in mapper and "query_categories(" not in mapper
    assert "self.queries.query(" in mapper and "self.queries.query_many(" in mapper
    assert "self.model.encoder.clip" not in mapper
    assert "SemanticFeatures" not in encoder
    assert "self.fastsam(" not in encoder
    assert encoder.count("self.semantic_injector.inject(") == 2
