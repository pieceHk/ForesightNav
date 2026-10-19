"""Read actual bridge methods for the existing isolated fault fixtures."""

import ast
from pathlib import Path


BRIDGE_METHODS = {
    "_compute_scsp_shadow_decision": "compute_decision",
    "_record_scsp_shadow_decision": "record_decision",
    "_prepare_scsp_refresh": "prepare_refresh",
}


def planning_function_source(function_name):
    root = Path(__file__).resolve().parents[2]
    tree = ast.parse((root / "experiments/planning.py").read_text())
    bridge = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    node = next(
        node
        for node in bridge.body
        if isinstance(node, ast.FunctionDef) and node.name == BRIDGE_METHODS[function_name]
    )
    names = {value: key for key, value in BRIDGE_METHODS.items()}
    names.update(
        project_goals="_project_goals_to_navigable",
        geodesic="_strict_geodesic_distance",
        clock="time",
    )

    class ExternalBindings(ast.NodeTransformer):
        def visit_Attribute(self, item):
            if isinstance(item.value, ast.Name) and item.value.id == "self":
                if item.attr in names:
                    return ast.copy_location(ast.Name(id=names[item.attr], ctx=item.ctx), item)
            return self.generic_visit(item)

    node = ExternalBindings().visit(node)
    assert node.args.args[0].arg == "self"
    node.args.args.pop(0)
    node.name = function_name
    return ast.unparse(node)
