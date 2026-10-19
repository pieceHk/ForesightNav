import ast
from pathlib import Path


def test_cross_dataset_visualization_contract_is_enabled():
    root = Path(__file__).resolve().parents[2]
    for profile_name in ("goat_val_seen.toml", "hm3dv1_objectnav_val.toml"):
        profile = (root / "Exp/profiles" / profile_name).read_text()
        assert "[visualization]" in profile
        assert "per_keyframe_snapshots = true" in profile


def test_runners_and_visualizer_share_the_snapshot_contract():
    root = Path(__file__).resolve().parents[2]
    hm_runner = (root / "Exp/scripts/run_hm3dv1_objectnav.py").read_text()
    goat_runner = (root / "Exp/scripts/run_goat.py").read_text()
    visualizer = (
        root / "Navigation/runtime/lagmemo/src/lagmemo/lagmemo/agent/imagenav_agent/visualizer.py"
    ).read_text()
    goat_entrypoint = (root / "Navigation/runtime/evaluators/goat/entrypoint.py").read_text()
    navigator = (root / "pbsp/navigation/navigator.py").read_text()

    assert 'command.append("--print-images")' in hm_runner
    assert '"--print-images"' in goat_runner
    assert '"--no-print-images"' in goat_runner
    assert "CELL_W = 480" in visualizer
    assert "CELL_H = 450" in visualizer
    assert "(480, 480)" not in visualizer
    assert 'name = f"snapshot_{timestep}_{i}.png"' in visualizer
    assert "make_goal_placeholder(goal_category)" in visualizer
    assert "_append_status_footer(frame, goal_text)" in visualizer
    assert "import pdb; pdb.set_trace()" not in visualizer
    assert "* frame_w / obstacle_map.shape" in visualizer
    assert "* frame_h / obstacle_map.shape" in visualizer
    assert 'info["goal_text"] = goal_text_desc' in navigator
    assert "pdb.set_trace" not in navigator
    assert "ipdb.set_trace" not in navigator
    assert "except:" not in navigator
    assert "ES_LagMemo_Integration" not in goat_entrypoint


def test_reachable_evaluator_modules_do_not_enter_an_interactive_debugger():
    root = Path(__file__).resolve().parents[2]
    agent_root = root / "Navigation/runtime/lagmemo/src/lagmemo/lagmemo/agent/lagmemo_agent"
    sources = [
        agent_root / name
        for name in ("glue_agent.py", "lagmemo_matching.py", "frontier_agent_module.py")
    ]
    sources += [
        root / "pbsp/navigation" / name
        for name in ("navigator.py", "local_map.py", "verification.py")
    ]
    sources += [root / "adapters/lagmemo_agent.py", root / "adapters/perception.py"]
    for source in sources:
        tree = ast.parse(source.read_text())
        debugger_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "set_trace"
        ]
        assert not debugger_calls, f"interactive debugger in evaluator dependency: {source}"
