from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LAGMEMO = ROOT / "Navigation/runtime/lagmemo/src/lagmemo/lagmemo"


def test_seem_overlay_does_not_use_matplotlib_visualizer():
    source = (LAGMEMO / "agent/lagmemo_agent/seem_model.py").read_text(
        encoding="utf-8"
    )
    assert "from seem.utils.visualizer import Visualizer" not in source
    assert "def render_mask_overlay" in source


def test_semantic_mapping_has_no_unconditional_pyplot_images():
    for filename in (
        "categorical_2d_semantic_map_module.py",
        "categorical_2d_semantic_map_module_frontier.py",
    ):
        source = (LAGMEMO / "mapping/semantic" / filename).read_text(
            encoding="utf-8"
        )
        assert "plt.imshow(rotated[0, 0].cpu())" not in source
