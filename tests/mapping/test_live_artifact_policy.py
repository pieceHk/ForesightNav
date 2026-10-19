from pathlib import Path


def test_live_mapper_never_persists_tensor_checkpoints():
    """Live navigation may emit PNG diagnostics, never per-episode tensors."""
    root = Path(__file__).resolve().parents[2]
    runtime = (root / "Exp" / "scripts" / "_runtime.py").read_text(
        encoding="utf-8"
    )
    mapper = (root / "Mapping" / "runtime" / "server" / "online_es_mapper.py").read_text(
        encoding="utf-8"
    )
    encoder = (
        root
        / "Mapping/runtime/embodied_splat/src/model/encoder/encoder_test_unet3d_online.py"
    ).read_text(encoding="utf-8")

    assert '"ES_DISABLE_P3_SNAPSHOT": "1"' in runtime
    assert "ES_EXPORT_FINAL_MAP_DIR" not in mapper
