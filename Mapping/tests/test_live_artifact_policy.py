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


def test_resumed_rebuild_keeps_observation_history_on_cpu():
    """Long MON episodes must not restage their complete RGB-D history."""
    root = Path(__file__).resolve().parents[2]
    mapper = (root / "Mapping/runtime/server/online_es_mapper.py").read_text(
        encoding="utf-8"
    )
    encoder = (
        root
        / "Mapping/runtime/embodied_splat/src/model/encoder/encoder_test_unet3d_online.py"
    ).read_text(encoding="utf-8")

    assert "observations_on_cpu=self._online_state is not None" in mapper
    assert "observation_device = 'cpu' if observations_on_cpu else dev" in mapper
    assert "self._online_state = _move_tree_to_device(resume_state, 'cpu')" in mapper
    assert "cur_image = all_image[idx:idx+1].to(device, non_blocking=True)" in encoder
    assert "].to(device, non_blocking=True)" in encoder
    assert "key: [] for key in depth_outputs_all" in encoder
