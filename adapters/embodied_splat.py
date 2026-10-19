"""Construct the retained EmbodiedSplat network for the canonical mapper."""

from pathlib import Path
import os
import sys


def load_model(*, engine_root, checkpoint, work_directory, input_spec, device):
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from pbsp.mapping.model import OnlineMappingModel

    engine_root = Path(engine_root).resolve()
    work_directory = Path(work_directory).resolve()
    work_directory.mkdir(parents=True, exist_ok=True)
    for path in (engine_root, engine_root / "src/third_party", engine_root / "src"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    # The retained network resolves its auxiliary checkpoints relative to this root.
    os.chdir(engine_root)
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(engine_root / "config"), version_base=None):
        config = compose(config_name="main", overrides=[
            "+experiment=scannet/embodiedsplat_fast_test", "mode=test",
            f"model.encoder.output_path={work_directory}",
            "model.encoder.use_gt_depth=true", "wandb.mode=disabled",
        ])
    if (list(config.dataset.image_shape) != [input_spec.image_height, input_spec.image_width]
            or float(config.dataset.near) != input_spec.near or float(config.dataset.far) != input_spec.far):
        raise ValueError("Model input specification disagrees with the network configuration")

    from src.global_cfg import set_cfg
    from src.config import load_typed_config
    from src.model.encoder.encoder_config import EncoderEpipolarCfg
    from src.model.encoder import get_encoder
    from src.model.decoder import get_decoder

    set_cfg(config)
    encoder_config = load_typed_config(config.model.encoder, EncoderEpipolarCfg)
    encoder_config.output_path = str(work_directory)
    encoder, _ = get_encoder(encoder_config, depth_range=[input_spec.near, input_spec.far])
    encoder = encoder.to(device).eval()
    decoder = get_decoder(config.model.decoder, config.dataset).to(device).eval()
    model = OnlineMappingModel(encoder, decoder, input_spec, work_directory)
    report = model.load_checkpoint(checkpoint)
    if report["encoder"]["missing_keys"]:
        print(f"[OnlineMapper] encoder missing keys: {report['encoder']['missing_keys'][:3]}", flush=True)
    print(f"[OnlineMapper] Model loaded: enc={sum(p.numel() for p in encoder.parameters()):,} params", flush=True)
    return model
