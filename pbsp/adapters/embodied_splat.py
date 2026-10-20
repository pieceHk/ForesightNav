"""Construct the retained EmbodiedSplat network for the canonical mapper."""

from pathlib import Path
import os
import sys


def load_model(*, engine_root, checkpoint, work_directory, input_spec, device, semantic_injector=None, model_resources=None):
    from omegaconf import OmegaConf
    from pbsp.mapping.model import OnlineMappingModel

    engine_root = Path(engine_root).resolve()
    work_directory = Path(work_directory).resolve()
    work_directory.mkdir(parents=True, exist_ok=True)
    for path in (engine_root.parent,):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    os.chdir(engine_root)
    config = OmegaConf.load(engine_root.parents[1] / "configs/models/network.yaml")
    config.model.encoder.output_path = str(work_directory)
    if (list(config.dataset.image_shape) != [input_spec.image_height, input_spec.image_width]
            or float(config.dataset.near) != input_spec.near or float(config.dataset.far) != input_spec.far):
        raise ValueError("Model input specification disagrees with the network configuration")

    from embodied_splat.global_cfg import set_cfg
    from pbsp.adapters.embodied_splat_config import load_typed_config
    from embodied_splat.model.encoder.encoder_config import EncoderEpipolarCfg
    from pbsp.adapters.embodied_splat_model import EmbodiedSplatEncoder
    from embodied_splat.model.decoder import get_decoder

    set_cfg(config)
    encoder_config = load_typed_config(config.model.encoder, EncoderEpipolarCfg)
    encoder_config.output_path = str(work_directory)
    encoder = EmbodiedSplatEncoder(
        encoder_config, depth_range=[input_spec.near, input_spec.far],
        semantic_injector=semantic_injector, model_resources=model_resources,
    )
    encoder = encoder.to(device).eval()
    decoder = get_decoder(config.model.decoder, config.dataset).to(device).eval()
    model = OnlineMappingModel(encoder, decoder, input_spec, work_directory)
    report = model.load_checkpoint(checkpoint)
    if report["encoder"]["missing_keys"]:
        print(f"[OnlineMapper] encoder missing keys: {report['encoder']['missing_keys'][:3]}", flush=True)
    print(f"[OnlineMapper] Model loaded: enc={sum(p.numel() for p in encoder.parameters()):,} params", flush=True)
    return model
