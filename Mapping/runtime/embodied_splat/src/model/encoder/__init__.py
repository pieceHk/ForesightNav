from typing import Optional
from pbsp.mapping.encoder import OnlineEncoder

from .encoder import Encoder
from .encoder_config import EncoderEpipolarCfg

from .visualization.encoder_visualizer import EncoderVisualizer

EncoderCfg = EncoderEpipolarCfg


def get_encoder(
    cfg: EncoderCfg, depth_range=[0.5, 15.0]
) -> tuple[OnlineEncoder, Optional[EncoderVisualizer]]:
    from adapters.embodied_splat_model import EmbodiedSplatEncoder
    from .visualization.encoder_visualizer_epipolar import EncoderVisualizerEpipolar

    if cfg.name != "encoder_test_unet3d_online":
        raise KeyError(cfg.name)
    encoder = EmbodiedSplatEncoder(cfg, depth_range=depth_range)
    visualizer = EncoderVisualizerEpipolar(cfg.visualizer, encoder)
    return encoder, visualizer
