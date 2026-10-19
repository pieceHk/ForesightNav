from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Type, TypeVar
from typing_extensions import Literal

from adapters.embodied_splat_config import TYPE_HOOKS, load_typed_config
from omegaconf import DictConfig, OmegaConf

from .dataset.data_module import DataLoaderCfg, DatasetCfg
from .loss import LossCfgWrapper
from .model.decoder import DecoderCfg
from .model.encoder import EncoderCfg
from .model.model_wrapper import OptimizerCfg, TestCfg, TrainCfg

from .dataset.dataset_re10k import DatasetRE10kCfg
from .dataset.dataset_scannet import DatasetScannetCfg


@dataclass
class CheckpointingCfg:
    load: Optional[str]  # Not a path, since it could be something like wandb://...
    every_n_train_steps: int
    save_top_k: int


@dataclass
class ModelCfg:
    decoder: DecoderCfg
    encoder: EncoderCfg


@dataclass
class TrainerCfg:
    max_steps: int
    val_check_interval: int | float | None
    gradient_clip_val: int | float | None
    num_sanity_val_steps: int | float | None


@dataclass
class RootCfg:
    wandb: dict
    mode: Literal["train", "test", "test_fvs"]
    data_loader: DataLoaderCfg
    dataset: DatasetCfg
    model: ModelCfg
    optimizer: OptimizerCfg
    checkpointing: CheckpointingCfg
    trainer: TrainerCfg
    loss: list[LossCfgWrapper]
    test: TestCfg
    train: TrainCfg
    seed: int
    output_dir: str
    strict: bool = True


T = TypeVar("T")

DatasetCfgs = {'re10k': DatasetRE10kCfg, 'scannet': DatasetScannetCfg}


def separate_loss_cfg_wrappers(joined: dict) -> list[LossCfgWrapper]:
    # The dummy allows the union to be converted.
    @dataclass
    class Dummy:
        dummy: LossCfgWrapper

    return [
        load_typed_config(DictConfig({"dummy": {k: v}}), Dummy).dummy
        for k, v in joined.items()
    ]


def load_typed_root_config(cfg: DictConfig) -> RootCfg:
    rootcfg = RootCfg
    return load_typed_config(
        cfg,
        rootcfg,
        {list[LossCfgWrapper]: separate_loss_cfg_wrappers},
    )


