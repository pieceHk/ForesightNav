"""Convert inference configuration without importing the training model wrapper."""

from pathlib import Path
from typing import Type, TypeVar

from dacite import Config, from_dict
from omegaconf import DictConfig, OmegaConf


TYPE_HOOKS = {Path: Path}
T = TypeVar("T")


def load_typed_config(
    cfg: DictConfig,
    data_class: Type[T],
    extra_type_hooks: dict = {},
) -> T:
    return_dict = from_dict(
        data_class,
        OmegaConf.to_container(cfg),
        config=Config(type_hooks={**TYPE_HOOKS, **extra_type_hooks}),
    )
    return return_dict
