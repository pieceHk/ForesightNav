"""Bind the released dataset reader for retained evaluator entrypoints."""

from functools import partial
import os
from pathlib import Path

from experiments.dataset import load_runtime_tasks
from runtime.mapping_client import MappingClient


class OnlineESProvider(MappingClient):
    def __init__(
        self,
        socket_path="/tmp/es_mapper.sock",
        top_k=10,
        connect_timeout=60.0,
        content_root=None,
    ):
        if content_root is None:
            root = Path(__file__).resolve().parents[1]
            engine = Path(os.environ.get("LAGMEMO_ROOT", root / "Navigation/runtime/lagmemo"))
            content_root = engine / "data/datasets/goat/hm3d/3_episode_data/content"
        self._content_root = str(content_root)
        super().__init__(
            socket_path,
            task_reader=partial(load_runtime_tasks, self._content_root),
            top_k=top_k,
            connect_timeout=connect_timeout,
            live_mode=False,
        )
