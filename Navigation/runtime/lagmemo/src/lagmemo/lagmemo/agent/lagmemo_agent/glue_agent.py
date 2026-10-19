"""Compatibility constructor for retained LagMemo entrypoints."""

from adapters.lagmemo_agent import LagMemoAgent
from runtime.legacy_navigation import navigation_settings_from_environment


class GLUEAgent(LagMemoAgent):
    def __init__(
        self, config, device_id=0, seem_cfg=None, clip_cfg="", *, settings=None
    ):
        super().__init__(
            config=config,
            device_id=device_id,
            seem_cfg=seem_cfg,
            clip_cfg=clip_cfg,
            settings=(
                navigation_settings_from_environment() if settings is None else settings
            ),
        )
