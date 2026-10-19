"""Semantic injection returns aligned image features and stable codebook indices."""

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class SemanticInjection:
    """Dense features BxDxHxW, pixel codebook IDs BxHxW, and updated semantic state.

    Either dense features or pixel IDs may be None, according to the selected
    semantic representation. The injector must preserve existing codebook IDs.
    """

    dense_features: Any
    pixel_indices: Any
    state: Any

    def values(self):
        return self.dense_features, self.pixel_indices, self.state


class SemanticInjector(Protocol):
    def inject(
        self, images, caches, image_shape, semantic_state
    ) -> SemanticInjection: ...
