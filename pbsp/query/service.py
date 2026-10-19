"""Compose semantic retrieval from a snapshot reader and a text encoder."""

from pbsp.settings import QuerySettings
from .engine import query_categories, query_category
from .types import TextEncoder


class SemanticQuery:
    def __init__(
        self, *, settings: QuerySettings, text_encoder: TextEncoder, read_index, device
    ):
        self.settings = settings
        self.top_k = settings.top_k
        self.text_encoder = text_encoder
        self.read_index = read_index
        self.device = device
        self._closed = False

    def _index(self, state):
        if self._closed:
            raise RuntimeError("Query backend is closed")
        return self.read_index(state)

    def query(
        self, state, category, top_k=None, *, return_records=False, map_revision=0
    ):
        return query_category(
            self.text_encoder,
            self._index(state),
            category,
            top_k or self.top_k,
            self.device,
            settings=self.settings,
            return_records=return_records,
            map_revision=map_revision,
        )

    def query_many(
        self, state, categories, top_k=None, *, return_records=False, map_revision=0
    ):
        return query_categories(
            self.text_encoder,
            self._index(state),
            categories,
            top_k or self.top_k,
            self.device,
            settings=self.settings,
            return_records=return_records,
            map_revision=map_revision,
        )

    def close(self):
        self._closed = True
        self.text_encoder = None
        self.read_index = None
