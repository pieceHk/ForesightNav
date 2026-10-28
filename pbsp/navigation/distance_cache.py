"""Reuse exact CPU distance fields while their level set and mask are unchanged."""

from collections import OrderedDict
import hashlib

import numpy as np


class DistanceFieldCache:
    def __init__(self, max_entries=8, max_bytes=64 * 1024 * 1024):
        if max_entries < 1 or max_bytes < 1:
            raise ValueError('Distance cache limits must be positive')
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self.clear()

    def clear(self):
        self._entries = OrderedDict()
        self.bytes = self.hits = self.misses = 0

    def compute(self, level_set, *, dx, solver):
        values = np.ascontiguousarray(np.ma.getdata(level_set))
        mask = np.ascontiguousarray(np.ma.getmaskarray(level_set))
        digest = hashlib.sha256(memoryview(values).cast('B'))
        digest.update(memoryview(mask).cast('B'))
        key = (solver, values.shape, values.dtype.str, float(dx), digest.digest())
        if key in self._entries:
            self.hits += 1
            result, size = self._entries.pop(key)
            self._entries[key] = result, size
            return result.copy()
        self.misses += 1
        result = solver(level_set, dx=dx)
        self._remember(key, result)
        return result

    def point_source(self, shape, goal, *, dtype, dx, solver):
        """Reflect a corner FMM field for a single source on an empty grid."""
        dtype = np.dtype(dtype)
        key = ('empty_grid_corner', solver, tuple(shape), dtype.str, float(dx))
        if key in self._entries:
            self.hits += 1
            template, size = self._entries.pop(key)
            self._entries[key] = template, size
        else:
            self.misses += 1
            level_set = np.ones(shape, dtype=dtype)
            level_set[0, 0] = 0
            template = solver(np.ma.array(level_set, mask=False), dx=dx)
            self._remember(key, template)
        rows = np.abs(np.arange(shape[0]) - goal[0])
        columns = np.abs(np.arange(shape[1]) - goal[1])
        return template[rows[:, None], columns[None, :]]

    def _remember(self, key, result):
        size = np.ma.getdata(result).nbytes + np.ma.getmaskarray(result).nbytes
        if size <= self.max_bytes:
            while self._entries and (
                len(self._entries) >= self.max_entries or self.bytes + size > self.max_bytes
            ):
                _, (_, old_size) = self._entries.popitem(last=False)
                self.bytes -= old_size
            self._entries[key] = result.copy(), size
            self.bytes += size
