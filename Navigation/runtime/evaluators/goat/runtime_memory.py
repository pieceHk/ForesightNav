#!/usr/bin/env python3
"""Small Linux process-memory helpers used at episode boundaries."""

import ctypes
import gc
import os


def _rss_mib() -> float:
    try:
        with open("/proc/self/statm", "r", encoding="ascii") as handle:
            resident_pages = int(handle.read().split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE") / (1024 ** 2)
    except (OSError, ValueError, IndexError):
        return -1.0


def _empty_torch_host_cache() -> bool:
    """Release free blocks held by PyTorch's pinned-host allocator."""
    try:
        import torch

        empty_cache = getattr(torch._C, "_host_emptyCache", None)
        if not callable(empty_cache):
            return False
        empty_cache()
        return True
    except (ImportError, RuntimeError):
        return False


def release_host_memory(reason: str = "") -> dict:
    """Return free Python, pinned-host, and glibc allocations to the OS."""
    before = _rss_mib()
    collected = gc.collect()
    host_cache_emptied = _empty_torch_host_cache()
    trimmed = False
    try:
        libc = ctypes.CDLL(None)
        malloc_trim = libc.malloc_trim
        malloc_trim.argtypes = [ctypes.c_size_t]
        malloc_trim.restype = ctypes.c_int
        trimmed = bool(malloc_trim(0))
    except (AttributeError, OSError):
        pass
    after = _rss_mib()
    suffix = f" ({reason})" if reason else ""
    print(
        "[HostMemory] release{}: rss={:.1f}->{:.1f}MiB "
        "gc={} torch_host_cache={} malloc_trim={}".format(
            suffix, before, after, collected, int(host_cache_emptied),
            int(trimmed)
        ),
        flush=True,
    )
    return {
        "rss_before_mib": before,
        "rss_after_mib": after,
        "gc_collected": collected,
        "torch_host_cache_emptied": host_cache_emptied,
        "malloc_trimmed": trimmed,
    }
