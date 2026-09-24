#!/usr/bin/env python3
"""Small Linux process-memory helpers used at episode boundaries."""

import ctypes
import gc
import os
import tracemalloc


_TRACE_ENABLED = os.environ.get("ES_HOST_MEMORY_TRACE", "0") == "1"
_TRACE_SNAPSHOT = None
if _TRACE_ENABLED:
    tracemalloc.start(1)


def _rss_mib() -> float:
    try:
        with open("/proc/self/statm", "r", encoding="ascii") as handle:
            resident_pages = int(handle.read().split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE") / (1024 ** 2)
    except (OSError, ValueError, IndexError):
        return -1.0


def release_host_memory(reason: str = "") -> dict:
    """Collect dead Python objects and return free glibc arenas to the OS."""
    before = _rss_mib()
    collected = gc.collect()
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
        "gc={} malloc_trim={}".format(
            suffix, before, after, collected, int(trimmed)
        ),
        flush=True,
    )
    global _TRACE_SNAPSHOT
    if _TRACE_ENABLED:
        snapshot = tracemalloc.take_snapshot()
        traced_mib = sum(
            stat.size for stat in snapshot.statistics("filename")
        ) / (1024 ** 2)
        print(
            "[HostMemoryTrace] traced={:.1f}MiB reason={}".format(
                traced_mib, reason or "unspecified"
            ),
            flush=True,
        )
        if _TRACE_SNAPSHOT is not None:
            for stat in snapshot.compare_to(_TRACE_SNAPSHOT, "lineno")[:12]:
                if stat.size_diff > 0:
                    print("[HostMemoryTrace] {}".format(stat), flush=True)
        _TRACE_SNAPSHOT = snapshot
    return {
        "rss_before_mib": before,
        "rss_after_mib": after,
        "gc_collected": collected,
        "malloc_trimmed": trimmed,
    }
