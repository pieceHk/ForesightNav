"""Episode-boundary release for Python, pinned-host, glibc, and CUDA caches."""

import ctypes
import gc
from dataclasses import dataclass


@dataclass(frozen=True)
class ReleaseResult:
    collected_objects: int
    torch_host_cache: bool
    malloc_trim: bool
    cuda_cache: bool


def release_episode_resources() -> ReleaseResult:
    collected = gc.collect()
    host_cache = False
    cuda_cache = False
    try:
        import torch

        empty_host = getattr(torch._C, "_host_emptyCache", None)
        if callable(empty_host):
            empty_host()
            host_cache = True
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            cuda_cache = True
    except (ImportError, RuntimeError):
        pass
    trimmed = False
    try:
        libc = ctypes.CDLL(None)
        trim = libc.malloc_trim
        trim.argtypes = [ctypes.c_size_t]
        trim.restype = ctypes.c_int
        trimmed = bool(trim(0))
    except (AttributeError, OSError):
        pass
    return ReleaseResult(collected, host_cache, trimmed, cuda_cache)
