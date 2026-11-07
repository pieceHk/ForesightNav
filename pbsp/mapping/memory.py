"""CPU-resident map state and transient CUDA resource cleanup."""

import ctypes
import gc
import os

import torch

def _release_cuda_cache(device: str, reason: str = ''):
    """Return cached CUDA blocks after dropping Python references.

    This does not free live tensors, but it prevents transient rebuild/context
    allocations from staying reserved in PyTorch's caching allocator after a
    rebuild or episode reset.
    """
    if isinstance(device, str) and device.startswith('cuda') and torch.cuda.is_available():
        try:
            import gc
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            alloc = torch.cuda.memory_allocated() / (1024 ** 2)
            reserved = torch.cuda.memory_reserved() / (1024 ** 2)
            suffix = f' ({reason})' if reason else ''
            print(f'[OnlineESMapper] CUDA cache released{suffix}: '
                  f'allocated={alloc:.0f}MiB reserved={reserved:.0f}MiB',
                  flush=True)
        except Exception as exc:
            print(f'[OnlineESMapper] CUDA cache release warn: {exc}', flush=True)



def _move_tree_to_device(obj, device: str, *, host_keys=()):
    """Move nested state in place, completing CUDA-to-CPU copies before return."""
    target = torch.device(device)
    host_copy_streams = {}

    def move(value):
        if value is None:
            return None
        if torch.is_tensor(value):
            if target.type == 'cpu' and value.device.type == 'cuda':
                host_copy_streams[value.device] = torch.cuda.current_stream(value.device)
            return value.to(device, non_blocking=True)
        if isinstance(value, torch.nn.Module):
            return value
        if isinstance(value, dict):
            for key, item in list(value.items()):
                value[key] = (_move_tree_to_device(item, 'cpu')
                              if key in host_keys else move(item))
            return value
        if isinstance(value, list):
            for index, item in enumerate(value):
                value[index] = move(item)
            return value
        if isinstance(value, tuple):
            return tuple(move(item) for item in value)
        if hasattr(value, '__dict__'):
            for name, item in list(vars(value).items()):
                setattr(value, name, move(item))
            if isinstance(getattr(value, 'device', None), str):
                value.device = device
            return value
        return value

    moved = move(obj)
    for stream in host_copy_streams.values():
        stream.synchronize()
    return moved


def _log_cuda_memory(device: str, label: str):
    if isinstance(device, str) and device.startswith('cuda') and torch.cuda.is_available():
        try:
            alloc = torch.cuda.memory_allocated() / (1024 ** 2)
            reserved = torch.cuda.memory_reserved() / (1024 ** 2)
            print(f'[OnlineESMapper] CUDA memory {label}: '
                  f'allocated={alloc:.0f}MiB reserved={reserved:.0f}MiB',
                  flush=True)
        except Exception:
            pass



def _release_encoder_transient_caches(encoder, *, offload=False):
    """Drop encoder-side forward caches that are safe to rebuild next pass."""
    try:
        released = []
        cost_volume = getattr(encoder, 'cost_volume', None)
        if cost_volume is not None and getattr(cost_volume, 'depth_planes_bdhw', None) is not None:
            cost_volume.depth_planes_bdhw = None
            released.append('cost_volume.depth_planes_bdhw')

        fastsam = getattr(encoder, 'fastsam', None)
        predictor = getattr(fastsam, 'predictor', None)
        if predictor is not None:
            for attr in ('results', 'batch', 'dataset', 'plotted_img'):
                if hasattr(predictor, attr) and getattr(predictor, attr) is not None:
                    setattr(predictor, attr, None)
                    released.append(f'fastsam.predictor.{attr}')
            if offload:
                predictor_model = getattr(predictor, 'model', None)
                if isinstance(predictor_model, torch.nn.Module):
                    predictor_model.to('cpu')
                    released.append('fastsam.predictor.model->cpu')
                fastsam.predictor = None
                released.append('fastsam.predictor')

        if offload and fastsam is not None:
            fastsam_model = getattr(fastsam, 'model', None)
            if isinstance(fastsam_model, torch.nn.Module):
                fastsam_model.to('cpu')
                released.append('fastsam.model->cpu')

        if released:
            print('[OnlineESMapper] Released encoder transient caches: '
                  + ', '.join(released), flush=True)
    except Exception as exc:
        print(f'[OnlineESMapper] Encoder cache release warn: {exc}', flush=True)



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
    return {
        "rss_before_mib": before,
        "rss_after_mib": after,
        "gc_collected": collected,
        "malloc_trimmed": trimmed,
    }
