"""Load TensorFlow and apply OpenSeg device controls when that backend is selected."""

from functools import lru_cache
import os


@lru_cache(maxsize=1)
def tensorflow_runtime():
    import tensorflow as tensorflow_v2
    import tensorflow.compat.v1 as tensorflow_v1

    if os.environ.get("EMBODIEDSPLAT_OPENSEG_CPU", "0") == "1":
        try:
            tensorflow_v2.config.set_visible_devices([], "GPU")
            print(
                "[EmbodiedSplat] TensorFlow GPU hidden for OpenSeg; "
                "PyTorch still uses CUDA_VISIBLE_DEVICES."
            )
        except RuntimeError as exc:
            print(f"[EmbodiedSplat] TensorFlow GPU visibility was already initialized: {exc}")
    else:
        try:
            for gpu in tensorflow_v2.config.list_physical_devices("GPU"):
                tensorflow_v2.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as exc:
            print(f"[EmbodiedSplat] TensorFlow memory-growth setup skipped: {exc}")
    return tensorflow_v2, tensorflow_v1
