"""Supply region models to the semantic injector without owning geometry layers."""

from types import SimpleNamespace
import weakref

from pbsp.semantics.features import SemanticFeatures
from pbsp.semantics.types import SemanticInjection


class EmbodiedSplatSemanticInjector:
    """Reuse loaded region/CLIP resources and keep checkpoint parameter names stable."""

    RESOURCE_NAMES = (
        "cfg",
        "device",
        "fastsam",
        "fastsam_prompt",
        "maskclip",
        "clip_preprocess",
        "PIXEL_MEAN",
        "PIXEL_STD",
        "clip",
        "maskadapter",
        "openseg",
        "openseg_text_emb",
        "extract_openseg_img_feature",
        "sam",
    )

    def __init__(self, encoder):
        self._resources = weakref.ref(encoder)

    def inject(self, images, caches, image_shape, semantic_state):
        owner = self._resources()
        if owner is None:
            raise RuntimeError("Semantic resources have been released")
        resources = SimpleNamespace(
            **{name: getattr(owner, name, None) for name in self.RESOURCE_NAMES}
        )
        methods = {
            "fastsam": SemanticFeatures.extract_fastsam_clip_feats,
            "sam": SemanticFeatures.extract_sam_clip_feats,
        }
        try:
            extract = methods[resources.cfg.sam_model_type]
        except KeyError as exc:
            raise ValueError("Unsupported region segmenter") from exc
        return SemanticInjection(
            *extract(resources, images, caches, image_shape, semantic_state)
        )
