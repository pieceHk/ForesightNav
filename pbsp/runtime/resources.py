"""Locate the default backend resources used by a resolved run."""

from dataclasses import dataclass
from pathlib import Path
import os
import json

HABITAT_REVISION = "8df2661e5b2203c0ddc47798d5da0132d5686bbe"
DETIC_CHECKPOINT = "Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"


@dataclass(frozen=True)
class ResourcePaths:
    root: Path
    data: Path
    weights: Path

    @property
    def mapping(self):
        return self.root / "third_party/embodied_splat"

    @property
    def navigation(self):
        return self.root / "third_party/lagmemo"

    @property
    def habitat(self):
        return self.root / "third_party/habitat-lab"

    @property
    def scene_directory(self):
        return self.data / "scenes/hm3dv1"

    @property
    def scene_dataset(self):
        return (
            self.scene_directory / "hm3d/hm3d_annotated_basis.scene_dataset_config.json"
        )

    @property
    def torch_home(self):
        return self.weights / "torch"

    @property
    def hf_home(self):
        return self.weights / "hf_home"

    def mapping_resources(self):
        from pbsp.adapters.model_resources import MappingModelResources

        return MappingModelResources(
            self.weights / "FastSAM-x.pt",
            self.weights / "maskadpater_ckpt.pth",
            self.weights / "tf_efficientnetv2_s_21ft1k-d7dafa41.pth",
        )

    def navigation_resources(self):
        from pbsp.runtime.navigation_worker import NavigationResources
        from pbsp.adapters.model_resources import MatchingResources

        detic = self.root / "third_party/detic"
        return NavigationResources(
            habitat_config=self.root / "configs/habitat.yaml",
            agent_config=self.root / "configs/navigation.yaml",
            categories=self.root / "configs/categories.json",
            seem_config=self.root / "third_party/seem/configs/seem/focall_unicl_lang_demo.yaml",
            seem_checkpoint=self.weights / "seem_focall_v0.pt",
            mobileclip_checkpoint=self.weights / "mobileclip_s0.pt",
            detic_config=detic
            / "configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml",
            detic_checkpoint=self.weights / "detic" / DETIC_CHECKPOINT,
            scene_directory=self.scene_directory,
            scene_dataset=self.scene_dataset,
            seem_tokenizer=self.weights / "clip-vit-base-patch32",
            detic_category_frequency=self.weights / "detic/datasets/metadata/lvis_v1_train_cat_info.json",
            matching=MatchingResources(
                self.weights / "superglue/superpoint_v1.pth",
                self.weights / "superglue/superglue_indoor.pth",
                self.weights / "clip/ViT-B-32.pt",
            ),
        )

    def required_files(self):
        nav = self.navigation_resources()
        return [
            self.scene_dataset,
            self.root / "configs/models/embodied_splat.yaml",
            self.root / "configs/models/network.yaml",
            self.weights / "freesplatpp_scannet_gtdepth.ckpt",
            self.weights / "detic/datasets/metadata/lvis_v1_train_cat_info.json",
            self.torch_home / "hub/checkpoints/depth-save.pth",
            self.torch_home / "hub/checkpoints/disk_lightglue_v0-1_arxiv-pth",
            *vars(self.mapping_resources()).values(),
            *vars(nav.matching).values(),
            *(nav.seem_tokenizer / name for name in ("vocab.json", "merges.txt", "tokenizer_config.json", "special_tokens_map.json")),
            *(getattr(nav, name) for name in ("habitat_config", "agent_config", "categories", "seem_config", "seem_checkpoint", "mobileclip_checkpoint", "detic_config", "detic_checkpoint")),
        ]


def validate_resources(resources, runtime):
    """Check provisioned resources without importing models or changing assets."""
    for path in (
        *resources.required_files(),
        runtime.mapping_python,
        runtime.navigation_python,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Required runtime resource is missing: {path}")
    if not (resources.hf_home / "hub").is_dir():
        raise FileNotFoundError(resources.hf_home / "hub")
    for python in (runtime.mapping_python, runtime.navigation_python):
        if not os.access(python, os.X_OK):
            raise PermissionError(f"Interpreter is not executable: {python}")
    from pbsp.runtime.source_provenance import habitat_source, verify_source
    from pbsp.runtime.model_assets import verify_model_set

    models = verify_model_set(resources.weights, resources.root / "configs/resources/models.json")
    lock = json.loads((resources.root / "configs/environments/navigation/linux-64.lock.json").read_text())
    source = habitat_source(lock)
    proof = verify_source(resources.habitat, source)
    from pbsp.runtime.external_sources import pinned_superglue_manifest, verify_external_source

    source_manifest = pinned_superglue_manifest(resources.root)
    matching_source = verify_external_source(resources.root / "third_party/superglue", source_manifest)
    return {
        "habitat_revision": source["revision"],
        "habitat_source": proof,
        "superglue_source": matching_source,
        "models": models,
        "files": list(map(str, resources.required_files())),
    }
