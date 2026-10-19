"""Locate the default backend resources used by a resolved run."""

from dataclasses import dataclass
from pathlib import Path
import os
import subprocess

HABITAT_REVISION = "8df2661e5b2203c0ddc47798d5da0132d5686bbe"
DETIC_CHECKPOINT = "Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"


@dataclass(frozen=True)
class ResourcePaths:
    root: Path
    data: Path
    weights: Path

    @property
    def mapping(self):
        return self.root / "Mapping/runtime/embodied_splat"

    @property
    def navigation(self):
        return self.root / "Navigation/runtime/lagmemo"

    @property
    def habitat(self):
        return self.root / "Exp/assets/dependencies/habitat-lab"

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

    def navigation_resources(self):
        from runtime.navigation_worker import NavigationResources

        detic = self.navigation / "src/lagmemo/lagmemo/perception/detection/detic/Detic"
        return NavigationResources(
            habitat_config=self.navigation / "project/config/habitat/lagmemo_hm3d.yaml",
            agent_config=self.navigation / "project/config/agent/hm3d_eval.yaml",
            categories=self.navigation / "data/goal_list.json",
            seem_config=self.navigation
            / "src/third_party/seem/configs/seem/focall_unicl_lang_demo.yaml",
            seem_checkpoint=self.weights / "seem_focall_v0.pt",
            mobileclip_checkpoint=self.weights / "mobileclip_s0.pt",
            detic_config=detic
            / "configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml",
            detic_checkpoint=self.weights / "detic" / DETIC_CHECKPOINT,
            scene_directory=self.scene_directory,
            scene_dataset=self.scene_dataset,
        )

    def required_files(self):
        nav = self.navigation_resources()
        return [
            self.scene_dataset,
            self.root / "configs/models/embodied_splat.yaml",
            self.weights / "freesplatpp_scannet_gtdepth.ckpt",
            self.weights / "FastSAM-x.pt",
            self.weights / "maskadpater_ckpt.pth",
            self.weights / "tf_efficientnetv2_s_21ft1k-d7dafa41.pth",
            self.weights / "detic/datasets/metadata/lvis_v1_train_cat_info.json",
            self.weights / "superglue/superpoint_v1.pth",
            self.weights / "superglue/superglue_indoor.pth",
            self.torch_home / "hub/checkpoints/depth-save.pth",
            self.torch_home / "hub/checkpoints/disk_lightglue_v0-1_arxiv-pth",
            *(path for name, path in vars(nav).items() if name != "scene_directory"),
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
    # These vendor constructors still resolve auxiliary weights relative to
    # their source. Make that bridge explicit until resource migration finishes.
    bridges = {
        resources.mapping
        / "pretrained/FastSAM-x.pt": resources.weights
        / "FastSAM-x.pt",
        resources.mapping
        / "src/third_party/maskadapter/maskadpater_ckpt.pth": resources.weights
        / "maskadpater_ckpt.pth",
        resources.navigation
        / "src/lagmemo/lagmemo/perception/detection/detic/Detic/datasets/metadata/lvis_v1_train_cat_info.json": resources.weights
        / "detic/datasets/metadata/lvis_v1_train_cat_info.json",
    }
    matching = (
        resources.navigation
        / "src/lagmemo/lagmemo/agent/imagenav_agent/SuperGluePretrainedNetwork/models/weights"
    )
    for name in ("superpoint_v1.pth", "superglue_indoor.pth"):
        bridges[matching / name] = resources.weights / "superglue" / name
    for vendor, selected in bridges.items():
        if not vendor.is_file() or not vendor.samefile(selected):
            raise ValueError(
                f"Vendor resource bridge differs from the selected weights: {vendor} -> {selected}"
            )
    revision = subprocess.check_output(
        ["git", "-C", str(resources.habitat), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != HABITAT_REVISION:
        raise ValueError(f"Habitat revision differs: {revision}")
    subprocess.run(
        ["git", "-C", str(resources.habitat), "diff", "--quiet", "HEAD"], check=True
    )
    return {
        "habitat_revision": revision,
        "files": list(map(str, resources.required_files())),
        "vendor_resource_bridges": {str(k): str(v) for k, v in bridges.items()},
    }
