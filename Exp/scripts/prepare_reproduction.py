#!/usr/bin/env python3
"""Provision local assets for the frozen Full PBSP profiles without editing policy."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

from _runtime import ROOT
from build_identity_sidecar import FROZEN, build_sidecar, expected_manifest_hash


DETIC_CHECKPOINT = "Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"
POOL = "HM3Dv1_MON_balanced1200_v1_runtime_compat_20260829"
SPLITS = {2: "2MON_600x2", 3: "3MON_400x3", 4: "4MON_300x4"}
SCENES = "HM3Dv1_objectnav_val_20scenes_3donav/data/scene_datasets"
HABITAT_REVISION = "8df2661e5b2203c0ddc47798d5da0132d5686bbe"
HABITAT_REPOSITORY = "https://github.com/weekgoodday/habitat-lab.git"
HABITAT_DEPENDENCY = ROOT / "Exp/assets/dependencies/habitat-lab"


def prepare_habitat(source=None):
    """Copy a committed dependency without sharing objects with an older checkout."""
    if not HABITAT_DEPENDENCY.exists():
        HABITAT_DEPENDENCY.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--no-hardlinks", "--dissociate", "--no-checkout",
                        str(source or HABITAT_REPOSITORY), str(HABITAT_DEPENDENCY)], check=True)
        subprocess.run(["git", "-C", str(HABITAT_DEPENDENCY), "checkout", "--detach", HABITAT_REVISION], check=True)
        subprocess.run(["git", "-C", str(HABITAT_DEPENDENCY), "remote", "set-url", "origin", HABITAT_REPOSITORY], check=True)
    observed = subprocess.check_output(["git", "-C", str(HABITAT_DEPENDENCY), "rev-parse", "HEAD"], text=True).strip()
    if observed != HABITAT_REVISION:
        raise RuntimeError(f"Habitat dependency revision differs: {observed}")
    subprocess.run(["git", "-C", str(HABITAT_DEPENDENCY), "diff", "--quiet", "HEAD"], check=True)


def habitat_pythonpaths():
    paths = [HABITAT_DEPENDENCY / "habitat-lab", HABITAT_DEPENDENCY / "habitat-baselines"]
    if not (paths[0] / "habitat/datasets/goat/goat_dataset.py").is_file():
        raise FileNotFoundError("Pinned Habitat dependency is missing; run prepare_reproduction.py")
    prepare_habitat()
    return paths


def require_file(path):
    if not path.is_file():
        raise FileNotFoundError(path)


def asset_link(path, target):
    """Keep existing valid links; preserve the exported scene placeholder."""
    target = target.absolute()
    if not target.exists():
        raise FileNotFoundError(target)
    if path.is_symlink():
        if path.resolve() != target.resolve():
            raise RuntimeError(f"Existing link has a different target: {path}")
        return
    if path.exists():
        if not path.is_dir() or {p.name for p in path.iterdir()} != {".gitkeep"}:
            raise RuntimeError(f"Refusing to replace existing assets: {path}")
        backup = ROOT / "Exp/assets/scene_datasets.placeholder"
        if backup.exists():
            raise RuntimeError(f"Placeholder backup already exists: {backup}")
        path.rename(backup)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Keep links portable inside the checkout. The dataset entrypoint is external.
    value = os.path.relpath(target, path.parent) if target.is_relative_to(ROOT) else str(target)
    path.symlink_to(value, target_is_directory=True)


def copy_missing(source, target):
    if target.exists():
        if target.stat().st_size != source.stat().st_size:
            raise RuntimeError(f"Existing asset has a different size: {target}")
        if file_sha256(target) != file_sha256(source):
            raise RuntimeError(f"Existing asset has different contents: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def runtime_text(mapping_python, navigation_python):
    text = (
        '[python]\n'
        f'mapping = {json.dumps(str(mapping_python))}\n'
        f'navigation = {json.dumps(str(navigation_python))}\n\n'
        '[datasets]\n'
    )
    aliases = [
        ("hm3dv1_mon_val", 3), ("hm3dv1_2mon_val", 2), ("hm3dv1_4mon_val", 4),
        ("hm3dv1_mon_balanced_2mon", 2), ("hm3dv1_mon_balanced_3mon", 3),
        ("hm3dv1_mon_balanced_4mon", 4),
    ]
    for name, mon in aliases:
        text += f'{name} = "${{TDONAV_ROOT}}/Exp/assets/datasets/{POOL}/{SPLITS[mon]}"\n'
    for name, folder in [
        ("goat_val_seen", "GOATbench_object_valseen_36scenes"),
        ("hm3dv1_objectnav_val", "HM3Dv1_objectnav_val_20scenes"),
    ]:
        text += f'{name} = "${{TDONAV_ROOT}}/Exp/assets/datasets/{folder}"\n'
    return text + '\n[shared]\nhf_home = "${TDONAV_ROOT}/Exp/assets/weights/hf_home"\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping-python", type=Path, required=True)
    parser.add_argument("--navigation-python", type=Path, required=True)
    parser.add_argument("--datasets", type=Path, required=True,
                        help="Existing shared dataset catalog, normally /data/3DONav_datasets")
    parser.add_argument("--weights", type=Path, required=True,
                        help="Provisioned EmbodiedSplat/SEEM/MobileCLIP/HF weight directory")
    parser.add_argument("--detic", type=Path, required=True,
                        help="Detic asset source containing models/ and datasets/metadata/")
    parser.add_argument("--habitat-source", type=Path,
                        help="Optional local Git repository to copy the pinned Habitat dependency from")
    args = parser.parse_args()
    mapping_python = args.mapping_python.expanduser().absolute()
    navigation_python = args.navigation_python.expanduser().absolute()
    datasets = args.datasets.expanduser().absolute()
    weights = args.weights.expanduser().absolute()
    detic = args.detic.expanduser().absolute()

    # Validate inputs before copying files or replacing the scene placeholder.
    for python in (mapping_python, navigation_python):
        require_file(python)
        if not os.access(python, os.X_OK):
            raise PermissionError(python)
    identities = {}
    frozen = json.loads(FROZEN.read_text())
    for mon, split in SPLITS.items():
        expected = expected_manifest_hash(mon)
        manifest = datasets / POOL / split / "dataset_manifest.json"
        require_file(manifest)
        if hashlib.sha256(manifest.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Dataset manifest does not match the frozen {mon}-MON profile: {manifest}")
        identities[mon] = build_sidecar(datasets / POOL / split, mon, expected, frozen)
    scene_source = datasets / SCENES
    if not scene_source.is_dir():
        raise FileNotFoundError(scene_source)
    for name in ("freesplatpp_scannet_gtdepth.ckpt", "FastSAM-x.pt", "maskadpater_ckpt.pth",
                 "mobileclip_s0.pt", "seem_focall_v0.pt"):
        require_file(weights / name)
    model_source = detic / "models" / DETIC_CHECKPOINT
    metadata_source = detic / "datasets/metadata/lvis_v1_train_cat_info.json"
    require_file(model_source)
    require_file(metadata_source)
    text = runtime_text(mapping_python, navigation_python)
    runtime = ROOT / "Exp/runtime.local.toml"
    if runtime.exists() and runtime.read_text() != text:
        raise RuntimeError(f"Existing runtime configuration differs; review it before replacing: {runtime}")

    assets = ROOT / "Exp/assets"
    assets.mkdir(exist_ok=True)
    prepare_habitat(args.habitat_source)
    target_weights = assets / "weights"
    if not target_weights.exists():
        shutil.copytree(weights, target_weights, symlinks=True)
    asset_link(assets / "datasets", datasets)
    detic_assets = target_weights / "detic"
    copy_missing(model_source, detic_assets / DETIC_CHECKPOINT)
    copy_missing(metadata_source, detic_assets / "datasets/metadata" / metadata_source.name)
    navigation = ROOT / "Navigation/runtime/lagmemo"
    detic_runtime = navigation / "src/lagmemo/lagmemo/perception/detection/detic/Detic"
    asset_link(detic_runtime / "models", detic_assets)
    asset_link(detic_runtime / "datasets", detic_assets / "datasets")
    asset_link(navigation / "data/scene_datasets", assets / "datasets" / SCENES)
    if not runtime.exists():
        runtime.write_text(text)
    for mon, payload in identities.items():
        output = assets / "identities" / f"full_pbsp_{mon}mon.json"
        output.parent.mkdir(exist_ok=True)
        identity_text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if output.exists() and output.read_text() != identity_text:
            raise RuntimeError(f"Existing identity sidecar differs: {output}")
        if not output.exists():
            output.write_text(identity_text)
    print(f"Prepared {runtime}")
    print("Algorithms and tracked experiment profiles were not edited.")
    print("Model and dataset contents remain under ignored asset paths.")
    print(f"Set ES_SCSP_IDENTITY_SIDECAR={assets / 'identities/full_pbsp_4mon.json'} for 4-MON.")


if __name__ == "__main__":
    main()
