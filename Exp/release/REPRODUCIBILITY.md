# Reproduction guide

## Environment and assets

Use Linux with CUDA-capable GPUs. The tested deployment used separate mapping and navigation Python environments. `pyproject.toml` lists only lightweight project dependencies and is **not** a complete GPU environment lockfile. No fresh GPU environment installation or end-to-end evaluation was validated during packaging.

Inspect the preserved runtime setup/configuration and upstream license files:
- `Navigation/runtime/lagmemo/src/environment.yml`
- `Navigation/runtime/lagmemo/src/lagmemo/setup.py`
- vendored Habitat-Lab, Detectron2, SEEM and MobileCLIP installation notes
- `Mapping/runtime/embodied_splat/src/ops/setup.py`, `simple-knn`, MinkowskiEngine and LangSplat rasterization CUDA extensions

Obtain HM3D scenes and model checkpoints under their respective licenses. Place datasets under `Exp/assets/datasets/` and model files under `Exp/assets/weights/`. Runtime source references EmbodiedSplat, Detic, SEEM, MobileCLIP and associated model/configuration assets. Media and weights are excluded; file omissions are listed in SOURCE_MANIFEST.json. Relative symlinks must remain symlinks on Linux (Windows checkouts may materialize them as text).

Use the exact **balanced common1200 v1** episode pool: 2MON_600x2, 3MON_400x3 and 4MON_300x4, each with 20 scenes and 1200 ordered targets. Dataset manifests for 2/3/4-MON have SHA256:
- 2-MON: dfcf14d1fe7edef9c9d7bd8286dfe8067c91f56ef46ca2be4524d7e01461ac62
- 3-MON: c3245c4bc0322d618b12e2845f138842c5a2553c2b0f4bfbdfffba641c415165
- 4-MON: e1e93c34fd5cadcf3d001f9d5e68f4bbc4dec69ba5df88ad230885b319c4e892

This repository does not distribute those generated episode files or guarantee that a newly generated split matches them. Exact numerical reproduction requires the original manifests and ordered episodes, not simply the same total count.

Copy `Exp/release/runtime.example.toml` to ignored `Exp/runtime.local.toml` and set your interpreter paths. Supply the matching local datasets. No credentials belong in this file.

## Profile selection

Full PBSP:
- 3-MON tag: `Exp/profiles/hm3dv1_mon_scsp_suffix.toml`
- 2-MON tag: `Exp/profiles/hm3dv1_2mon_scsp_suffix.toml`
- 4-MON tag: `Exp/profiles/hm3dv1_4mon_scsp_suffix.toml`

The launcher for **MON** is `Exp/scripts/run_goat.py`: its historical name refers to the multi-target evaluator. `run_hm3dv1_objectnav.py` is single-target ObjectNav, not this experiment.

Example after selecting the 3-MON tag and provisioning assets:
```bash
python Exp/scripts/run_goat.py --runtime Exp/runtime.local.toml   --profile Exp/profiles/hm3dv1_mon_scsp_suffix.toml   --gpus 0,1 --scene-start 0 --scene-end 20 --tag paper_3mon
```
GPU order is mapping,navigation. Use the corresponding profile and tag for 2/4-MON. Run records are generated under ignored `Exp/runs/`. Output provenance will contain the new export commit; SOURCE_MANIFEST.json connects it to the historical source commit.

Baseline: use a matching captured profile from `Exp/release/experiment_profiles/baseline_*mon_*.toml` with the corresponding Baseline tag. Do not silently substitute the older generic `semantic_source=auto` profile for the captured experiment profile. The Baseline archive explicitly records `semantic_source = "disabled"`; preserve this field rather than relabeling it as the Full PBSP Detic control. Full PBSP captured profiles are also retained there for comparison; the frozen Full PBSP provenance logic requires the tracked `Exp/profiles` path, so launch with the tracked profiles above.

All captured profiles are exported byte-for-byte with SHA256 verification. The historical Baseline 3-MON result combines earlier and later source versions; selecting one tag for a fresh full run does not reproduce that mixed recovery history exactly. Similarly, Baseline 4-MON includes runtime-fix stages.

## Lightweight checks
```bash
python -m pip install numpy tomli pytest
python -m pytest Mapping/tests Navigation/tests -q
python Exp/release/verify_source.py
```
Baseline's GOAT sensor test additionally needs OmegaConf/Habitat dependencies. Full `Exp/tests` and `Exp/scripts/validate.py` depend on populated local assets, runtime configuration and platform-specific behavior. Do not interpret lightweight unit tests as full simulation validation.
