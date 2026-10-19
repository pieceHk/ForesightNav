# Linux Full PBSP reproduction

Run from the repository root with the navigation Python environment activated.
The default smoke run executes one complete episode, including all four ordered
targets with the original limit of 500 actions per target.

## Prepare resources

The benchmark scenes, generated ordered episodes and model weights are local
resources. The dataset catalog must contain the balanced common-1200 runtime
views and the HM3Dv1 scene assets described in `REPRODUCIBILITY.md`.

```bash
python Exp/scripts/prepare_reproduction.py \
  --mapping-python /path/to/mapping-env/bin/python \
  --navigation-python /path/to/navigation-env/bin/python \
  --datasets /data/3DONav_datasets \
  --weights /path/to/provisioned/weights \
  --detic /path/to/provisioned/Detic
```

The weights directory supplies the EmbodiedSplat checkpoint, FastSAM,
MaskAdapter, SEEM, MobileCLIP, SuperGlue and Hugging Face cache. The Detic source
supplies `models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth` and
`datasets/metadata/lvis_v1_train_cat_info.json`. Resource files are copied into
ignored `Exp/assets/weights/`. Shared benchmark data is linked through
`Exp/assets/datasets/`.

Preparation checks all three dataset manifest hashes and the compressed episode
contents. It creates identity sidecars from the source trace and checks each
task's scene, runtime episode ID, source episode ID, position and category.
Success labels and outcomes are omitted from these sidecars.

Existing runtime configuration or links with different contents are rejected.
The repository tracks relative scene and Detic resource links. The original
scene placeholder is preserved under
`Exp/release/placeholders/scene_datasets.gitkeep`; its original hash and new
release path are recorded in `SOURCE_MANIFEST.json`.

Preparation also clones the Habitat fork at
[`8df2661e5b2203c0ddc47798d5da0132d5686bbe`](https://github.com/weekgoodday/habitat-lab/commit/8df2661e5b2203c0ddc47798d5da0132d5686bbe)
under ignored `Exp/assets/dependencies/`. On an offline server, add
`--habitat-source /path/to/existing/habitat-lab` to copy committed Git objects
locally. The copy does not share objects with the old checkout. The launcher
verifies the revision and loads Habitat and Habitat Baselines from this copy.

## Check and launch

```bash
python Exp/scripts/run_full_pbsp.py --mon 4 --scope smoke --check-only
python -m pytest Mapping/tests Navigation/tests Exp/tests -q
python Exp/scripts/validate.py
python Exp/scripts/run_full_pbsp.py --mon 4 --scope smoke --gpus 0,1
```

GPU order is mapping,navigation. The wrapper chooses the tracked Full PBSP
profile and supplies its identity sidecar to the original evaluator. It requires
`ordered_suffix_mvp` and keeps the original policy settings and episode order.

The smoke defaults are scene range `[0, 1)` and episode range `[0, 1)`. Use the
range flags to choose a different deterministic slice. A complete 4-MON split
contains 20 scenes, 300 episodes and 1,200 ordered targets:

```bash
python Exp/scripts/run_full_pbsp.py --mon 4 --scope full --gpus 0,1 --tag full_4mon
```

`--mon 2` and `--mon 3` select another profile and dataset within the current
checkout. They do not switch to the historical 2-MON or 3-MON source snapshot.
Use the branch/tag table in `CODE_GUIDE.md` when the earlier code version matters.

## Outputs and review

Each launch creates a fresh directory under `Exp/runs/`. Check `status.json`,
`provenance.json`, `logs/mapping.log`, the scene navigation log,
`scsp_decisions.jsonl`, `per_episode_metrics.json` and the aggregated
`mon_metrics.json`. The last three may be nested under the scene results
directory. In-place resumption of a metric run is unsupported.

The tracked 4-MON profile disables per-keyframe image writes. Planner maps remain
enabled for SCSP. Enabling screenshots changes the frozen experiment controls;
record it as a separate diagnostic run.

`verify_source.py` checks committed retained source blobs and requires tracked
working files to match the commit. During review, use
`git diff origin/full-pbsp` to inspect the pending packaging changes. The source
manifest preserves the original hashes, including the relocated placeholder.

## Server environment inventory

The 2026-10-19 reproduction uses two existing environments on an A40 server:

| Runtime | Python | PyTorch | torchvision | NumPy |
| --- | --- | --- | --- | --- |
| Mapping | 3.10 | 2.7.1+cu118 | 0.22.1+cu118 | 2.1.3 |
| Navigation | 3.9 | 2.1.2+cu118 | 0.16.2+cu118 | 1.23.5 |

Navigation reports Habitat-Lab 0.2.3, Habitat-Sim 0.3.3 and Detectron2 0.6.
The original environment installs customized Habitat from an older source
checkout. The new launcher selects the separate pinned copy under this checkout;
scene reset and a real simulator action have been checked with this copy.
PBSP's LagMemo and Detectron2 Python modules resolve from this repository.
Mapping's compiled deformable aggregation extension and MinkowskiEngine resolve
from its existing environment.

This environment inventory supports reproduction on the current server. A
fresh-machine installation still needs its Python environments and
CUDA extension builds validated. Keep this distinction when
describing the scope of the release.

## Recorded smoke result

The 2026-10-19 check completed scene `4ok3usBNeis`, runtime episode `0`, with
four ordered targets and 2,000 actions. All four targets reached the 500-action
limit: successful subtasks were `0/4`, and SR, SPL, PSR and PeSPL were zero.
This validates process execution and metric output. Full-split performance and
agreement with the historical results remain untested.

The telemetry contains 204 decisions: 147 suffix dispatches and 57 Idea-1
fallbacks. It records 18 suffix interventions and zero stale suffix dispatches.
Every decision carries the complete episode identity. The test suite passes
all 231 checks, including reference mapping equivalence and the new identity
alignment/outcome-isolation tests.

`REPRODUCTION_VALIDATION.json` records the source revisions, dataset and profile
hashes, sample metrics and validation scope. Detailed generated evidence is under
`Exp/runs/reproduction_setup/` and the recorded smoke run directory.
