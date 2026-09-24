# Exp

`Exp` owns experiment identity and evidence, not mapping or navigation policy.

- `assets/`: local datasets, explicit model weights and HF cache; ignored by Git but fixed by manifest/layout.
- `datasets/`: tracked dataset catalog.
- `profiles/`: tracked GOAT and HM3D controls.
- `scripts/`: launch, lifecycle, monitor and architecture/runtime validation.
- `runs/`: ignored generated artifacts, logs, images and metrics.

`runtime.local.toml` may select local Python interpreters only. All project data/runtime paths must expand beneath
`${TDONAV_ROOT}`; it must not name another `/nvme/khy/...` worktree.

## Shared visualization contract

Both tracked profiles enable `[visualization].per_keyframe_snapshots`. The launchers pass that control to the
owned evaluator, and `Navigation/runtime/lagmemo/.../visualizer.py` writes the same `snapshot_<step>_<index>.png`
artifact for GOAT and HM3Dv1. Each artifact uses a fixed 3×2 grid with `480×450` panels: Goal Image, Observation,
Semantic, Predicted Map, Oracle Top-Down Map, and Navigable Map. ObjectNav keeps the Goal Image cell with a target
category placeholder; task metadata is a compact footer rather than an overlay on the map panels.
