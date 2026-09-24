# LagMemo runtime layout

This is Navigation-owned runtime source, not another checkout.

- `src/lagmemo/`: 3DONav's action, perception, mapping and policy runtime.
- `src/third_party/`: vendored Detectron2, Habitat-Lab, MobileCLIP, SEEM and frontier code. The adapter adds the
  required package roots explicitly; it never adds the parent `third_party/` directory, avoiding the SEEM package
  shadowing problem.
- `project/`: evaluator configuration and Habitat glue.
- `data/`: a small internal goal list and a symlink to the canonical 3DONav HM3D asset root.
- `datadump/`: generated runtime output only.

Upstream revision, local patches and generated-path rules are recorded in `../vendor_manifest.toml`. Model files
are always read from `Exp/assets/weights` through internal symlinks or explicit evaluator arguments.
