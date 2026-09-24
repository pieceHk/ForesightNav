# Navigation runtime

`lagmemo/` contains the evaluator runtime. `evaluators/goat` and `evaluators/hm3dv1` are dataset entrypoints
plus their small socket/resource helpers. Checkpoints resolve to `Exp/assets/weights` through internal links.
The adapter selects an entrypoint by name; it cannot take an external source or working directory.
