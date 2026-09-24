# Method and Runtime Alignment

Source method note: `Online ES-LagMemo-Integration`, read 2026-08-09.

| Method responsibility | 3DONav owner | Internal runtime anchor |
|---|---|---|
| Habitat RUB?RDF, keyframes, persistent fusion, query/NMS | Mapping | `runtime/server/es_mapper_server.py`, `runtime/embodied_splat/` |
| Candidate projection, visual STOP, trapped recovery | Navigation | `runtime/lagmemo/`, `runtime/evaluators/{goat,hm3dv1}/` |
| Dataset identity, profiles, launch, provenance, monitoring | Exp | `datasets/`, `profiles/`, `scripts/`, `runs/` |

## Stage 3: owned runtime boundary

The trained mapper, EmbodiedSplat implementation, LagMemo/Habitat runtime, and both evaluator entrypoints now live
under their respective owner. The adapters select only internal entrypoints; they no longer accept a source path or a
working directory. `runtime.local.toml` carries only interpreter paths and internal asset paths expanded from
`${TDONAV_ROOT}`.

| Acceptance check | Result |
|---|---|
| Functional root owners | `Mapping`, `Navigation`, `Exp` only |
| Source/runtime legacy worktree paths | none outside ignored historical run artifacts |
| Mapper, GOAT evaluator, HM3D evaluator direct import/help smoke | passed |
| Structural unit suite | 7 passed, 1 skipped |
| Runtime validator | passed; rejects escaping symlinks for selected runtime assets |
| Local data/weights | explicit weights, HF cache, CLIP tokenizer, GOAT data, and a physical HM3D scene copy are under `Exp/assets/` |

The prior matched-prefix records remain baseline evidence, but a fresh full benchmark is required before claiming that
the relocation preserves final SR/SPL. Existing GPU processes that began before this migration are historical legacy
runs; newly launched 3DONav experiments resolve only the internal runtime tree.
