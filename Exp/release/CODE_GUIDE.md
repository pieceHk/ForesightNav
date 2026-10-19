# Full PBSP code guide

Full PBSP runs online geometric and semantic mapping, maintains evidence about
candidate goals, and uses the remaining ordered tasks to rank the next candidate.
The `full-pbsp` snapshot selects `ordered_suffix_mvp`. Its 4-MON profile is
`Exp/profiles/hm3dv1_4mon_scsp_suffix.toml`.

The package name `tdonav` and the experiment launcher's historical name
`run_goat.py` are retained so that the frozen experiment remains traceable.

## Repository boundaries

| Directory | Responsibility | Start reading here |
| --- | --- | --- |
| `Mapping/src/tdonav_mapping/` | Observation, coordinate, keyframe, query and resource contracts | `contracts.py`, `coordinates.py`, `query.py` |
| `Mapping/runtime/server/` | Live RGB-D input, Gaussian reconstruction and semantic candidate retrieval | `online_es_mapper.py`, `es_mapper_server.py` |
| `Mapping/runtime/embodied_splat/` | Neural mapping engine and its dependencies | Enter through the server adapter |
| `Navigation/src/tdonav_navigation/` | Candidate evidence, ranking, ordered suffix planning and MON metrics | Reading order below |
| `Navigation/runtime/evaluators/` | Connects the planner, mapper and Habitat action loop | `goat/entrypoint.py` |
| `Navigation/runtime/lagmemo/` | Navigation execution, verification, simulator integration and vendored libraries | `src/lagmemo/lagmemo/agent/lagmemo_agent/glue_agent.py` |
| `Exp/` | Dataset identities, experiment controls, launch, validation and output records | `profiles/`, `scripts/run_goat.py` |

## Core reading order

1. Read `Navigation/src/tdonav_navigation/scsp_contracts.py` for episode identity,
   map/query version tokens, candidate records and explicit geodesic status.
   These contracts keep scene, episode and task indices attached to a decision.
2. Read `evidence.py`, `ranking.py` and `category_cooldown.py` for recurring
   candidate evidence, candidate utility and controls on switching active goals.
3. Read `suffix_planning.py`. `plan_ordered_suffix` consumes the current candidate
   ordering and strict geodesic matrices, then returns a deterministic current-task
   ordering. It is a pure module: the caller supplies candidates and distances.
4. Read `Navigation/runtime/evaluators/goat/entrypoint.py` to follow candidate
   refresh, geodesic construction, plan dispatch, verification and action execution.
5. Read `scsp_telemetry.py` and `mon_metrics.py` to connect a decision to the
   recorded episode identity and the SR, SPL, PeSPL and PSR computations.

`Mapping/src/tdonav_mapping/` provides interfaces and small policies. The actual
online neural mapper lives under `Mapping/runtime/server/` and
`Mapping/runtime/embodied_splat/`; reading only the interface package does not
cover the mapping implementation.

## Experiment flow

```text
tracked profile + local runtime paths + ordered episode dataset
    -> Exp/scripts/run_goat.py
    -> mapping process receives live RGB-D observations
    -> mapper returns spatial candidates and query metadata
    -> evaluator updates evidence and candidate ordering
    -> ordered suffix planner considers remaining tasks
    -> LagMemo executes actions and verifies goals
    -> episode telemetry and MON metrics are written under Exp/runs/
```

The planner receives strict geodesic availability and reachability status. The
4-MON profile sets `strict_geodesic = true` and `euclidean_fallback = false`.
The protocol contains reserved evidence states and experimental modes; their
presence in a type definition does not mean that the active profile uses them.
For example, frontier reranking, the approach ring and candidate-exhaustion
exploration are disabled in the tracked 4-MON profile.

## Snapshot selection

| Branch | Saved source revision | Evaluation shape |
| --- | --- | --- |
| `full-pbsp-2mon` | `ea5efef0` | 600 episodes with 2 ordered targets |
| `full-pbsp-3mon` | `96e91424` | 400 episodes with 3 ordered targets |
| `full-pbsp` | `7f232ca7` | 300 episodes with 4 ordered targets |

Each balanced pool has 20 scenes and 1,200 ordered targets. These are exported
snapshots with new Git commit IDs. `SOURCE_MANIFEST.json` records their original
source lineage. The `baseline*` branches preserve separate reference and runtime
recovery snapshots; consult the version and evaluation notes before comparing
their recorded results.

## Local reproduction assets

`Exp/scripts/prepare_reproduction.py` checks the frozen dataset manifest hashes,
copies local model assets into ignored `Exp/assets/weights/`, connects shared
datasets and writes ignored `Exp/runtime.local.toml`. It also supplies Detic's
checkpoint and LVIS category-frequency metadata, which are needed at inference
time.

`build_identity_sidecar.py` validates each compressed episode file against the
dataset manifest and matches every ordered task to `source_trace.json`. Its output
contains scene ID, runtime episode ID, source episode IDs and ordered categories.
It omits success labels, outcomes and goal geometry. The 3-MON sidecar omits the
48 identities already embedded in the frozen protocol, avoiding duplicate records.
Pass the resulting file through `ES_SCSP_IDENTITY_SIDECAR` when launching.

Scene and Detic resource entrypoints are tracked relative links into `Exp/assets/`.
The original scene placeholder is retained under
`Exp/release/placeholders/scene_datasets.gitkeep`, with its original source hash
in `SOURCE_MANIFEST.json`. Model and dataset contents remain ignored.

`run_full_pbsp.py` supplies the selected profile and identity sidecar to the
original launcher. Its default smoke scope runs one complete episode. See
`LOCAL_REPRODUCTION.md` for commands and the tested environment inventory.

The server reproduction reuses existing mapping and navigation environments.
The launcher loads a separate Habitat checkout under ignored
`Exp/assets/dependencies/`, pinned to the same clean fork revision as the
installed dependency. The existing environments supply compiled CUDA packages;
a fresh installation still requires those builds to be provisioned.

## Reviewing changes

Keep algorithm edits, resource preparation and generated results separate in
review. Run output, private runtime paths, licensed scenes and model weights
belong under the existing ignored paths. Review changes against
`origin/full-pbsp`; keep the frozen profiles and original source hashes intact
when establishing the reproduction baseline. Resource-placeholder relocations
retain the original source hash and an explicit release path in the manifest.

A single completed episode demonstrates the mapping/navigation runtime path.
Dataset-wide SR/SPL claims require the complete selected split and its recorded
provenance. The dataset's episode order and task identities must remain unchanged.
