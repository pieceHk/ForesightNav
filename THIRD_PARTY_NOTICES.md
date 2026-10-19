# Third-party notices

PBSP-authored code is licensed under the MIT license in [LICENSE](LICENSE).
Vendored software, adaptations of upstream components, model weights and datasets
retain their original licenses and attribution notices.

The mapping runtime derives from EmbodiedSplat, and the navigation runtime derives
from LagMemo. Their source revisions and retained paths are recorded in
[Mapping/runtime/vendor_manifest.toml](Mapping/runtime/vendor_manifest.toml),
[Navigation/runtime/vendor_manifest.toml](Navigation/runtime/vendor_manifest.toml)
and the original [source manifest](Exp/release/SOURCE_MANIFEST.json).

The Gaussian storage in `pbsp/mapping/gaussians.py` and the projection, fusion and
refinement operations in `pbsp/mapping/fusion.py` were extracted from
`Mapping/runtime/embodied_splat/src/model/encoder/encoder_test_unet3d_online.py`.
They retain their EmbodiedSplat provenance and upstream license scope.
The online encoding and region-feature operations in `pbsp/mapping/encoder.py`
and `pbsp/semantics/features.py`, and the view-distance and image-grid helpers in
`pbsp/mapping/geometry.py`, were extracted from the same runtime and its geometry
and tensor utility modules. The same attribution and license scope apply.
`adapters/embodied_splat_model.py` retains the network constructor and optional
sparse/OpenSeg backend methods from that encoder.

Candidate refresh, visual-confirmation history and recovery-state operations in
`pbsp/navigation/candidates.py`, `verification.py` and `recovery.py` were extracted
from LagMemo's `glue_agent.py`. The upstream Meta copyright notice and MIT scope
are preserved in those modules.
The Agent state and candidate execution in `pbsp/agent.py`, the navigation kernel
in `pbsp/navigation/navigator.py`, local-map updates in
`local_map.py`, geometric transforms in `geometry.py`, and the matching/assembly
code in `adapters/perception.py` and `adapters/lagmemo_agent.py` have the same
LagMemo provenance. Observation and action types are supplied through adapter
boundaries; the upstream names remain available through compatibility imports.

The sensor/action conversion in `adapters/habitat.py`, measurement reset in
`experiments/evaluator.py`, Habitat assembly in `adapters/goat.py`, and
top-down visualization in `experiments/visualization.py` derive from LagMemo's
Habitat environment and agent. They retain the same Meta copyright and MIT scope.
Evaluation maps are read by the experiment observer separately from policy input.
The episode/action loop in `experiments/runner.py` was extracted from the retained
GOAT evaluator entrypoint, which adapts LagMemo's evaluation loop. The same
attribution applies to that retained control flow.
The Habitat configuration adapter in `adapters/habitat_config.py` and the
navigation service assembly in `runtime/navigation_worker.py` retain the same
LagMemo attribution. The old Habitat and configuration modules forward to these
implementations.

Upstream license texts are retained unchanged:

- `third_party/licenses/clip/LICENSE`
- `third_party/licenses/gaussian_rasterizer/LICENSE.md`
- `third_party/licenses/gaussian_rasterizer/glm-copying.txt`
- `Mapping/runtime/embodied_splat/src/third_party/MinkowskiEngine/LICENSE`
- `Mapping/runtime/embodied_splat/src/third_party/langsplat-rasterization/LICENSE.md`
- `Mapping/runtime/embodied_splat/src/third_party/langsplat-rasterization/third_party/glm/copying.txt`
- `Navigation/runtime/lagmemo/src/lagmemo/lagmemo/agent/imagenav_agent/SuperGluePretrainedNetwork/LICENSE`
- `Navigation/runtime/lagmemo/src/lagmemo/lagmemo/perception/detection/detic/Detic/LICENSE`
- `Navigation/runtime/lagmemo/src/lagmemo/lagmemo/perception/detection/detic/Detic/third_party/CenterNet2/LICENSE`
- `Navigation/runtime/lagmemo/src/lagmemo/lagmemo/perception/detection/detic/Detic/third_party/Deformable-DETR/LICENSE`
- `third_party/licenses/detectron2/LICENSE`
- `Navigation/runtime/lagmemo/src/third_party/frontier_exploration/LICENSE`
- `third_party/licenses/habitat_lab/LICENSE`
- `Navigation/runtime/lagmemo/src/third_party/ml-mobileclip/LICENSE`
- `Navigation/runtime/lagmemo/src/third_party/ml-mobileclip/LICENSE_weights_data`
- `Navigation/runtime/lagmemo/src/third_party/seem/LICENSE`

The mapping installer builds [OpenAI CLIP](https://github.com/openai/CLIP/tree/d05afc436d78f1c48dc0dbf8e5980a9d471f35f6)
and the [depth rasterizer](https://github.com/JonathonLuiten/diff-gaussian-rasterization-w-depth/tree/cb65e4b86bc3bd8ed42174b72a62e8d3a3a71110)
from pinned source archives. Their license texts above are copied byte for byte
from those archives. The depth rasterizer retains the Gaussian-Splatting license;
PBSP's MIT license applies to PBSP-authored code. Versions, artifact checksums and
tracked extension source hashes are recorded in [the mapping lock](envs/mapping/linux-64.lock.json).

Detectron2 and Habitat-Lab use complete pinned external source archives from the
[navigation lock](envs/navigation/linux-64.lock.json). Their unchanged license
texts are retained above. The navigation installer builds Detectron2, Habitat-Lab
and Habitat-Baselines; runtime preparation exports the same Habitat-Lab source
revision for the sensor and task adapters.

The semantic pooling/codebook operations now live in `pbsp/semantics/features.py`;
`pbsp/mapping/features.py` retains a compatibility import. The same EmbodiedSplat
attribution and license scope apply. Semantic retrieval now lives in `pbsp/query/`;
its Gaussian storage and coordinate adapter is `adapters/embodied_splat_query.py`.
