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
and `pbsp/mapping/features.py`, and the view-distance and image-grid helpers in
`pbsp/mapping/geometry.py`, were extracted from the same runtime and its geometry
and tensor utility modules. The same attribution and license scope apply.
`adapters/embodied_splat_model.py` retains the network constructor and optional
sparse/OpenSeg backend methods from that encoder.

Candidate refresh, visual-confirmation history and recovery-state operations in
`pbsp/navigation/candidates.py`, `verification.py` and `recovery.py` were extracted
from LagMemo's `glue_agent.py`. The upstream Meta copyright notice and MIT scope
are preserved in those modules.
The navigation kernel in `pbsp/navigation/navigator.py`, local-map updates in
`local_map.py`, geometric transforms in `geometry.py`, and the matching/assembly
code in `adapters/perception.py` and `adapters/lagmemo_agent.py` have the same
LagMemo provenance. Observation and action types are supplied through adapter
boundaries; the upstream names remain available through compatibility imports.

Upstream license texts are retained unchanged:

- `Mapping/runtime/embodied_splat/src/third_party/MinkowskiEngine/LICENSE`
- `Mapping/runtime/embodied_splat/src/third_party/langsplat-rasterization/LICENSE.md`
- `Mapping/runtime/embodied_splat/src/third_party/langsplat-rasterization/third_party/glm/copying.txt`
- `Navigation/runtime/lagmemo/src/lagmemo/lagmemo/agent/imagenav_agent/SuperGluePretrainedNetwork/LICENSE`
- `Navigation/runtime/lagmemo/src/lagmemo/lagmemo/perception/detection/detic/Detic/LICENSE`
- `Navigation/runtime/lagmemo/src/lagmemo/lagmemo/perception/detection/detic/Detic/third_party/CenterNet2/LICENSE`
- `Navigation/runtime/lagmemo/src/lagmemo/lagmemo/perception/detection/detic/Detic/third_party/Deformable-DETR/LICENSE`
- `Navigation/runtime/lagmemo/src/third_party/detectron2/LICENSE`
- `Navigation/runtime/lagmemo/src/third_party/frontier_exploration/LICENSE`
- `Navigation/runtime/lagmemo/src/third_party/habitat-lab/LICENSE`
- `Navigation/runtime/lagmemo/src/third_party/ml-mobileclip/LICENSE`
- `Navigation/runtime/lagmemo/src/third_party/ml-mobileclip/LICENSE_weights_data`
- `Navigation/runtime/lagmemo/src/third_party/seem/LICENSE`
