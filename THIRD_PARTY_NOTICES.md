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
