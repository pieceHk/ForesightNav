# Third-party notices

PBSP-authored code uses the MIT license in [LICENSE](LICENSE). Adapted upstream code, dependencies, weights and datasets retain their original licenses and attribution notices.

The reconstruction layers derive from EmbodiedSplat revision `b5e1e90fc9917fbfe34231684b823725b7e1e948`. Its [source record](third_party/embodied_splat/UPSTREAM.toml) identifies the retained inference scope. Gaussian storage, projection, fusion and refinement in `pbsp/mapping/gaussians.py` and `fusion.py`, online encoding in `encoder.py`, semantic pooling in `pbsp/semantics/features.py`, and geometry helpers retain their EmbodiedSplat provenance and license scope. `pbsp/adapters/embodied_splat_model.py` constructs the upstream neural layers and optional feature backends.

Navigation support derives from LagMemo revision `df48cfca2d0c5646ea012c72bc006da279512898`; see its [source record](third_party/lagmemo/UPSTREAM.toml). Agent execution, local-map updates, candidate history, target verification, recovery, Habitat conversion, experiment visualization and the evaluation loop retain the upstream Meta copyright and MIT scope. These adaptations are in `pbsp/agent.py`, `pbsp/navigation/`, `pbsp/adapters/` and `experiments/`.

Retained license texts:

- [OpenAI CLIP](third_party/licenses/clip/LICENSE)
- [Gaussian depth rasterizer](third_party/licenses/gaussian_rasterizer/LICENSE.md) and its [GLM notice](third_party/licenses/gaussian_rasterizer/glm-copying.txt)
- [Language rasterizer](third_party/langsplat-rasterization/LICENSE.md) and its [GLM notice](third_party/langsplat-rasterization/third_party/glm/copying.txt)
- [MinkowskiEngine](third_party/licenses/MinkowskiEngine-LICENSE)
- [Detic](third_party/detic/LICENSE), [CenterNet2](third_party/detic/third_party/CenterNet2/LICENSE) and [Deformable-DETR](third_party/detic/third_party/Deformable-DETR/LICENSE)
- [Detectron2](third_party/licenses/detectron2/LICENSE)
- [Frontier exploration](third_party/frontier_exploration/LICENSE)
- [Habitat-Lab](third_party/licenses/habitat_lab/LICENSE)
- [MobileCLIP code](third_party/mobileclip/LICENSE) and [weights/data](third_party/mobileclip/LICENSE_weights_data)
- [SEEM](third_party/seem/LICENSE)
- [SuperGlue](third_party/licenses/superglue/LICENSE)

FastSAM and the retained Ultralytics files carry their upstream AGPL-3.0 source notices. Those notices remain in the corresponding source files.

The mapping installer builds OpenAI CLIP revision `d05afc436d78f1c48dc0dbf8e5980a9d471f35f6` and the depth rasterizer revision `cb65e4b86bc3bd8ed42174b72a62e8d3a3a71110` from pinned source archives. The rasterizer retains its Gaussian-Splatting license. Detectron2 and Habitat use complete external source archives pinned in the [navigation lock](configs/environments/navigation/linux-64.lock.json). The unchanged license texts above accompany these dependencies.

SuperGlue inference files are prepared separately from revision `d2d6dcc21f67690c07a2787acfb357b7ed59b920` of the fork used by LagMemo. Its license limits use to noncommercial research and restricts distribution. The [source manifest](configs/resources/superglue.json) pins six downloaded files and two explicit checkpoint-path adaptations.

Fixed MON task orders and source identities are packaged in `configs/resources/episodes/`. Preparation reads selected annotations from the official Habitat ObjectNav archive. Scene data and generated annotations retain the [dataset notice](configs/resources/episodes/NOTICE.md) and Matterport terms. Model identities and source locations are listed in [the model manifest](configs/resources/models.json).
