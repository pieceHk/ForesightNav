# Packaging validation — 2026-09-24

Read-only export of committed Git trees, not uncommitted server files. Seven source snapshots retained. Source hashes checked before import; verify_source.py can recheck retained original files. Common credential patterns and the active SSH secret/endpoint were scanned without logging secret values. No detections were exported. This is a scoped scan, not a guarantee of exhaustive security review.

## Tests actually run on the Windows packaging host (Python3.10)
- Baseline5ec04a8f: Mapping/tests + Navigation/test_mon_metrics.py + test_navigation_policies.py: **26 passed**.
- Full PBSP96e91424: Mapping/tests + Navigation/tests: **176 passed**.
- Full PBSPea5efef0: same suite: **178 passed**.
- Full PBSP7f232ca7: same suite: **178 passed**.

The broader initial Baseline test collection was blocked by missing OmegaConf. Excluding that sensor test but including Exp/tests gave33 passed,3 failed,1 skipped (missing local runtime configuration and uninitialized export Git context). The initial3-MON broad suite gave215 passed,6 failed,1 skipped: local runtime configuration, export Git context, a vendored Detic import dependency, and an upstream root-document architecture conflict. These are disclosed, not silently counted as passing. The root design note was relocated under Exp/release during packaging; algorithms/tests were not rewritten to hide failures.

Pytest reports an upstream unknown cache_dir option warning. No CUDA extension builds, GPU inference, Habitat episodes, full dataset evaluation or clean-room Linux installation were performed for this release. The three historical Baseline runtime variants were preserved but not separately unit-tested in this packaging pass.
