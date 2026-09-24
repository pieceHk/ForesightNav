# PBSP: Persistent Belief-Aware Suffix Planning

Private paper-code repository containing the **Baseline** and **Full PBSP** source versions used in the recorded experiments. Full PBSP here means the historical complete-C implementation (`ordered_suffix_mvp`), not the later Suffix-only checked-candidate ablation. The paper name does not imply implementation of every item in historical design notes; consult the executable code and profiles.

## Select the experiment version

| Variant | Branch | Fixed tag | Original source commit |
|---|---|---|---|
| Baseline reference / 2-MON | `baseline` | `baseline-5ec04a8f` | `5ec04a8f` |
| Baseline early 3-MON records | `baseline-3mon-legacy` | `baseline-3mon-legacy-a3719e68` | `a3719e68` |
| Baseline 4-MON recovery | `baseline-runtime-fix` | `baseline-runtime-36e6c403` | `36e6c403` |
| Baseline final 4-MON runtime | `baseline-4mon-runtime` | `baseline-runtime-cf0e056f` | `cf0e056f` |
| Full PBSP 3-MON | `full-pbsp-3mon` | `pbsp-3mon-96e91424` | `96e91424` |
| Full PBSP 2-MON | `full-pbsp-2mon` | `pbsp-2mon-ea5efef0` | `ea5efef0` |
| Full PBSP 4-MON / latest tracked snapshot | `main`, `full-pbsp` | `pbsp-4mon-7f232ca7` | `7f232ca7` |

These are **new export commits**, not the original repository's Git history. Original full SHAs, file hashes, exclusions and packaging changes are recorded in [SOURCE_MANIFEST.json](Exp/release/SOURCE_MANIFEST.json). No algorithm source or experiment outcomes were edited for this release.

```bash
git clone git@github.com:pieceHk/PBSP.git
cd PBSP
git switch --detach pbsp-3mon-96e91424  # choose the required fixed tag
```

## Layout and reproduction

- `Mapping/`: online geometric-semantic mapper, server and vendored runtime.
- `Navigation/`: candidate execution, evidence/ranking, suffix planning, metrics and evaluator runtime.
- `Exp/`: profiles, dataset catalog, launch/monitor scripts and tests.
- [Reproduction guide](Exp/release/REPRODUCIBILITY.md): assets, environment, profile selection and launch commands.
- [Version and evaluation notes](Exp/release/VERSIONS_AND_RESULTS.md): source lineage and raw-versus-post-hoc distinction.
- [Validation record](Exp/release/VALIDATION.md): tests actually run and remaining limitations.
- [Third-party notices](Exp/release/THIRD_PARTY_NOTICES.md): preserved licenses; no new blanket license is granted.

This is a **source release**, not a self-contained GPU image. Licensed HM3D assets, generated episodes, model weights, machine-local configuration, logs and images are not uploaded. Relative asset symlinks are retained. Linux, suitable CUDA/PyTorch environments, Habitat and separate mapping/navigation dependencies are required.

The repository is private; reviewers cannot access it until explicitly granted access or given an appropriate submission archive. No collaborator access or public release is performed automatically.

Current snapshot source: `96e9142493969ea98e4883bde2824e97678ba9bf`.
