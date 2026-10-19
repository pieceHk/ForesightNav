# Historical source snapshots

| Variant | Branch | Fixed tag | Original source commit |
|---|---|---|---|
| Baseline reference / 2-MON | `baseline` | `baseline-5ec04a8f` | `5ec04a8f` |
| Baseline early 3-MON records | `baseline-3mon-legacy` | `baseline-3mon-legacy-a3719e68` | `a3719e68` |
| Baseline 4-MON recovery | `baseline-runtime-fix` | `baseline-runtime-36e6c403` | `36e6c403` |
| Baseline final 4-MON runtime | `baseline-4mon-runtime` | `baseline-runtime-cf0e056f` | `cf0e056f` |
| Full PBSP 3-MON | `full-pbsp-3mon` | `pbsp-3mon-96e91424` | `96e91424` |
| Full PBSP 2-MON | `full-pbsp-2mon` | `pbsp-2mon-ea5efef0` | `ea5efef0` |
| Full PBSP 4-MON / latest tracked snapshot | `main`, `full-pbsp` | `pbsp-4mon-7f232ca7` | `7f232ca7` |

These are **new export commits**, not the original repository's Git history. Original full SHAs, file hashes, exclusions and packaging changes are recorded in [SOURCE_MANIFEST.json](../Exp/release/SOURCE_MANIFEST.json). No algorithm source or experiment outcomes were edited for this release.

```bash
git clone --branch full-pbsp https://github.com/pieceHk/PBSP.git
cd PBSP
```

Select a fixed tag from the table when inspecting a historical source version.
Current shared 2/3/4-MON reproduction uses `refactor/full-pbsp`; see the [README](../README.md).


Historical observed and post-hoc results are described in [version and evaluation notes](../Exp/release/VERSIONS_AND_RESULTS.md). New refactor validation is recorded in [migration progress](migration.md).
