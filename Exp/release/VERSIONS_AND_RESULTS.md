# Version and evaluation notes

Baseline source lineage: 2-MON uses 5ec04a8f; the archived 3-MON full400 contains 280 records from a3719e68 and 120 from 5ec04a8f. Baseline 4-MON accepted global indices use 5ec04a8f [0,36], 36e6c403 [37,266], cf0e056f [267,299]. Partial interrupted episodes were not counted in the archived result.

Full PBSP: 3-MON behavior source96e91424; 2-MON ea5efef0; 4-MON7f232ca7. The `exact_474a752` arm in historical frozen protocol files is a different internal comparison, **not** the user-selected5ec04a8f Baseline. Historical design notes are retained as historical records, not assertions about implemented functionality.

## Raw observed metrics (%)
| Method | MON | Episodes | SR | SPL | PSR | PeSPL |
|---|---|---:|---:|---:|---:|---:|
| Baseline | 2 | 600 | 21.67 | 8.82 | 45.67 | 20.78 |
| Baseline | 3 | 400 | 12.00 | 4.73 | 47.25 | 20.50 |
| Baseline | 4 | 300 | 8.67 | 3.67 | 49.17 | 22.05 |
| Full PBSP | 2 | 600 | 25.83 | 10.72 | 48.17 | 22.39 |
| Full PBSP | 3 | 400 | 18.25 | 7.56 | 50.33 | 23.03 |
| Full PBSP | 4 | 300 | 8.67 | 3.07 | 49.42 | 22.10 |

Full PBSP 4-MON logs contain202 mapper OOM entries and9 Flush incomplete entries. Complete output records do not establish complete fusion. This limitation remains in any derived statistics. No rerun is claimed.

## Post-hoc results are not direct benchmark outputs
The previously selected paper tables (2/3/4-MON SR32.67/23.75/12.00, SPL13.60/9.65/4.28, PSR54.83/56.17/56.92, PeSPL25.57/25.07/25.81) use **different outcome-aware scene weights for each MON**, not raw observations:
- 2-MON: qyAac8rV8Zk,XB4GS9ShBRE,Dd4bFSTQ8gi weight0; QaLdnwvtxbs,4ok3usBNeis,mv2HUxq3B53 weight2; others1.
- 3-MON: qyAac8rV8Zk,wcojb4TFT35 weight0; QaLdnwvtxbs,svBbv1Pavdk weight2; others1.
- 4-MON: qyAac8rV8Zk,wcojb4TFT35,6s7QHgap2fW weight0; QaLdnwvtxbs weight3; svBbv1Pavdk weight2; others1.

Unique episodes are510/360/255 despite weighted totals600/400/300. No source success flags or path lengths were changed. Such outcome-selected sensitivity analyses are not new independent experiments, unbiased primary benchmarks, or evidence of causal algorithm improvement. The runtime does not apply these weights or hard-code these outcomes. Do not claim those table values will be emitted by a fresh unweighted evaluation.

Metrics: SR is episode-all success; SPL is its summed-path-efficiency weighted mean; PSR is successful targets/(M*N); PeSPL sums success-weighted per-target path efficiency/(M*N). Combine by (scene,runtime local episode index), not by dataset source ID; failed/unreachable tasks remain in denominators.
