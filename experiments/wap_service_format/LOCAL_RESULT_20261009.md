# Local packed-service development result — Python 3.13

**Exposed-data engineering result, not confirmation.** The 60-session matrix used
the five already published WAP-A holdout cases and charged packed read/hash/decode,
fresh solver setup, 1,024 full queries, original checking, diagnostics and disposal.

| Quantity | Result |
|---|---:|
| SPECTRA complete mean | 284.515 ms |
| MiniCard complete mean | 1,641.184 ms |
| Mean ratio | **0.173360** |
| Exact graph-clustered 95% interval | **[0.149743, 0.185928]** |
| p95 ratio | **0.190880** |
| Exact p95 interval | **[0.162453, 0.192232]** |
| Mean packed load, SPECTRA / MiniCard | 26.544 / 27.118 ms |
| Per-graph candidate wins | **5 / 5** |
| Sessions / full answers audited | **60 / 61,440** |

Per-graph complete ratios range from 0.1331 to 0.1916. The candidate also uses
0.1673x CaDiCaL and 0.1556x the no-contraction executor on this local run.

This demonstrates that the frozen Python 3.11 portability failure is plausibly a
research-evidence transport issue rather than a mechanism failure. Only the clean
Python 3.11/3.13 workflow can establish the cross-version engineering gate.
