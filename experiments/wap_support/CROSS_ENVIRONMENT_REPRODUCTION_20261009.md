# Cross-environment reproduction — one pass, one performance-gate failure

This report concerns **exposed-data reproduction**, not a new confirmation study. Both jobs used the untouched frozen source commit `0af6fda36c60a110584e7db836ee5724e530fbd4`, reacquired the five exact upstream graph blobs, rebuilt the deterministic cases, ran all 105 sessions, and audited all 107,520 complete answers. Neither job changed the solver, workload, threshold, or analysis.

## Summary

| Environment | CPU | SPECTRA complete mean | MiniCard complete mean | Complete ratio | Exact graph 95% interval | Warm-session ratio | Frozen 0.50 gate |
|---|---|---:|---:|---:|---:|---:|---|
| Original one-shot: Ubuntu 24 / Python 3.13 | AMD EPYC 9V45 | 569.290 ms | 1,566.119 ms | **0.363503** | **[0.346789, 0.372944]** | 0.136014 | **PASS** |
| Reproduction: Ubuntu 24 / Python 3.13 | AMD EPYC 7763 | 984.038 ms | 2,735.008 ms | **0.359794** | **[0.340752, 0.368473]** | 0.135780 | **PASS** |
| Reproduction: Ubuntu 22 / Python 3.11 | AMD EPYC 7763 | 1,695.404 ms | 3,204.897 ms | **0.529004** | **[0.513299, 0.536117]** | 0.155132 | **FAIL** |

The Ubuntu 22/Python 3.11 run is not a correctness failure. It completed every scheduled call, audited all answers, and SPECTRA remained faster on every graph:

| Graph | SPECTRA / MiniCard complete ratio |
|---|---:|
| WAP02a | 0.511194 |
| WAP03a | 0.537367 |
| WAP04a | 0.537730 |
| WAP07a | 0.520152 |
| WAP08a | 0.512449 |

Its p95 ratio was `0.537062`, exact graph-clustered interval `[0.510976, 0.537062]`, well below the frozen tail limit `1.10`. It failed only the deliberately demanding requirement that the upper mean-ratio bound be at most `0.50`.

## Why the complete ratio changed

The frozen headline uses `complete_ns`. In the worker this timer starts after loading the native runtime and, for competitors, after preloading PySAT. It then includes immutable case JSON decoding, validation, order construction, fresh preparation, 1,024 queries, complete output materialisation and original checking, diagnostics, and disposal.

`session_ns` starts after case decoding, validation, and order construction. The observed decomposition is:

| Environment | Arm | Front-end before session | Warm session | Complete |
|---|---|---:|---:|---:|
| Original Ubuntu 24 / Py3.13 | SPECTRA | 415.001 ms | 154.288 ms | 569.290 ms |
|  | MiniCard | 431.762 ms | 1,134.357 ms | 1,566.119 ms |
| Reproduction Ubuntu 24 / Py3.13 | SPECTRA | 711.088 ms | 272.951 ms | 984.038 ms |
|  | MiniCard | 724.763 ms | 2,010.245 ms | 2,735.008 ms |
| Reproduction Ubuntu 22 / Py3.11 | SPECTRA | 1,417.095 ms | 278.309 ms | 1,695.404 ms |
|  | MiniCard | 1,410.880 ms | 1,794.018 ms | 3,204.897 ms |

The warm repeated-query mechanism remains strongly favorable on all three executions: SPECTRA uses approximately 13.6%, 13.6%, and 15.5% of MiniCard's warm-session time. The Ubuntu 22/Python 3.11 front-end cost is approximately twice the Ubuntu 24/Python 3.13 reproduction front-end and is nearly identical for the two arms. Adding this large common cost moves the stricter complete ratio above 0.50.

This is an observed decomposition, not yet a causal attribution solely to Python. The two missing combinations—Ubuntu 22/Python 3.13 and Ubuntu 24/Python 3.11—are being executed from the same frozen source to separate Python-version effects from operating-system/runner effects. Their results must be reported whether favorable or unfavorable.

## Integrity receipts

### Original confirmation

- Run: `37883619073`
- Artifact ID: `11595284861`
- ZIP SHA-256: `5410db4d5832544c3ec55b19c1449395020319516d20823038de66216b45527c`
- Delivery manifest entries independently checked: 382

### Ubuntu 24 / Python 3.13 reproduction

- Run: `37941889799`
- Artifact ID: `11621951388`
- ZIP SHA-256: `81b36015bcbd86cf6c96a2f42184157b829962b315b9e150489fd951dab98ccb`
- Internal SHA-256 manifest entries checked: 359, zero mismatches
- Verdict: `PASS`

### Ubuntu 22 / Python 3.11 reproduction

- Run: `37941889799`
- Artifact ID: `11622101487`
- ZIP SHA-256: `ed1a65650e0a55240a7481c7430d927935de1549becbef63a6776722631ea332`
- Internal SHA-256 manifest entries checked: 359, zero mismatches
- Verdict: `FAIL` for the numeric 0.50 threshold; semantic/correctness reproduction completed

## Scientific interpretation

The original prospectively frozen result remains a valid pass on its declared host and environment, and a second Ubuntu 24/Python 3.13 host closely reproduces both correctness and effect size. The Ubuntu 22/Python 3.11 result prevents a stronger claim that the frozen 2× complete-call advantage is already portable across supported Python environments.

The strongest justified wording is therefore:

> SPECTRA has a prospectively confirmed flagship result on the frozen Ubuntu 24/Python 3.13 WAP support-query contract, independently recomputed and reproduced on a second Ubuntu 24/Python 3.13 server. The exact complete-call 2× gate is not portable to the tested Ubuntu 22/Python 3.11 environment, although SPECTRA remains faster on every graph and its warm repeated-query session remains more than 6× faster.

The failure is retained. The threshold is not relaxed, and the faster warm-session ratio is not substituted for the predeclared complete-call endpoint.
