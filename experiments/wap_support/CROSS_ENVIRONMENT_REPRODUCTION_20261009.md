# Cross-environment reproduction — Python 3.13 passes, Python 3.11 misses the strict 2x complete-call gate

This report concerns **exposed-data reproduction**, not a new confirmation study. Every job used the untouched frozen source commit `0af6fda36c60a110584e7db836ee5724e530fbd4`, reacquired the five exact upstream graph blobs, rebuilt the deterministic cases, ran all 105 sessions, and audited all 107,520 complete answers. No job changed the solver, workload, threshold, or analysis.

## Complete four-cell result

| Environment | CPU | SPECTRA complete mean | MiniCard complete mean | Complete ratio | Exact graph 95% interval | Warm-session ratio | Frozen 0.50 gate |
|---|---|---:|---:|---:|---:|---:|---|
| Original: Ubuntu 24 / Python 3.13 | AMD EPYC 9V45 | 569.290 ms | 1,566.119 ms | **0.363503** | **[0.346789, 0.372944]** | 0.136014 | **PASS** |
| Reproduction: Ubuntu 24 / Python 3.13 | AMD EPYC 7763 | 984.038 ms | 2,735.008 ms | **0.359794** | **[0.340752, 0.368473]** | 0.135780 | **PASS** |
| Diagnostic: Ubuntu 22 / Python 3.13 | AMD EPYC 7763 | 1,155.545 ms | 3,175.345 ms | **0.363912** | **[0.344955, 0.373872]** | 0.123408 | **PASS** |
| Diagnostic: Ubuntu 24 / Python 3.11 | AMD EPYC 7763 | 1,706.029 ms | 3,192.533 ms | **0.534381** | **[0.517585, 0.541937]** | 0.153772 | **FAIL** |
| Reproduction: Ubuntu 22 / Python 3.11 | AMD EPYC 7763 | 1,695.404 ms | 3,204.897 ms | **0.529004** | **[0.513299, 0.536117]** | 0.155132 | **FAIL** |

All five executions completed every scheduled call and audited every full answer. The two Python 3.11 runs are not correctness failures: SPECTRA remains faster on every graph and easily passes the frozen tail-ratio limit. They fail only the deliberately demanding requirement that the upper **complete-call mean-ratio** bound be at most `0.50`.

### Exact p95 results

| Environment | p95 ratio | Exact graph-clustered 95% interval | Frozen 1.10 p95 gate |
|---|---:|---:|---|
| Original Ubuntu 24 / Py3.13 | 0.364735 | [0.338157, 0.383226] | PASS |
| Ubuntu 24 / Py3.13 reproduction | 0.373544 | [0.338440, 0.374074] | PASS |
| Ubuntu 22 / Py3.13 diagnostic | 0.377894 | [0.339055, 0.378264] | PASS |
| Ubuntu 24 / Py3.11 diagnostic | 0.541010 | [0.510408, 0.544537] | PASS |
| Ubuntu 22 / Py3.11 reproduction | 0.537062 | [0.510976, 0.537062] | PASS |

### Per-graph ratios for the two Python 3.11 failures

| Graph | Ubuntu 24 / Py3.11 | Ubuntu 22 / Py3.11 |
|---|---:|---:|
| WAP02a | 0.510901 | 0.511194 |
| WAP03a | 0.543876 | 0.537367 |
| WAP04a | 0.542066 | 0.537730 |
| WAP07a | 0.530591 | 0.520152 |
| WAP08a | 0.519184 | 0.512449 |

The near-repetition across Ubuntu versions is strong evidence that the portability break is associated with the tested Python 3.11 front end rather than the Ubuntu image or a different quotient relation.

## Where the difference enters

The frozen headline uses `complete_ns`. In the worker this timer starts after loading the native runtime and, for competitors, after preloading PySAT. It includes immutable case JSON decoding, recursive tuple reconstruction, validation, deterministic order construction, fresh preparation, all 1,024 queries, complete output materialisation and original checking, diagnostics, and disposal.

`session_ns` starts only after case decoding, validation, and order construction. The observed decomposition is:

| Environment | Arm | Front end before session | Warm session | Complete |
|---|---|---:|---:|---:|
| Original Ubuntu 24 / Py3.13 | SPECTRA | 415.001 ms | 154.288 ms | 569.290 ms |
|  | MiniCard | 431.762 ms | 1,134.357 ms | 1,566.119 ms |
| Reproduction Ubuntu 24 / Py3.13 | SPECTRA | 711.088 ms | 272.951 ms | 984.038 ms |
|  | MiniCard | 724.763 ms | 2,010.245 ms | 2,735.008 ms |
| Diagnostic Ubuntu 22 / Py3.13 | SPECTRA | 872.577 ms | 282.969 ms | 1,155.545 ms |
|  | MiniCard | 882.396 ms | 2,292.949 ms | 3,175.345 ms |
| Diagnostic Ubuntu 24 / Py3.11 | SPECTRA | 1,435.813 ms | 270.217 ms | 1,706.029 ms |
|  | MiniCard | 1,435.278 ms | 1,757.255 ms | 3,192.533 ms |
| Reproduction Ubuntu 22 / Py3.11 | SPECTRA | 1,417.095 ms | 278.309 ms | 1,695.404 ms |
|  | MiniCard | 1,410.880 ms | 1,794.018 ms | 3,204.897 ms |

The warm repeated-query mechanism remains strongly favorable in every environment: SPECTRA uses 12.3–15.5% of MiniCard's warm-session time. The decisive portability difference is a large **common** front-end cost under Python 3.11, nearly identical for candidate and baseline, which dilutes the stricter complete ratio.

A direct audit of the frozen case loader identifies the costly operations: multi-megabyte canonical JSON parsing, recursive list-to-tuple conversion, full structural validation, `dataclasses.asdict` deep copying, sorted canonical JSON reserialization, and SHA-256 recomputation. On the largest exact holdout case, those evidence-oriented operations consumed hundreds of milliseconds even under Python 3.13 and roughly double under Python 3.11 in the four-cell matrix.

This diagnosis does **not** remove that cost from the published endpoint. A future binary/prevalidated service format would be a new post-confirmation engineering study and may not retroactively replace the frozen complete-call result.

## Integrity receipts

| Environment | Run | Artifact | ZIP SHA-256 | Internal manifest |
|---|---:|---:|---|---:|
| Original Ubuntu 24 / Py3.13 | `37883619073` | `11595284861` | `5410db4d5832544c3ec55b19c1449395020319516d20823038de66216b45527c` | 382 delivery entries |
| Ubuntu 24 / Py3.13 | `37941889799` | `11621951388` | `81b36015bcbd86cf6c96a2f42184157b829962b315b9e150489fd951dab98ccb` | 359 entries, zero mismatches |
| Ubuntu 22 / Py3.11 | `37941889799` | `11622101487` | `ed1a65650e0a55240a7481c7430d927935de1549becbef63a6776722631ea332` | 359 entries, zero mismatches |
| Ubuntu 22 / Py3.13 | `37944047042` | `11622899680` | `e3413b7480bfc9db295e8839e0718e27decfc7eb96b5ffbd5cbe7ebd9fcd07e5` | 358 entries, zero mismatches |
| Ubuntu 24 / Py3.11 | `37944047042` | `11622624683` | `550cd2a7187168aedb76ecca0c8299b3a88bceedd2102e5373cfb8597c8afbd2` | 358 entries, zero mismatches |

All four post-confirmation jobs checked out the untouched frozen source, verified the same freeze SHA-256, acquired the same graph blobs, rebuilt identical deterministic cases, and independently audited the same complete relation. Separate certificate reproduction regenerated byte-identical canonical certificates on Python 3.11 and 3.13.

## Scientific interpretation

The prospectively frozen result remains a valid pass on its declared Ubuntu 24/Python 3.13 environment and closely reproduces on a second Ubuntu 24/Python 3.13 server and on Ubuntu 22/Python 3.13. The full four-cell experiment establishes that the strict 2× **complete-call** advantage is portable across the two tested Ubuntu images under Python 3.13, but not across the two tested Python 3.11 environments.

The strongest justified wording is:

> SPECTRA has a prospectively confirmed flagship result for the frozen Python 3.13 WAP support-query contract, independently recomputed and reproduced on two additional server environments. Its exact 2× complete-call gate is not portable to the tested Python 3.11 evidence-decoding path, although semantic identity, every witness, every certificate, per-graph wins, and a 6.45–8.10× warm-session advantage reproduce.

The failures are retained. The threshold is not relaxed, and the faster warm-session ratio is not substituted for the predeclared complete-call endpoint.
