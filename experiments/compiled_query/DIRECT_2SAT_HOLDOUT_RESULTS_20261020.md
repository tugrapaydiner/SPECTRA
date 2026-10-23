# Mixed direct 2-SAT holdout result

**Date:** 2026-10-20  
**Verdict:** PASS on x86-64 and native ARM64  
**Frozen source:** `7962ffe4033a8409659c917816b719259b16fab9`  
**Opening workflow:** `79cfc618d5939f210ec514eccfde0f366b650ae6`  
**GitHub Actions run:** `39355144820`

## Scope

This was the prospectively frozen comparison against the fastest median of three
separately implemented direct 2-SAT engines: closure, scalar condensation-DAG, and
tile-64. All arms consumed the same immutable `SBC1` packet and returned complete
proof-carrying `SCQ2` output.

The four graph names, upstream identities, generator, 512 SAT / 512 UNSAT outcome
schedule, six measured arms, repetition schedule, and numerical gates were frozen
before any candidate graph bytes were acquired. The graph-free validation passed in
run `39354604555` before the opening workflow began.

The result is narrow exact-service evidence. It is not a general SAT,
ordinary graph-colouring, production, operator-value, or independent-replication
claim.

## Public `auto` result

Ratios are `SPECTRA auto / fastest direct engine`; lower is better for SPECTRA.
Every correctness and performance gate passed on both architectures.

| Requests | x86-64 ratio | x86 speedup | ARM64 ratio | ARM speedup |
|---:|---:|---:|---:|---:|
| 1 | 0.1241 | 8.06× | 0.1808 | 5.53× |
| 8 | 0.1243 | 8.04× | 0.1826 | 5.48× |
| 64 | 0.1301 | 7.69× | 0.1896 | 5.28× |
| 1,024 | 0.4571 | 2.19× | 0.3997 | 2.50× |

SPECTRA `auto` won all four request counts on all four graphs on both
architectures: **32 of 32 measured cells**.

### Frozen primary gates

| Gate | x86-64 | ARM64 |
|---|---:|---:|
| 64-query geometric mean ≤ 0.90 | 0.1301 | 0.1896 |
| Every 64-query graph ≤ 1.05 | 0.1625 max | 0.2371 max |
| 1,024-query geometric mean ≤ 0.75 | 0.4571 | 0.3997 |
| Every 1,024-query graph ≤ 0.90 | 0.5155 max | 0.4678 max |

## Graph-level 1,024-query ratios

| Graph family | Graph | x86-64 | ARM64 |
|---|---|---:|---:|
| random DIMACS | `DSJC250.5.col` | 0.4463 | 0.3820 |
| flat/planted | `flat300_28_0.col` | 0.4353 | 0.3730 |
| Leighton | `le450_15c.col` | 0.4359 | 0.3828 |
| geometric | `miles1000.col` | 0.5155 | 0.4678 |

## Important negative finding

The public automatic policy selected SCC/tile execution for the 1,024-query mixed
banks, but the explicit `raw` arm was faster on every graph. Against the same
fastest direct competitor, explicit raw would have produced an approximately
**4.84× x86-64** and **3.99× ARM64** geometric-mean advantage at 1,024 queries.

This does not change the frozen PASS verdict. It identifies a real policy and proof
construction opportunity: many UNSAT rows contain a direct conflict between forced
literals, but the current tile path rediscovers those contradictions through a
full closure replay. The next development step is a general direct-conflict proof
fast path, followed by a newly frozen transfer study. The opened holdout will not
be reused as prospective confirmation for that change.

## Evidence identities

- x86-64 artifact:
  `sha256:7dd15232272787af9022a985db65b0fd09ac24b67dd5b0cc64ada2d41ac3be7a`
- ARM64 artifact:
  `sha256:6774377577e7b4cf6376447f5ad84f759baf7ae780c20efd67d18115b8d2ed7c`
- x86-64 benchmark results:
  `324fabfd32763ded867f9bb6694b81e2d780b81859c264f077aa373fcf6129f7`
- ARM64 benchmark results:
  `b1feb0d1060be03ee7cce71799e0b2674e57463e730783758170c533c59765a1`

The retained artifacts contain the opened cases and generation receipts, complete
benchmark samples, verdicts, build receipts, native binaries, logs, and exit codes.
