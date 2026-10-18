# First-use exact sessions — final results, 17 October 2026

## Result

The installed exact API now preserves a large SCC advantage across both relevant
first-use boundaries.

The normal public path starts from an admitted fixed problem:

```python
packet = pack_session_packet(problem, requests)
answer = runtime.solve_session_packet(packet, mode="scc")
```

Both calls are timed. This includes request validation, SBC1 request-array
serialization, two independent native parses, fresh solver and observer
construction, all 1,024 decisions, complete labels or contradiction proofs,
independent native checking, SCQ2 serialization, immutable Python-byte ownership,
and disposal.

A second guard begins before `ExactProblem.create`, then performs the same packet
packing and native call. It prevents the primary result from looking good merely
because fixed-graph admission moved outside the timer.

Both frozen gates passed on x86-64 and native ARM64.

### Admitted-problem bank boundary

| Environment | WAP05 SCC/raw | WAP06 SCC/raw | Geometric mean | Reduction | Speedup | Gate |
|---|---:|---:|---:|---:|---:|---|
| Linux x86-64, CPython 3.13 | 0.475526 | 0.475522 | **0.475524** | **52.4%** | **2.103×** | PASS |
| Native Linux ARM64, CPython 3.13 | 0.538862 | 0.532060 | **0.535450** | **46.5%** | **1.868×** | PASS |

The predeclared limits were a geometric-mean ratio below `0.60` and every graph
below `0.65`.

### One-bank application guard

| Environment | WAP05 SCC/raw | WAP06 SCC/raw | Geometric mean | Reduction | Speedup | Gate |
|---|---:|---:|---:|---:|---:|---|
| Linux x86-64, CPython 3.13 | 0.660880 | 0.653783 | **0.657322** | **34.3%** | **1.521×** | PASS |
| Native Linux ARM64, CPython 3.13 | 0.745922 | 0.737088 | **0.741492** | **25.9%** | **1.349×** | PASS |

The predeclared limits were a geometric-mean ratio below `0.75` and every graph
below `0.80`. ARM64 passed narrowly but honestly: its geometric mean is about
`0.0085` below the fixed limit, and its weaker graph is about `0.0541` below the
per-graph limit. Neither threshold was changed after timing.

Each environment ran 31 paired repetitions per graph after five untimed warmups.
Mode order was shuffled deterministically within every pair. Packet bytes and
output bytes remained stable across repetitions, raw and SCC outputs were
byte-identical, and the complete SCQ2 banks passed standard-library replay.

## What changed

The earlier JSON-bound audit was dominated by research-envelope parsing and
canonical reconstruction shared by both engines. This change removes avoidable
work from the installed `spectra.exact` path without changing the native solver
or weakening raw:

- `ExactProblem` validates canonical fixed graphs in one pass and stores the
  immutable SBC1 mask/edge bytes once at the explicit admission boundary;
- noncanonical, reversed or duplicate edge input still takes the normalizing
  fallback path;
- `pack_session_packet` validates each current pin once while building compact
  native integer arrays;
- graph and request inputs are no longer admitted twice for one packet; and
- standard-library SCQ2 replay preprocesses immutable graph data once per answer
  bank instead of rebuilding edge and colour tables for every row.

Strict SBC1, SBQ1, SCQ2 and SBP1 bytes, exact API 1, native ABI 3, solver logic,
raw mode, decisions, labels, proofs, default modes, resource limits and the
independent checking contract are unchanged.

## Correctness and portability

The final measured source passed 19 focused public contracts on both x86-64 and
ARM64 before timing. These include:

- byte equality with the previous scalar SBC1 serializer;
- identical invariants for direct and factory problem construction;
- canonical and noncanonical graph admission;
- early rejection of oversized request banks;
- a 65-row bank spanning two tile64 blocks;
- SAT, empty-domain UNSAT and implication-path UNSAT;
- malformed packet and proof mutation rejection; and
- public standard-library replay without loading native code.

The benchmark represented 4,096 distinct graph/query decisions across the two
raw and two SCC output banks in each environment. No decision, complete label,
proof, packet byte or output byte differed.

The same implementation line also passed the repository's full pull-request
matrix before this final timing run:

- installed exact API on Python 3.11 and 3.13;
- Linux x86-64 and native ARM64;
- Windows x64/MSVC;
- UBSan and ASan+UBSan;
- compiler-free wheel installation;
- public-package checks;
- full CPU regression on Python 3.11 and 3.13;
- WAP flagship regression; and
- repository and SVM portability audits.

Leak detection remained disabled in the ASan job, so this is not a
LeakSanitizer claim.

## Relationship to the competition audit

The earlier JSON-bound composite audit remains a recorded failure: complete
SCC/raw ratios were `0.929` on x86-64 and `0.948` on ARM64 because unrelated
JSON and case reconstruction dominated that research endpoint. Its thresholds
are not edited or retroactively reinterpreted.

That audit independently established large complete-cost advantages over serious
external competitors on the same exposed WAP banks:

- **2.81×** over persistent MiniCard on x86-64 and **2.83×** on ARM64;
- **2.86×** over persistent CaDiCaL on x86-64 and **2.93×** on ARM64; and
- **3.94×** over reduced native CUDD on x86-64.

The final first-use result closes the remaining product-boundary problem for this
scope: SPECTRA now beats its optimized same-problem raw engine by a meaningful
margin even when a one-bank caller pays fixed-problem admission, and it already
beats the external SAT/BDD controls by much larger margins.

## Claim boundary

This remains exposed development evidence on two all-SAT WAP banks of 1,024
restrictions. It is not a new holdout, outside-team replication, production
trace, one-query result, high-pin ITC result, CPython 3.11 performance result, or
universal SAT/graph-colouring claim.

Within the declared long-bank first-use scope, however, the result is no longer
a tie or a small win. Both internal first-use gates and all external competition
gates pass on both tested architectures. Automatic engine selection remains a
separate issue and should not inherit these results without its own frozen test.

## Identities and retained evidence

- Base `main`: `4a647e54e7e7e0756e74b52575f86f03d2764455`
- Final measured source commit: `3d9eb74af6573d6482b250e719d0b066f3d01746`
- Final measured source tree: `17b2c54e20d21a2e88fac7f43add28d108e4ba8e`
- Benchmark workflow run: `39085921452`
- x86-64 artifact: `12121051137`, ZIP SHA-256
  `14e9e98c3b56b525184aeaabc7186cd51ce1aa24d6a86979aaa0eddc9ab6b2c2`
- ARM64 artifact: `12120583068`, ZIP SHA-256
  `bc916075ea0d22b3d1b052144e8d05f6f5e08fff555d629442e69476bc8c1b47`

The artifacts' internal SHA-256 manifests replay without error. The executable
source identities recorded in both artifacts are identical:

```text
998e8bae939ecd925d4d305589acdec75030285fcec06337ad776d89e2801426  spectra/exact.py
21107ea4c729d5c6cc37555c694c0f4551ba39764222899fbd62f9d5a0f89b87  spectra/exact_problem.py
136a09d42b23054d813ea4c5a2927b428cbcbbca53fc6d8a5c87219d91944312  spectra/exact_proof.py
570c5fe9204342535adb766d01a923001c49881e31b7bd2da42fc972a27c73d3  spectra/_native/exact_binary.cpp
6f1bb00f185e5b6f2f70136b3f2d6ac92a0468e318032dd1831ee63fbb4f9457  experiments/compiled_query/session_packet_cold_benchmark.py
40e7c13789ba7383d4f6c3d6c8877416ddff3119e147587be6b98e828e0754ca  experiments/compiled_query/SESSION_PACKET_COLD_PROTOCOL_20261017.md
```
