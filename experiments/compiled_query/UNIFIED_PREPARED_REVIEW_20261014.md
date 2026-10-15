# SPECTRA full code review and unified prepared-bank checkpoint

**Review date:** 14 October 2026, America/Toronto
**Tested implementation:** `4d7649b0667f938254245f3ff780b11e2597de49`
**Tested tree:** `45b92b883fa31cfef9a499f1cdf36d0284d5d4e1`
**Baseline prepared implementation:** `3d085050ff48a907d95571900c1f6481af8c2972`
**Local branch:** `review/unified-prepared-bank-20261014`

## Executive conclusion

SPECTRA is no longer a 67/100 codebase. The strongest current local checkpoint is
an excellent research-engineering artifact with exact mixed SAT/UNSAT decisions,
portable original-input contradiction proofs, independently owned checkers, a
strict one-call `SBC1` session boundary, serious native controls, preserved negative
evidence, and a newly unified 64-lane solver/checker kernel.

Under the original **whole-program flagship** standard, however, the current grade
is **75/100**, not 90–100. The gap is no longer basic implementation quality. It is
mostly scientific scope, unseen transfer, final-endpoint portability, repository
publication/integration, independent replication, adoption, and novelty.

Separate grades:

| Object | Grade | Interpretation |
|---|---:|---|
| Unified prepared exact-session engineering checkpoint | **94/100** | Excellent internal engineering; not yet stable API or externally replicated. |
| Narrow exact proof-carrying systems result | **87/100** | Strong publishable systems result on exposed development workloads. |
| Whole SPECTRA flagship research program | **75/100** | Very strong research prototype; not yet field-leading or independently impactful. |
| Current public GitHub state as seen by an outsider | **~70/100** | The newest tested local state is not yet cleanly published and PR/branch identity remains behind. |

## What was reviewed

The tested tree contains **905 tracked files** and about **60.6 MB of tracked
content**. Its active source/documentation footprint includes approximately:

| Area | Relevant files | Lines |
|---|---:|---:|
| `spectra/` stable package | 52 | 8,863 |
| `experiments/` | 138 | 17,717 |
| `tests/` | 145 | 17,003 |
| `scripts/` | 91 | 19,662 |
| `model/` | 19 | 2,844 |
| `train/` | 12 | 2,305 |
| `eval/` | 38 | 6,889 |
| `docs/` | 61 | 7,648 |

The review covered the public package and CLI, native build paths, retained model
tracks, WAP evidence, exact mixed-outcome service, CUDD/SAT controls, proof replay,
prepared packet endpoint, experiment/audit harnesses, release navigation,
historical-byte identity, full fast CPU regression suite, sanitizer runs, and the
separate hardened exact-bank line.

## Workstream status

### Finished or strong

1. **Stable public package and installation contract.** The dependency-free wheel
   path, CLI, public CNF state, source inventory, and 11 installed-only checks have
   a clean prior package validation. The current executable change is confined to
   `experiments/compiled_query`, outside the stable wheel package set.
2. **Frozen WAP evidence discipline.** Positive and negative outcomes, portability
   failures, source identities, raw evidence, and claim boundaries are retained.
3. **Exact mixed outcomes.** The companion service returns either a complete checked
   colouring or a portable checked contradiction proof; completed admitted calls do
   not hide behind `UNKNOWN`.
4. **Independent checking.** Solver and observer own separate parses and structures.
   SAT witnesses and UNSAT paths are checked against original domains, edges, and
   current restrictions. Python replay provides an additional independent path.
5. **Strong controls.** Matched raw traversal, SCC, MiniCard, CaDiCaL, native CUDD,
   multiple variable orders, and generic logical reduction are represented.
6. **Prepared one-call ownership.** `SBC1` contains the fixed problem and request
   bank; a single native call constructs both owners, solves, proves, verifies,
   emits strict `SCQ2`, and disposes.
7. **Resource and parser hardening.** Exact schemas are strict, malformed trailing
   inputs fail closed, native ABI names are versioned, and output/scratch envelopes
   are charged before execution.
8. **Regression depth.** Full fast CPU, compiled-query modules, sanitizers,
   navigation, and 449-file historical identity all pass at the tested tree.

### Partially finished

1. **GitHub publication/integration.** The tested local unified commit is ahead of
   the public PR state. A clean publication branch and updated CI/result narrative
   are still required.
2. **Stable API promotion.** The strongest exact endpoint remains in
   `experiments/compiled_query`; it is not yet an installed, versioned public API.
3. **Final-endpoint portability.** Earlier exact work covered Python 3.11 and 3.13,
   but this unified timing checkpoint is local Linux x86-64 only.
4. **Memory and operational characterization.** Conservative native payload
   admission exists, but allocator overhead, total RSS, concurrent service behavior,
   and production-tail latency need broader evidence.
5. **Repository coherence.** Historical tracks are preserved responsibly, but the
   branch count, stale PR narrative, and public repository identity still obscure
   the strongest result.
6. **Problem-family breadth.** The exact service remains a binary-domain,
   repeated-query problem rather than a broad general reasoning result.

### Still open

- A prospectively frozen unseen topology/application for the **final** endpoint.
- Outside-team clean-room reproduction.
- An operator-defined workload and independently maintained adoption.
- ARM/Windows/ordinary-laptop final-endpoint evidence.
- Peer-reviewed closest-work and novelty determination.
- Measured real-world utility or durable scientific impact.

## New implementation completed in this review

The separate exact-bank branch contained a useful 64-lane SCC propagation kernel,
but it diverged from the stricter ABI-3 prepared architecture and originally had a
proof-output resource-accounting defect. The useful kernel was therefore ported
into the `SBC1` path rather than replacing the prepared architecture.

The new implementation:

- preserves ABI 3 and `SBC1` / `SCQ2` / `SBP1`;
- uses serial execution for raw mode or banks smaller than 64 requests;
- propagates up to 64 independent request lanes through the SCC DAG using one
  `uint64` word per component;
- constructs full original labels directly for feasible lanes;
- replays only contradictory lanes through the ordinary exact solver to reconstruct
  the same portable proof bytes;
- independently reconstructs original equal-colour conflicts in the observer;
- verifies all SAT rows bit-parallel in 64-lane words;
- continues to check every UNSAT proof individually against original implications;
- expands conservative scratch admission for parsed copies, answer lanes, output,
  proof replay, and metadata;
- exposes `bank_kernel` and `bank_tiles` diagnostics; and
- adds threshold and byte-exact equivalence regression contracts.

The implementation delta is **2 files, +192/-20**. `exact_native.cpp` is now
approximately 965 lines and remains a maintainability hotspot; splitting packet
parsers, solver, observer, and Python bindings is sensible before stable API
promotion, but not before the result is frozen.

## Correctness and regression verification

The stable fast CPU suite passed with a writable isolated native-extension cache:

**2590 passed, 1 skipped, 16 deselected, 3 warnings in 273.67s (0:04:33)**

The one skipped test is the repository's documented Windows-only narrow legacy
file-path contract. The 16 excluded tests are the existing `slow` marker set; they
were not renamed or weakened.

### Compiled-query modules

| Module | Result |
|---|---|
| `test_bdd` | 83 passed, 2 warnings in 8.39s |
| `test_construction` | 39 passed, 1 warning in 13.50s |
| `test_reduced` | 32 passed, 2 warnings in 6.73s |
| `test_exact` | 75 passed in 7.19s |
| `test_exact_study` | 26 passed in 10.77s |
| `test_proof_replay` | 19 passed in 4.42s |
| `test_study` | 83 passed, 30 warnings in 9.81s |
| `test_exact_ubsan` | 75 passed in 3.46s |
| `test_exact_asan_ubsan` | 75 passed in 5.23s |

The seven ordinary modules total **357 passed**. The warnings are upstream CUDD
statistics warnings, not failed assertions.

Additional gates:

- Git whitespace check: **PASS**
- Python compilation: **PASS**
- Release/navigation audit: **PASS**
- Historical workspace audit: **PASS; 449 protected files byte-identical**
- Exact module under UBSan: **75 passed**
- Exact module under ASan + UBSan: **75 passed**
- Leak detection was disabled; no LeakSanitizer claim is made.

An initial full-suite attempt failed because the shared Torch-extension cache was
read-only. It produced 2,231 passing tests before 339 failures and 19 errors with
the same cache-lock `PermissionError`. That run is retained as an environment
failure and is not counted as a code regression or a passing suite.

## Measured unified result

### Protocol

- Two already exposed WAP development graphs: WAP05 and WAP06.
- Six immutable `SBC1` packets: forward, reverse, and shuffled order per graph.
- 1,024 requests per packet.
- 5 warmups and 21 timed ready-session calls per configuration.
- Configuration order shuffled.
- Same local interpreter, host, packet bytes, and complete endpoint boundary.
- Each call includes native construction, solve, complete witness/proof production,
  independent verification, `SCQ2` materialization, and disposal.
- Packet creation and arbitrary JSON decoding are outside this ready-input timer.

### Per-packet medians

| Graph | Order | Old prepared route (ms) | Unified route (ms) | Old prepared raw (ms) | Unified raw (ms) | Outputs identical |
|---|---|---:|---:|---:|---:|---|
| WAP05 | forward | 20.345 | 6.393 | 20.992 | 7.154 | yes |
| WAP05 | reverse | 19.875 | 6.909 | 20.792 | 7.406 | yes |
| WAP05 | shuffle | 19.853 | 6.025 | 20.493 | 7.339 | yes |
| WAP06 | forward | 20.334 | 6.832 | 23.041 | 7.451 | yes |
| WAP06 | reverse | 19.584 | 6.858 | 21.469 | 7.321 | yes |
| WAP06 | shuffle | 20.016 | 7.040 | 21.050 | 7.297 | yes |

All route outputs match the prior route endpoint byte-for-byte. All raw outputs
match the prior raw endpoint byte-for-byte.

### Aggregate interpretation

Using geometric aggregation of per-packet medians:

- Unified route / previous prepared route: **0.333327**
  — **66.67% lower** ready-session time.
- Unified raw / previous prepared raw: **0.344165**
  — **65.58% lower**.
- Unified route / unified raw: **0.909778**
  — **9.02% lower**.

The large baseline improvement is a genuine endpoint result, but it combines two
mechanisms: 64-lane solver propagation and 64-lane independent SAT checking. The
narrower route-versus-equally-optimized-raw comparison is about 9%, not a universal
algorithmic breakthrough. Both graphs favour route, but there are only two exposed
clusters, so no population or confidence claim is justified.

Every 1,024-request route call used 16 tiles. WAP05 returned 610 SAT / 414 UNSAT;
WAP06 returned 596 SAT / 428 UNSAT. Repeated timings and request orders are not
independent scientific instances.

## Strict grade

| Category | Score | Assessment |
|---|---:|---|
| Problem importance and central focus | 9 / 12 | Useful exact repeated-query and proof-carrying service, but still narrow rather than a general reasoning result. |
| Correctness, contracts, and verification | 15 / 15 | Exact SAT/UNSAT outcomes, original-input observers, portable proofs, fail-closed parsing, exhaustive contracts, sanitizers, and byte-identical output equivalence. |
| Experimental rigor and reproducibility | 17 / 18 | Frozen packets, complete-cost boundary, 21 timed repetitions, raw records, source/binary hashes, preserved failures, and full replay; final speed evidence remains exposed development data. |
| Competitor strength and causal attribution | 14 / 15 | Matched same-contract raw control plus SCC, MiniCard, CaDiCaL and reduced CUDD; route advantage over equally optimized raw remains modest and is separated from observer gains. |
| Novelty and scientific depth | 8 / 15 | Strong systems composition, but SCC condensation, bit-parallel propagation, 2-SAT proofs, batching and independent checking are established foundations. |
| Generalization, robustness, and portability | 6 / 10 | Broad internal testing and prior Python 3.11/3.13 evidence, but final unified timing is local Linux x86-64 on two exposed graph families. |
| Repository coherence and release quality | 6 / 8 | Stable package and evidence audits pass, but the endpoint remains experimental, remote publication trails the tested state, and historical branch/identity debt remains. |
| Independent replication, adoption, and external impact | 0 / 7 | No outside-team reproduction, operator-defined workload, independently maintained adoption, or production impact. |

**Total: 75/100.**

### Why it is not higher

The strongest internal engineering is already near flagship quality, but the
whole-program score cannot inherit points that have not happened:

- only two exposed graph clusters support this latest timing result;
- the final unified endpoint has no prospective unseen confirmation;
- there is no outside reproduction or adoption;
- no external reviewer has established algorithmic priority;
- the public GitHub state does not yet match the tested local state;
- the endpoint is experimental rather than a stable installed API; and
- the result is a narrow binary-domain service, not general reasoning.

### Fastest credible route upward

1. Publish this exact tested tree cleanly, with immutable raw benchmark evidence
   and CI on Python 3.11/3.13.
2. Freeze the endpoint, controls, resource caps, and analysis before opening any
   new data.
3. Run a genuinely unseen second family or operator-defined workload.
4. Obtain at least one outside clean-room reproduction.
5. Promote only after compatibility, RSS, concurrency, ARM/laptop, and API evidence.
6. Seek closest-work review around the combined proof-carrying prepared service,
   not around established SCC or bitset primitives.

Internal optimization can raise the engineering subgrade further, but a strict
90–100 whole-program score requires external confirmation and impact rather than
more self-authored tests.

## Publication status

The tested implementation is preserved as a local Git commit and recovery bundle.
The public `research/compiled-query-validation` PR state remains behind this tree.
No merge, default promotion, package-version change, or broad flagship claim is
recommended from the current exposed development benchmark alone.
