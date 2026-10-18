# Automatic engine selection v4 — frozen protocol, 18 October 2026

## Purpose

Structural-v3 passed its exposed development panel but failed prospective
confirmation and the real WAP selector guard. The failed evidence is retained in
`AUTO_SELECTOR_V3_RESULTS_20261018.*`. This protocol defines one replacement
policy before implementation or any v4 confirmation case is opened.

Issue 2 remains the only active issue. Explicit raw, SCC and route engines are not
changed.

## Failure-derived feature

v3 used an absolute threshold of eight pins per request. That incorrectly grouped
very different workloads:

- failed confirmation: 15.09 pins over 1,024 vertices = 1.47% density; SCC won;
- ITC set9: 9.47 pins over 169 exams = 5.60% density; raw is conservative;
- ITC set10: 9.09 pins over 214 exams = 4.25% density; raw is conservative;
- ITC set12: 10.12 pins over 78 exams = 12.97% density; raw is conservative.

v4 therefore uses both absolute pin count and graph-relative density. The fixed
binary threshold is one pinned vertex per 32 original vertices on average.

## Frozen structural-v4 policy

Selection order is fixed:

1. base-infeasible fixed graph: `raw`;
2. at least eight pins per request **and** average pin density at least `1/32`:
   `raw`;
3. feasible bank of at least 64 requests: `scc`;
4. feasible bank of at least eight requests whose SCC count is at most one quarter
   of the original literal count: `scc`;
5. otherwise: `raw`.

The native density test is integer-only:

```text
bank_pins >= 8 * bank_size
and
32 * bank_pins >= original_vertices * bank_size
```

All public admission bounds keep these products within uint64. The selector uses
only immutable graph/request geometry. It cannot inspect answers, timings, trial
solves, proof paths, competitor output or cached query results. Automatic route
selection remains disabled; route stays an explicit oracle competitor.

## Unchanged contracts

Exact API 1, native ABI 3, SBC1/SBQ1/SCQ2/SBP1 bytes, raw/SCC/route semantics,
complete original assignments, portable contradiction proofs, independent replay,
default resource limits and explicit-engine behavior remain unchanged.

## Development evidence

Before v4 confirmation, all of the following must pass on x86-64:

1. the original 48-case structural development matrix;
2. the complete failed-v3 confirmation panel, now openly classified as v4
   development evidence;
3. a new density sweep generated under seed namespace
   `SPECTRA-auto-selector-v4-density-development`, including matched absolute pin
   counts on both sides of the `1/32` threshold;
4. exposed WAP05/WAP06 at 64, 128 and 1,024 requests;
5. ITC set9/set10/set12 at 1,024 requests; and
6. focused public policy, packet, proof and replay contracts.

Development acceptance:

- global realised-regret geometric mean `<= 1.025`;
- global selection-regret geometric mean `<= 1.015`;
- every family/count geometric mean `<= 1.05`;
- at least 97% of cells within 5% of the explicit oracle;
- every cell `<= 1.10`;
- all structural decision rules exactly followed;
- WAP 1,024 auto/raw geometric mean `<= 0.75`, every graph `<= 0.80`, and all
  tested WAP banks select SCC;
- every ITC case selects raw and auto/raw paired median `<= 1.05`; and
- zero correctness, replay, timeout, crash or completeness failure.

## Fresh prospective confirmation

Only after code and thresholds are committed may a new panel be generated under
seed namespace `SPECTRA-auto-selector-v4-confirmation`. It must not reuse v3
confirmation cases.

The fixed panel contains eight topology families and five regimes per family:

1. 320 vertices, strong two-colour contraction, sparse pins;
2. 512 vertices, about 2.7% average density with high absolute pin count;
3. 384 vertices, about 3.6% average density with the same high absolute pin count;
4. 1,536 vertices, about 1.8% density with still higher absolute pin count; and
5. 512 vertices, prospectively base-infeasible.

Counts are `1, 8, 32, 64, 128, 1024`. Each case contains deterministic mixed
SAT/UNSAT requests, duplicate pins and query-blind topology generation. Explicit
`raw`, `scc`, and `route` plus `auto` run for 17 paired repetitions after two
warmups. The same frozen cases run on x86-64 and native ARM64.

Confirmation acceptance, independently on each architecture:

- global realised-regret geometric mean `<= 1.04`;
- global selection-regret geometric mean `<= 1.025`;
- every family/count geometric mean `<= 1.06`;
- at least 95% of cells within 5% of the explicit oracle;
- every cell `<= 1.10`;
- no family aggregate more than 3% slower than raw;
- all structural rules exactly followed; and
- zero incorrect decision, invalid witness/proof, incomplete cell, timeout or
  crash.

The workflow must use `set -euo pipefail` and explicitly re-read
`results.json` to enforce `summary.pass`; a green outer pipeline cannot mask a
failed benchmark process.

## Product and regression guards

Before merge:

- Issue 1 first-use SCC/raw gates remain green on x86-64 and ARM64;
- the previously frozen MiniCard, CaDiCaL and CUDD advantages are not weakened;
- WAP flagship regression remains green;
- ITC source-only replay accepts both retained structural-v2 evidence and current
  structural-v4 output;
- Linux x86-64, native ARM64, Windows/MSVC, Python 3.11/3.13, UBSan,
  ASan+UBSan, compiler-free wheel, public-package and full CPU matrices pass.

## Interpretation

A pass establishes a reliable automatic dispatcher for the tested exact-query
scope. It does not establish universal SAT/colouring dominance, outside-team
replication, production p99 behavior, operator adoption or one-query leadership
against every external solver.
