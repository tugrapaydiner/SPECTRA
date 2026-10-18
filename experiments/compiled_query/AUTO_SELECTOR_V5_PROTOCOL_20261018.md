# Automatic engine selection v5 — frozen protocol

## Why v5 exists

Structural-v4 fixed the absolute-pin mistake that caused v3 to fail prospective
confirmation, and it passed the real WAP and ITC guards. It still failed the
full 720-cell development oracle gate:

- realised-regret geometric mean: `1.007368` — pass;
- selection-regret geometric mean: `1.005854` — pass;
- worst family/count geometric mean: `1.032266` — pass;
- cells within 5% of the explicit oracle: `95.42%` — fail versus `97%`;
- worst realised regret: `1.162020` — fail versus `1.10`.

The complete failed run is retained as workflow `39092003260`, artifact
`12123004409`. No v4 threshold is reinterpreted and no v4 confirmation panel was
opened.

Two exposed structural mistakes explain the systematic misses:

1. A `1/32` pin-density cutoff is too conservative. The 384-vertex, 14-pin
   development banks have about 3.67% density and consistently favour SCC at 64,
   128 and 1,024 requests. The lowest real ITC density is about 4.25%, so `1/24`
   remains on the raw side for every real ITC guard case.
2. On uncompressed graphs with at least four original edges per vertex, raw wins
   the first 64–128 requests before SCC amortises construction. At 1,024 requests,
   SCC wins again. This pattern appears in the exposed communities cases and is
   not inferred from confirmation data.

Issue 2 remains the only active issue. Explicit raw, SCC and route engines remain
unchanged.

## Frozen structural-v5 policy

Selection order is fixed:

1. base-infeasible fixed graph: `raw`;
2. at least eight pins per request and average pin density at least `1/24`:
   `raw`;
3. at most 128 requests, no SCC contraction (`components == 2 * vertices`), and
   at least four original edges per vertex: `raw`;
4. feasible bank of at least 64 requests: `scc`;
5. feasible bank of at least eight requests whose SCC count is at most one
   quarter of the original literal count: `scc`;
6. otherwise: `raw`.

The native tests are integer-only:

```text
bank_pins >= 8 * bank_size
and
24 * bank_pins >= original_vertices * bank_size
```

and

```text
bank_size <= 128
and
components == 2 * original_vertices
and
original_edges >= 4 * original_vertices
```

All products fit the existing uint64 admission bounds. The selector may inspect
only immutable graph/request geometry. It may not inspect answers, measured
latency, trial solves, proof paths, competitor output or cached query results.
Automatic route selection remains disabled; route stays an explicit oracle arm.

## Unchanged contracts

Exact API 1, native ABI 3, SBC1/SBQ1/SCQ2/SBP1 bytes, explicit raw/SCC/route
semantics, complete original assignments, portable contradiction proofs,
independent replay, resource limits and default API shape remain unchanged.

## Development gate

Before v5 confirmation, the same exposed evidence must pass on x86-64:

- the original 48-case structural matrix;
- the complete failed-v3 confirmation panel;
- the v4 density sweep;
- the v4 failure cells and the existing edge-density contrast;
- WAP05/WAP06 at 64, 128 and 1,024 requests;
- ITC set9/set10/set12 at 1,024 requests; and
- focused packet, policy, proof and replay contracts.

The oracle matrix runs explicit raw, SCC and route plus auto for **17 paired
repetitions** after two warmups. Development acceptance is unchanged:

- global realised-regret geometric mean `<= 1.025`;
- global selection-regret geometric mean `<= 1.015`;
- every family/count geometric mean `<= 1.05`;
- at least 97% of cells within 5% of the explicit oracle;
- every cell `<= 1.10`;
- all structural rules exactly followed;
- WAP 1,024 auto/raw geometric mean `<= 0.75`, every graph `<= 0.80`, and every
  tested WAP bank selects SCC;
- every ITC case selects raw and auto/raw paired median `<= 1.05`; and
- zero correctness, replay, timeout, crash or completeness failure.

The pre-implementation simulation using the retained v4 explicit-engine medians
produced selection-regret geometric mean `1.003255`, worst selection regret
`1.068459`, worst family/count geometric mean `1.013238`, and `98.89%` of cells
within 5%. These numbers are design evidence only, not a v5 result.

## Fresh prospective confirmation

Only after the v5 implementation and development gates are committed may cases
be generated under the new namespace
`SPECTRA-auto-selector-v5-confirmation`. Neither v3 nor the unopened v4
confirmation namespace may be reused.

Eight topology families are crossed with six query-blind regimes:

1. strong two-colour contraction with sparse pins;
2. pin density just below `1/24`;
3. pin density just above `1/24`;
4. high-edge, uncompressed graphs that exercise the 64/128-request raw guard;
5. large graphs with high absolute pins but low relative density; and
6. prospectively base-infeasible graphs.

Counts are `1, 8, 32, 64, 128, 1024`. Each cell runs explicit raw, SCC and route
plus auto for 17 paired repetitions after two warmups on GitHub x86-64 and native
ARM64.

Confirmation acceptance, independently on each architecture, remains:

- global realised-regret geometric mean `<= 1.04`;
- global selection-regret geometric mean `<= 1.025`;
- every family/count geometric mean `<= 1.06`;
- at least 95% of cells within 5% of the explicit oracle;
- every cell `<= 1.10`;
- no family aggregate more than 3% slower than raw;
- all structural rules exactly followed; and
- zero incorrect decision, invalid witness/proof, incomplete cell, timeout or
  crash.

Every workflow must use `set -euo pipefail`, upload failed evidence, and reread
`results.json` to enforce `summary.pass`.

## Product and regression guards

Before merge:

- Issue 1 first-use SCC/raw gates remain green on x86-64 and ARM64;
- the frozen MiniCard, CaDiCaL and CUDD advantages remain documented and
  unchanged;
- WAP flagship regression remains green;
- current structural-v5 ITC output and retained structural-v2/v3/v4 evidence
  remain auditable at their pinned source commits;
- Linux x86-64, native ARM64, Windows/MSVC, Python 3.11/3.13, UBSan,
  ASan+UBSan, compiler-free wheel, public-package and full CPU matrices pass.

## Interpretation

A pass establishes a reliable automatic dispatcher for the tested exact-query
scope. It does not establish universal SAT/colouring dominance, outside-team
replication, production p99 behaviour, operator adoption or one-query leadership
against every external solver.
