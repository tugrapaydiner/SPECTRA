# Automatic engine selection v7 — frozen protocol

**Frozen:** 18 October 2026  
**Base implementation:** `performance/automatic-engine-selection-final@d515e70d4f88ef451b16ed54b45e2bee13559d15`

## Purpose

V6 passed every correctness, global-regret, group-regret and worst-cell gate, but missed the fixed `95% within 5%` confirmation gate on both architectures. The retained failures were systematic: feasible long high-pin banks remained on raw after SCC had become faster. This protocol changes only that structural boundary and opens a fresh confirmation namespace.

## Frozen rule

Selection uses immutable geometry only.

1. Banks of at most eight requests on graphs above 512 vertices use raw directly.
2. Fixed-graph contradictions use raw.
3. Before 64 requests, dense high-pin banks use raw.
4. Dense high-pin banks with fewer than 256 original vertices remain raw at every bank length. This is the application guard for ITC-sized problems.
5. At exactly 64 requests, an uncompressed graph with at least four edges per vertex remains raw.
6. Every other feasible bank of at least 64 requests uses SCC.
7. Between eight and 63 requests, SCC is used only when the SCC quotient compresses the literal graph by at least fourfold.
8. All remaining banks use raw.

The selector may not inspect graph names, answers, timings, machine identity, proof sizes or retry outcomes. Explicit raw, SCC, sparse and route modes remain unchanged.

## Fresh confirmation panel

The confirmation seed namespace is `SPECTRA-auto-selector-v7-confirmation`. It is generated only after this protocol and implementation rule are committed.

For each of the eight established topology families, generate these seven regimes:

- strong compression: `n=352`, two pins/request;
- high-pin below the density boundary: `n=704`, 27 pins/request;
- high-pin above the density boundary: `n=480`, 21 pins/request;
- edge-heavy uncompressed warmup: `n=832`, 13 pins/request, at least five edges/vertex;
- low-density large: `n=1408`, 26 pins/request;
- small high-pin application guard: `n=224`, 12 pins/request;
- base-infeasible: `n=576`, four pins/request.

Every case is evaluated at `1, 8, 32, 64, 128, 1024` requests with 31 paired repetitions after two warmups. No case may be filtered or regenerated after timing.

## Fixed gates

The existing v6 gates remain unchanged:

- global realized-regret geometric mean `<= 1.04`;
- global selection-regret geometric mean `<= 1.025`;
- worst family/count group geometric mean `<= 1.06`;
- at least `95%` of cells within `1.05` of the explicit oracle;
- worst cell `<= 1.10`;
- every family no more than `1.03` slower than raw overall;
- zero structural-rule failures;
- complete, independently replayed and decision-equivalent outputs.

The explicit oracle is the fastest paired median among raw, SCC and route. Auto output must be byte-identical to its reported selected explicit mode.

## Application guards

- WAP05/WAP06 at 1,024 requests must select SCC.
- ITC set9/set10/set12 at 1,024 requests must remain raw or stay within 5% of the best raw/SCC paired median.
- Base-infeasible banks must remain raw.
- Large banks below nine requests must retain the direct raw path.

A failed gate is retained as a failed result. Thresholds are not relaxed after confirmation.
