# Automatic exact-engine selection v8 — frozen protocol

**Frozen before v8 implementation or timing: 18 October 2026.**

## Why v8 exists

The retained v7 prospective panel passed every aggregate gate except the fixed
coverage gate. Its failure was concentrated in one structural class: feasible
high-pin graphs with 224 vertices. V7 kept every dense graph below 256 vertices
on raw, but SCC was 4–11% faster at 64, 128, and 1,024 requests on both x86-64
and ARM64.

The same rule protected three ITC exam-repair banks. Their explicit engines show
a more precise boundary:

- 78 vertices: raw is about 1.4% faster than SCC;
- 169 vertices: SCC is about 8.2% faster than raw;
- 214 vertices: SCC is about 9.8% faster than raw.

V8 therefore changes only the dense-long-bank size boundary. It does not alter
any acceptance threshold after observing a result.

## Frozen selector

The selector may inspect only immutable structural data already parsed by the
native endpoint: original vertices and edges, implication-component count,
base satisfiability, request count, and total pin count. It may not execute a
candidate engine, inspect answers, read timings, recognize dataset names, or
retry a choice.

Apply the following rules in order:

1. For at most eight requests on a graph larger than 512 vertices, use raw.
2. If the fixed problem is already infeasible, use raw.
3. Define a dense high-pin bank as at least eight pins per request and aggregate
   pin density of at least `1/24` of the original vertices.
4. Before 64 requests, every dense high-pin bank uses raw.
5. At 64 or more requests, a dense high-pin bank below 128 vertices uses raw;
   a dense high-pin bank with at least 128 vertices continues to the long-bank
   rule.
6. At exactly 64 requests, an uncompressed graph with at least four original
   edges per vertex uses raw.
7. Every other feasible bank of at least 64 requests uses SCC.
8. From 9 through 63 requests, use SCC only when the implication quotient has at
   least fourfold literal compression (`2 * components <= vertices`).
9. Otherwise use raw.

The selected mode must be reported in the existing session metadata. Exact API
1, native ABI 3, packet formats, outputs, proofs, resource limits, and explicit
engine modes remain unchanged.

## Fresh confirmation panel

V7's confirmation cases are now development evidence and will not be reused as
v8 confirmation. V8 uses a new namespace, new graph/query seeds, and the eight
existing query-blind topology families. For every family, generate these seven
regimes:

| Regime | Vertices | Pins/request | Purpose |
|---|---:|---:|---|
| strong-sparse | 320 | 3 | preserve short-bank compression win |
| dense-small | 96 | 12 | raw side of the new boundary |
| dense-boundary-low | 120 | 12 | raw immediately below the boundary |
| dense-boundary-high | 128 | 12 | SCC exactly at the boundary |
| dense-medium | 192 | 12 | SCC in the real-ITC size range |
| edge-heavy-warmup | 768 | 12 | preserve the 64-request raw warmup |
| base-infeasible | 448 | 4 | preserve immediate raw fallback |

Each case is measured at 1, 8, 32, 64, 128, and 1,024 requests. Graphs and query
banks are generated before timing and independently of selector decisions.

Run five untimed warmups and 31 paired repetitions of raw, SCC, route, and auto
on CPython 3.13 Linux x86-64 and native ARM64. Mode order is deterministically
shuffled within every paired cell. All complete outputs must be byte-stable,
mode-equivalent, and independently replayable.

## Unchanged acceptance gates

All gates are evaluated separately on x86-64 and ARM64:

- zero decision, label, proof, packet, or replay failures;
- zero structural-rule mismatches;
- global realized-regret geometric mean `<= 1.04`;
- selector-decision regret geometric mean `<= 1.025`;
- every family/regime group regret `<= 1.06`;
- at least 95% of cells within `1.05` of the measured oracle;
- worst cell regret `<= 1.10`;
- every topology family's auto/raw geometric mean `<= 1.03`.

A failed gate remains failed. No threshold, case, repetition, architecture, or
family may be removed after timing.

## Real-workload guards

The established WAP guard must continue to select SCC for 1,024-request banks
and retain its large advantage over raw.

The ITC regression guard is updated before v8 timing to reflect the structural
boundary rather than dataset identity:

- set9 (169 vertices): select SCC;
- set10 (214 vertices): select SCC;
- set12 (78 vertices): select raw.

For set9 and set10, automatic mode must remain within 3% of explicit SCC and be
at least 2% faster than raw. For set12, automatic mode must remain within 5% of
raw. All proof-carrying outputs must remain complete and independently checked.

The paired WAP dispatch-overhead guard must remain within 3% of explicit SCC on
each architecture.

## Interpretation boundary

A pass establishes that the deterministic selector chooses an engine close to
the per-cell oracle across a fresh synthetic structural panel while preserving
the two disclosed real-workload guards. It is project-owned confirmation, not
outside replication, production adoption, or universal solver dominance.
