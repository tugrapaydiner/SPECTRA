# Structural auto policy v6 — failed confirmation

## Status

Policy v6 passed the exposed 720-cell development panel, the WAP long-bank guard,
and the three-case ITC execution guard. It then failed the independently opened
confirmation panel on both x86-64 and native ARM64. The failure is retained and
v6 must not be merged as a successful selector.

The confirmation contained 48 newly generated graph cases: eight topology
families crossed with six regimes, evaluated at 1, 8, 32, 64, 128, and 1,024
requests. Every cell retained paired raw, SCC, route, and automatic timings over
31 repetitions. All answer banks passed independent replay; the only failure was
the predeclared performance gate.

## Frozen-gate results

| Environment | Global regret | Selection regret | Cells within 5% | Worst cell | Worst family auto/raw | Result |
|---|---:|---:|---:|---:|---:|---|
| x86-64 | 1.010578 | 1.005320 | **94.79%** | 1.088852 | 0.873258 | **FAIL** |
| ARM64 | 1.014293 | 1.007454 | **92.36%** | 1.091138 | 0.882216 | **FAIL** |

The frozen confirmation required at least 95% of cells within 5% of the explicit
per-cell oracle. No threshold was changed after measurement.

## Failure mechanism

The dominant failure class was the `above-density` regime at 64, 128, and 1,024
requests. These cases had 448 vertices, about 19 pins per request, and no useful
SCC contraction. V6 treated every sufficiently dense pin bank as raw. On long
banks, however, tile64 SCC propagation still amortised its construction and was
about 5–9% faster than raw.

The retained ITC evidence supplies the complementary boundary:

- set9, 169 vertices: SCC was faster than raw;
- set10, 214 vertices: SCC was faster than raw;
- set12, 78 vertices: raw remained faster.

These results motivate a new development rule for v7: dense banks use SCC only
when the bank contains at least 64 requests and the graph has at least 128
vertices. Smaller dense graphs stay raw. The failed v6 panel is now development
evidence and can never be reused as v7 confirmation.

## Evidence identities

Workflow run: `39096807922`

- x86 artifact `12125586385`, ZIP SHA-256
  `af7bea5698344cbb52aba0c6f57ab11cc5841cc555b9c142e573ce75d46a17c5`;
- ARM64 artifact `12125269318`, ZIP SHA-256
  `b1a7dd203f6747eeaa5ff862cb4868316fa2d9f5394ba278418a120af8a3dd0a`.

Both environments completed every benchmark cell, produced zero rule or
correctness failures, retained all paired timing vectors, and failed only the
unchanged fraction-within-5% gate.
