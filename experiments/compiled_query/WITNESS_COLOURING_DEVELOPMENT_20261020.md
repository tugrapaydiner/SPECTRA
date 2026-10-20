# Witness-colouring development audit

**Date:** 2026-10-20  
**Classification:** exposed development evidence

The first six-family opening failed before timing because
`DSJC500.5.col` used 65 colours under the single deterministic DSATUR
witness, above the predeclared 63-colour limit. The complete failed opening
remains recorded separately.

## Exposed six-graph audit

The acquired graphs were then used only as development inputs. The audit
compared the original deterministic DSATUR, largest-first, smallest-last,
256 deterministic DSATUR tie orders, and 64 deterministic recursive-largest-
first tie orders. Every retained colouring was checked against the original
graph.

| Graph | Original DSATUR | Best 256-run DSATUR | Best audited method | Best colours |
|---|---:|---:|---|---:|
| `DSJC500.5.col` | 65 | 63 | 64-run RLF | 59 |
| `flat300_20_0.col` | 42 | 39 | 64-run RLF | 36 |
| `le450_25c.col` | 29 | 28 | DSATUR/RLF tie | 28 |
| `miles1500.col` | 73 | 73 | all tested methods tied | 73 |
| `queen16_16.col` | 23 | 20 | 64-run RLF | 19 |
| `school1.col` | 17 | 17 | smallest-last | 15 |

Audit artifact:

```text
sha256:347f91629a93d8103f6214aa67bc23f874aa831d0699617e966b757dbbbdaef7
```

The result explains both parts of the original admission failure:

- `DSJC500.5.col` was a weak single-tie-order failure; a fixed structural
  multi-start DSATUR portfolio reaches the exact 63-colour admission boundary.
- `miles1500.col` remains above the boundary under every tested full-graph
  heuristic. A stronger tie order alone is therefore not a robust general
  admission rule.

## Replacement construction

The development generator now uses a deterministic two-stage procedure:

1. Run a fixed portfolio consisting of the historical DSATUR, smallest-last,
   and 256 SHA-256-derived DSATUR tie orders. Keep the lowest-colour proper
   colouring, breaking ties by the complete colouring bytes and method identity.
2. If the full graph still needs more than 63 witness colours, sort original
   edges by a SHA-256 structural key and greedily retain a maximal subset in
   which every vertex has degree at most 62. Run the same colouring portfolio
   on that retained topology.

The fallback has a direct admission proof: a graph of maximum degree 62 is
greedily colourable with at most 63 colours, leaving colour 64 available as the
shared trap colour. It uses no SAT solver, request answer, benchmark timing, or
post-hoc case deletion.

Every receipt records whether the full topology was retained, the original and
retained edge counts, the deterministic prepared-topology hash, the degree cap,
the selected witness method/trial, and the complete witness hash.

## Claim boundary

This change repairs workload admission; it is not itself a solver speed result.
The six graphs are already exposed and can only support development testing.
Any new prospective comparison must freeze this generator against genuinely new,
previously unacquired graph data.
