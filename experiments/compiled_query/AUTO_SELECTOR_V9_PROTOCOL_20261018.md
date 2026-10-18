# Automatic selector v9 protocol — 18 October 2026

## Status

The v8 prospective panel is now development evidence because its failures are
used here. The policy, thresholds, fresh namespace, and v9 confirmation panel
below are frozen before v9 timing.

## Correction

V8 kept every dense high-pin bank below 128 vertices on raw. The retained v8
result showed a stable crossover instead: at 64 or more requests, the 96- and
120-vertex banks preferred SCC across all eight graph families on x86-64 and
ARM64, while the fixed ITC set12 holdout at 78 vertices still preferred raw.

V9 therefore keeps dense high-pin work on raw when the bank has fewer than 64
requests or the graph has fewer than 96 vertices. Dense banks at 96 vertices or
more use the ordinary SCC long-bank rule. Obvious structural raw decisions are
also dispatched before native SCC construction, without consulting answers or
measured timings.

## Fresh confirmation panel

The new namespace is `SPECTRA-auto-selector-v9-confirmation`. It uses the same
eight graph generators with new seeds and these regimes: strong sparse n=336;
dense high-pin n=80, n=95, n=96, n=112, and n=176; edge-heavy n=736; and base
infeasible n=432.

Each case is tested at 1, 8, 32, 64, 128, and 1,024 requests with 31 paired
repetitions on CPython 3.13 x86-64 and native ARM64.

## Unchanged gates

- at least 95% of cells within 5% of the measured oracle;
- selection-regret geomean <= 1.025;
- realized-regret geomean <= 1.04;
- worst group regret <= 1.06;
- worst cell <= 1.10;
- no graph family slower than explicit raw by more than 3%; and
- every structural-rule assertion passes.

The existing WAP long-bank, WAP overhead, and ITC set9/set10/set12 guards must
also pass unchanged. Set12 must remain raw; set9 and set10 must remain SCC.

This is project-owned prospective confirmation, not outside replication or a
universal solver claim.
