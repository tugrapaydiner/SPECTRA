# Stop repeating work in feasible query batches

This continues `faster-queries` from
`2afaf001c78bee5a3ea5a3918193acd012213150`. The previous 10% target failed.
All 48 structural-panel graphs have already been exposed. This is development,
with no new holdout, changed threshold or novelty claim.

An instrumented diagnostic of that commit found that contradictory lanes repeat
request sorting and SCC propagation after the 64-lane pass has already found a
conflict. In the two preferential-attachment cases, the repeated scalar solver
took roughly 0.21–0.27 ms of a 0.91–0.95 ms instrumented batch call in auto mode.
Instrumentation overhead is included; these are diagnostic timings, not the
primary speed result. SAT witness construction also allocates and copies a
separate output vector for every successful lane.

## Changes to investigate

1. Reuse the current lane's prepared roots during proof reconstruction. For SCC
   and sparse modes, go directly to original-edge proof search after the batch
   has detected the conflict. Routed mode must still reconstruct its predecessor
   chain. Preserve the exact proof bytes and current-request isolation.
2. Reduce allocations and repeated component lookups while materializing SAT
   answers. Every original vertex must still be emitted, and the separate
   observer must check the complete serialized output.

Neither change may skip pins, cache request answers, change engine selection,
relax memory admission, or share mutable scratch between calls. Existing modes,
defaults, packet schemas and ABI stay unchanged. Apply common improvements to
the raw control too wherever applicable.

## Comparison and acceptance

Retain the original six-arm comparison as the default. Add an explicit option
to compare against a named full commit, including its auto engine. For this
continuation, the baseline is the commit above. Use seven arms: previous
raw/route/auto and candidate raw/SCC/route/auto. Keep all 48 graphs, five bank
lengths, nine paired repetitions, both warmups, immutable records, independent
replay, complete-call timing, memory/deadline limits and graph-level statistics
from [the earlier plan](QUERY_LATENCY_PLAN.md).

Report candidate auto against previous auto as well as candidate raw. Preserve
all failed cells and regressions above 25%. The existing 10% target against
candidate raw remains unchanged: long-bank ratio <= 0.90, upper 95% bound < 1,
every family <= 1.10, and single-request ratio <= 1.10. Improvements against the
previous implementation do not by themselves pass that stronger target.

Run exhaustive small-problem and MiniCard checks, malformed-input and proof
mutation tests, byte equality against the previous modes, Linux x86/ARM CI,
Windows builds, and address/undefined sanitizers. The final comparison must use
committed, uninstrumented source. Keep diagnostics distinct from benchmark data.

```bash
python -m experiments.compiled_query.query_latency run --out feasible-batches \
  --baseline-commit 2afaf001c78bee5a3ea5a3918193acd012213150 --include-baseline-auto
python -m experiments.compiled_query.query_latency audit --out feasible-batches
```
