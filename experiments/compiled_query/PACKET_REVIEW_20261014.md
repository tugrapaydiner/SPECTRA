# SPECTRA prepared exact sessions — development review

**Date:** 14 October 2026
**Classification:** development-only, exposed inputs, no external replication
**Result file:** [`PACKET_RESULTS_20261014.json`](PACKET_RESULTS_20261014.json)

## Decision

The exact binary-domain service now has an explicitly separate ready-input path.
A complete immutable **SBC1** packet contains the original graph, original one/two-
colour domains and the ordered current-request bank. One native call independently
parses that packet for the solver and observer, constructs both owners, solves every
request, creates every witness or contradiction path, emits strict SCQ2, reparses
and checks the final serialized bank, and disposes native state.

On the two already exposed mixed-WAP development cases, the paired process-isolated
study completed all **24/24** scheduled sessions and independently replayed
**24,576 exact decisions**: **14,472 SAT witnesses** and **10,104 UNSAT proofs**.
Prepared routed execution used a graph-geometric mean **0.121024x** the old exact-
raw JSON service's complete time—**87.90% lower**. Prepared raw used **0.125495x**.

This is a substantial end-to-end systems improvement, not an algorithmic discovery.
The gain is primarily from removing repeated Python object reconstruction,
per-request native crossings and Python-side SCQ2 assembly. Prepared route and
prepared raw are not consistently separated across the two graphs, so no new
proof-routing speed claim is justified.

## What was implemented

### SBC1 full-session format

SBC1 is a little-endian packet containing:

- schema, vertex/palette/edge/query/pin counts and zero reserved words;
- all original uint64 domains;
- sorted unique original edges;
- complete monotone query offsets;
- original vertex identifiers and uint64 current masks.

The solver and observer use separately written full-packet parsers and separately
owned graph/domain arrays. Every mask, edge, offset and query vertex is checked.
Malformed later rows cannot be hidden by an earlier contradiction. Conservative
admission covers retained owners, borrowed packet bytes, working state and the
worst permitted SCQ2 proof bank before solving begins.

### One-call certified execution

The native endpoint performs all solver/observer construction, exact decisions,
SAT labelling, SBP1 proof construction, independent checking, SCQ2 serialization
and native disposal before returning one immutable Python `bytes` object. It does
not publish intermediate per-query answer objects. ABI 3 uses versioned capsule
names and rejects ABI 2 handles before dereference.

The existing tuple-based single-query endpoint and SBQ1 bank endpoint remain
available. Existing frozen evidence and default solvers are unchanged.

### Reproducible prepared studies

The study freezer now writes exact SBC1 bytes for every paired case/prefix/order,
including SHA-256, byte count and a preparation-time receipt. Runtime file read and
hashing are charged to the prepared endpoint. Packet construction is explicitly
marked `preparation_charged=false` and remains outside ready-input latency.
Independent post-run replay still uses the original JSON case, original graph,
original domains and current requests rather than trusting the packet solver.

## Process-isolated result

Each row below is a mean of three paired sessions at 1,024 requests. All four arms
return the same portable-certificate contract.

| Exposed development case | Exact raw JSON | Exact route JSON | Prepared raw SBC1 | Prepared route SBC1 |
|---|---:|---:|---:|---:|
| WAP05 mixed common-mask | 185.295 ms | 199.018 ms | 25.186 ms | **22.530 ms** |
| WAP06 mixed common-mask | 204.428 ms | 177.677 ms | **23.687 ms** | 24.626 ms |

Graph-geometric complete-session ratios:

| Candidate / baseline | Ratio | Reduction |
|---|---:|---:|
| Prepared route / exact raw JSON | **0.121024x** | **87.90%** |
| Prepared raw / exact raw JSON | **0.125495x** | **87.45%** |
| Prepared route / exact route JSON | **0.125260x** | **87.47%** |
| Prepared route / prepared raw | 0.964374x | 3.56% descriptive advantage |

The final row is not a robust routing result: its per-graph ratios are **0.894556x**
and **1.039641x**, pointing in opposite directions. With only two exposed graph
clusters and three repetitions, the correct conclusion is a tie, not a universal
3.56% win.

Ready-packet input read plus SHA-256 costs approximately **0.30–0.41 ms** in these
sessions. The rest of each 22–25 ms call includes both native constructions, all
1,024 exact decisions, every proof/witness check, SCQ2 materialization and disposal.

## Preparation is not free

The six frozen packet preparations take **21.37–25.14 ms** from already validated
`BinaryListProblem` objects and frozen request rows. Adding mean packet preparation
to the prepared-route call gives:

| Case | Preparation + prepared route | Exact raw JSON | Ratio |
|---|---:|---:|---:|
| WAP05 | 44.444 ms | 185.295 ms | 0.239853x |
| WAP06 | 47.488 ms | 204.428 ms | 0.232298x |

The graph-geometric ratio is **0.236045x**, or **76.40% lower**, for first use from
an already validated in-memory problem. This combination is descriptive because
preparation was recorded at study freeze, not interleaved as another process arm.

Arbitrary JSON decoding remains additional work before packet preparation. The
ready-SBC1 result therefore must not be described as an 87.9% first-ever speedup
from arbitrary JSON. Its intended use is a persisted deployment artifact or a
reused session input.

## Verification

| Check | Result |
|---|---:|
| Focused exact/study/audit contracts | **179 passed** |
| Exact contracts under UBSan | **70 passed** |
| Exact contracts under ASan + UBSan | **70 passed** |
| Process-isolated paired sessions | **24/24 complete** |
| Independently replayed decisions | **24,576** |
| False or unverifiable SAT/UNSAT rows | **0** |

The contracts cover malformed packet regions, exact lengths, reserved words,
invalid domains and edges, offset monotonicity, out-of-range later pins, corrupted
SCQ2 rows, all proof kinds, old ABI handles, concurrency, lifecycle and worst-case
output admission. ASan used leak detection disabled; this is not a LeakSanitizer
claim.

## What this changes—and what it does not

This checkpoint materially improves deployment engineering and complete-session
cost. It also makes the cost boundary cleaner: preparation, ready-input execution,
and arbitrary JSON decoding are now distinct measured objects.

It does **not** close the principal flagship gates:

- both WAP graphs and their synthetic list construction were already exposed;
- there is no independent team, operator-defined workload or maintained adoption;
- SBC1/SCQ2 composition is a systems contribution built from established exact
  graph/SCC/2-SAT techniques, not a proven algorithmic first;
- prepared routing does not consistently beat prepared raw traversal;
- no stable package API, default solver or frozen WAP claim is changed.

Under the existing harsh rubric, this supports a meaningful engineering/release
increase, but it does not justify a 90–100 whole-program grade. The next credible
large scientific jump requires an outcome-blind second problem family or operator-
defined workload, ordinary-hardware transfer, and outside reproduction of the
final endpoint—not another internally selected timing table.

## Reproduce

Build the current native extensions, then run the packet-enabled exact workflow:

```bash
python -m pytest experiments/compiled_query/test_exact.py \
  experiments/compiled_query/test_exact_study.py \
  experiments/compiled_query/test_study.py -q

python -m experiments.compiled_query.study new \
  --cases <case-directory> \
  --library <quotient-extension> \
  --exact-library <exact-extension> \
  --output <study-directory> \
  --arms exact-raw exact-route exact-packet-raw exact-packet-route \
  --prefixes 1024 --repeats 3
python -m experiments.compiled_query.study run --output <study-directory>
python -m experiments.compiled_query.study audit --output <study-directory>
```

The exact mathematical, packet and certificate boundaries are in
[`EXACT_BINARY_CONTRACT.md`](EXACT_BINARY_CONTRACT.md).
