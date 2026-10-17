# Tile64 external-competition audit protocol — 17 October 2026

## Status and purpose

This is an **exposed-development audit**, not a new holdout, outside-team
replication, production trace, or universal SAT/colouring claim. It is frozen
before timing the current `main` implementation at commit
`4a647e54e7e7e0756e74b52575f86f03d2764455`.

The preceding tile64 checkpoint established a large same-contract improvement
against SPECTRA's current raw implication engine on the exposed 48-graph
structural panel. That result is necessary but is not, by itself, an external
competitive win. This audit therefore challenges the exact SCC bank endpoint
against independently maintained native solvers on a matched complete-cost
boundary. No automatic-selection work begins until this audit is resolved.

## Fixed workloads

Use only the already exposed WAP development graphs `wap05a.col` and
`wap06a.col` from `marijnheule/clicolcom` at commit
`4932048642da2144f387961b595112277afff82f`.

The existing `experiments.compiled_query.study export-wap` path reconstructs the
historical cases. These are real public conflict topologies with synthetic
binary lists and 1,024 historically SAT-accepted restrictions per graph. The
selection is disclosed. The graphs and query banks cannot become new holdouts.
They are useful here because every arm must return a complete original-vertex
witness; no competitor receives credit for a weaker UNKNOWN response or an
unverifiable decision.

Each upstream graph blob must match
`experiments/quotient_application/WAP_UPSTREAM_INVENTORY.json` before case
construction.

## Fixed arms

All arms receive the same canonical validated case JSON and the same query order.

1. `exact-bank-scc` — candidate. Current proof-carrying exact engine, explicit
   SCC mode, with the optimized 64-lane bank preparation merged in PR #69.
2. `exact-bank-raw` — current internal control, not the older routed endpoint.
3. `minicard` — persistent native MiniCard from pinned
   `python-sat==1.9.dev15`, retaining learned state across all 1,024 queries.
4. `cadical195` — persistent native CaDiCaL 1.9.5 from the same pinned PySAT
   distribution, retaining learned state across the bank.
5. `cudd-reduced-natural` — independently maintained native CUDD at upstream
   `tulip-control/dd@4442d498ddea1bec0071315f538cb5ef37443985`, with the
   repository's disclosed generic signed-equivalence elimination and unit
   propagation. This arm is required on x86-64; it is not required on ARM64.

No arm may consume another arm's prepared answers, quotient, query cache, or
measured timings. The external SAT solvers use their public persistent-assumption
interfaces. The CUDD manager is rebuilt per complete session.

## Matched endpoint and accounting

Use `exact-bank-scc`, **not** the ready-SBC1 packet fast path. Every arm starts
from the same canonical case JSON. The complete timer charges:

- input read, decode, validation and hashing;
- deterministic query ordering;
- candidate query-bank packing;
- fresh solver/checker construction;
- all 1,024 decisions;
- complete original-vertex witness materialisation;
- a separately owned native original-input check for every SAT answer;
- output bank serialisation and hashing;
- diagnostics; and
- explicit disposal.

Process launch/import remains in a separately retained cold-wall receipt, as in
the existing study harness. Compiler/build time, graph acquisition and evidence
transport remain outside the endpoint and are disclosed.

Because the WAP development banks are all SAT, exact SCQ2 rows carry no UNSAT
proof payload. Thus the relevant output obligation is matched: every arm must
return and independently validate a full original-vertex witness for every
query. Exact output still passes the independent SCQ2 replay path.

## Schedule and resource rules

- Python: CPython 3.13.
- Environments: GitHub Ubuntu 24.04 x86-64 and native Ubuntu 24.04 ARM64.
- Bank length: exactly 1,024.
- Repetitions: nine paired complete sessions per graph and arm.
- Orders: forward, reverse and fixed shuffle, cycling by repetition.
- Arm order: deterministic randomized pairing from the existing study harness.
- CPU affinity: one available CPU per worker.
- Worker wall deadline: 120 seconds.
- Address-space ceiling: 2 GiB.
- Native admission ceiling: 512 MiB.
- CUDD soft node ceiling: 5,000,000.
- Failed, timed-out, unavailable and incomplete cells are retained and fail the
  competitive gate. There is no retry-until-pass selection.

## Correctness gate

Every scheduled job must be `COMPLETE`. The independent audit must verify:

- exact source/build/input identities;
- timing-stage accounting and cold-wall domination;
- output byte geometry and hashes;
- all full witnesses against original graph edges, lists and current query;
- every exact SCQ2 answer through the independent original-input replay path;
- exactly `scheduled_jobs × 1,024` audited SAT answers;
- zero UNKNOWN, false UNSAT, malformed output, crash, timeout or unavailable
  baseline; and
- two distinct graph clusters in every comparison.

Any correctness failure blocks all performance wording.

## Predeclared meaningful-win gate

Ratios are `exact-bank-scc complete time / baseline complete time`; lower is
better for SPECTRA. Aggregation is the geometric mean of the two per-graph mean
ratios. Repetitions and queries do not create additional graph clusters.

### External competition

On **both x86-64 and ARM64**, against each of MiniCard and CaDiCaL:

- geometric-mean complete ratio must be `<= 0.50` (at least 2× lower complete
  latency overall); and
- each graph ratio must be `<= 0.67` (at least about 1.49× lower latency on every
  graph).

On **x86-64**, the same thresholds apply against reduced native CUDD.

### Current internal control

On both architectures, against `exact-bank-raw`:

- geometric-mean complete ratio must be `<= 0.75`; and
- each graph ratio must be `<= 0.85`.

This internal gate is deliberately weaker than the external 2× gate because raw
is a narrowly optimized same-problem engine. It still excludes a slight or
single-graph-only win.

The complete audit passes only if every applicable threshold passes. Thresholds,
arms, inputs and schedules are not relaxed after seeing results. A failed gate
keeps this issue open and triggers optimization or a narrower honest claim; it
does not become a victory through a changed denominator.

## Interpretation boundary

A pass would establish that the merged tile64 SCC engine has a large current-main
complete-cost advantage over three serious exact alternatives on these two
exposed, all-SAT WAP development banks, while preserving full checked answers.
It would strengthen the already confirmed frozen WAP result, but it would not
create a new holdout, prove one-query superiority, validate high-pin ITC banks,
show universal SAT dominance, or replace the need for outside-team replication
and an application-owner workload.
