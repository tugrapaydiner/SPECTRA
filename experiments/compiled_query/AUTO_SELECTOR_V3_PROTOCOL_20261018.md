# Automatic engine selection — protocol, 18 October 2026

## Question

Can `mode="auto"` choose the right exact engine from graph and request-bank
geometry alone, without consulting answers or measured timings, while staying
within 5% of the fastest explicit admissible engine across short, long,
compressible, weakly compressible, base-infeasible and high-pin workloads?

Issue 1 is already closed for explicit long-bank SCC execution. This protocol
addresses only Issue 2: making the public automatic mode deliver the correct
engine without forcing users to know the workload in advance.

## Protocol chronology

The development panel and target metrics were committed before measuring the
existing structural-v2 selector. That exposed diagnosis was then used to choose
one structural-v3 policy. Before opening the prospective confirmation panel, two
clarifications are recorded here:

1. strong SCC contraction may select SCC from eight requests onward; this is the
   explicit development-tuned exception to the ordinary short-bank raw rule; and
2. route mode can serialize a different but equally valid contradiction path, so
   correctness requires identical decisions plus independent replay of every
   output, not byte-identical proof bytes across all explicit modes. Raw, SCC and
   auto must remain byte-identical whenever auto selects that explicit engine.

No prospective confirmation result existed when these clarifications and the v3
implementation were frozen. After confirmation begins, no selector threshold,
feature, exception or acceptance gate may change.

## Non-negotiable boundaries

The selector may inspect only immutable input geometry available before solving:

- graph and domain sizes;
- implication/SCC geometry computed during construction;
- bank length and total pins;
- per-row pin statistics;
- duplicate pins and immediately empty intersections;
- quotient size/depth/arc counts; and
- native memory admission estimates.

It may not inspect SAT/UNSAT answers, measured timings, a prior trial solve,
cached query answers, proof paths, or competitor output. It may not run several
engines and choose the fastest afterward.

Exact API 1, native ABI 3, SBC1/SBQ1/SCQ2/SBP1 bytes, raw/SCC/route semantics,
complete witnesses, portable proofs, independent checking, defaults and resource
limits remain unchanged.

## Development panel

The exposed development panel is the existing 48-case structural matrix:

- eight graph families;
- six deterministic replicas per family;
- 128- and 512-vertex cases;
- three domain regimes;
- counts `1, 8, 32, 63, 64, 65, 128, 1024`;
- explicit `raw`, `scc`, and `route` arms plus `auto`;
- nine paired repetitions per case/count/arm;
- deterministic request order and arm shuffling; and
- two untimed warmups per arm/case/count.

The two exposed WAP development banks are added at counts 64, 128 and 1,024 to
ensure the selector actually chooses SCC where Issue 1 established a large
first-use advantage.

The exposed ITC set9, set10 and set12 repair banks are added at count 1,024 to
ensure high-pin application workloads remain on raw when raw is the fastest
explicit engine.

These inputs are tuning evidence. They are not new holdouts.

## Frozen structural-v3 policy

The prospective policy is fixed before confirmation:

1. base-infeasible fixed graph: `raw`;
2. at least eight pins per request on average: `raw`;
3. feasible low-pin bank of at least 64 requests: `scc`;
4. feasible low-pin bank of at least eight requests where SCC count is at most
   one quarter of the original literal count: `scc`;
5. otherwise: `raw`.

Automatic route selection is removed. Route remains an explicit engine and an
oracle competitor. Development evidence showed that the optimized SCC tile path
was normally as fast or faster while retaining the same decision/proof contract.

## Prospective confirmation panel

After the selector implementation and all thresholds are committed, a separate
confirmation workflow generates a fixed panel under seed namespace
`SPECTRA-auto-selector-v3-confirmation`:

- eight topology families distinct in seed from the development panel;
- four replicas per family;
- 256- and 1,024-vertex cases;
- low-, medium- and high-pin banks;
- feasible and base-infeasible fixed graphs;
- mixed SAT/UNSAT request banks;
- counts `1, 8, 64, 128, 1024`;
- explicit `raw`, `scc`, and `route` plus `auto`; and
- nine paired repetitions.

The confirmation cases may be generated only after the selector code is frozen.
No threshold or selector change is permitted after opening their results. A
failure remains a failure.

## Timing and correctness

Primary timing begins before ordinary request-bank packing and ends after the
complete independently checked SCQ2 result returns. Problem admission is outside
this primary boundary because an `ExactProblem` is the public admitted fixed
problem. A secondary application guard includes problem admission for one-bank
cases.

All arms must return the same SAT/UNSAT decision for every request. Every unique
SCQ2 bank is replayed from original inputs with the standard-library verifier.
When auto reports `raw` or `scc`, its output bytes must equal the corresponding
explicit arm. Route proof bytes may differ only if both certificates independently
replay. Missing cells, crashes, timeouts, UNKNOWN answers, decision disagreements,
malformed certificates or replay failures invalidate the entire result.

## Oracle and regret

For each case/count/environment, the oracle is the explicit engine with the
lowest paired median complete time among `raw`, `scc`, and `route`. The oracle is
used only for evaluation; automatic mode cannot see it.

Two metrics are retained:

- **selection regret:** median time of the explicit engine selected by auto,
  divided by the fastest explicit median;
- **realized auto regret:** median auto complete time divided by the fastest
  explicit median.

The first isolates policy quality; the second includes dispatch overhead and
measurement noise. A value of 1.00 is oracle-equivalent.

## Acceptance gates

### Development

All must pass on x86-64:

1. global case/count geometric-mean realized regret `<= 1.03`;
2. every family/count geometric-mean realized regret `<= 1.05`;
3. at least 95% of case/count cells have realized regret `<= 1.05`;
4. every case/count cell has realized regret `<= 1.10`;
5. global selection-regret geometric mean `<= 1.02`;
6. WAP 1,024-request banks select `scc` and auto/raw ratio `<= 0.75`;
7. all base-infeasible cells select `raw`;
8. counts below eight select `raw`;
9. counts from eight to 63 select SCC only under the frozen fourfold-contraction
   exception; and
10. ITC set9/set10/set12 select `raw` and remain within 5% of explicit raw.

### Prospective confirmation

All must pass independently on x86-64 and native ARM64:

1. global geometric-mean realized regret `<= 1.05`;
2. every family/count geometric-mean realized regret `<= 1.07`;
3. at least 90% of case/count cells have realized regret `<= 1.05`;
4. every cell has realized regret `<= 1.12`;
5. global selection-regret geometric mean `<= 1.03`;
6. no workload family is more than 5% slower than raw at its aggregate level;
7. all frozen structural decision rules are followed; and
8. zero correctness or completeness failures.

### Product guard

The existing Issue 1 first-use gates must remain green:

- admitted-problem bank SCC/raw geomean below 0.60 and every case below 0.65;
- one-bank application SCC/raw geomean below 0.75 and every case below 0.80; and
- existing external competition and WAP regression results remain unchanged.

## Interpretation

A pass establishes that the automatic policy is a reliable structural dispatcher
for the tested exact-query scope. It does not establish universal solver
superiority, one-query leadership against every external SAT solver, production
p99 behavior, outside-team replication or operator adoption.
