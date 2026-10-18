# Automatic engine selection — frozen protocol, 18 October 2026

## Question

Can `mode="auto"` choose the right exact engine from graph and request-bank
geometry alone, without consulting answers or measured timings, while staying
within 5% of the fastest explicit admissible engine across short, long,
compressible, weakly compressible, base-infeasible and high-pin workloads?

Issue 1 is already closed for explicit long-bank SCC execution. This protocol
addresses only Issue 2: making the public automatic mode deliver the correct
engine without forcing users to know the workload in advance.

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
- deterministic request order and arm shuffling;
- two untimed warmups per arm/case/count.

The two exposed WAP development banks are added at counts 64, 128 and 1,024 to
ensure the selector actually chooses SCC where Issue 1 established a large
first-use advantage.

The exposed ITC set9, set10 and set12 repair banks are added at count 1,024 to
ensure high-pin application workloads remain on raw when raw is the fastest
explicit engine.

These inputs are tuning evidence. They are not new holdouts.

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
- explicit `raw`, `scc`, and `route` plus `auto`;
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

Every explicit and automatic arm must return byte-identical certified output for
the same packet. Every unique SCQ2 bank is replayed from original inputs with the
standard-library verifier. Missing cells, crashes, timeouts, UNKNOWN answers,
decision disagreements, malformed certificates or replay failures invalidate the
entire result.

## Oracle and regret

For each case/count/environment, the oracle is the explicit engine with the
lowest paired median complete time among `raw`, `scc`, and `route`. The oracle is
used only for evaluation; automatic mode cannot see it.

Regret is:

`median(auto complete time) / min(median(raw), median(scc), median(route))`.

A value of 1.00 is oracle-equivalent; lower values may occur only through timing
noise and are retained as measured.

## Frozen acceptance gates

### Development

All must pass on x86-64:

1. global case/count geometric-mean regret `<= 1.03`;
2. every family/count geometric-mean regret `<= 1.05`;
3. at least 95% of case/count cells have regret `<= 1.05`;
4. every case/count cell has regret `<= 1.10`;
5. WAP 1,024-request banks select `scc` and auto/raw ratio `<= 0.75`;
6. all base-infeasible cells select `raw`;
7. all counts below 64 select `raw` unless a frozen explicit exception is
   documented before confirmation;
8. ITC set9/set10/set12 select `raw` and remain within 5% of explicit raw.

### Prospective confirmation

All must pass independently on x86-64 and native ARM64:

1. global geometric-mean regret `<= 1.05`;
2. every family/count geometric-mean regret `<= 1.07`;
3. at least 90% of case/count cells have regret `<= 1.05`;
4. every cell has regret `<= 1.12`;
5. no workload family is more than 5% slower than raw at its aggregate level;
6. zero correctness or completeness failures.

### Product guard

The existing Issue 1 first-use gates must remain green:

- admitted-problem bank SCC/raw geomean below 0.60 and every case below 0.65;
- one-bank application SCC/raw geomean below 0.75 and every case below 0.80;
- existing external competition and WAP regression results remain unchanged.

## Interpretation

A pass establishes that the automatic policy is a reliable structural dispatcher
for the tested exact-query scope. It does not establish universal solver
superiority, one-query leadership against every external SAT solver, production
p99 behavior, outside-team replication or operator adoption.
