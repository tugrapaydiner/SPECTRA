# Compiled-query validation track

An opt-in, complete-cost challenge to SPECTRA's binary-list feasibility service.
This directory does not replace the public solver or the frozen WAP study.

The newer [exact mixed-outcome checkpoint](EXACT_REVIEW_20261013.md) adds a separate
proof-carrying binary-domain service and documents its regressions as well as
completed checks. Its [exact contract](EXACT_BINARY_CONTRACT.md) is stronger than
the earlier feasibility-only arms. The studies linked below are the earlier CUDD
checkpoint; their frozen outcomes are not overwritten or silently upgraded.

Start with the [review and results](REVIEW_20261013.md), the
[development protocol](PROTOCOL.md), the [reduction contract](REDUCTION_CONTRACT.md),
and the [roadmap](ROADMAP.md). [Machine-readable results](RESULTS_20261013.json)
include failed cells; `null` is not an infinite speedup.

## What is implemented

The independently maintained native CUDD backend is exercised directly through
pinned `dd`, alongside persistent MiniCard, CaDiCaL and the noncontracting native
control. Explicit CUDD variants cover legacy balanced construction, frontier
construction, redundant equivalence seeding, and lossless generic equivalence
elimination plus unit propagation. The last variant reconstructs every original
variable and colour. No SPECTRA quotient, prepared answer or query cache is given
to the comparator. The preprocessing and adapters are project code, not an
outside team's replication.

Every successful query receives a full original-vertex witness and a separately
owned original-input check. The complete timer includes input, preprocessing,
compilation, queries, checking, output materialization/hashing, diagnostics and
disposal. Cold process cost is retained separately. All failures remain visible.

## Run the native contracts

Use the pinned dependency build in
[compiled-query-validation.yml](../../.github/workflows/compiled-query-validation.yml).
It checks Python 3.11 and 3.13, builds native CUDD at upstream commit
`4442d498ddea1bec0071315f538cb5ef37443985`, and uses `python-sat==1.9.dev15`.
After installing those dependencies, from the repository root:

```bash
SPECTRA_REQUIRE_CUDD=1 python -m pytest experiments/compiled_query -q
```

Missing native CUDD is a test failure when that environment variable is set.
The explicitly named `autoref` reference tests never stand in for the native arm.

## Run an unfiltered development study

The following commands are for the Linux process-isolated harness. They use the
actual extension path returned by the build, rather than a hardcoded Python ABI.
Use fresh destination directories. Five synthetic structures are generated
without accepting or rejecting requests based on solver answers.

```bash
library="$(python -c "from spectra.cnf.quotient_query import build_quotient_runtime; print(build_quotient_runtime('cq-build'))")"
python -m experiments.compiled_query.study synthetic --output cq-cases --n 96 --queries 128
python -m experiments.compiled_query.study new \
  --cases cq-cases --library "$library" --output cq-study \
  --arms scc none minicard cadical195 cudd-reduced-natural cudd-reduced-bfs cudd-reduced-occurrence \
  --prefixes 1 128 --repeats 3 --deadline 15
python -m experiments.compiled_query.study run --output cq-study
python -m experiments.compiled_query.study audit --output cq-study > cq-audit.json
```

A successful audit means the evidence is internally consistent, not that every
arm completed. In particular, SPECTRA's current `UNKNOWN` responses on infeasible
requests make those sessions incomplete. Read `outcomes`,
`all_scheduled_decisions_complete` and every comparison's `eligible` field.
The example above is a reproduction recipe, not a claim that its exact final
seven-arm synthetic matrix has already been run.

`run --max-new N` stops after N new jobs. Re-running the same command resumes only
with matching source, inputs, binaries and environment. It does not retry failed
jobs until they pass. A retry or changed source requires a new identified study.

## Evidence and limits

The final WAP development matrix is in
[run 38482658200](https://github.com/tugrapaydiner/SPECTRA/actions/runs/38482658200).
The required native contracts are in
[run 38482327907](https://github.com/tugrapaydiner/SPECTRA/actions/runs/38482327907).
Source bundles, raw outputs, logs, build receipts and audits are workflow
artifacts; the configured retention is 30 days. Preserve the delivered recovery
archive rather than relying on permanent Actions artifact availability.

WAP05/06 are already exposed graphs with synthetic lists and historically
SAT-filtered query banks. There are two graph clusters, not thousands of
independent problems. The final development result is not new held-out evidence,
not outside replication, and not a repair of the older frozen Python 3.11 gate.
