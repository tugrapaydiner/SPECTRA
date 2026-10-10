# Propagation and backtracking on public structured tasks

**Decision: retain the new DPLL backend as an explicit optional solver.** It
solved all 85 evaluation puzzles where the existing focused and deductive arms
solved zero and one, respectively. It used 37.2% less mean complete wall time than
focused search on this panel. Native SAT and the direct domain solver solved the
same 85 puzzles and were 2.52× and 6.65× faster than DPLL. No default changes or
neural-learning claims follow from this result.

## What changed and why

The previous [focused study](../focused_evaluation/RESULTS.md) improved planted
random CNF but did not demonstrate transfer to external task inputs. The first
ten public Norvig top95 Sudoku puzzles reproduced a concrete gap: both existing
SPECTRA arms solved 0/10 under 2,048 flips. A one-hot Sudoku CNF couples exactly-one
constraints; a Boolean flip can break other satisfied constraints. The remaining
85-case run left median residuals of 41 clauses for focused search and 20 for
deductive search. These observations motivate propagation/backtracking, but do not
constitute an isolated causal ablation of each algorithmic component.

The new generic [solver](../../spectra/cnf/dpll.py) maintains two watched literals
per clause, propagates forced assignments, and undoes decisions on conflict with
an iterative trail. It branches on the smallest unresolved positive clause before
arbitrary clauses. This fixed classical heuristic exposes remaining one-hot
domains; it is not a learned policy or a novel SAT algorithm. Profiling development
cases motivated cheaper binary-clause normalization and direct positive-literal
lookups. All ten development paths were preserved. Both attempts, initial source,
profile and final source are in [development evidence](development-20261005.zip).

## Frozen evaluation

Published source commit: `a0d7efaafd0573a4655d1e3965bbc1e7560f72ec`.
Tree: `64b4fe052008474944623da3735d64e88cbe37d0`.
The candidate, encoder, runner, analysis and [protocol](PROTOCOL.md) were published
in draft PR #53 before any local solver execution on rows 11–95. The exact
24-file source/data inventory is in [FROZEN.json](FROZEN.json). No candidate or
analysis code changed after evaluation exposure.

Inputs are the public top95 corpus at upstream commit
`bfcdac7e2c74f44a9c2ff36217a8dd834ad6b73b`. [UPSTREAM.json](UPSTREAM.json) pins the
raw puzzle file, unmodified direct solver and MIT license. Rows 1–10 were development;
rows 11–95 were the 85 evaluation cases. The split was by file order. The corpus
is public and familiar, not a secret or author-independent confirmation set.

Measurements on 2026-10-05: Python 3.12.14, Linux x86-64, AMD EPYC 9V74, one CPU
affinity, python-sat 1.9.dev15. Three timing rounds produced 1,275 raw rows. Counts
below refer to **85 unique puzzles**, not 255 repeated observations per arm.

| Arm | Verified / 85 | Mean complete wall ms | Mean CPU ms | p95 wall ms | Max wall ms |
|---|---:|---:|---:|---:|---:|
| Existing focused | 0 | 64.53 | 64.52 | 71.07 | 72.03 |
| Existing deductive | 1 | 57.45 | 57.43 | 65.28 | 76.58 |
| New watched DPLL | **85** | **40.54** | **40.55** | 57.77 | 91.72 |
| Native Glucose4 | 85 | 16.11 | 16.09 | 18.37 | 29.07 |
| Original Norvig Sudoku | 85 | 6.09 | 6.09 | 14.54 | 34.20 |

Quantiles and maxima use each puzzle's three-round mean. UNKNOWN costs remain in
all denominators. Whole-call cost includes CNF construction/validation, solver
setup, search, result construction/destruction and independent clause/grid checks.
Norvig directly consumes the puzzle and needs no CNF encoding. Imports, its static
units/peers tables, interpreter startup, input loading and JSON/fsync are outside
timing. These are imported-library task costs, not process startup latency.
Subphase observations remain in the raw rows and [summary](SUMMARY.json).

Budgets differ: existing arms use 2,048 flips at seed 17; DPLL permits 2,048 branch
attempts including retries; native requests 2,000 conflicts; Norvig is uncapped.
DPLL used median 33 / maximum 446 attempts and maximum 10,773 propagated assignments.
Native's observed maximum was 38 conflicts. This is not equal-compute comparison.
The bounded DPLL API supplies no UNSAT proof or hard wall-clock guarantee.

Both predeclared gates passed: at least 80/85 verified solutions and 20 more than
each existing SPECTRA control; mean complete wall cost no higher than focused.
The worst per-case round-mean DPLL latency exceeds focused's worst latency, despite
the lower mean and p95. No broad latency guarantee is implied.

## Memory is a separate measurement

Five declared evaluation puzzles (rows 11–15), fresh worker per arm/task; 25 RSS
rows and separate traced calls for Python arms. Values are medians over that subset.

| Arm | Peak RSS MiB | Incremental high-water RSS MiB | Python allocation peak bytes |
|---|---:|---:|---:|
| focused | 25.52 | 5.75 | 5,467,684 |
| deductive | 24.26 | 3.88 | 3,579,984 |
| dpll | 24.14 | 3.75 | 3,354,048 |
| glucose4 | 23.31 | 3.38 | Not measured |
| norvig | 19.99 | 0.13 | 52,291 |

Peak RSS includes common interpreter/native/domain imports. Incremental RSS is
the high-water difference, not live allocated bytes. Python peaks come from
separate tracemalloc runs and do not count native allocations. This small subset
does not establish worst-case memory; full baseline and maximum values are retained.

## Independent checks and reproduction

All claimed successes passed original-grid givens, row, column and box checks.
CNF arms additionally passed the original signed clauses and decoded-grid checks.
The retained verifier checks every inventory and binding and recomputes the report.
All 340 Python arm/task paths replayed exactly (focused, deductive, DPLL and Norvig).
Native answers are independently checked but native search is not replayed by the
dependency-free verifier. No verification step recreates historical latency.

```bash
python -I -S experiments/structured_search/analyse.py experiments/structured_search/evidence-20261005.zip
python -I -S experiments/structured_search/analyse.py experiments/structured_search/evidence-20261005.zip --replay
```

For a new timing run, create a separate checkout at the published frozen commit,
copy FROZEN.json into it, install python-sat 1.9.dev15, and use a new output path:

```bash
python -I experiments/structured_search/run.py run --out new-structured-run --freeze experiments/structured_search/FROZEN.json
```

This reruns consumed inputs and is not a fresh confirmation experiment. A new
solver/candidate requires another declared evaluation before selection.

[Raw evaluation archive](evidence-20261005.zip), [archive hashes](EVIDENCE.json),
[machine-readable summary](SUMMARY.json), [installed API guide](../../docs/DPLL_GUIDE.md).
No failed evaluation attempt occurred; all 1,275 rows completed. Development
timings before and after profiling remain distinguishable: attempt 01 was one
unpinned round, attempt 02 was three pinned rounds, so they are not a controlled
claim of a 41% optimization speedup.

## Limits and next question

This fills one concrete capability gap in the dependency-free toolkit on one
external structured family. It does not show industrial SAT strength, broad
reasoning, neural generalization or state-of-the-art performance. The native CDCL
and direct-domain controls dominate on this workload. One seed per stochastic
baseline leaves its wider seed sensitivity unmeasured. One host and five memory
cases limit efficiency claims. No historical model, sealed input, tolerance,
default or raw observation changed. A useful next investigation would test a new
declared structured family and the cost of generic search versus a direct domain
representation, without reusing these puzzles for confirmation.
