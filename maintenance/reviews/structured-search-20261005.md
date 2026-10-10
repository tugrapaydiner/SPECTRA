# Structured capability checkpoint — 2026-10-05

Draft [PR #53](https://github.com/tugrapaydiner/SPECTRA/pull/53), isolated branch
`research/structured-search-20261005`, stacked on #52 at
`0bcb3793d26d25a05c84099301d60ad4477e79a6`. This is an algorithm/capability experiment,
not a maintenance-only pass. The recurring automation remains paused; no merge,
release, default replacement or historical model change is part of this work.

## Saved result

Add explicit opt-in `spectra.cnf.dpll.solve_dpll`: dependency-free watched-literal
propagation, iterative backtracking, bounded branch attempts and original-CNF
witness validation. The old evidence-bound modules and package exports remain
byte-identical. Strong controls include existing focused/deductive search,
Glucose4 and the unmodified upstream direct Sudoku solver.

Freeze commit `a0d7efaafd0573a4655d1e3965bbc1e7560f72ec`, tree
`64b4fe052008474944623da3735d64e88cbe37d0`, was published before evaluation execution.
The [24-file freeze](../../experiments/structured_search/FROZEN.json) binds the
candidate, imported runtime, encoder, runner, analysis, protocol and upstream data.
Public top95 rows 1–10 are consumed development; rows 11–95 are now consumed
evaluation. They must not be presented as a fresh confirmation split in future.

All 85 evaluation puzzles are solved by DPLL, versus 0 focused and 1 deductive.
Full wall means: 40.543 / 64.532 / 57.446 ms respectively. Native SAT and direct
Sudoku also solve all 85 and are faster: 16.107 / 6.093 ms. Candidate peak RSS
median is 24.145 MiB versus focused's 25.523 MiB on five cases; imports are included.
Both declared capability and efficiency gates pass. This is one public structured
family on one CPU host with unequal algorithmic budgets, not neural learning,
general reasoning or state-of-the-art SAT performance. See
[all controls and limitations](../../experiments/structured_search/RESULTS.md).

## Verification and reproducibility

Local Python 3.12.14 validation:

- 403 public tests pass (21 added tests included, not counted twice). Solver
  contracts include 512 exhaustive two-variable formulas, 800 brute-force checked
  noncanonical five-variable formulas, strict budgets and depth above 1,000.
- Full new evidence: 85 tasks, 1,275 timing rows, 25 memory observations; all
  original-clause/grid checks pass and all 340 Python arm/task paths replay.
- Previous focused evidence remains valid; 768 previous paths replay exactly.
- Built an sdist, then the actual wheel from that sdist. All 149 shipped source
  files byte-match; 11 installed-only checks pass outside the checkout without
  optional dependencies. Installed DPLL passes 512 truth-table checks; installed
  focused passes 512 formula checks plus two CLI commands.
- All 449 historical files remain byte-identical. Navigation, metadata and
  deterministic historical README figures pass. No numerical tolerance changed.

Commands, interpreter identity, test logs, wheel/sdist hashes and source identities
are retained in [validation](../../experiments/structured_search/VALIDATION.json).
CI additionally runs the compact-state benchmark and repeats public, installed and
both retained studies on Python 3.11 and 3.13. Exact final-head status is reported
in the PR. Full neural training/slow/native-model suites were not rerun; no such
code changed. Native SAT was a comparator, not a substitute for those suites.

## Development and failure retention

Attempt 01 measured the first ten development puzzles once. Profiling showed
clause-normalization and selection overhead. Attempt 02 reduced those allocations
and lookups, retained all ten decision paths and measured three pinned rounds.
Initial source, both raw runs, profile and test outputs remain in the development
archive. The two timing setups differ; their means are not a controlled speedup
estimate. A development-only harness preflight replayed all four Python arms and
tested memory inventory structure with explicitly synthetic fixture rows.

All 1,275 evaluation rows completed without an evaluation retry, drop or candidate
retuning. Upstream original bytes and MIT license are retained unchanged, including
their original whitespace. Build metadata warnings remain visible in the logs.
Hashes bind artifacts but do not prove independent authorship or scientific validity.

Next useful research: choose a new structured family, compare generic search with
its domain-specific representation and strong solvers, and freeze before fresh
evaluation. Neural training/generalization, multi-host behavior and production
scale remain open gaps. Do not repeat this corpus and count that as new capability.
