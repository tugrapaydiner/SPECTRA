# Current status

This page separates released implementation results, experimental work, and
historical scientific outcomes. A systems optimization does not turn a failed
scientific gate into a pass.

## Current efficiency upgrade: 0.7.1

An opt-in ranked/prepared search implementation removes repeated residual sorting
without changing seeded trajectories. The frozen local primary full-call ratio
is 0.237505 [0.216128, 0.281891]; 1152 answers and 384 backend paths verify exactly.
Small-case regressions and increased cold allocation remain visible. Large cases
return UNKNOWN. This is not a learned or external-solver superiority result.

An output-only CPU API preserves final outputs without retaining the diagnostic
trajectory. Its storage measurements are returned tensor storage, not peak RAM.
See [the efficiency guide](EFFICIENCY_GUIDE.md) for evidence, scope and commands.
The 0.7.1 GitHub release was published from `9fd6682` on September 12, 2026.
Its release receipt reports 1,157 fast tests, with 16 slow tests excluded. These
are historical release counts, not a dynamically updated test badge. All original
research dispositions below are unchanged.

## Experimental integrated runtime

The opt-in [integrated runtime](RUNTIME_GUIDE.md) combines owned validated weights
with final-only execution and adds a separate ordered four-vector blocked kernel.
Historical operators remain available and unchanged. Current branch validation
and measurement scope are documented in that guide, not folded into the older
0.7.1 search result.

## Public release surface

Version 0.7.0 adds a compiler-free base installation, one `spectra` command, strict
DIMACS interchange, exact compact CNF state, capped search and witness checking.
Research dependencies are explicit extras. Native source files ship in wheels and
source distributions; ordinary installation no longer silently attempts a native
build depending on whether PyTorch happens to be installed.

A useful unmerged branch contribution is recovered as `spectra.evidence`:
`common/evidence_freeze.py` from commit
`095c1ed65ffa3beabc55e42052c2f3b038a3150e` is copied byte-for-byte, with its tests
ported only to the public import path. Its unmerged search framework is not
promoted. Hash checks do not certify the correctness of a scientific claim.

## Compact CNF state

The previous cache represents every clause's true variables as a bitset indexed
by the largest global variable ID. The new optional backend stores a true-literal
count and XOR of true variable IDs. Only a count of one permits interpreting the
XOR as a sole variable; a nonempty support can otherwise have XOR zero. Single
and simultaneous flips preserve exact original-clause make/break semantics.

The [protocol](COMPACT_CNF_PROTOCOL.md) was committed as
`82bb2b8b3777354657c253d52e5dcf2022b0a769` before measurement. Twelve unfiltered
formulas, two search seeds, three timing rounds and a 128-flip cap produce 144
timing observations and 24 paired paths. All corresponding paths and outcomes
agree. A separate constructor pass records 24 Python allocation observations.

Local Python 3.13 measurement; old bitset / compact values:

| Variables:family | Mean full execution, ms | Time ratio | Peak Python allocations, MB | Peak ratio |
|---|---:|---:|---:|---:|
| 16384:planted | 347.136 / 283.637 | 0.8171 | 99.589 / 25.412 | 0.2552 |
| 16384:uniform | 364.969 / 289.893 | 0.7943 | 99.467 / 25.411 | 0.2555 |
| 4096:planted | 65.347 / 68.405 | 1.0468 | 9.814 / 6.281 | 0.6400 |
| 4096:uniform | 61.386 / 56.098 | 0.9139 | 9.855 / 6.283 | 0.6375 |
| 512:planted | 7.443 / 7.198 | 0.9670 | 0.579 / 0.648 | 1.1184 |
| 512:uniform | 7.278 / 7.265 | 0.9982 | 0.577 / 0.647 | 1.1204 |

The pooled peak allocation ratio at 4,096 and 16,384 variables is
**0.28980**, approximately **71.0% less**. At 16,384 variables the reduction is
approximately 74.5%; at 512 variables peak allocation is about 12% higher.
Every declared cell's mean-time point ratio is at most 1.10, and the allocation
gate passes. Some descriptive timing intervals cross 1; the protocol's timing
gate is a point-ratio condition, not a claim of significant speedup in every cell.

This measures `tracemalloc` Python construction allocations, not RSS, native
memory, inference memory or energy. Timings include initialization, weight-table
construction, state construction, search, original-formula checking and release.
Parsing, imports and serialization are outside. The cap is flips, not a hard
wall-clock deadline. Repetitions are not independent new formulas, and only two
formulas per family/size cell give limited generalization evidence. No new learned
quality, SAT superiority or universal memory improvement follows.

The initial local invocation was interrupted by a tool timeout after all 144
timing rows and 11 allocation records had been written. Only the 13 missing
allocation records were resumed; no existing observation was discarded or retimed.
The interruption/resume log is retained. The complete verifier checks all 144
answers and replays 48 engine-specific non-timing trajectories. CI runs reproduce
the fixed workload; their results belong to the PR receipts, not fabricated here.

## Research disposition

PR23's learned selector remains a completed negative result, not the default.
Its CI development witness rate was 42.97%, below the 47.22% greedy repair control.
The earlier Sudoku/maze results, symmetry overlap and much stronger classical
comparators remain described in [the research review](RESEARCH_REVIEW_20260911.md).

The next capability experiment is the neighborhood-repair protocol on
`research/neighborhood-repair`. Its neighborhood component is not implemented by
this cleanup. Establish cross-fitted useful repair headroom before training a
larger selector; then compare full execution cost, classical baselines and
proposal/selector factorial controls on untouched cases. Another packaging or
cache improvement is not a substitute for that learned result.

## Repository cleanup

The starting inventory had 32 branches: 25 non-main tips already in main and six
unmerged tips. Five completed/superseded unmerged branches are retained through
exact archive tags; the neighborhood-repair branch is kept active. The cleanup
branch is also archived after merge. The post-merge workflow checks expected tips
and archives/deletes the selected refs atomically; an advanced branch stops the
operation. Actual completion is recorded in the PR and workflow receipt.

Thirty-two historical documents and nineteen one-off workflows are removed from
the working tree but remain at the immutable base commit, with original blob and
SHA256 receipts. Three active workflows are renamed. Full fast regression still
runs on Python 3.11 and 3.13; redundant full-suite invocations in native/task jobs
are removed, not their unique native and task checks. M01's energy API and small
optimizer smoke are moved into the canonical CPU workflow. Retained checkpoint,
fixed-pool, restart, controller-refit and SAT-admission audits remain active.

## Historical 0.7.0 / PR25 local acceptance receipt

The final local fast suite has 999 passes, 16 historical slow tests deselected,
and two warnings. This includes 108 public contracts and 16 new compact-evidence
contracts; they are not additional to the 999 count. The actual wheel built from
a source distribution passes eight outside-checkout commands in a clean venv
with no PyTorch/NumPy. A separate research-environment check compiles and executes
all three JIT extensions from the installed wheel in a fresh build directory.

That first native-installation check failed because a nested venv did not see
PyTorch installed in its parent venv. The corrected check explicitly hands off
the declared research site while requiring all SPECTRA modules to resolve inside
the new installation. The original failure and the corrected success are retained;
this does not change the wheel or any native arithmetic. Final-head CI and branch
retirement completion are recorded separately in PR25 and its artifacts.
