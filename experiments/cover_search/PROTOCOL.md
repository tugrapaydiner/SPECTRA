# Cover/exclusion execution — prospective protocol

Status: freeze before generating the new graph inventory or running the complete matrix.
Base: PR53, eabcc9c5a8ebe0beee364da3ecd1a68bedeb7253. Default solver, fitted models,
all baseline source files and historical observations remain unchanged.

## Central question
Can direct native execution of covering and mutual-exclusion groups materially
reduce complete verified CPU problem-solving cost, rather than merely accelerate
a search kernel or defeat a slow internal reference?

This is classical constraint execution, not a neural-learning or algorithmic-first
claim. Algorithm X, dancing links, bitsets, minimum-remaining-values branching and
incremental cardinality counts are prior techniques. The implementation maintains
remaining-group counts as available choices are removed; the dense control
recomputes exactly the same counts and must follow the same search path.

## Development boundary and stop rule
Development used the previously exposed first ten Norvig top95 puzzles and twelve
synthetic graphs with n=18/36/72, planted/uniform, seeds 100+n+j for j=0,1. Preserve
the Python prototype, first native CNF prototype, direct-group pilot and incremental
pilot, including slow measurements and losses to MiniCard. Three implementation
revisions are sufficient: no tuning after this freeze. Code correctness failures
invalidate the experiment and require a new prospective study, not selected reruns.

All 95 public puzzles were already consumed in earlier SPECTRA work. Rows 11–95
are compatibility only, NOT a fresh holdout, training-generalization result or
independent replication. This study does not train or tune a model.

## Inputs, controls and scope
Use exactly the 85 public compatibility puzzles plus 144 newly generated graphs:
n=18/36/72, planted or uniform, 24 cases per cell, three colours. The source-fixed
SplitMix64 master seed is 0x9AF490C711DE7301. Edges follow the documented generator,
with no solver filtering, rejection of hard cases or generator replacement.
The planted and uniform families differ and are reported separately. Graphs are
synthetic, not measured deployments or realistic application adoption.

Check the graph colour-refinement invariant against every development graph and
all other prospective graphs. Different signatures exclude graph isomorphism,
assuming collision-resistant hashing; matching signatures cause explicit refusal,
not regeneration or an assertion of distinctness. This invariant is not a complete
isomorphism canonicalizer. There are no fitting/validation/test model roles to leak.

Arms: direct incremental cover; direct dense-count ablation; the identical native
incremental search behind expanded original CNF; existing Python watched DPLL;
unmodified native MiniCard (native at-most-one constraints, no pairwise handicap);
unmodified native Glucose4.2, CaDiCaL300 and Kissat404 (pairwise CNF), plus GlueCard4
with native cardinality constraints; original preserved Norvig
solver on Sudoku only. Pin python-sat 1.9.dev15. No solver retuning on fresh outputs.
No comparison against a compiled DLX, CP-SAT or every SAT solver is claimed.

## Complete cost and finite resource boundaries
Time raw task record → freshly encoded/validated constraints → index construction
→ search → fresh witness → independent original constraint and task checks →
index/solver disposal and diagnostic result construction. Every failed attempt is
retained. Imports, process creation, filesystem reads, compilation, JSON output and
original dataset acquisition are outside primary timing and disclosed separately.
Do not call UNKNOWN return cost time-to-solution. All-verified-SAT panels can be
interpreted as complete-solve latency; mixed panels are complete-attempt cost.

One pinned Linux CPU; one numerical thread; four-GiB virtual-address-space guard.
Both custom and DPLL node/decision guardrails are 1,000,000, not equivalent units.
All arms have the same one-second supervisor wall deadline after worker readiness;
terminated attempts are charged their observed parent elapsed time, not silently
dropped or clamped. Implementation has no implicit hard real-time guarantee.
Native index payload and logical simultaneous search state payload are each capped
at 64 MiB. These are not total RSS bounds; original inputs, temporary buffers,
allocator capacities and interpreter overhead remain real costs.

Run three shuffled rounds of the complete matrix. One fresh isolated worker and
fresh solver/index per job: no retained CDCL learning between tasks, no shared
prepared-index assumption benefit to either side. Load baseline libraries before
primary timing; use separate fresh-exec resource/cold-start probes for five public
puzzles and the first two examples in every graph stratum, all applicable arms.
Those probes include ordinary imports and record own-process VmHWM/VmRSS; no
inherited supervisor high-water mark. They have a separate 15-second outer cap;
resource probe failures stay explicit. A reused service, real-time tail guarantee,
energy efficiency and maximum concurrent-worker RSS are not established.

## Decisions and statistics
Primary external gate: on all 72 prospectively generated PLANTED graphs, the
candidate and MiniCard both return independently verified SAT on every case and
round; the upper descriptive paired 95% case-bootstrap interval on the mean-cost
ratio must be <=0.50, and the upper p95-cost ratio interval <=1.10.

Secondary internal engineering gate: the same conditions against existing DPLL on
all 85 exposed public puzzles. This can justify optional implementation retention,
not scientific promotion. Other comparisons and uniform-graph stress are explicitly
secondary/descriptive. Neither passing the internal gate nor a different external
comparison replaces failure of the primary gate. No claim of general novelty,
independent external replication, universal superiority or field adoption follows.

Bootstrap 2,000 times with fixed analysis seed 519018, resampling independent task
IDs within their fixed strata and retaining all three rounds and both arms together.
Rounds are not independent problems. Report full-call mean, median, p95, maximum,
verified case counts, all statuses, deadline overruns, scaling strata and failure
reasons. These are single-host descriptive intervals, not a preregistered population
sampling theorem or production service SLO. No multiple-comparison discovery claim.

MiniCard/Glucose/CaDiCaL UNSAT reports are not proof-checked. The new engine returns
UNKNOWN on exhaustive failure, node exhaustion or state-budget exhaustion, never
an unverified UNSAT certificate. Every SAT output must pass the original task checker.
A valid solution ends search immediately and cannot be displaced by later branches.

## Reproduction
Install the pinned optional research dependency and use a C++17 compiler. From the
repository, with a new output path:

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -I -S experiments/cover_search/run.py --out /tmp/cover-study
    python -I -S experiments/cover_search/analyse.py /tmp/cover-study --out /tmp/cover-summary.json

FREEZE.json binds source files and the complete baseline SHA256 manifest. Raw task
records, schedule, every timing, fresh-process resource rows, native build command,
compiler version, library hash and host metadata accompany the run. The analyser
reconstructs task inventory, checks every returned witness and complete schedules,
recomputes summaries and refuses missing or changed data. Hashes do not authenticate
clocks or protect against wholesale replacement of evidence and its trust anchor.

A second-host automation may run the unchanged complete protocol after the local
run. It is automated cross-host reproduction, not an outside researcher's replication.
No parameter/default/release promotion is automatic. Keep all failed gates visible.
