# Frozen focused-search evaluation — 2026-10-05

Question: does the recovered break/age policy improve verified SAT successes per
complete CPU solve cost over indexed search, and is the change due to dense-pool
bookkeeping or the variable-selection policy? This is classical search. No neural
model is trained and no general-intelligence or algorithmic-novelty claim is made.

## Candidate and exposure

Use exactly the recovered `spectra/cnf/focused.py` at PR #51 commit
`54b91553e5d3be356db7011ff622fb16e495256f`: default `novelty_break`, no restarts.
Do not tune on this study. The old lost development seeds and unrun earlier
evaluation declaration remain in `experiments/focused_search/RECOVERY.md`.
This new protocol supersedes neither historical evidence nor its missing data.
Publish all runtime, generator, runner, analysis and protocol sources before
generating these evaluation inputs. FROZEN.json binds that commit/tree and the
complete declared source inventory. Unit/smoke fixtures use different tiny inputs.

## Inputs and controls

96 fresh synthetic formulas: planted and uniform 3-SAT at density 4.2; sizes
128, 512 and 1,024; 16 formulas per family/size cell. Generation seeds are
`202610055000` through `202610055095`, in family/size/index order. No filtering or
replacement based on any solver result. Planted inputs are not a hardness claim;
uniform inputs may be SAT, UNSAT or unresolved. No historical sealed set is opened.

Every formula uses search seeds 17 and 73, a 2,048-flip total budget, and three
timing rounds. Arms: indexed reference; dense `poly` bookkeeping control;
dense `minbreak` policy control; selected `novelty_break`; native Glucose4 through
python-sat 1.9.dev15 requesting 2,000 conflicts. Flips and conflicts are unequal
work units. Record actual native counters/overshoots; do not infer native
superiority or inferiority at equal time from those caps. Glucose4 has no exposed
search seed here: its two seed slots are repeated deterministic controls, not
independent native trials. Native UNSAT answers are reported without proof checking.

Rotate/reverse arm order deterministically. Pin one available CPU; import all
solver modules before measurement. Time the whole call with monotonic wall and
process-CPU clocks: input-record conversion, index/native setup, solve, witness
extraction, result construction, native destruction and an independent check
against the original signed literals. Exclude module imports, disk reads, JSON
serialization and durable logging, which are common harness costs. Keep GC enabled.
There is no hard solve-time deadline. Logging fsync occurs outside measured calls.

## Memory and statistics

Separate untimed memory pass on the first two declared formulas in each of the
six cells (12 formulas, no outcome-based selection), seed 17, for indexed,
novelty_break and Glucose4. Report full-process peak RSS including input and a
common interpreter/native-import baseline. Use a fresh small relay then a fresh
worker to avoid inheriting the timing supervisor's memory floor. For the Python
arms, a separate fresh worker records tracemalloc peak during the complete call.
Native allocations are not measured by tracemalloc. Memory is descriptive, not a
96-formula estimate, model size, energy or speed measurement.

Verify all 2,880 timing rows, 36 memory rows and 96 formulas. Check exact inventory,
hashes, original-clause Boolean SAT witnesses, local residuals/budgets and
non-timing determinism across rounds. Analysis runs without PySAT/NumPy.
Report per-cell and pooled outcomes (192 formula/seed pairs), mean/p95 complete
wall and CPU time; paired wins/losses; per-formula coverage; common observed
5/20/100-ms outcome counts. Those thresholds are retrospective observations, not
enforced deadlines. UNKNOWN cases remain in every cost denominator.

Predeclared optional-backend gate: at least five additional SAT-verified pairs
over indexed, pooled mean complete wall time no greater than indexed, no cell
losing more than two SAT pairs, and no invalid SAT witness. Report all conditions;
a failed gate stays failed. This does not authorize a default change. Use 2,000
paired, family/size-stratified formula bootstrap resamples (seed 20261005017) for
descriptive 95% intervals of solve-rate difference and pooled mean time ratio.
Seeds and rounds stay clustered within formulas; they do not increase independent
sample size. The gate uses the stated point conditions, not a post-hoc CI rule.

## Retention and limits

Durably flush every raw row. Refuse existing output paths. A missing/duplicate row
or source binding rejects a completed report. Retain partial files and FAILURE.json
if execution fails; do not silently overwrite or rerun successful observations.
Publish a checksummed archive, machine-readable summary and concise case study,
including unfavorable arms and honest limits. This is one host and synthetic
distribution; it cannot establish real-world generalization or job readiness.
