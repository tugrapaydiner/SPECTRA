# Frozen structured-search comparison — 2026-10-05

## Question and decision

Can bounded classical propagation/backtracking fill the observed failure of local
Boolean flips on public 9×9 Sudoku? This is an optional generic CNF solver, not
neural training, a new algorithm, general intelligence or a default replacement.
Candidate: watched-literal DPLL with a reversible iterative trail, shortest
unresolved positive clause first, then shortest unresolved arbitrary clause.
Literal order breaks ties. There is no clause learning or UNSAT certificate.

The development declaration predates execution. First ten top95 puzzles in file
order were consumed for development. Attempt 01 solved 10/10 versus 0/10 for each
existing SPECTRA control. Profiling motivated allocation/lookup reductions;
attempt 02 preserved all ten DPLL paths. Both attempts, source snapshots and
observations remain available. No changes follow evaluation exposure.

The capability admission threshold is at least 80/85 verified solutions and at
least 20 more solutions than **each** SPECTRA control. A separate efficiency
threshold requires mean complete wall cost no greater than the focused arm's.
Passing capability alone justifies retaining an explicit optional backend, not
changing defaults or claiming a speed improvement. Strong controls are reported
even when they dominate. This is one externally sourced task family; the cut is
by upstream order, not randomized or statistically representative.

## Inputs and controls

Peter Norvig's public `py/sudoku-top95.txt`, pytudes commit
`bfcdac7e2c74f44a9c2ff36217a8dd834ad6b73b`: rows 1–10 development, rows 11–95
evaluation. UPSTREAM.json binds exact bytes and the MIT license. The remaining
85 rows have not been run locally before publication of this protocol and source.
The entire corpus is publicly known; this is not a secret or author-independent
confirmation set. Historical SPECTRA sealed inputs are not accessed.

| Arm | Work limit | Role |
|---|---|---|
| focused | 2,048 flips; seed 17; existing defaults | Current local search |
| deductive | 2,048 flips; seed 17; existing defaults | Existing propagation/local search |
| dpll | 2,048 branch attempts, counting both polarities | Candidate |
| glucose4 | 2,000 conflicts requested; python-sat 1.9.dev15 | Native CDCL control |
| norvig | Unmodified upstream Python search; no search cap | Direct domain control |

These limits are different work units. No equal-compute claim is allowed.
Propagation adds input-dependent cost; branch limits are not time limits. DPLL
returns UNKNOWN even for exhausted unsatisfiable search, since it has no proof.
Native UNSAT is only reported, not certified. Public puzzles are expected SAT.

All SAT arms except Norvig use the same 729-variable CNF: 324 exactly-one groups,
each one positive clause plus 36 binary exclusions (11,988 clauses), plus givens.
No constraint deduplication or clue-specific preprocessing is shared in advance.
Check every claimed Boolean witness against original clauses; decode a grid and
check original givens, all rows, columns and boxes independently. Norvig consumes
the original puzzle directly and receives the same independent grid check.

## Measurement and retained analysis

Freeze runtime, task adapters, complete source/data inventory, runner and analysis
in a published Git commit before evaluation execution. Runner checks that HEAD
and every frozen source match, pins to the first allowed Linux CPU, records the
interpreter/platform/CPU/native module identities, and refuses reused output paths.

Three serial timing rounds across all 85 cases and five arms: 1,275 raw rows.
Rotate and reverse arm order by case/round. Count **85 unique tasks**, not timing
rounds as extra successes. One fixed seed per stochastic arm. Charge encoding,
CNF validation, solver setup/search/destruction, result construction, original-CNF
checking, decoding and grid checking to the full wall/process CPU timer. Retain
encoding and solve/result-construction subphases separately. Imports and Norvig's
static units/peers tables are outside timing; each call creates fresh task/search
state. Serialization, fsync, input file loading and interpreter launch are outside
timing. These are in-process imported-library costs, not end-to-end CLI latency.

Retain every UNKNOWN and include its cost. Report unique success counts and
mean/median/p95/max of per-case round-mean whole-call cost, plus mean CPU and phase
costs. Timing repetitions are not independent problem samples. Do not extrapolate
uncertainty to industrial SAT or other structured domains.

Memory: first five evaluation rows (11–15), all five arms. Fresh small-process
relay prevents inherited supervisor RSS floors. Common native and domain imports
precede each worker. Linux peak RSS includes that import baseline and is recorded
without tracing. Separate fresh runs measure Python allocation peaks for the
four Python arms; no tracemalloc number for native allocations. Report baseline,
peak and incremental high-water RSS separately; do not add allocation peaks to
RSS or imply this five-case subset covers worst-case memory.

Manifest binds all four required raw artifacts: LOCK.json, cases.jsonl,
rows.jsonl and memory.jsonl. Missing, duplicate or extra inventory is rejected.
Analysis checks complete case/arm/round inventories, source identities, solver
limits, phase timing, round determinism and independent solutions, then recomputes
the result. Optional replay reexecutes all four Python arms once per case (340
paths); it never attempts to recreate historical timing. Native output can be
checked without loading its optional runtime, but native search is not replayed.

Failures and partial files remain visible. If a harness fails after any evaluation
exposure, retain that attempt, explain the fault and do not tune the candidate.
No change to historical models, package defaults or old experiment evidence.
