# CNF workload admission: establish headroom before learning another controller

## Decision and scope

SPECTRA's consumed Sudoku/maze development inventory is not a convincing flagship
for another selector: the existing restart schedule reaches its full measured
candidate coverage, and the maze validator already performs BFS. The next useful
question is whether a task has a meaningful solve/check gap before neural training.
This change supplies a **development admission experiment**, not a new learned
solver, task-capability improvement, novelty claim or independent confirmation.

`config/sat_workload_gate_v1.json` freezes 48 generated formulas: uniform and
planted 3-SAT, 64/128/256 variables, floor(4.2n) distinct clauses, eight examples
per distribution/size, two explicit classical solver configurations and three
rounds. Seeds and every input are fixed without solver-based acceptance filtering.
Planted assignments are retained for provenance but never passed to a solver
worker or used to construct a repair policy's features. Planting defines a
separate, potentially easy distribution. These are **two distributions of one
task**, not two unrelated task families.

The initial protocol/source were committed as `d47859a` before native solver measurement.
That is an auditable ordering, not independent external preregistration. Repeated
CI executions of the same inputs are runtime regressions, not new confirmation.
All observations, unknowns and unsuccessful gates must remain available.

## Measured outcome

The first native pilot completed all **288 observations**: **189 independently
verified SAT witnesses, 69 budget-limited unknowns and 30 UNSAT reports**. There
were no execution errors or process timeouts. **None of the six tiers passes** the
unchanged admission rule. Every qualifying-case count is zero. This does not
classify the unknown inputs as UNSAT or establish that all formulas are easy.

All raw measurement payloads are durably retained in
[`results/sat_workload/`](../results/sat_workload/README.md), not just a temporary
CI link. The byte-identical original pilot ZIP includes every input,
measurement and provenance payload. Input regeneration additionally checks the
source-frozen protocol and original hash. Both normal
CPU CI versions require offline replay. The original outer archive was separately
downloaded and checked against its GitHub digest, every member checksum and the
complete 543-file source tree. See the retained manifest for exact identifiers.

```bash
python scripts/verify_sat_admission.py
```

## Exact environment and repair identity

`data/cnf.py` checks only the supplied complete Boolean witness. It never invokes
a SAT solver or consults a reference assignment. The independent replay checker
uses signed-literal set intersections, separately from the direct predicate and
incremental repair implementation. Contradictory unit clauses and empty clauses
are handled without pretending that every input is satisfiable.

`eval/cnf_repair.py` maintains exact per-clause satisfied-literal counts. Let U(a)
be the number of unsatisfied clauses under assignment a. For a variable v,

```
U(a XOR e_v) = U(a) - make_v(a) + break_v(a).
```

The implementation aggregates all signed occurrences before changing a clause's
status. Thus duplicate literals and tautologies cannot corrupt make/break counts.
Its tests enumerate tiny assignments and compare every single flip with a full
independent rescan. Flips and make/break feature extraction have separate literal
work counters. Feature computation is not free, and a one-step improvement is
**not** an absolute state value or a guarantee of eventual solvability. This is a
classical local-search environment, not a learned algorithm or a novelty claim.

## Admission rule and accounting

Each proposed tier has eight inputs. A case qualifies only if **both solvers in
every round** return a complete independently valid SAT witness, the faster
solver's median full in-process solve takes at least 10 ms, and that cost is at
least 100 times the slower median witness check. A tier requires at least two
such cases and at least 25% of **all generated inputs**, including unknown,
timeout and UNSAT-reported inputs in the denominator.

Passing means `FEASIBILITY_LEAD_ONLY`. Failing means `NO_FEASIBLE_TIER` **under this
particular inventory, host and budget**; it does not prove that the distribution
is intrinsically easy or that a neural solver cannot help. These thresholds are a
proposed triage decision, not a statistical power calculation or hiring standard.

Each solver is given a 2,000-conflict request and a three-second supervising
process deadline. Conflicts have solver-specific semantics and are not comparable
FLOPs. The external deadline includes process startup/import/IPC; OS scheduling,
startup and termination can overrun it. Timeout durations are not fabricated as
successful or completed solves.

The in-process timer includes solver construction, clause insertion, budget
setup, solve, model extraction, work counters, destruction and the first independent
witness check. Cold process/import/IPC/cleanup time is separately recorded as
`supervised_wall_ns`; it is not silently included in or excluded from the same
number. Additional checker repetitions are separate measurements. Every declared
input/solver/round, hash, counter, witness and execution order must verify before
summarization. Errors invalidate the evidence; an honest negative gate does not.
No UNSAT claim is independently certified without a checked proof. Physical
energy remains null. CPU model, affinity, package version and source identity are
recorded; these measurements do not certify cross-machine performance.

### Observed budget overshoot

A 2,000-conflict request was **not a strict observed-work cap**. CaDiCaL
recorded up to **2,004 conflicts** (21 observations above the request); Glucose
recorded up to **37,797 conflicts** (33 observations above the request). The
independent replay exposes these diagnostics from the original counters. Do not
claim equal actual conflicts or at-most-2,000-conflict execution. Full measured
latency, raw counters and the separate process deadline remain the valid scopes.
The original protocol already labels this a solver-specific request; the result
is retained without rewriting counters, censoring overshoot or changing gates.

## Reproduction

```bash
python -m pip install -r requirements-sat-research.txt
python scripts/sat_workload_gate.py smoke
python scripts/sat_workload_gate.py run --out outputs/sat-admission-new
python scripts/sat_workload_gate.py verify --out outputs/sat-admission-new
python -m pytest tests/test_cnf_workload.py -q
```

The run command refuses an existing destination and preserves partial rows and
exceptions. Verification needs only the standard library; it accepts a directory
or a bounded seven-file ZIP. It checks source hashes, regenerates all inputs,
checks every SAT witness with a separate predicate and reproduces the report.
Hashes bind the records but do not magically authenticate external measurements.
The normal two-version CPU suite and the bounded native-solver workflow are both
required before merge. Synthetic timing fixtures are tests, never research data.

## Prior art and the next scientific gate

Recent [PTRM](https://arxiv.org/abs/2605.19943) and
[Energy-guided Recursive Model](https://arxiv.org/abs/2607.10128) already study
stochastic recursive trajectories and selection. Those are relevant baselines
for any later claim about the recursive mechanism, not comparable Sudoku4 scores.

Neural SAT initialization is established, e.g. [NLocalSAT (2020)](https://arxiv.org/abs/2001.09398).
Make/break feedback and focused local search are established; random formulas near
this density can be tractable for appropriate local-search heuristics, e.g.
[ASAT analysis](https://arxiv.org/abs/cond-mat/0601703). The two default CDCL solvers
here are **not** a tuned SAT-competition portfolio or adequate final SLS baselines.
[PySAT's documented solver API](https://pysathq.github.io/docs/html/api/solvers.html)
is used at pinned wrapper version 1.9.dev15; actual smoke tests check both backends.

Before promoting a tier to a learned-repair experiment, add strong focused local
search and solver-tuning controls, independent duplicate/isomorphism exclusions,
and genuinely untouched test families. Do not filter the test set to the examples
that favor the learned method. Compare direct prediction, ordinary recurrence,
random restarts, deterministic make/break repair, learned input-only control and
learned residual control with the same complete-solve accounting. The candidate
coverage ceiling must improve before an allocation-only hypothesis is attractive.

An impactful result would need a replicated full quality/cost frontier improvement
against strong controls, a mechanism-specific ablation, a relevant deployment
regime and outside reproduction. Adding this instrument alone leaves the impact
assessment near **35/100**, not 100/100 and not an OpenAI level equivalence.
