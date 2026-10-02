# Deductive CNF search development experiment — 2026-10-02

Goal: improve actual checked solving capability per CPU cost over the current
indexed local-search API. This is a classical, dependency-free solver extension,
not a learned-model or general-intelligence claim. No historical confirmation
inputs, model weights or numerical tolerances are changed.

Candidate: canonicalize clauses, propagate forced literals with occurrence
counts, solve any remaining binary-clause formula by strongly connected components
of its implication graph, otherwise use the existing indexed search on the
remaining formula. Preserve the original formula for final witness validation.
Contradictions return UNKNOWN until there is a separate proof-bearing UNSAT API.
`max_flips` caps only residual local search; deduction/graph work is additional,
linear in the literal inventory. No hard elapsed-time cap is claimed.

Development: four families, 128 variables, six formulas per family; generator
seed base 202610021000. Evaluation: the same four families, sizes 128 and 512,
eight formulas per cell; generator seed base 202610022000. Generate every declared
formula, with no solver-based filtering or replacement. Identity/order hashes and
all raw outcomes are retained. All these inputs are fresh synthetic development
and evaluation evidence, not an independent real-world confirmation set.

Families:
- Planted random 2-SAT, four clauses per variable. Demonstrates implication
  reasoning, with a known witness withheld from every solver.
- Forced mixed CNF: one unit, a shuffled implication chain forcing the planted
  assignment, and three planted 3-clauses per variable. Tests deductions through
  mixed arities; this is intentionally deduction-friendly synthetic input.
- Planted random 3-SAT, 4.2 clauses per variable.
- Uniform random 3-SAT, 4.2 clauses per variable; satisfiability is not assumed.

Controls: existing `solve_indexed` and actual Glucose4 through python-sat
1.9.dev15. Candidate/indexed use seeds 17 and 73 and a 2048-flip cap. Glucose4
requests 2000 conflicts; it is a mature native reference, not an equal-work arm.
Report observed counters and any budget overshoot. The same cases go to every
arm. Repeat each timed call three times in fixed, alternated arm order. Verify
SAT witnesses against original signed literals independently of solver caches.

Primary reports: SAT-verified count per distinct formula/seed, unknowns and any
UNSAT reports; complete-call mean/p95 CPU wall time by family and size; paired
candidate/indexed ratios. Complete time includes preparation, deduction, search,
result construction and original-formula witness checks, excluding parsing/imports.
Native setup/extraction/checking/destruction are included. Repeat rounds are not
independent inputs. Also report results at common observed 5/20/100-ms deadlines
(post-hoc censoring of complete outcomes, not an enforced online time bound).

Memory: separate untimed tracemalloc peak per case/arm for Python paths, plus
isolated-process peak RSS including interpreter/native imports. Neither is model
storage or energy. Do not label Python allocations as total RAM. Time calls are
not run under tracemalloc. Input objects exist before timers and tracemalloc.

Advance only if every returned SAT witness is valid, the candidate solves every
2-SAT and forced case, its added success or lower full cost survives the stronger
indexed control, and general-3-SAT outcomes do not regress at the same flip budget.
Report setup overhead, general-family regressions and the native reference even
when unfavorable. No universal speed or novelty claim. If a candidate needs
revision after evaluation, retain that evaluation as consumed and declare a new
fresh evaluation seed range before running it.
