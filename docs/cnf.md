# CNF tools

`spectra.cnf` is a dependency-free toolkit for Boolean CNF: parsing, witness
checking, seeded local search, deductive solving and DPLL. Every `SAT_VERIFIED`
result has been checked against the original clauses. These solvers do not emit
UNSAT proofs; a failed search returns `UNKNOWN`. For proofs use
[`spectra.sat`](sat.md).

## Formulas

```python
from io import StringIO
from spectra.cnf import CNF, read_dimacs

problem = read_dimacs(StringIO("p cnf 2 2\n1 2 0\n-1 2 0\n"))
assert problem == CNF(2, ((1, 2), (-1, 2)))
assert problem.satisfied((False, True))
```

`CNF(nvars, clauses)` uses signed one-based literals; witnesses are tuples of
exactly `nvars` Python `bool` values. An empty formula is satisfied and an empty
clause makes a formula unsatisfiable. `read_dimacs` requires one ASCII `p cnf`
header, exact counts, in-range literals and zero-terminated clauses; variable,
clause and literal limits are keyword arguments. `problem.sha256()` identifies a
formula exactly; `violated(witness)` lists unsatisfied clause indices.

## Local search

```python
from spectra.cnf import PreparedCNF, solve, solve_focused, solve_indexed

reference = solve(problem, seed=7, max_flips=128)
indexed = solve_indexed(problem, seed=7, max_flips=128)
repeated = PreparedCNF(problem).solve(seed=7, max_flips=128)
assert reference.witness == indexed.witness == repeated.witness

focused = solve_focused(problem, seed=7, max_flips=2048)
```

| Backend | Description |
|---|---|
| `solve` | Compact incremental make/break state; the reference trajectory |
| `solve_indexed` | Ranked, reusable clause indexes; the same trajectory as `solve`, faster on large formulas |
| `PreparedCNF(p).solve` | Index built once and shared by many calls; each call gets independent search state |
| `solve_focused` | Dense pool of unsatisfied clauses with a break-count / flip-age policy (`policy="novelty_break"`) and optional restarts |

Seeds are unsigned 64-bit integers and `max_flips` bounds variable flips (across
restarts for `solve_focused`). `result.record()` returns a JSON-stable dictionary
with status, witness, unsatisfied clauses, flips, queries, a path hash and elapsed
time. On `UNKNOWN` the witness is only the final candidate.

## Deductive solving

```python
from spectra.cnf import CNF, solve_deductive

result = solve_deductive(CNF(3, ((1,), (-1, 2), (-2, 3))), seed=7, max_flips=0)
assert result.status == "SAT_VERIFIED" and result.strategy == "propagation"
```

Unit propagation first, then an implication-graph (2-SAT) solver for binary
residuals, then indexed local search for anything left. `max_flips` bounds only
the residual search. Records add `strategy`, `propagated_variables` and
`implication_edges`. Contradictions return `UNKNOWN`.

## DPLL

```python
from spectra.cnf.dpll import solve_dpll

answer = solve_dpll(CNF(3, ((1, 2), (-1, 3), (-2, 3), (3,))), max_decisions=2048)
print(answer.status, answer.reason, answer.decisions)
```

Watched-literal unit propagation with an iterative trail (no recursion limit) and
a branching rule that prefers the smallest unresolved all-positive clause, which
exposes the remaining domain in one-hot encodings. `reason` is `satisfied`,
`budget` or `exhausted`. An exhausted search is still `UNKNOWN`: no clause learning
and no proof.

## Repair state

`CompactCNFRepairState` (and `spectra.cnf.cached.CachedCNFRepairState`) maintain
exact make/break counts under single-variable flips, with duplicate literals,
tautologies, empty clauses and unused variables handled exactly.
`spectra.cnf.repair.CNFRepairState` is the plain reference implementation used to
test them.

## Command line

```bash
spectra cnf solve formula.cnf --backend focused --seed 7 --max-flips 2048 --out answer.json
spectra cnf check formula.cnf answer.json
```

`--backend` is one of `compact` (default), `indexed`, `deductive` and `focused`.
See [the CLI reference](cli.md) for exit codes and input limits.
