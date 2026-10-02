# Optional deductive CNF solving

The unreleased `deductive` backend uses classical unit propagation and an iterative
implication-graph solver for binary residuals. Remaining general constraints use
the existing indexed local search. It adds no runtime dependencies or learned
parameters. The historical `compact` default is unchanged.

```python
from spectra.cnf import CNF, solve_deductive

problem = CNF(3, ((1,), (-1, 2), (-2, 3)))
result = solve_deductive(problem, seed=7, max_flips=0)
assert result.status == "SAT_VERIFIED"
assert problem.satisfied(result.witness)
print(result.strategy)  # propagation
```

```bash
python -m spectra cnf solve examples/tiny.cnf --backend deductive --max-flips 2048 --out deductive.json
python -m spectra cnf check examples/tiny.cnf deductive.json
```

`max_flips` limits residual local search. Propagation, clause normalization,
implication-graph construction and final checking cost additional time and memory,
even with a zero flip budget. Propagation/SCC passes are linear in their
inventories; normalization sorts each clause, so total arbitrary-width
preprocessing is not strictly linear. There is no hard elapsed-time/RAM limit.

SAT witnesses are checked against original clauses. Contradictions return
`UNKNOWN`, since this API emits no checkable UNSAT certificate. On UNKNOWN, the
witness is only a candidate. Result records add `strategy`, `propagated_variables`
and `implication_edges`; path hashes for a reduced formula do not identify the
historical full-formula trajectory. General formulas routed directly to indexed
search retain its seeded trajectory. Output files must be new.

The [fresh synthetic comparison](../experiments/deductive_search/RESULTS.md)
solves 124/160 formula/seed cases versus indexed search's 97/160, with 31.4% lower
mean full solve time. Gains concentrate in forced/binary constraints; general
3-SAT quality is unchanged, its mean overhead reaches 5.1%, and overall p95 is
2.3% higher. Binary implication graphs sometimes use more memory. Native Glucose4
is faster on those structured cells. These are workload-specific classical gains,
not evidence of better learned reasoning or a universal SAT solver.
