# Experimental focused CNF search

`solve_focused` is an opt-in classical local-search candidate. It uses a dense
swap-delete pool of unsatisfied clauses and a break-count/last-flip-age move
policy. It does not train a model, prove UNSAT, or replace the historical default.
Its quality, time and memory advantage has **not** been established by a retained
fresh evaluation. See the [recovery record](../experiments/focused_search/RECOVERY.md).

```python
from spectra.cnf import CNF, solve_focused

problem = CNF(2, ((1, 2), (-1, 2)))
result = solve_focused(problem, seed=7, max_flips=2048)
if result.status == "SAT_VERIFIED":
    assert problem.satisfied(result.witness)
```

```bash
spectra cnf solve examples/tiny.cnf --backend focused --seed 7 --max-flips 2048 --out focused.json
spectra cnf check examples/tiny.cnf focused.json
```

The Python defaults are `policy="novelty_break"` and `restart_interval=0`.
The CLI uses those defaults. `max_flips` bounds total variable flips across all
restarts; a positive restart interval starts a new random assignment after that
many flips, without replenishing the budget. Seeds are unsigned 64-bit integers;
flip and restart settings are nonnegative integers, excluding booleans.

The selected policy prefers low break counts, breaking ties by oldest last-flip
age and variable index. If the best variable has nonzero break count and is the
youngest in the clause, a fair coin may select the second-best variable. This is
a break-only age heuristic, not a claim of equivalence to the full Novelty
algorithm or of algorithmic novelty. Explicit research controls remain available
as `poly`, `freebie`, `minbreak`, `sharp` and `anti_reverse`; their retention does
not imply they improve on indexed search.

Dense-pool ordering changes seeded trajectories relative to indexed search.
SAT witnesses are checked against the original clauses before returning.
`UNKNOWN` means the returned candidate does not satisfy the formula; it is not an
UNSAT proof. An empty clause stops search immediately. Tautologies and duplicate
literals retain the original CNF semantics.

`FocusedResult.record()` adds `algorithm="dense_focused_search"`, `policy`,
`restarts` and `restart_interval` to the usual solve record, with `learned=false`.
`elapsed_ns` covers per-call index/state preparation, search, original-clause
verification and state disposal. It excludes import and input parsing; there is
no hard time or memory cap. Flip counts are not comparable to native conflict
budgets. No new neural, general-intelligence or benchmark speed claim is made.
