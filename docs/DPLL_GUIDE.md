# Optional watched-literal CNF search

This unreleased backend adds unit propagation and reversible search decisions to
SPECTRA's dependency-free CNF tools. It is classical DPLL, with no learned model,
clause learning or proof-producing UNSAT interface. It is explicit opt-in:

```python
from spectra.cnf import CNF
from spectra.cnf.dpll import solve_dpll

problem = CNF(3, ((1, 2), (-1, 3), (-2, 3), (3,)))
answer = solve_dpll(problem, max_decisions=2048)
print(answer.status, answer.reason, answer.decisions)
if answer.status == "SAT_VERIFIED":
    assert problem.satisfied(answer.witness)
```

Import from `spectra.cnf.dpll`; existing top-level exports and CLI defaults retain
their previous behavior. `max_decisions` is a nonnegative integer; both the first
branch and its opposite consume one attempt. Root propagation still runs at zero.
Propagation and input processing add unbounded input-dependent work, so this is
not a wall-clock or instruction limit.

`SAT_VERIFIED` means the complete Boolean witness passes the original clauses.
`UNKNOWN` is returned after a budget stop or exhausted unsuccessful search.
`reason` distinguishes `budget`, `exhausted` and `satisfied`; an exhausted search
does not emit an independently checkable UNSAT proof. The returned residual lists
original clause indices. `record()` is JSON-stable and includes counters, a
deterministic assignment/backtrack trace hash, algorithm identity and elapsed time.
Trace hashes support same-source replay, not authenticity or a proof certificate.

Two watched literals reduce propagation rescans. An iterative trail makes decision
depth independent of Python's recursion limit. Branch selection favors the
smallest unresolved all-positive clause, then falls back to all clauses. This
syntactic heuristic exposes remaining domains in one-hot encodings; no task type
is inferred. Binary setup avoids dictionary/set allocations. Arbitrary-clause
normalization preserves input literal order and original-clause validation.

See the [public Sudoku comparison](../experiments/structured_search/RESULTS.md)
for solve quality, complete CPU/wall cost, native and domain controls, and memory.
The branch budget, flip budget and native conflict budget are different work
units. A measured gain on this corpus does not establish broad SAT performance
or improved neural reasoning. Native CDCL and a direct domain solver remain
important alternatives. Historical search implementations and models are intact.

From the full checkout, verify retained observations without optional packages:

```bash
python -I -S experiments/structured_search/analyse.py experiments/structured_search/evidence-20261005.zip
python -I -S experiments/structured_search/analyse.py experiments/structured_search/evidence-20261005.zip --replay
```

Replay checks all four Python arms against retained answers; it does not reproduce
historical latency. Native output is independently checked, not reexecuted by this
command. For a fresh benchmark, use the published frozen commit and pinned
python-sat 1.9.dev15, then run the command in the study's result document with a new
output directory. Evaluation inputs are now consumed and cannot become a new
untouched confirmation split through another run.
