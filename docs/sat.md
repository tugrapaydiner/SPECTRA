# General CNF solving

`spectra.sat` solves arbitrary CNF formulas and never returns an unchecked answer.
SAT models are checked against the original clauses; UNSAT answers carry a DRUP
proof that is replayed by a separate watched-clause checker before the result is
returned. Proofs are expressed over the original variable numbering and can also be
written as standard DRAT for an external checker.

| Module | Role |
|---|---|
| `spectra.sat.native_cdcl` | Native CDCL solver: watched literals, first-UIP learning, phase saving, restarts, clause-database reduction, DRUP output |
| `spectra.sat.binary_core` | Proof-replayable simplifier for unit and binary structure (tautology/duplicate removal, implication SCCs, equivalent-literal substitution, unit propagation) |
| `spectra.sat.native_binary_core` | Native implementation of the binary core for large inputs |
| `spectra.sat.admission` | Strict, solver-free DIMACS admission with identity hashes and structural statistics |
| `spectra.sat.drat` | DIMACS/DRAT writers and an adapter for the DRAT-trim checker |

## Solving

```python
from itertools import product
from spectra.sat.native_cdcl import (NativeCDCLRuntime, build_native_cdcl,
                                     solve_cnf_native, verify_direct_native_result,
                                     write_drup)

runtime = NativeCDCLRuntime(build_native_cdcl("build/sat"))

# All eight sign patterns over three variables: unsatisfiable.
clauses = [tuple(v if s else -v for v, s in zip((1, 2, 3), signs))
           for signs in product((False, True), repeat=3)]

result = solve_cnf_native(runtime, 3, clauses, max_conflicts=10_000)
assert result.status == "UNSAT_VERIFIED"
assert verify_direct_native_result(3, clauses, result)
write_drup("proof.drat", result.proof)
```

`solve_cnf_native` solves the formula as given. `solve_general_native` first applies
the binary core and solves the residual, then composes the core's derivation with
the residual proof into a single proof over the original formula; check it with
`verify_native_result`.

| `status` | `reason` | Meaning |
|---|---|---|
| `SAT_VERIFIED` | `model` / `binary_core_model` | `result.model` covers every declared variable and satisfies every clause |
| `UNSAT_VERIFIED` | `drup` / `binary_core` | the proof was replayed against the original clauses |
| `UNKNOWN` | `conflict_budget` / `learned_literal_budget` | a budget was exhausted; no claim is made |

`max_conflicts` and `max_learned_literals` are unsigned 64-bit budgets. Sparse
inputs whose declared variable range is at least three-quarters unused are
compacted internally; models and proofs are always reported in the original
numbering. `result.stats` reports decisions, propagations, conflicts, restarts and
learned/deleted clauses.

## Admission

```bash
python -m spectra.sat.admission formula.cnf.gz
```

`admit_cnf(path)` reads plain or gzip DIMACS without simplifying it and returns a
record with source and decompressed SHA-256 hashes, an order-independent semantic
hash, declared and used variables, clause-length histogram, Horn/dual-Horn counts
and a problem class. Malformed headers, out-of-range literals, missing terminators
and inputs over the byte limits are rejected.

## External proof checking

```python
from spectra.sat.drat import check_drat, write_dimacs

write_dimacs("formula.cnf", 3, clauses)
check = check_drat("/path/to/drat-trim", "formula.cnf", "proof.drat")
assert check.verified
```

A proof counts as verified only when the checker exits with status 0 and prints
an exact `s VERIFIED` line. The test suite runs this path when the `DRAT_TRIM`
environment variable points at a DRAT-trim binary.

## Performance

The solver is designed for verifiable answers, not for competition speed. On hard
industrial instances it is substantially slower than state-of-the-art solvers such
as CaDiCaL or Kissat. [`benchmarks/sat`](../benchmarks/README.md#sat) contains a
campaign runner that measures it against other solvers with per-process memory and
time limits on Linux and Windows.
