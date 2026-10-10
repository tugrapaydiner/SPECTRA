# Public API and CLI

The supported base interfaces use Python's standard library. They do not train a
model, download data or import PyTorch. Optional inference is documented separately
in [the efficiency guide](EFFICIENCY_GUIDE.md).

## Boolean CNF

```python
from io import StringIO
from spectra.cnf import CNF, read_dimacs, solve, solve_indexed, PreparedCNF

problem = read_dimacs(StringIO("p cnf 2 2\n1 2 0\n-1 2 0\n"))
reference = solve(problem, seed=7, max_flips=128)
indexed = solve_indexed(problem, seed=7, max_flips=128)
prepared = PreparedCNF(problem)
repeated = prepared.solve(seed=7, max_flips=128)
assert reference.witness == indexed.witness == repeated.witness
```

`CNF(nvars, clauses)` uses signed, one-based literal IDs; witness positions and
repair-state variable indices are zero-based. A complete witness must contain
exactly `nvars` genuine Boolean values, not integer 0/1 substitutes. An empty
formula is satisfied; an empty clause prevents satisfaction.

`read_dimacs` requires a single ASCII `p cnf` header, exact counts, in-range
literals and zero-terminated clauses. Parsed variable/clause/literal limits are
configurable Python keyword arguments. They do not cap arbitrary input-file
bytes, process memory or elapsed time.

`solve` is the historical compact backend. `solve_indexed` uses reusable ranked
indexes; `PreparedCNF.solve` amortizes formula preparation. The same formula,
seed and flip cap preserve non-timing search results across these backends.
A prepared object gives each call independent mutable search state. Preparation
excluded from a warm call must be accounted for separately in performance claims.

Seeds are unsigned 64-bit integers; `max_flips` is a nonnegative integer.
`SolveResult.record()` returns a `spectra.cnf.solve.v1` dictionary with status,
witness, unsatisfied-clause IDs, flips, queries, path hash and elapsed nanoseconds.
The witness on `UNKNOWN` is only the final candidate. Neither backend supplies
an UNSAT proof or a hard wall-clock deadline.

`solve_deductive` is an optional classical backend that propagates forced literals,
solves binary residuals by an implication graph, then uses indexed search for
general residuals. Select `--backend deductive` from the CLI. Its flip cap bounds
only residual search, and its trajectory can differ from the original formula's
walk. See [deductive solving](DEDUCTIVE_GUIDE.md) for costs, result fields and
measured scope. The historical default is unchanged.

`solve_focused` and CLI `--backend focused` expose an experimental dense-clause
pool and break/age move policy. This classical candidate has a retained fresh
synthetic evaluation and independent witness checks. See [focused search](FOCUSED_GUIDE.md)
for its measured scope, budget semantics and recovery limitations.

## Commands and exit codes

```bash
spectra --version
spectra doctor
spectra cnf solve examples/tiny.cnf --seed 7 --max-flips 128 --out answer.json
spectra cnf check examples/tiny.cnf answer.json
```

| Exit | Meaning |
|---|---|
| `0` | Command completed; inspect a solve's `status`. A check accepted the witness. |
| `1` | Witness is well-formed but does not satisfy the formula. |
| `2` | Invalid input/settings, missing file, manifest failure or refused overwrite. |

`solve` adds `formula_sha256`, backend and timing-scope fields to CLI reports.
`check` accepts a JSON object with a Boolean-list `witness`; if a formula hash is
present it must match. Extra report fields are permitted, but duplicate object
keys, NaN/Infinity, floating-point overflow and invalid UTF-8 are rejected.
The default witness/manifest limit is **16 MiB**, measured before decoding:

```bash
spectra cnf check examples/tiny.cnf answer.json --max-json-bytes 1048576
```

`--max-json-bytes` must be an integer from 1 through `sys.maxsize - 1`.
Reads use chunks of at most 64 KiB; a large cap does not preallocate the cap.
The limit bounds input bytes, not all decoder allocations, CPU time or filesystem access. Excessive nesting is reported as an
input error. `--out` uses exclusive creation; use a new path for each solve.

## Evidence integrity

```python
from spectra.evidence import capture, verify, write_new

manifest = capture("run", ["config.json", "observations.jsonl"])
write_new("manifest.json", manifest)
assert verify("run", manifest, required=["config.json", "observations.jsonl"])["verified"]
```

Create these example artifacts before calling `capture`. The manifest format is
`spectra.evidence_freeze.v1`; each relative POSIX path has a SHA-256 digest and
byte count. `verify` rejects changed/missing artifacts, malformed records,
traversal and symlinked artifacts. It verifies only the named inventory; extra
files in the root are not automatically rejected. Use Python's `required`
argument to require an exact expected inventory.

```bash
spectra evidence run manifest.json --max-json-bytes 16777216
```

The CLI applies the same strict JSON reader as witness checking. The historical
Python evidence API is unchanged and accepts an already decoded dictionary.
Matching hashes do not establish the manifest's origin, protect a concurrently
mutating directory or certify scientific correctness. See [Security](../SECURITY.md).
