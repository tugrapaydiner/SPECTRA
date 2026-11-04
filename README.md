# SPECTRA

**Certified repeated-query solving on CPUs.** SPECTRA compiles a fixed constraint
problem once and then answers many restricted queries against it, returning a
complete witness or an independently checkable proof for every answer. Nothing is
trusted on the solver's word: SAT answers are checked against the original input,
and UNSAT answers carry proofs that a separate, standard-library verifier replays.

```bash
pip install .
spectra doctor
```

The package has no third-party dependencies. Native kernels ship as C++ source and
are compiled only when you call an explicit build function, into a directory you
choose, with a receipt that binds the source, compiler and binary.

## What is in the package

| Module | What it does |
|---|---|
| [`spectra.exact`](docs/exact.md) | Proof-carrying list-colouring queries for domains of one or two colours. Every answer is a full colouring or an original-literal contradiction proof. |
| [`spectra.cnf.quotient_query`](docs/exact.md#quotient-runtime) | Strongly-connected-component quotient compiler for repeated list-colouring support queries, addressed in the original vertex space. |
| [`spectra.sat`](docs/sat.md) | General CNF: a native CDCL solver with checked models and DRUP proofs, a proof-replayable unit/binary simplifier, strict DIMACS admission and a DRAT-trim adapter. |
| [`spectra.cnf`](docs/cnf.md) | Dependency-free CNF tools: seeded local search (compact, indexed, focused), deductive and DPLL solvers, DIMACS I/O and witness checking. |
| [`spectra.migration`](docs/migration.md) | Two-plan migration: the unique minimum set of changes between two valid colourings, or a contradiction path. |
| [`spectra.exam_repair`](docs/exact.md#exam-repair) | Bounded exam rescheduling on top of `spectra.exact`, with named, human-readable infeasibility explanations. |
| [`spectra.svm`](docs/svm.md) | Native execution of exported RBF SVMs with certified pairwise voting, fitted preprocessing plans, decision receipts and a streaming CLI. |

## Quick start

```python
from spectra.exact import (ExactProblem, ExactRuntime, build_exact_runtime,
                           pack_session_packet, verify_session_output)

# A path 0-1-2-3; each vertex may take one of two colours (bit masks over 3 colours).
problem = ExactProblem.create(n=4, palette=3,
                              edges=((0, 1), (1, 2), (2, 3)),
                              masks=(0b011, 0b110, 0b101, 0b011))
runtime = ExactRuntime(build_exact_runtime("build/exact"))   # explicit C++ build

with runtime.prepare(problem) as session:
    answer = session.solve(((0, 0b001), (3, 0b010)))       # pin vertex 0 and 3
    print(answer.status, answer.labels)                     # SAT_VERIFIED + colouring

# Or answer a whole bank in one call and verify it without native code:
packet = pack_session_packet(problem, ((), ((0, 0b001),), ((0, 0b001), (1, 0b001))))
result = runtime.solve_session_packet(packet)
assert verify_session_output(packet, result.output)
```

```bash
spectra cnf solve examples/tiny.cnf --seed 7 --out answer.json
spectra cnf check examples/tiny.cnf answer.json
python -m spectra.sat.admission examples/tiny.cnf     # strict DIMACS admission record
```

## Results

**Repeated list-colouring support queries (WAP).** On five optical-network conflict
graphs from the public WAP benchmark set, SPECTRA's quotient runtime answered
1,024-query sessions in **36.4%** of the time of a persistent native MiniCard
solver (mean ratio 0.364, 95% CI 0.347–0.373; p95 ratio 0.365), winning on all five
graphs. All 107,520 answers were checked against the original graph, lists and
restrictions. Timings cover the complete session: input decoding and validation,
solver construction, every query, full answer materialisation, independent
checking and disposal. Graph topologies are real; availability lists and queries
are synthetic.

| Environment | Ratio vs. MiniCard | 95% upper bound |
|---|---:|---:|
| Ubuntu 24, Python 3.13 | 0.364 | 0.373 |
| Ubuntu 24, Python 3.13 (rerun) | 0.360 | 0.368 |
| Ubuntu 22, Python 3.13 | 0.364 | 0.374 |
| Ubuntu 24, Python 3.11 | 0.534 | 0.542 |
| Ubuntu 22, Python 3.11 | 0.529 | 0.536 |

Under Python 3.11 the warm query loop is unchanged (15% of MiniCard's time), but
slower JSON decoding and validation of the multi-megabyte inputs dominates the
complete-session cost. The original source tree, inputs, outputs and analysis are
preserved at commit
[`0af6fda`](https://github.com/tugrapaydiner/SPECTRA/tree/0af6fda36c60a110584e7db836ee5724e530fbd4);
[`benchmarks/`](benchmarks/README.md) contains the workload generator.

**Scope.** These are results on a specific repeated-query workload. They are not a
claim that SPECTRA outperforms SAT solvers or graph-colouring systems in general.
The general-CNF solver in `spectra.sat` is correct by construction-and-check but is
not competitive with state-of-the-art SAT solvers on hard instances.

## Documentation

- [Exact queries, quotient runtime and exam repair](docs/exact.md)
- [General CNF solving](docs/sat.md)
- [CNF local search and DPLL](docs/cnf.md)
- [Two-plan migration](docs/migration.md)
- [SVM execution](docs/svm.md)
- [Command-line interface](docs/cli.md)
- [Development, testing and releases](docs/development.md)
- [Benchmarks](benchmarks/README.md)

## Platform support

Python 3.10–3.13. Every native component builds on Linux x86-64 and ARM64 with
GCC. `spectra.exact`, the `spectra.sat` CDCL solver and `spectra.svm` also build on
Windows x64 with MSVC. Building requires a C++17 compiler and the CPython
development headers; nothing is downloaded.

## License

MIT. See [LICENSE](LICENSE). Upstream datasets and third-party components retain
their own licenses.
