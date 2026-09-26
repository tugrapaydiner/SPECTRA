# SPECTRA

**Verified CNF search and reproducible CPU reasoning research.**

SPECTRA provides a dependency-free Python API and command-line interface for
bounded Boolean search, witness checking and artifact integrity. Neural models,
native kernels and historical experiments are a separate, optional research layer.
This is not a frontier-model replacement or a complete SAT solver.

[API and CLI](docs/API.md) · [Results and limitations](docs/STATUS.md) ·
[Development](docs/DEVELOPMENT.md) · [Documentation](docs/README.md)

## Quick start

Use Python 3.10 or newer, from this checkout:

```bash
python -m pip install .
spectra doctor
spectra cnf solve examples/tiny.cnf --seed 7 --max-flips 1024 --out answer.json
spectra cnf check examples/tiny.cnf answer.json
```

The base installation requires neither PyTorch nor a C++ compiler. Commands emit
JSON and refuse to overwrite existing output files. A completed solve can return
`SAT_VERIFIED` or `UNKNOWN`: always inspect `status`, not just the exit code.
`UNKNOWN` is neither a valid solution nor an UNSAT proof.

```python
from spectra.cnf import CNF, solve

problem = CNF(2, ((1, 2), (-1, 2)))
result = solve(problem, seed=7, max_flips=128)
if result.status == "SAT_VERIFIED":
    assert problem.satisfied(result.witness)
```

For repeated or larger searches, explicitly select the indexed backend:

```bash
spectra cnf solve examples/tiny.cnf --backend indexed --seed 7 --max-flips 4096 --out indexed.json
```

## Supported surface

| Interface | Purpose | Boundary |
|---|---|---|
| `spectra.cnf` / `spectra cnf` | CNF state, bounded search, Boolean witnesses | Classical algorithms; no learned solver claim |
| `spectra.evidence` / `spectra evidence` | Portable SHA-256 and size manifests | Byte integrity, not scientific validity or authenticity |
| `spectra.inference` | Output-only historical CPU inference | Optional research environment; trusted model artifacts |
| `model/`, `train/`, `eval/`, `deploy/` | Research and native implementations | Original protocols and compatibility paths are retained |

See [API contracts](docs/API.md) for formats, resource limits and exit codes, and
[security boundaries](SECURITY.md) before accepting external inputs or checkpoints.

## Results, without mixing experiments

The v0.7.1 indexed backend preserves seeded search paths while removing repeated
sorting. Its frozen local large-case comparison reports a **0.237505 cold-call
latency ratio** against the reference, including preparation. Small inputs can
regress and cold allocation increases. The large timed cases return `UNKNOWN`;
this measures cheaper identical bounded search, not higher SAT success.
The [efficiency guide](docs/EFFICIENCY_GUIDE.md) retains all cells and limitations.

Output-only inference preserves final numerical outputs in its checked fixtures.
Reduced returned tensor storage is not reduced peak RAM. Historical learned
capability gates and stronger classical baselines remain in
[the research review](docs/RESEARCH_REVIEW_20260911.md).

The published GitHub release is
[v0.7.1](https://github.com/tugrapaydiner/SPECTRA/releases/tag/v0.7.1).
This cleanup is an **unreleased maintenance change**; it does not replace that
release or publish to PyPI. Later prepared-runtime and compiler work is tracked
separately in [current status](docs/STATUS.md#delivery-state).

## Work on the project

```bash
python -m pip install ".[test]"
python -m pytest tests/public --confcutdir=tests/public
python scripts/check_release_readiness.py
```

For the pinned CPU research environment, full regressions, native compilation and
outside-checkout wheel testing, use [Development](docs/DEVELOPMENT.md).
Release operators should use [Releasing](docs/RELEASING.md), not historical logs.

```text
spectra/       public API and CLI
examples/      runnable inputs
model/ train/  historical neural implementation
common/ data/ eval/ deploy/  evidence-bound compatibility modules
scripts/       experiments, replay and maintenance commands
tests/         regression suite; public/ needs only pytest
config/        historical experiment configurations
results/       retained evidence, including negative results
docs/          current guides, fixed protocols and history index
maintenance/   byte-retention and archived-branch receipts
```

Research evidence and old module paths are intentionally kept in place. The
[history index](docs/history/README.md) explains work already archived; this
cleanup neither rewrites Git history nor deletes unfavorable experiments.
