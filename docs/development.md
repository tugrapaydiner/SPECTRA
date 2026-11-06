# Development

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -e ".[dev]"        # pytest, the PySAT reference solvers, ruff
python -m pip install -e ".[svm]"        # optional: NumPy/scikit-learn SVM exporters
```

A C++17 compiler (GCC, Clang, or MSVC on Windows) and the CPython development
headers are needed for tests that build native code.

## Repository layout

| Path | Contents |
|---|---|
| `spectra/` | The installed package. Native C++ sources live next to the Python modules that build them (`spectra/_native`, `spectra/sat`, `spectra/migration`). |
| `tests/` | Tests, grouped by package area: `cnf`, `exact`, `sat`, `migration`, `svm`, `package`, plus `benchmarks` for the benchmark tooling. |
| `benchmarks/` | Workload generators, baseline solvers and benchmark runners. Not installed. |
| `scripts/` | Installed-wheel checks and the release metadata check used by CI. |
| `docs/` | User guides. |

## Tests

```bash
python -m pytest                 # everything; native code is built into temp dirs
python -m pytest tests/exact     # one area
ruff check .
python scripts/check_release_readiness.py
```

Tests that need optional external tools skip with the reason shown in the summary:

| Variable | Enables |
|---|---|
| `DRAT_TRIM=/path/to/drat-trim` | external replay of UNSAT proofs |
| `CADICAL_SOURCE=/path/to/cadical` | the official CaDiCaL baseline adapter (3.0.1) |
| `SPECTRA_EXACT_LIBRARY`, `SPECTRA_QUOTIENT_LIBRARY`, ... | run tests against a prebuilt (e.g. sanitizer) library |

Windows job-object tests run only on Windows.

## Continuous integration

`ci.yml` runs lint and the release check; the full suite on Linux x86-64 (Python
3.10–3.13) and ARM64; the MSVC-capable components on Windows; the exact runtime,
baselines and quotient runtime under UBSan and ASan+UBSan; DRAT-trim and CaDiCaL
cross-checks; and GCC/Clang builds of the CDCL solver.

`package.yml` builds the sdist and wheel once, checks that every shipped file
matches the checkout, installs the wheel into clean environments on Linux, ARM64
and Windows, exercises the installed APIs from outside the checkout, replays the
SVM fixture corpus, and builds and exercises the installed native CDCL solver on
Linux and Windows. It requires `spectra exact selftest` bundles from all five
environments, including Windows Python 3.11, to agree and retains the comparison
report as a CI artifact.

All third-party actions are pinned to full commit SHAs;
`scripts/check_release_readiness.py` enforces this.

## Native code conventions

- Importing a module never compiles. Each component exposes an explicit
  `build_*` function that writes the library and a `build.json` receipt with
  source and binary hashes and the compiler command into a fresh directory.
  CPython extension receipts also record interpreter compatibility metadata.
- The exact runtime and native CDCL solver verify their build receipts before
  loading. Other native loaders accept a trusted library path; receipt checking
  is component-specific.
- The exact runtime and native CDCL solver independently check completed SAT and
  UNSAT answers. The quotient runtime checks returned colourings but emits no
  UNSAT proof. SVM voting certificates check the winner implied by disclosed pair
  outcomes; decision receipts also replay those outcomes against the model.

## Releases

1. Set the version in both `pyproject.toml` and `spectra/__init__.py`.
   Unreleased source carries a `.devN` suffix.
2. Update `CHANGELOG.md`.
3. Make sure `ci` and `package` pass on the release commit.
4. Build with `python -m build` and verify with
   `python scripts/check_current_installation.py --dist dist --check-sdist --out .install`.
5. Tag `vX.Y.Z` and publish the verified artifacts. Never move or replace an
   existing tag.
