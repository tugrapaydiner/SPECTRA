# Development and reproduction

## Public tools

Use Python 3.10 or newer and a virtual environment. From a full checkout:

```bash
python -m pip install ".[test]"
python -m spectra doctor
python -m pytest tests/public --confcutdir=tests/public
python scripts/check_release_readiness.py
```

The base package has no mandatory third-party dependencies; `test` adds pytest
without the research stack. `spectra` and `python -m spectra` are equivalent.
See [API and CLI](API.md) for `UNKNOWN`, exit codes and bounded JSON input.

## Build and test the actual distribution

Use a new build/output directory for each attempt. The selector rejects stale or
multiple wheels rather than guessing which one to test.

```bash
python -m pip install build
python -m build --outdir .release-check/dist
python scripts/check_current_installation.py --dist .release-check/dist --out .release-check/install
python scripts/check_release_readiness.py --out .release-check/navigation.json
python scripts/audit_workspace.py --out .release-check/workspace.json
```

The default `python -m build` builds an sdist, then builds the wheel from that
sdist. The installation check creates a separate virtual environment, installs
without dependencies, and runs **11 installed-only checks outside the checkout**.
It verifies the command, APIs and packaged native source inventory. It does not
substitute an editable install for a wheel test. The public-package CI performs
the same path; the indexed-efficiency workflow also checks Python 3.10 and 3.13.

Do not reuse existing report paths: evidence tools refuse replacement. The
workspace audit requires full Git history (`git fetch --unshallow` for a shallow
clone), not a downloaded source ZIP. Its 449-file historical identity contract
is separate from metadata/navigation checks. The sdist includes guides and the
example CNF; complete historical evidence and replay require the Git checkout.

## CPU research environment

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.10.0
python -m pip install -r requirements-cpu-research.txt
python -m pip install --no-deps -e .
python -m pytest -m "not slow"
python scripts/verify_retained_results.py --out outputs/retained-check.json
```

Create the output parent directory first. The pinned environment is a historical
reproduction target, not a promise that old dependency versions remain secure.
`.[research]` installs unpinned research extras; `.[dev]` adds pytest to them.
Sixteen historical slow tests retrain models and are excluded from the fast-suite
count. Full CPU CI additionally replays checkpoint, fixed-pool, restart,
controller-refit and SAT-admission evidence. Reproducing a stored claim does not
open a new confirmation experiment or allow selecting a more favorable result.

## Native execution

With PyTorch, Ninja and a suitable compiler installed,
`python setup.py build_ext --inplace` preserves the historical build command.
An explicit binary build can use `SPECTRA_BUILD_NATIVE=1` with
`pip wheel --no-build-isolation .`; compilation failure is an error.
Ordinary wheels include sources for the existing JIT loaders without compiling
during installation. Native checks are separate from the dependency-free wheel
checks. Cross-host bitwise equality is not a blanket runtime guarantee.

## Reproduce systems comparisons

```bash
python scripts/bench_compact_cnf.py run --out compact-run
python scripts/bench_compact_cnf.py verify --out compact-run --replay
python -m eval.cnf_cache_benchmark run --out original-cache-run
python -m eval.cnf_cache_benchmark verify --out original-cache-run --replay
```

For indexed search, use [the efficiency guide](EFFICIENCY_GUIDE.md). NumPy is
needed for experiment statistics, not public CNF search. Preserve each complete
run directory, source identity and failed attempt. Deterministic trajectory
replay does not regenerate historical wall-clock times. Download CI evidence
before its retention expires; temporary CI artifacts are not a durable archive.

## History and compatibility

Use one branch per hypothesis and distinguish protocol, implementation, result
and promotion candidate. Do not rename old import paths, mass-format retained
research code, regenerate old measurements or relax tolerances for tidiness.
The [history index](history/README.md) and [contribution policy](../CONTRIBUTING.md)
cover the existing archived tips. The branch-retirement script remains dry-run
by default and refuses advanced tips; no history rewrite is part of this cleanup.

## Optional runtime tracks in unreleased main

The prepared FP32 and compiler baselines are included without changing the
default solver. Use [prepared execution](TRAINED_FP_GUIDE.md),
[packed execution](RUNTIME_GUIDE.md) and [compiler controls](COMPILER_BASELINE_GUIDE.md).
A fresh named distribution check can also be run with:

```bash
python -m build --outdir dist/current
python scripts/check_current_installation.py --dist dist/current --out install-check
```

## README figures

From the full checkout, `python scripts/render_readme_charts.py --check` validates
all generated chart bytes against hash-pinned published summaries. Omit `--check`
to regenerate the three assets. This needs only the standard library and does
not run benchmarks or update historical measurements. See
[chart provenance](../assets/readme/README.md).
