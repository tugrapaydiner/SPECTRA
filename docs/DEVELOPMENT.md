# Development and reproduction

## Install the public tools

From a checkout, run `python -m pip install .`. The base package has no mandatory
third-party dependencies. The `spectra` command and `python -m spectra` are
identical entry points. Run `spectra doctor` to inspect the environment without
importing PyTorch.

```bash
spectra cnf solve examples/tiny.cnf --seed 7 --max-flips 128 --out answer.json
spectra cnf check examples/tiny.cnf answer.json
```

A false witness check exits 1. Malformed input, invalid settings and refused output
overwrite exit 2. A completed capped solve exits 0, but callers must inspect its
`status`: `UNKNOWN` is not a solution or an UNSAT proof. DIMACS limits bound parsed
variables, clauses and literals, not arbitrary file size. Only Boolean-list
witnesses are accepted; a report with a different formula fingerprint is rejected.

## Test and build

```bash
python -m pip install pytest
python -m pytest tests/public --confcutdir=tests/public
python -m pip install build
python -m build --outdir dist/current
python scripts/check_current_installation.py --dist dist/current --out install-check
```

Use a fresh `dist/current` directory. The selector checks distribution metadata and
rejects multiple wheels, a stale version, or missing native sources rather than
guessing which artifact to test. The installation check uses a new virtual environment outside the repository,
installs the actual wheel without dependencies, and exercises the public command,
imported API and packaged native sources. It does not run from an editable checkout.
CI also builds the wheel from the source distribution rather than only the source
tree. Public tests do not need the parent research test setup or PyTorch.

For historical experiments use the CPU environment in
[the repository README](../README.md). `pip install ".[dev]"` retains all former
research dependencies plus pytest; `.[research]` omits pytest. Full regression is
`python -m pytest -m "not slow"`. Slow tests retrain historical models and are not
silently included in a fast-test count.

Native builds are explicit. With PyTorch, Ninja and a suitable compiler installed,
`python setup.py build_ext --inplace` preserves the historical build command.
A binary wheel can be requested with `SPECTRA_BUILD_NATIVE=1` and
`pip wheel --no-build-isolation .`; compilation failure is an error. Ordinary
wheels include the native sources used by the existing JIT loaders.

## Reproduce bounded systems comparisons

```bash
python scripts/bench_compact_cnf.py run --out compact-run
python scripts/bench_compact_cnf.py verify --out compact-run --replay
python -m eval.cnf_cache_benchmark run --out original-cache-run
python -m eval.cnf_cache_benchmark verify --out original-cache-run --replay
```

NumPy is needed for experiment statistics, not for the public CNF tools. Preserve
the complete directory and its exact execution source. Timing and allocation
numbers are observations; a deterministic trajectory replay does not regenerate
historical wall-clock times. Public-package and replay artifacts use 90-day
retention; native/task artifacts use 14 days. Keep the delivered bundle for
long-term evidence retention. Historical raw results remain in `results/`.

## Branch and history policy

Use one branch for each active hypothesis and state whether it is a protocol,
implementation, failed result or promotion candidate. A completed experiment can
be closed without merging its unsuccessful learner. Retired tips receive exact
`archive/2026-09-12/...` tags. Never force-push over an advanced branch for cleanup.
The [history index](history/README.md) restores removed working-tree files using
immutable Git commits. No Git history is rewritten, no evidence is erased and no
historical numerical tolerance is weakened.

The archive script defaults to a dry run. Applying it requires the precise merge
commit, the checked manifest and the main-only maintenance workflow. It refuses
wrong tags, changed branch tips, missing unarchived tips and non-merge commits.
Tests exercise these cases against disposable local Git remotes.

## Integrated CPU runtime

The optional [runtime guide](RUNTIME_GUIDE.md) documents supported concrete runtime
types, bitwise equivalence tests, full-cost benchmarking and profiling. Historical
runtimes remain unchanged. No new runtime is selected silently by the public API.

## Release intent

The indexed-efficiency workflow compares `[project].version` with the previous
main commit before any publication steps. Packaging-only edits do not request a
new release; an unknown prior version fails closed. The existing protections
against replacing tags/assets and the version-specific evidence packaging contract
remain unchanged. A future version still requires its own compatible publication
protocol; changing metadata alone does not authorize rewriting version 0.7.1.
