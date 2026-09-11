# CPU research installation and replay

The default `requirements.txt` now resolves the already-declared M16/M17 CPU
research versions, including the explicit `torch==2.10.0+cpu` wheel. Broad
package dependencies remain in `pyproject.toml` for runtime consumers. A broad
minimum-version range is not a bit-exact historical reproduction environment.

This distinction was exposed by PR20 run `34561817811`: the legacy M03 workflow
installed Torch 2.14.0+cpu, NumPy 2.4.6 and pytest 9.1.1. Thirteen new
version-bound replay tests correctly rejected that environment; the remaining
548 fast tests passed. No test is disabled to resolve this failure. The default
research installation is now consistent with the pinned CPU regression matrix.

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -m 'not slow' -ra
python scripts/verify_retained_results.py --out /tmp/spectra-retained.json
python scripts/verify_checkpoint_replay.py --out /tmp/spectra-complete-replay
python scripts/verify_fixed_pool_replay.py --cpu-profile historical-ordered --out /tmp/spectra-fixed-pools
python scripts/audit_sudoku_symmetry.py --out /tmp/spectra-symmetry --verify-report results/reliability/sudoku4_symmetry_audit.json
```

Use new output paths on each invocation; verification refuses to overwrite a
previous record. These commands use CPU only. The explicit historical arithmetic
profile is Linux x86-64 AVX2/FMA and Torch 2.10.0+cpu only. It is separate from
ordinary B=1 inference, which has its own fidelity tests and timing scope.

The replay CLIs select numerical dispatch before importing Torch or NumPy. A
sitecustomize hook that imports these libraries first is rejected, rather than
silently applying an ineffective profile. Run in a clean virtual environment.

Activate that environment before running native builds: PyTorch invokes the
`ninja` executable through PATH. The research requirements now install Ninja
explicitly, rather than depending on a CI runner's preinstalled build tools.

Long legacy training tests are a separate scope (`pytest -m slow`). Fast test
counts must not be described as the full training suite. Complete transitive
package versions and source archives are retained by CI; the direct package
pins are not a universal transitive lockfile.
