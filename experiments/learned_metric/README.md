# Label-trained integer metrics

Read [RESULTS.md](RESULTS.md) first: a small Letter quality gain, no gain on Pendigits
or the pre-acquisition-frozen Satellite transfer, all original joint gates FAILED.
The model is trained on integer-weighted distances and deployed without changing
those distances. See [CONTRACT.md](CONTRACT.md) for the exact numerical scope.

This folder is an additive experiment, not the production API. Python3.11+ and a
C++17 compiler are needed; accepted native targets here are Linux x86-64 portable
and explicitly AVX2. Training needs NumPy2.3.5, SciPy1.17.0, scikit-learn1.8.0 and
threadpoolctl. Deployment needs the standard library and the compiled component.
No compilation happens on import. Native libraries and benchmark pickles must be
trusted; never load an unknown pickle to reproduce these commands.

## Reproduce fitting and evaluation

The evidence packet includes the original official archives, exact parsed data,
selection outputs, frozen models, native controls and matching sources. Use fresh
output directories; do not replace the previous evidence. These commands are not
permission to select different models from the already opened evaluation sets.

```bash
python experiments/learned_metric/build.py --out /new/native --target portable
python -m pytest experiments/learned_metric/test_metric.py \
  experiments/learned_metric/test_learning.py --import-mode=importlib \
  --confcutdir=experiments/learned_metric
python experiments/learned_metric/prepare_data.py --archives /evidence/archives --out /new/data
python experiments/learned_metric/train.py selection --data /new/data --out /new/selection
python experiments/learned_metric/train.py refit --data /new/data --selection /new/selection --out /new/models
python experiments/learned_metric/controls.py --models /new/models --out /new/controls
python experiments/learned_metric/evaluate.py --data /new/data --models /new/models \
  --controls /new/controls --library /new/native/metric.so --out /new/evaluation
```

The fixed fitter runs one numerical thread. The biggest full Letter Gram matrix is
2.048GB on disk; RAM, scratch disk, inference table cost and fit costs are separate.
A new environment may produce different fitted bytes: record them as a new run,
not the original receipt. Use supplied frozen models for exact execution replay.

Satellite uses `satellite.py prepare/selection/refit` with the original archive,
then `satellite_evaluate.py`. Its `--reference-library` is the original-reference
observer built during primary evaluation. The pre-acquisition fixed parameters
and integer-budget projection are in SATELLITE_PROTOCOL.md. Do not silently apply
the Letter gamma grid or unit budget to this dataset.

## Complete cost comparison and independent audit

`bench.py` compares eleven arms on both original tasks, including the old finite
executor. The four unchanged previous runtime files under finite_kernel and
adaptive_kernel are compatibility prerequisites, not new code in the learner.
The exact old models/data and their library build receipts are in the evidence.

```bash
python experiments/learned_metric/bench.py --data /evidence/datasets \
  --models /evidence/models --evaluation /evidence/evaluation \
  --controls /evidence/controls --library /evidence/build-metric/metric.so \
  --old-library /path/to/finite.so --old-models /evidence/historical --out /new/timing
python experiments/learned_metric/audit.py --root /evidence --source /matching/source --out /new/primary-audit.json
python experiments/learned_metric/audit_satellite.py --root /evidence/satellite \
  --source /matching/source --out /new/satellite-audit.json
```

The auditors use only JSON, checked NPY data and the standard library. They check
saved labels, same-feature splits, complete selection/timing grids and the derived
outcomes. They do not unpickle models, import the learner, independently certify
all bootstrap assumptions, authenticate clocks or prove absence of human adaptation.

`negative_audit.py` exercises sixteen corruption cases only on disposable copies.
`resource_probe.py` uses the extracted SDK and measures each fresh process's own
Linux high-water memory. Its row cap, table allocation and platform limits remain.

## Quick deployment replay

The delivered SDK contains all nine trained SVMs, seven simpler controls, their
frozen evaluation inputs and both accepted Linux libraries. From its root:

```bash
python -I -S selftest.py --target portable --out replay.json
# Only on AVX2-capable x86-64 hardware:
python -I -S selftest.py --target avx2 --out replay-avx2.json
```

These replay fixed models without training, network access or numerical-framework
imports. They do not establish new held-out model quality or another operating
system's compatibility. Never assign old SPECTRA Windows/ARM results to this new
experimental executor. The previous production models and defaults are unchanged.
