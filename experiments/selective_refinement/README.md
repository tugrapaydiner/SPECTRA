# Selective refinement: two frozen models, one conditional native call

Read RESULTS.md and CONTRACT.md first. Only Letter passes the fixed quality/cost
condition; the overall two-task gate fails. Pendigits exceeds the nominal added-harm
budget on its official test. These are four previously exposed benchmarks, not
new independent confirmation or production-calibrated models.

## What is in the continuation

The training-only head-conditioning pilot is under `experiments/conditioned_heads`.
It produced no convincing accuracy gain and was not promoted. This experiment then
trained a cheap256-prototype model, stronger SVC and properly scaled MLP, separating
model fitting/selection from calibration groups. No teacher or pretrained model.
The MLP receives an ordinary FP32 OpenBLAS deployment, not a Python slow baseline.

The old PR40 canonical source and CI records were recovered, but the missing local
auditor and24 old final models were not. Their previous audit mismatch therefore
remains unclassified. Do not assign this experiment's acceptance to the old packet.
New evidence includes the actual12 fitted models, roles, every calibration row,
all test predictions, all588 timed jobs and source snapshots at every stage.

## Reproduction

Tested Python3.13.5, NumPy2.3.5, SciPy1.17.0, sklearn1.8.0, PyTorch2.10.0+cpu and
pytest9.0.2 on Linux x86-64. Other Python versions/platforms require their own tests.
Inference needs no numerical Python package. Original fitted pickles in evidence
are trusted reconstruction artifacts only; never execute an untrusted pickle.
All output directories must be fresh. Use source snapshots to reconstruct the
literal recorded stages; current files include later audits/documentation.

```bash
python -m experiments.selective_refinement.build --out /new/native --target portable
python -m pytest experiments/selective_refinement/test_refinement.py \
  experiments/selective_refinement/test_deployment.py \
  experiments/conditioned_heads/test_optimizer.py \
  experiments/source_receipts/test_receipts.py \
  --import-mode=importlib --confcutdir=experiments

python -m experiments.selective_refinement.study \
  --data /evidence/inputs --out /new/models
python -m experiments.selective_refinement.calibrate_run \
  --models /new/models --data /evidence/inputs \
  --library /new/native/refinement.so --out /new/calibration
```

`evaluate.py` requires models, calibration, input data, refinement library,
FP32-MLP library and a fresh output folder. `bench.py` uses the frozen evaluation
plus the same libraries and policies. Run their module `--help` commands for all
arguments. The FP32 control builder is the canonical parent
`experiments.budgeted_prototypes.float_control.build`; its `blas` argument must
be the matching LP64 SciPy OpenBLAS shared library exposing `scipy_cblas_sgemm`.
It is a comparator dependency, not a refinement inference dependency. All baseline
packing/scaling and allocations are in the warm timer. Do not compare a new run's
clock values directly with old-host measurements or use already opened tests to
retune the policy while retaining an old confirmation label.

The supplied small deployment kit contains both models for each task, bounded
model-bound policy JSON, portable/AVX2 shared libraries, matching source and inputs.
From that kit, `python -I -S selftest.py --target portable` replays all routes;
`demo.py --task letter` runs one included example. AVX2 is opt-in and requires
compatible hardware. Windows/ARM binaries are not supplied or newly accepted.

## Source-bound evidence audit

`audit.py` is standalone standard-library code: it does not import the learner,
load native libraries, execute pickles or trust the reported calibration quantile.
It inverts the binomial CDF independently, checks all role splits, exports,
selection choices, policies, routing, prediction counts and timing arithmetic.
Its PASS means those recorded checks pass, including the recorded FAILED scientific
gate and observed distribution-shift limitation; not that the model is safe.

```bash
python -I -S experiments/source_receipts/runner.py \
  --auditor experiments/selective_refinement/audit.py \
  --evidence /extracted/evidence --output /new/source-bound-audit.json \
  --expected-sha256 <trusted-auditor-sha256-from-delivery-manifest>
```

The runner hashes and executes the SAME source bytes, refuses a mismatch before
execution, and atomically publishes a new receipt with the exact auditor digest.
The file-closure checker separately binds every delivered source/data member:
`python -I -S experiments/source_receipts/seal.py /extracted/evidence --sha256 ...`.
Expected hashes must come from a trusted reference; nothing here is a digital
signature, sandbox, authentication of clocks or proof of research independence.

`negative_audit.py` mutates disposable copies only and checks rejection. Input data
attribution and hashes are in DATA_SOURCES.md / input provenance. Complete model
file sizes, counted native state and fresh-process memory are separate measures.
No original SPECTRA source, learned default, public release or old negative result
is replaced by this experiment.
