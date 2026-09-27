# HAR execution study and fixed linear deployment

Read [PROTOCOL.md](PROTOCOL.md), [RESULTS.md](RESULTS.md) and
[DEVIATIONS.md](DEVIATIONS.md) together. The primary RBF packing hypothesis failed;
the installed addition is the explicitly separate fixed-model linear evaluator.
Existing public SVM numerical code and defaults remain unchanged.

## Reproduce without selecting a new result

The evidence archive is a study-root directory containing `inputs/har`, `models`,
`evaluation`, `linear-export`, `linear-build`, `linear-evaluation`, `application`,
rejected prototypes and source snapshots. Its UCI feature files retain original
bytes, authorship and CC BY4.0 attribution. Training artifacts include trusted
joblib files; NEVER load them from an untrusted source. Deployment loads no pickle.

The original environment was Python3.13.5, NumPy2.3.5, SciPy1.17.0, scikit-learn1.8.0,
joblib1.5.3 and threadpoolctl3.6.0 on Linux, GCC14.2. No GPU. Pin numerical threads
before import: OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1. Use fresh
output directories. A different library/CPU can change fitted parameters or timing;
record that difference rather than calling it the original frozen result.

From the checkout, with STUDY set to a fresh root containing the original inputs:

```bash
python experiments/har_execution/fit.py --data "$STUDY/inputs/har" --out "$STUDY/models"
python -c "from spectra.svm import build_runtime; print(build_runtime('$STUDY/build-baseline-avx2',target='avx2'))"
python experiments/har_execution/evaluate.py --data "$STUDY/inputs/har" \
  --models "$STUDY/models" --library "$STUDY/build-baseline-avx2/libspectra_svm.so" \
  --out "$STUDY/evaluation"
```

The AVX2 build requires a capable x86-64 CPU. Use portable elsewhere, reporting
that different implementation/host rather than assigning it the observed timings.
The evaluation locks the model/source/data inventory before loading test vectors.
No original locked result is replaced when a new command runs.

For the fixed LinearSVC, export the already fitted trusted `linear_c1/model.joblib`
with `spectra.linear.export_linear_svc` to `$STUDY/linear-export`, then call
`build_linear` with a fresh `$STUDY/linear-build` directory. This reuses original
weights and does not train a new model. See [the public API](../../docs/LINEAR.md).

```bash
python experiments/har_execution/evaluate_linear.py --data "$STUDY/inputs/har" \
  --model "$STUDY/models/linear_c1/model.joblib" --bundle "$STUDY/linear-export" \
  --library "$STUDY/linear-build/libspectra_linear.so" --out "$STUDY/linear-evaluation"
python experiments/har_execution/bench_application.py --root "$STUDY" --source .
```

The parameterized application driver reproduces the measured process experiment;
its exact original fixed-path driver and interruption/resume receipt are retained
separately. It deliberately uses isolated child startup and feeds the same raw
feature JSONL to all arms. It is a Linux example harness, not support for linear
bundles in `spectra svm run` and not sensor feature extraction.

## Verify the original packet independently

```bash
python experiments/har_execution/audit.py --root /extracted/evidence \
  --source /extracted/evidence/source --out /new/audit.json
```

The standard-library auditor validates original subject/label identities, fixed
model files, complete timing grids, output hashes and aggregate arithmetic without
importing the runtime or benchmark aggregation. It maps the original recorded path
forms onto the supplied packet root; new experiments can retain their own identity
map. It is not independent timer attestation or outside researcher replication.
Numerical ordered-score replay is separately included in the extracted SDK.

Rejected prototype source/patches and all training-probe observations are under
`evidence/rejected`; [rejected/INDEX.json](rejected/INDEX.json) binds them. They are
not part of the installed runtime. Reproduction should never swap these rejected
prototypes into the public engine or omit the stronger NumPy/linear controls.
