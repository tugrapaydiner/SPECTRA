# SVM execution

`spectra.svm` runs exported RBF support-vector classifiers natively, without NumPy,
scikit-learn or a deep-learning framework at inference time. Predictions use
**certified pairwise voting**: one-vs-one comparisons are evaluated only until
integer vote bounds settle the winner, and the settled bounds can be checked
independently.

Exporting needs scikit-learn in the training environment (`pip install ".[svm]"`);
inference needs only the compiled runtime.

## Export and run

```python
# Training environment
from spectra.svm_export import export_prepared_svc
export_prepared_svc(fitted_svc, "model.srt")
```

```python
# Inference
from spectra.svm import build_runtime
from spectra.svm_shared import PreparedModel

library = build_runtime("build/svm")                 # explicit C++17 build
model = PreparedModel("model.srt", library, input_dtype="float64")
with model.session() as worker:
    result = worker.predict_with_certificate([0.0] * worker.features)
    print(result.label, result.class_index)
    labels = worker.predict_many(rows)
model.close()
```

`export_prepared_svc` writes the `SPCSVM02` format: binary64 coefficients, support
vectors, gamma, intercepts, the feature count (1–4,096) and an ordered mapping of
integer or UTF-8 string labels (2–128 classes). Loading never unpickles anything;
a CRC detects corruption. Export does not include preprocessing; apply the
original feature transform first, or use a [pipeline](#fitted-preprocessing).

`input_dtype` selects whether inputs are rounded to binary32 (`"float32"`, default)
or passed as binary64. Compare against other implementations only on identically
rounded inputs.

A `PreparedModel` holds immutable native model data; each `session()` worker owns
its own kernel cache, vote bounds and scratch, so separate workers run in
parallel. Closing the model stops new workers but does not invalidate existing
ones. The legacy `spectra.svm.Session` (16-feature `SPCSVM01` models written by
`export_svc`) remains available.

## Schedules and certificates

`predict_with_certificate` returns the label, the class index used for
deterministic tie-breaking, every evaluated pair outcome and work counters.
`spectra.svm.verify_certificate` checks that the reported outcomes settle the
winner; it does not re-evaluate the model.

| Schedule | Behaviour |
|---|---|
| `beretta_cert` (default) | tournament-champion selection with early certification |
| `beretta_array`, `beretta_ordered` | variants of the same selector |
| `knockout_cert`, `lazy` | alternative certified orderings |
| `exhaustive` | evaluate every pair |
| `binary_stream` | specialised two-class path; falls back to `beretta_cert` for multiclass |

A hint (class index) may change evaluation order but never the accepted winner.
Batches are limited to 65,536 rows and 8 million feature values, and every input
is validated before any output is written.

## Decision receipts

```python
from spectra.svm_receipt import create_receipt, verify_receipt

receipt = create_receipt(worker, features, schedule="binary_stream")
check = verify_receipt("model.srt", receipt, expected_input=features,
                       input_dtype="float64")
assert check["verified"]
```

A receipt binds a decision to the exact model bytes and input. The verifier
decodes the model independently, recomputes every disclosed pair in ordered
binary64 Python arithmetic and checks that undisclosed pairs cannot change the
winner. Unlike a certificate, a receipt cannot be satisfied by forged pair
outcomes. Replay work is bounded by `max_kernel_work` (default 8 million
kernel-feature operations). Receipts contain feature values; treat them as data.

## Fitted preprocessing

```python
# Training environment
from spectra.svm_pipeline_export import export_pipeline
export_pipeline(column_transformer, svc, "deployment")
```

```python
from spectra.svm import build_runtime
from spectra.svm_pipeline import PreparedPipeline
from spectra.svm_preprocess_native import build_preprocessor

svm = build_runtime("build/svm")
pre = build_preprocessor("build/preprocess")          # optional native preprocessing
with PreparedPipeline("deployment", svm, preprocessor_library=pre) as model:
    with model.session() as worker:
        labels = worker.predict_fused(raw_rows, columns=model.columns)
```

`export_pipeline` writes `preprocessing.json` and `model.srt` from a fitted
scikit-learn `ColumnTransformer` and `SVC`. Supported branches: median
`SimpleImputer` followed by `StandardScaler` for numeric columns, and dense
`OneHotEncoder` (string categories, `handle_unknown` of `ignore` or `error`) for
categorical columns. Operations replay in the original order with binary64
constants; no folding or refitting. Rows are ordered sequences in `model.columns`
order. Without the native preprocessor, use `predict_many`; with it,
`predict_fused` transforms and classifies in bounded tiles.

## Streaming CLI

```bash
spectra svm run deployment --library build/svm/libspectra_svm.so \
    --preprocessor build/preprocess/_spectra_preprocess.so \
    --input rows.jsonl --output predictions.jsonl --batch-rows 128
```

Each input line is a JSON array in the plan's column order (numbers, strings,
`null` for missing numerics). Input is processed in bounded chunks and the output
file is published atomically only after every row succeeds; an existing path is
never overwritten. The first output record identifies the model, plan and limits;
the last records row counts and SHA-256 hashes of the input and output. Use
`--engine two_stage` without a native preprocessor.

## Platforms and references

Builds are supported on Linux x86-64 and ARM64 and on Windows x64 (MSVC), with
strict floating-point flags (`-fno-fast-math -ffp-contract=off`, `/fp:strict`).
`target="avx2"` is an explicit, hardware-specific option. Build on the machine that
will run the model.

The selector follows Beretta et al., *An Optimal Algorithm for Finding Champions
in Tournament Graphs* ([arXiv:2111.13621](https://arxiv.org/abs/2111.13621)). The
model layout follows the
[scikit-learn one-vs-one SVC convention](https://scikit-learn.org/stable/modules/svm.html#multi-class-classification).
