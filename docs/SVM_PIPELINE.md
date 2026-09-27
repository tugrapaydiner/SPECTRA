# Fitted preprocessing and certified SVM inference

`spectra.svm_pipeline` executes a restricted, explicitly exported preprocessing
plan and an existing SVM without NumPy, pandas, scikit-learn or PyTorch at inference.
The numerical SVM engine, weights and vote-certificate contract are unchanged.
This is an optional deployment path, not a newly trained model or arbitrary
scikit-learn pipeline interpreter.

## Export once in a trusted training environment

Given a supported fitted ColumnTransformer `preprocessor` and a stock fitted
RBF `SVC` named `classifier`:

```python
from spectra.svm_pipeline_export import export_pipeline
export_pipeline(preprocessor, classifier, "deployment")
```

No fitting occurs. The fresh directory contains `preprocessing.json` and
`model.srt`. Existing destinations are never overwritten. A failed export can
leave a partial directory; inspect it and use a new destination. Never load an
untrusted training pickle. Deployment reads bounded inert files, not pickles,
Python code, or transformer import paths.

## Deploy raw schema rows

```python
from spectra.svm import build_runtime
from spectra.svm_pipeline import PreparedPipeline

library = build_runtime("native-build")  # explicit local C++17 compilation
with PreparedPipeline("deployment", library) as model:
    print(model.columns)  # exact fitted input column order
    with model.session() as worker:
        # raw_rows must contain values in model.columns order.
        labels = worker.predict_many(raw_rows, columns=model.columns)
        receipt = worker.predict_with_certificate(raw_rows[0])
```

Only Linux native builds have been accepted so far. AVX2 remains an explicit
hardware choice. The supplied native library is executable code and must be
trusted. The prepared owner can create multiple independent workers; closing it
prevents new workers but existing workers retain the model. Scratch and outputs
are per request/worker. This does not add parallel preprocessing or improve the
Python GIL's throughput.

## Supported preprocessing

The exporter accepts stock, fitted, dense ColumnTransformer branches with dropped
remainder and no transformer weights:

- Numeric: SimpleImputer with median strategy and NaN missing values, then
  StandardScaler with centering and scaling enabled. No missing-value indicators,
  dropped all-empty features, subclass/instance transform overrides, or extra steps.
- Categorical: dense binary64 OneHotEncoder with string categories, no dropped or
  grouped/infrequent categories, and unknown handling `ignore` or `error`.

The fitted input names must be unique strings. Numeric None and NaN use the stored
training imputation constant. Infinity, booleans and numeric strings are rejected.
Categories must be strings; unknown `ignore` produces the trained feature block's
all-zero vector. Application-specific tokenization, categorical missing-token
creation, unit conversion, raw-file parsing and domain feature extraction are
outside this format. Thus `raw_rows` means rows at the fitted transformer's input
boundary, not unprocessed images or arbitrary source records.

The plan retains category order and output offsets. Binary64 constants are stored
as canonical hexadecimal strings. Imputation, subtraction and division execute
separately, in the original order; no reciprocal multiplication, affine folding,
quantization, or refitting is introduced. The runtime checks model digest and
feature count against the bytes actually loaded by PreparedModel. Digests ensure
identity, not author authenticity or correctness of training.

## Bounds and ownership

Metadata is capped at 4 MiB; input and output dimensions at 4,096. A request is
bounded by 65,536 rows and 8 million raw/output elements. Row/schema iterators are
consumed only to the relevant cap. Duplicate JSON keys, unsupported operations,
invalid offsets, nonfinite constants and inconsistent model binding are rejected.
Rows are ordered sequences, not mappings or DataFrames. A supplied `columns`
sequence must match exactly; it is validation, not automatic reordering.

Each transform returns a fresh binary64 buffer. The input rows are never modified;
callers must not mutate them concurrently. The plan's internal operations are
immutable tuples and mapping proxies, but Python objects are not an adversarial
sandbox. Native pointers, executable libraries and concurrently mutated source
objects during export remain outside the security guarantee.

`predict_with_certificate` additionally copies and verifies a partial vote trace.
The certificate settles the engine's computed SVM vote, not the actual truth of the
label or exact-real-valued exponential arithmetic. Its additional work is not
included in class-only timing claims.

## Performance boundary

A specialized standard-library plan avoids generic transformer/DataFrame dispatch
and removes numerical-framework dependencies. A block-vectorized NumPy control
can still be faster for larger numeric batches. Keep `SharedSession.predict_buffer`
for applications that already have suitable transformed arrays. Do not claim the
raw-row path universally wins, or multiply its gains by earlier kernel results.

The accompanying raw-pipeline evidence records all 18 frozen six-domain models,
full feature/decision fidelity, one-row and batch timings, a stronger vectorized
NumPy control, the initial failed gate and separate corrected measurements. The
research scripts and old cross-domain records are supplied in that evidence;
this package integration does not rebrand them as new untouched accuracy tests.

See [shared model ownership](SVM_SHARED.md) and [SVM numerical limits](SVM.md).
