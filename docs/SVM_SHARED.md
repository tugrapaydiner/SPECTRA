# Shared and generic SVM execution

`spectra.svm_shared.PreparedModel` is an additive API. The original 16-feature
`spectra.svm.Session`, existing SPC SVM01 files and default selector are unchanged.
The new path supports 1–4096 features, up to8 million stored support-feature values,
2–128 classes, and unique integer or UTF-8 string labels. This is still a dense
RBF-SVC executor, not arbitrary sklearn pipelines, other kernels or probabilities.

## Export and precision

Export a trusted fitted stock `sklearn.svm.SVC` in a training environment:

```python
from spectra.svm_export import export_prepared_svc
export_prepared_svc(fitted_svc, "model.srt")
```

`SPCSVM02` stores feature count and an ordered label mapping alongside original
binary64 coefficients, support vectors, gamma and bias. Inference reads the file
once into bounded bytes; label parsing and native preparation use those same bytes.
The CRC detects corruption, not hostile model provenance. No pickle is loaded.
Export does not embed scaling or other preprocessing. Apply the original feature
transform before inference. Integer labels are signed64-bit; string labels are
UTF-8, at most256 bytes each, with no mixed-type mapping. Existing format01 models
are interpreted as16 features with labels0..C-1.

Select input precision explicitly: `float32` (the default) preserves the old API's
rounding; `float64` passes finite Python binary64 values without binary32 rounding.
A different input precision can change a decision close to the boundary. Comparisons
with sklearn must use the SAME transformed/rounded inputs. No universal bitwise
match with every sklearn/compiler build is asserted.

## Prepare once, keep independent workers

```python
from spectra.svm import build_runtime
from spectra.svm_shared import PreparedModel

library = build_runtime("svm-shared-build")  # explicit, fresh directory
model = PreparedModel("model.srt", library, input_dtype="float64", tables=False)
worker_a = model.session()
worker_b = model.session()
model.close()  # releases the owner's reference; existing workers remain usable
try:
    prediction = worker_a.predict_with_certificate([0.0] * worker_a.features)
    print(prediction.label, prediction.class_index)
    batch = worker_b.predict_many([[0.0] * worker_b.features])
finally:
    worker_a.close()
    worker_b.close()
```

Prepared support vectors, sparse coefficient arrays, exact dictionaries and
centroids are const native data held through shared ownership. Every worker owns
its own kernel cache, generation counters, vote bounds and feature-distance
scratch. No result, context or kernel value is reused across different inputs.

Worker methods and close use a per-worker lock. Separate workers may execute in
parallel; closing the owner cannot destroy data still retained by workers. Creating
a new worker after owner.close() raises. Shared library files are executable code
and must be trusted. Do not mutate private Python fields or raw C pointers.

`info` distinguishes `shared_prepared_bytes` (count once per prepared model) from
`worker_scratch_bytes` (count once per worker). Counts include the described C++
objects/vector capacities, not allocator metadata, shared_ptr control blocks,
Python objects, library mappings, loading peaks or process RSS. IDs identify shared
objects within one process, not persistent model identities. The model SHA256 is
reported separately.

## Decisions and certificates

`predict` and `predict_many` return original labels; `predict_index` returns the
class index used for deterministic vote tie-breaking. `predict_with_certificate`
returns both, all supplied pair outcomes and work counters. A `hint` is always a
CLASS INDEX, not a label. Wrong valid hints may affect speed, never acceptance.

The independent certificate checker verifies integer vote bounds, not model
origin, the truth of computed pair outcomes or real-world label correctness.
Ordinary schedules are those in the original API. New float64/generic arithmetic
is separately tested; the four historical numerical source files are unchanged.

Batches are capped by both65536 rows and8 million feature elements. All native
inputs are checked before output writes. A native execution error returns no
Python result and invalidates its certificate; raw C output on an unexpected
execution error is unspecified. Batch calls, including a one-row batch, never
expose an implicit last-row certificate. Round-to-nearest is required.

## Cost-aware scheduling: failed promotion, opt-in only

`cost_aware` is a first experimental scheduler, not the default. It selects a
possible-win pivot and compares incident edges by the estimate
`uncached_kernels*(features+16)+nonzero_terms`. Cache lookups and certificate checks
are charged; cost estimates can order queries but can never accept a winner.

The frozen first comparison failed its>=1.10x improvement gate on both retained
models. It performed MORE pair/kernel work plus extra scanning. See the receipt
in the accompanying report. Do not present this prototype as an efficiency win or
as a novel optimal selector. Established adaptive tournament selection and minimal
support literature remain relevant:

- Beretta et al., https://arxiv.org/abs/2111.13621
- Contet, Grandi and Mengin, https://arxiv.org/abs/2509.09312

The useful result in this change is generic model support and shared preparation,
not a new learned architecture, higher accuracy, external reproduction or release.
The old API remains preferable where its narrower interface is faster. Linux is
the only exercised build path; Windows/ARM performance and numerical acceptance
are not implied by support for arbitrary feature dimensions.

## Optional binary execution profile

`binary_stream` specializes the one-pair binary case and falls back to
`beretta_cert` for multiclass models. It is opt-in, not a new default. See
[SVM_BINARY_STREAM.md](SVM_BINARY_STREAM.md) for numerical order, temporary-buffer,
feature-table and CPU instruction-set boundaries. Rebuild the runtime and optional
preprocessing extension after updating.
