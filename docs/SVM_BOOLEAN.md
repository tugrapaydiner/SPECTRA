# Optional exact Boolean-domain RBF execution

This opt-in specialization preserves the existing computed decision. It does not
quantize arbitrary real features or change the trained classifier.

```python
from spectra.svm import build_runtime
from spectra.svm_shared import PreparedModel

library = build_runtime('fresh-native-build', target='portable')
with PreparedModel('model.srt', library, input_dtype='float64', boolean='lookup') as model:
    with model.session() as worker:
        labels = worker.predict_many(binary_rows)
        print(model.info['boolean_enabled'], worker.boolean_stats)
```

The example assumes a trusted supported exported model and rows in its original
feature order. `boolean='off'` is the default. `packed` enables exact bit-packed
distances; `lookup` additionally reuses repeated exponential values within ONE
input. `PreparedPipeline(..., boolean='lookup')` offers the same option AFTER the
existing fitted preprocessing; all pipeline/worker/schema/lifetime rules remain.
Rebuild the native SVM library. Off mode still accepts older libraries; requesting
an optional mode from an old library raises instead of silently changing behavior.
The preprocessing operation ABI remains 4; its numerical source is unchanged.

## Exact eligibility and arithmetic

Every support coordinate must be exactly zero or one; signed zero is accepted.
Then each incoming, post-precision-conversion input is checked by the same exact
predicate. Any fractional, negative, nonbinary or smallest nonzero subnormal value
uses the original floating-point path. There is no tolerance or implicit rounding
to enter the domain. Explicit `input_dtype='float32'` still performs its documented
rounding BEFORE this guard; choose float64 to preserve supplied binary64 values.

For this Boolean domain, `(x_i - s_i)^2` is exactly 0 or 1. With at most 4096
features, the original ordered nonnegative sum is an exactly representable integer.
An XOR/population count gives that same integer, NOT normalized Hamming distance.
Gamma multiplication and the system scalar `exp` are unchanged. Each pair's sparse
coefficients are still accumulated in original order; no fused multiply-add or
reassociation is enabled. Binary/multiclass zero and voting tie conventions stay.

The packed representation uses a portable unsigned population-count implementation.
An optimizing compiler may select an appropriate instruction under the explicitly
chosen target flags; no unconditional AVX2 instruction is added to portable builds.
Population-count/Hamming methods are established prior art, not a new theory.

The `lookup` mode lazily evaluates `exp(-gamma*h)` once for each integer distance h
actually encountered in that input. Generation markers invalidate all entries for
every new row, including repeated identical rows. Marker wrap clears both cache
inventories. No answer or exponential result is retained for reuse by a later
input. The existing round-to-nearest requirement applies. This is same-environment
computed FP64 fidelity, not an exact-real transcendental proof or a cross-libm claim.

## Costs and compatibility

The old support representation remains for nonbinary queries. The extra packed
bank costs `8 * supports * ceil(features/64)` bytes, plus object overhead. Lookup
workers add per-distance values and generation markers, and a packed query buffer.
This is NOT a model-compression or lower-process-RAM claim. Nonqualifying models
allocate no packed bank or per-distance arrays, but the added object fields still
exist. Preparation scans and packing are additional cold work; warm timing excludes
model loading and preparation. No automatic per-model performance choice is made.

`model.info` reports the mode, eligibility, packed words per support and counted
packed-bank bytes. `worker.boolean_stats` reports last-executed-row activity, word
comparisons, exponential calls and lookup hits. It is a diagnostic snapshot, not
a batch aggregate, signature or certificate. Request the existing decision receipt
for independent numerical replay. Reading diagnostics does not authenticate input
or make the predicted real-world label correct.

The opt-in packed path disables the separate direct binary four-input-lane path
for eligible models so domain checking and lookup cannot be bypassed. Other
schedules retain their selection semantics. Real-valued fallback uses the old
ordered formulas; its speed can differ due to dispatch, object layout or code size.
Both improvements and regressions belong in the measurements.

## Evidence boundary

See `experiments/boolean_kernel/PROTOCOL.md`, `AMENDMENT.md` and `RESULTS.md`.
Only Chess-101 among the seven retained models has an eligible support bank.
This experiment intentionally targets that known representation; it is not a fresh
confirmation dataset, general classifier improvement or evidence that an SVM is
preferable to a linear model on HAR. Generated C remains stronger on several other
models, and its earlier HAR generation timeout remains a missing comparison.

The independent test observer compares actual distances, kernels and pair margins
to the original source, not only labels. The timing auditor checks source/model
identities and every fixed record without importing the runtime. Neither is
outside researcher replication. Default execution, existing release assets and
trained model bytes are unchanged.

References:
https://github.com/facebookresearch/faiss/wiki/Binary-indexes
https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html
