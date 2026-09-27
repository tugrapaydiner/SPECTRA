# Optional binary streaming execution

`binary_stream` is an opt-in execution profile for the shared SVM and fitted
pipeline interfaces. It does not change the default schedule or trained model.
It specializes exactly two-class models; on multiclass models it delegates to
`beretta_cert`, preserving that scheduler's work and tie rules.

```python
with prepared_model.session() as worker:
    labels = worker.predict_buffer(binary64_rows, schedule="binary_stream")
    checked = worker.predict_with_certificate(one_row, schedule="binary_stream")

with prepared_pipeline.session() as worker:
    labels = worker.predict_fused(raw_rows, schedule="binary_stream")
```

These examples assume already prepared model/pipeline objects and valid inputs.
The existing buffer, schema, precision, model binding, worker ownership and
unsupported-input rules remain in force. Original `spectra.svm.Session` is unchanged.
Rebuild BOTH native libraries after updating. The CPython preprocessing operation
ABI is now 4 so stale extensions are rejected before binding a worker.

## Why specialize binary inference?

One two-class model has one pairwise classifier. There is no other class pair to
reuse its kernel cache, and no choice of class comparisons to schedule. The new
path streams kernel evaluations into the original ordered coefficient sum and
avoids that cache/centroid machinery. It still checks every kernel and obtains
the ordinary single-pair vote certificate. No support vector is omitted merely
because it appears unimportant. Exactly zero coefficients remain omitted as in
the existing sparse exhaustive implementation.

In AVX2 builds, a direct-distance batch of four or more rows puts four DIFFERENT
inputs in the four vector lanes. Each lane sums dimensions and coefficients in
the same order as the old scalar decision. It does not split a reduction across
lanes, fuse multiply/add or replace scalar system exp. Input features are packed
once per group; coefficients/support values can then be shared across those four
independent calculations. Partial groups use streaming support-vector packets.
There is no cross-input kernel cache or answer reuse.

Feature-table models retain per-row exact dictionaries and the support-packet
streaming path. Portable builds do not claim AVX2 row-lane execution. Numerical
exactness is relative to the existing deterministic computed binary64 outcome,
not an exact-real exponential or a blanket guarantee for other compilers/libm.

## Work and storage

The streamed path still performs ALL nonzero kernel/term evaluations for the binary
pair. Any speed change is execution organization, not fewer mathematical kernels.
Counts and single-row certificate output report the same work semantics. The
strictly ordered score uses `score >= 0` for the original binary SVC class convention;
multiclass zero/tie conventions are not substituted.

The AVX2 direct batch uses one temporary `4 * features` binary64 packing buffer,
allocated once per batch (at most 128 KiB under the existing 4096-feature cap).
This is additional to the existing worker/model storage and, for fused input,
the separately bounded preprocessing feature tile. It is not counted as persistent
worker scratch or claimed to reduce total memory. Fused preprocessing plus this
packing can therefore use up to 256 KiB of those two temporary feature buffers.
No per-support/per-row heap allocation is introduced. The old paths allocate no
new packing buffer. No concurrent-throughput, cold-start, RSS or energy claim is
implied by single-core measurements.

## Evidence

See `experiments/binary_stream/PROTOCOL.md`. All default, exhaustive and streaming
arms use matched values and unchanged models. Record portable results and table
fallbacks separately, include single-row regressions and multiclass controls, and
report complete raw-input pipeline costs as well as prepared-input batches. These
are adaptive systems experiments on retained convenience models, not a new
accuracy benchmark, newly invented SIMD technique or outside reproduction.
