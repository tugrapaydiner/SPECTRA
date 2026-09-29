# Native tree certificates: numerical and ownership contract

This experiment preserves a supplied classifier's class index. It does not make
its learned decision more accurate, establish the true label, or accept arbitrary
CatBoost features/backends. Only numeric oblivious trees, bounded uint8 input,
positive common output scale and explicit class bias are supported. Single-output
sources are interpreted as scores `[0, raw_score]`, with smaller-index ties.

## Exact-rational source verification

The unchanged `reference/certificate_oracle.py` compiles and verifies the policy.
Its SHA256 is `704556048f58844787a89c2079710abbce949cc7f5d427e821fcb25f64f5c6a5`.
Source numbers are interpreted as their stored binary64 values, not ideal decimal
literals. Feature thresholds must be exactly representable binary32 values. For
integer input, comparing with `floor(border)` (with outer sentinels) preserves the
original strict-greater-than predicate, including repeated and constant splits.

For each leaf, subtract the same leaf's class-zero value. This common offset does
not change the winning class. The positive source scale is factored out and bias
contrasts are divided by it. A model-wide power-of-two step makes every quantized
contrast an int8/int16 value. Quantization uses exact rational nearest-even rounding.

The compiler sums the minimum and maximum residual for each class over ALL leaves
of each tree, plus the bias residual. It also bounds the source's own roundoff.
All bound arithmetic is exact rational; stored integer endpoints round outward
at 20 fractional bits. A proof uses neither fitting examples nor a confidence
threshold. Optional pairwise residual bounds exist for the reference's bounded
small cases; the four measured large models use classwise box bounds.

The source arithmetic assumption is IEEE binary64 nearest-even, gradual underflow,
no overflow, at most T-1 additions per class to sum T leaves, then common positive
scale multiplication and bias addition. The reference uses conservative gamma
bounds and an underflow term. This is a stated conditional contract, NOT proof
that every CatBoost backend, compiler setting or future library follows it.
Actual source CBMs are separately checked against JSON leaf values/borders/scale
and all retained routing. Both official CPU libraries preserve all recorded labels.

## Certification, early exit and refinement

Completed integer sums plus outward residual/roundoff bounds enclose the source
scores up to a common offset and positive factor. A winner is accepted only if its
lower endpoint exceeds every rival's upper endpoint, accounting for lower-index
ties. Otherwise the API returns -1 (UNRESOLVED), never its guessed quantized winner.

Early checkpoints add conservative per-class suffix extrema from all remaining
trees. They may reduce leaf additions but the tiled implementation has already
routed the whole tile: do not call that eliminated routing. Scalar mode routes
on demand and retains encountered leaf indices for refinement. Logical evaluated
leaf-vector counts and routed-tree counts are recorded separately.

The owning refinement pipeline may load an8-bit compact model, the corresponding
16-bit model and one official CatBoost model. Both compact models must bind the
same source and identical route geometry. The second pass reuses route indices.
Unresolved16-bit rows are gathered for an actual official C API call. Full-coverage
mode therefore retains the original model and library: compact file savings are
NOT full-pipeline memory savings. An explicit16-bit-first profile is also available.

## Packed representation and implementation

SPCERT02 includes source/oracle digests, dimensions, scalar-policy metadata,
unique predicates, tree descriptors, integer leaves/biases and outward bounds.
A checksum-valid structural load is not a numerical proof. `VerifiedCompact`
reconstructs the exact packed bytes from source before accepting them. SHA256/CRC
check identity/corruption, not authorship, malicious replacement or actual clocks.
Use trusted source and native libraries; private-state tampering is not sandboxed.

Limits:1..256 features,2..64 classes,1..4096 trees, depth<=12,500000 leaf scalars,
64MiB packed files, input batches<=65536 rows/8million elements. Accumulator and
certificate envelopes are checked. Intermediate integer sums fit int32; certificate
comparisons fit int64. Negative input codes, floating buffers and readonly buffers
are refused rather than rounded or coerced.

Vector routing transposes a bounded32-row tile, evaluates predicates in SIMD lanes,
and combines split bits without per-row branching on supported depths. Depths>8
retain general routing. End-only register accumulation retains original class
contrasts exactly. Unused trailing lanes read within eight explicitly reserved
guard elements and never participate in class selection. Their allocated storage
is counted. Portable/general code paths remain. SIMD and partial-sum bounds are
established techniques, not a first-invention claim.

The prototype native model owns its arrays. Each operation has private scratch and
uses the existing `_NativeOwner` lease for serialization and deferred same-thread
close. Caller buffers stay exported and must not be mutated concurrently. Results,
work arrays and statistics are copied to caller memory only after successful whole
batch validation/inference/fallback. A forced process kill is not rollback.

## Supported acceptance boundary

Freshly tested here: Linux x86-64, CPython3.13 locally, explicit portable and AVX2
builds and AVX2 undefined-behavior sanitizer. No Windows/ARM, ASan, free-threaded
Python, live service, latency-tail, GPU or original historical-suite acceptance is
inherited from other SPECTRA branches. CatBoost libraries1.2.8 and1.2.10 are unmodified
upstream CPU assets with their own licensing and dependencies.

Verification and native model preparation are outside warm timing. The exact
Python verifier can be substantially more expensive than a native prediction;
it is not a fast cold-start claim. Storage reports distinguish packed bytes,
counted compact native allocation and full fallback dependencies. Process RSS is
not inferred from these counts.
