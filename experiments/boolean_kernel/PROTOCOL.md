# Exact Boolean-domain RBF specialization — prospective protocol

Base: c9f8bb6a2ca30d6840c69b463fd6dc1ea046ad6f, the native-baseline
research branch. Existing installed source and all previous findings are retained.
No model training, new data acquisition, or hyperparameter selection in this test.

## Mechanism and exactness boundary

For model support coordinates and input coordinates each exactly 0 or 1 (signed
zero allowed), squared Euclidean distance equals integer Hamming count. At at
most 4096 features, the count and every step of the old nonnegative sum are exact
binary64 integers. Use packed 64-bit XOR/population counts for this domain only.
Do not substitute normalized Hamming distance. Original gamma multiplication,
scalar system exp, coefficient ordering, zero/tie rules and vote certificate stay.

Two opt-in modes separate bit-packed distance from a second mechanism: memoize
exp(-gamma*h) for repeated integer h WITHIN one input. This table is invalidated
for every row, so no input answer or kernel value is reused across requests or
floating-point environments. Modes: off, packed, lookup. Non-Boolean support banks
or any non-Boolean query fall back to the original arithmetic. Retain original
support representation to support that fallback; do not claim model compression.
No tolerance decides domain membership. Float32 rounding, when explicitly chosen,
still precedes domain testing, as in the existing public precision contract.

Population-count/Hamming machinery is established prior art (Faiss binary indexes,
SciPy distance definitions). Do not claim a new distance identity, SIMD theory or
frontier model. Question: can a strictly guarded specialization beat generated C
on the known categorical workload while maintaining the general interface?

## Correctness before performance

Compare actual distance, kernel and margin bits to the unchanged base source,
not only labels. Exercise all six prior seed101 cases and HAR, fractional and
subnormal fallbacks, -0, NaN/Infinity rejection, exact boundary values, empty/odd
batches, 1/63/64/65/73/128/257/4096 dimensions, table+packed combination, multiclass
and binary, independently owned workers and generation-reset behavior. Exhaustive
small Boolean vector pairs check the packed primitive. Historical API defaults
remain off; old library binaries work in off mode and reject opt-in modes clearly.

## Fixed primary comparison

Only Chess-101 in the previous seven-model panel is expected to meet the Boolean
support predicate. That selection is based on the known representation, not new
timing. Measure ALL seven models to report fallback costs as negative controls.
Rebuild upstream LIBSVM 3.37 and the identical six prior m2cgen C sources with the
same strict O3 AVX2 flags. HAR generated C remains missing under its OLD generation
budget; do not relabel the timeout as victory. Also build original SPECTRA and
new SPECTRA, and measure off/packed/lookup crossed with exhaustive/beretta_cert,
plus the existing explicit binary_stream alternative. Use prepared binary64
input views symmetrically, fresh label lists, original full-case order, chunks
1/32/256 and 11 randomized repetitions, seed20260928. Do not run unrelated CPU
jobs concurrently. Retain every measurement, mismatch, build failure and outlier.

Primary gate: on Chess-101 at batch32, lookup+beretta_cert must preserve every
bitwise/numerical contract and have median full-job ratio <=1/1.20 vs BOTH the
same-build off+beretta_cert and generated C. Report generated C versus every
profile, not only the winning one. Timing precision differences of external
code are disclosed; no tolerance excuses changed labels. This is adaptive
engineering on previously consumed inputs, not fresh independent confirmation.

Secondary: cold preparation, extra immutable/worker buffers, bounded per-input
exp calls, and raw-input preprocessing+inference. New parameter-free synthetic
cases are correctness stress only, never task-accuracy evidence. If primary fails,
keep it failed; do not silently broaden the fast domain or replace the baseline.

## Deliverable and acceptance

Additive API/source change on a focused review branch. No version bump, release
replacement or default switch. Run full applicable SVM contracts, sanitizer
checks, an actual wheel install without numerical frameworks and reconstruction
of the reported timing summaries. Remote CI outcomes are reported only once
observed. Outside reproduction and real application usefulness remain unproven.

References:
https://github.com/facebookresearch/faiss/wiki/Binary-indexes
https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html
