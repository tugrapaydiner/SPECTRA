# Certificate and deployment contract

The output is a **source-model class index or UNRESOLVED (-1)**, not a claim about
the ground-truth label. Full refinement forwards unresolved rows to the original
CatBoost model using its unmodified native CPU prediction library. Both models
remain loaded. No calibrated probability or empirical confidence threshold is used.

## Exact offline reasoning

`certificate_oracle.py` is the unchanged exact-rational checkpoint, SHA256
704556048f58844787a89c2079710abbce949cc7f5d427e821fcb25f64f5c6a5. It parses only
numeric oblivious trees, exact binary32 thresholds and bounded integer features.
Subtracting each leaf's class-0 value is a real-arithmetic class-contrast change,
not an assertion that reassociation leaves floating-point scores bit-identical.
The compiler explicitly bounds source summation/scale/bias roundoff as well as
quantization error. It rounds class contrasts to a common power-of-two step and
stores residual lower/upper bounds with a finer integer denominator (2^20).
All rounding decisions and outward error calculations use exact rational arithmetic.
The binary compiler deterministically reproduces these objects; `verify_binary`
rebuilds the entire object from the source JSON and compares every output byte.

After k trees, each possible completed normalized source score lies within its
integer accumulated score plus the remaining-tree lower/upper contribution and
its saved quantization/source-arithmetic envelope. A class can be returned only
when its lower bound beats every other upper bound, using the documented first-
class tie convention. Optional correlated pairwise residuals can tighten a completed
comparison; the real four-model panel uses the smaller classwise box bounds.
If no certificate exists, completing the compact model still returns UNRESOLVED.
A raw approximate argmax is exposed only for diagnostics, never silently promoted.

The admitted source arithmetic is IEEE binary64 round-to-nearest with gradual
underflow, no overflow, at most T-1 additions to sum T leaf values per class,
followed by a common positive scale multiplication and a bias addition. Arbitrary
summation trees over those leaves are enclosed. Source magnitudes are bounded
conservatively below 2^900; the additive subnormal allowance remains explicit.
This is a stated contract, not a proof that every CatBoost backend satisfies it.
The tested CatBoost CPU library and C++ export are independently compared with
source scores and sampled exact-rational roundoff intervals. GPU, arbitrary
backend transformations, negative scales, categorical/CTR/text/embedding features,
missing/NaN inputs and asymmetric trees are not admitted.

## Native format and safety

SPTCQ001 stores signed8/16-bit class-contrast leaves, integer bias/error bounds,
canonical shared predicates and tree descriptors. SPTCF001 is a separate original
binary64 evaluator. Both bind source JSON identity. File CRC detects corruption;
expected SHA256 values must come from a trusted compiler receipt. Neither CRC nor
SHA256 authenticates a maliciously replaced source, verifier or shared library.
Runtime validation of bounds is not a substitute for offline source verification.

Runtime caps: 256 input features, 64 classes, 4,096 trees, depth12, 500,000 leaf
scalars, 64MiB binary, 65,536 rows and 8,000,000 input elements per call. Quantized
magnitudes exclude the signed minimum and accumulated scores stay in int32. Suffix
bounds and finer residual arithmetic have explicit int64 overflow checks. Queries
are original uint8 codes in the declared domain; no clipping or snapping. Structural
and input errors fail before predictions are written. A raw C pointer interface
still requires trusted pointers/caller discipline; this is not a security sandbox.

Models are immutable after construction; per-call temporary arrays are private.
Python buffers are exported for the entire call, and callers must not mutate them
concurrently. Existing SPECTRA native-resource leases serialize operations and
safely defer same-thread close until a borrowed pointer is released. The fused
refinement handle owns compact and original models together, not two per-call
Python ownership contexts. It validates and converts all unresolved inputs inside
the timed call, invoking the actual original library rather than a new approximation.

## Same-function execution and limits

The first implementation is a retained row-major control. The optimized path tiles
32 input rows and traverses a tree's leaf table across those rows before advancing.
Common depth6 and class counts receive compile-time loop specialization. Each row
still follows identical predicates, quantized additions, source envelope and tie
rules. Tails and other supported geometries use checked generic paths. Early
checkpoint execution remains a separate ablation; it often costs more here.

Acceptance applies to Linux x86-64 little-endian binary64. Portable own-code builds
omit AVX2 flags; they were tested on the same AVX2-capable host, not an old physical
CPU. No new Windows/ARM, ASan, free-threaded Python, power/energy, service-tail or
whole-project acceptance is implied. The official CPU shared library is pinned
CatBoost1.2.8; the reference export and native evaluator are separate controls.
The compact backend does not need CatBoost or a numerical framework. The complete
fallback pipeline requires the native CatBoost library, original model and system
runtime dependencies, but not CatBoost's Python package.
