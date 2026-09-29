# Learned integer-metric execution contract

This is a NEW trained kernel, not faster evaluation of an unchanged unweighted
classifier. A supervised diagonal metric is projected to nonnegative integer
weights, then an SVM is fitted using exactly that projected metric. Inference
receives original uint8 feature codes; arbitrary floating inputs are rejected,
not snapped onto the grid. The existing production API/default is unchanged.

For codes q,r, weights w, maximum D and normalized scale g:

    S(q,r) = sum_j w_j (q_j-r_j)^2
    coefficient = g / (D^2 * mean(w))
    K(q,r) = exp(-coefficient * S(q,r)).

Nonnegative diagonal weights define a squared Euclidean distance on the embedding
sqrt(w_j)*q_j. Thus the ideal real-valued RBF is positive semidefinite. Rounded
finite Gram matrices need not be mathematically exact PSD matrices; numerical
PSD tests are checks, not a proof that arbitrary roundoff is positive semidefinite.

## Exact computed integer signature

The loader checks a positive weight mass, 1..4096 dimensions, 1..255 feature
maximum, 0..255 integer weights, and sum(w)*D^2+1 <= 4,194,304 table entries. All
integer products and partial sums lie below 2^22. Binary64 represents them exactly.
For AVX2, signed16 differences weighted by <=127 cannot overflow, and signed32
lane sums remain bounded. Larger weights use the scalar path. Four-byte-aligned
reads are not assumed: SIMD loads are unaligned-safe.

A common positive weight is factored out of the table index to avoid duplicate
entries, then multiplied back as an exact integer before gamma multiplication.
Every query code must be in its declared domain. Each request resets its caches;
there is no cross-input prediction cache, learned acceptance threshold or omitted
support vector based on importance. Selective voting retains the existing exact
integer winner certificate over COMPUTED pair outcomes.

## Independent reference and limitations

The independent native observer uses unchanged SPECTRA FP64 RBF source with each
coordinate repeated w_j times. Since all squared differences and sums are bounded
integers, that ordered computation has exactly the same integer distance, gamma
multiplication, system exp argument and ordered coefficient sum. This equality
requires the same deterministic math library and admitted FP environment, not
correct rounding of a real exponential or a portable-C++ promise about every libm.
Zero and multiclass tie conventions remain unchanged. Prediction correctness here
means fidelity to the NEW trained model, not ground-truth label correctness.

The implementation checks round-to-nearest. Its exercised scope is little-endian
Linux x86-64 IEEE binary64 with gradual underflow and strict noncontracted compiler
flags. Windows, ARM, altered floating-point environments and arbitrary supplied
native libraries are not newly accepted by this experiment. The old production
runtime's broader CI must not be assigned to this new experimental module.

Models use distinct SPLMET01 magic, bounded lengths, CRC and an inner SPCSVM02
weight payload. CRC and SHA256 are integrity/identity checks, not authentication.
The original model and static kernel table coexist. Inference avoids numerical
Python frameworks; training still uses quadratic Gram matrices and LIBSVM.

The exposed byte-buffer API holds the exporting object while native code runs;
callers must not mutate it concurrently. A shared session serializes operations
through the existing native-resource lease. Model files and native libraries are
trusted inputs. Resource caps and tests do not make the API a security sandbox.
