# Integer embedding and product-kernel contract

A learned linear map is normalized and rounded to signed integer coefficients
BEFORE the SVM is fitted. An identity block preserves the original coordinates.
This defines a new classifier, not an optimization promising the same predictions
as a previously fitted real-valued metric or an old unweighted SVM.

For original integer features q, the model stores A and a fixed integer offset o:

    z(q) = A q + o
    S(q,r) = sum_j (z_j(q)-z_j(r))^2
    alpha = gamma / (D^2 * trace(A.T A)/raw_features)

The offset makes coordinates nonnegative and cancels exactly from distance. Raw
codes are uint8 in0..D, D<=255. Coefficients lie in[-31,31], raw dimensions<=64,
embedded dimensions<=128; every row's L1 bound times D must be<=32767. The loader
computes a conservative maximum S and rejects unsafe model geometry. Every query
is validated before any output writes; no implicit feature rounding occurs.

## Two-table floating-point definition

Let b=ceil(bit_length(S_max)/2), B=2^b. The actual fitted kernel is

    H[k] = system_exp(-alpha * double(k*B))
    L[k] = system_exp(-alpha * double(k))
    K(S) = H[S>>b] * L[S & (B-1)]

Table entries number floor(S_max/B)+1+B, rather than S_max+1. Under the admitted
bound S_max<2^48 and at most1,048,576 total entries, all integer sums are exactly
representable in binary64. In real arithmetic this product equals the usual RBF
on a linear embedding and is PSD. In floating arithmetic its rounding differs
from exp(-alpha*S). A rounded Gram matrix need not be exactly PSD. This work does
not claim a new kernel theorem, guaranteed positive eigenvalues under arbitrary
roundoff, or bitwise substitution into an existing direct-exp classifier.

Training uses this exact product definition. The independently compiled observer
computes the integer-projected coordinates with separate scalar code, uses the
original ordered binary64 distance summation, then calls the two exponentials
directly rather than reading the candidate tables. Coefficients are accumulated
in their original sparse order, and binary/multiclass zero conventions are retained.
All reported score equality is within the same deterministic math-library/FP
context, not a guarantee across arbitrary libraries or exact-real exponentials.

## CPU arithmetic and storage

Each int16 difference is in[-32767,32767]. Its square fits int32, and each two-term
SIMD madd sum is at most2*32767^2 < INT32_MAX. When the complete bound fits signed32,
lane accumulators use int32. Otherwise each pair sum is widened to uint64 before
accumulation. Scalar and vector paths implement the same integer expression.
Unaligned-safe loads are used. No floating-point horizontal reduction is substituted.

The engine recognizes exactly diagonal A.T A using model integers and bypasses
redundant projection for those controls. A forced-project mode separately checks
that representation. Centroid order is an execution hint only; acceptance uses
the unchanged discrete vote certificate. One can preserve class decisions while
changing order of comparisons; this is not mathematical knowledge of true labels.

Support coordinates are held as int16. After original coefficient/geometry
validation and center construction, the unused duplicate binary64 feature bank is
released before const model ownership is published. No original Worker::edge/run
can be invoked through this experimental engine; its calls use the integer bank.
The separate reference keeps its own full binary64 bank. Preparation still has
transient buffers, so reduced counted persistent storage need not reduce peak RSS.

## Interface and limits

Model magic SPINT001, explicit dimensions/lengths/CRC, strict label metadata and
bounded allocations distinguish it from all earlier model formats. Libraries and
model provenance remain trusted. Integrity hashes/CRC are not signatures. The
Python borrowed-buffer API holds the exporter and serializes calls/close through
the prior native ownership lease. Callers must not mutate inputs concurrently.
No partial Python result is returned on error; internal computation is not rolled
back. Diagnostic raw pointers are not a public safe interface.

The accepted new target is Linux x86-64, GIL-enabled CPython, round-to-nearest,
gradual underflow, strict noncontracted arithmetic; AVX2 is explicitly selected.
Portable here means no AVX2 requirement on that tested x86-64 environment, not a
new Windows/ARM/macOS acceptance claim. Tests reject non-nearest FP modes.

## Learning and scientific limits

NCA, covariance whitening and Mahalanobis geometry are established. The added
local-neighbor control ignores labels in neighbor matching, but the sample of
anchors/gallery is label-stratified: it is not a wholly label-free pipeline.
A full NCA matrix need not improve classifier quality. All original/secondary
hypotheses, controls and rejected outcomes must stay visible. The two-table form
removes an allocation barrier; it does not by itself increase intelligence.
