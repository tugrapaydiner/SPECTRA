# Exact dyadic reconstruction, not a changed certificate

The unchanged rational oracle is the numerical specification. The new `dyadic.py`
implementation must return the same complete object and packed bytes. This is an
implementation of exact arithmetic, not a new quantization theorem, approximation,
calibration or cached/trusted-verification bypass.

For every finite stored binary64 leaf v there are integers n and K with
v=n/2^K. The parser chooses a conservative common K from the exact binary
exponents, then converts with float.as_integer_ratio(). Conversion has no rounding:
only multiplication by an integer power of two. Subnormal leaves and negative
values are supported within the reference's original source/envelope limits.

Let e be the ORIGINAL quantization step exponent. For a multiclass leaf, its
class-zero contrast in quantization units is (n_c-n_0)/2^(K+e). If K+e<0,
shift the numerator left and use denominator one. Nearest-even division uses
integer quotient and remainder, including the negative-halfway cases. If q is
the chosen integer coefficient, r=n_c-n_0-q*2^(K+e) is an EXACT residual numerator.
Each tree's per-class minima/maxima and correlated pairwise residual extrema have
the same positive denominator, so integer comparison and summation reproduce the
rational extrema exactly. This eliminates per-leaf Fraction normalization and the
large retained Fraction/residual object graph.

The source bias normalized by a general positive binary64 scale need not be dyadic
(e.g. division by3). That class-sized calculation remains Fraction arithmetic.
The source summation/scale/bias roundoff envelope also remains exact rational,
including the original2^-1074 additive underflow term. Final bound floors/ceilings
are computed from exact numerators/denominators with precisely the old policy.
No source-operation assumption, overflow envelope, quantization step, tie rule,
output class, pairwise bound or optional early-exit behavior is relaxed.

Both source-schema implementations reject unsupported categorical/asymmetric
models, nonfinite numbers, invalid binary32 borders, overlarge inventories and
invalid precision policies. They intentionally share the strict bounded JSON
decoder, final wire packer and tiny exact power/log helpers. They do NOT share the
per-leaf parser, contrast/quantization/residual algorithm or roundoff calculation.
This is differential implementation evidence, not a formal proof of both parsers
or a claim of independently implemented every utility. The original oracle file
is byte-identical to the parent and stays selectable with backend='reference'.

`VerifiedCompact(..., backend='dyadic')` is the experimental default; the only
other allowed backend is 'reference'. Both rebuild all packed bytes on every
construction. Neither a matching hash alone, a previous receipt, an empirical
calibration set nor an option named trusted/skip can bypass reconstruction.

Source-model prediction agreement remains conditional on the ORIGINAL arithmetic
contract. This optimization changes no native executor, model, reference score,
coverage count, full-precision fallback or true-label accuracy. An exact compiler
can reproduce a flawed specification: independent source-model execution and
adversarial certification tests therefore remain necessary and are retained.

## Concurrent-delivery reconciliation

The previously local04173ea delivery had stronger manifest class/asset validation
and a nonnested8/16-bit acceptance regression. Those are ported onto actual remote
20e36d4 without replacing its CBM/source all-structure verifier or output auditor.
The v2 manifest enforces original class order and measured refined coverage; the
builder executes refinement instead of assuming it accepts exactly the16-bit set.
The former local branch remains separately addressable in the Git bundle.

References: Python float.as_integer_ratio and fractions documentation describe
exact binary-rational conversion; this is established arithmetic, not a novelty
claim. https://docs.python.org/3.13/library/stdtypes.html#float.as_integer_ratio
https://docs.python.org/3.13/library/fractions.html
