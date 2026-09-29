# Preserved comparator, not newly introduced arithmetic

The complete learning benchmark uses a historical lower-accuracy finite-kernel
model as one arm. Four files copied byte-for-byte from the delivered finite-kernel
source at e9e32743149bd15482f2144cbe90b32800da7781 make this arm reproducible from
the focused GitHub checkout: adaptive_kernel/runtime.cpp, finite_kernel/runtime.cpp,
finite_kernel/build.py and finite_kernel/session.py. They are unchanged comparison
prerequisites, not additional novel implementation in this learning experiment.

The historical runtime uses exact same-environment kernels on admitted dyadic grids
and conditional forward-error enclosures on non-dyadic grids. Those enclosures
assume an absolute original-libm exponential error <=2^-48 and the stated IEEE
round-to-nearest/gradual-underflow environment. Nonmember inputs delegate to the
previous adaptive implementation; it has its own conditional reference-libm bound.
Neither prior backend is an unconditional portable-C++ real-arithmetic certificate.
They do not improve the old model's real-world accuracy.

The NEW integer-metric model instead fits directly on weighted integer distances.
Its repeated-coordinate reference shares exactly representable sums and the same
computed system-exp arguments; it does not borrow the non-dyadic approximation
claim for its own model. See CONTRACT.md. The full old evidence and proofs remain
in the earlier delivery and full local source archive; their historical scores,
compiler failures and platform exclusions are unchanged.
