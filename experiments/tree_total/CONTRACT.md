# Total specified-source execution

SPCTOT01 is a new experimental format. It embeds the unchanged SPCERT02 16-bit
certificate and a lossless representation of the ORIGINAL binary64 leaf vectors,
scale and biases. It does not certify the real-world class or retrain a model.

## Meaning of total

For an admitted source, valid original uint8 input and supported floating-point
environment, normal execution returns a class index rather than UNRESOLVED. The
fast path uses the existing sufficient numerical certificate. An undecided row
uses its already computed routes to add the original leaf values, in original
tree order, starting at positive zero, then applies positive common scale and
bias with separate binary64 operations. First-index ties are retained. Zero
additions are not omitted and leaf contrasts are not used for exact fallback.

This is totality of the mathematical execution policy, not immunity to process
termination, out-of-memory, invalid libraries, hardware errors or malformed input.
Invalid/unsupported requests fail explicitly. The diagnostic certificate_only
policy intentionally retains abstention and must not be described as full coverage.
The exact policy always runs original arithmetic; audit checks every fast result
against original arithmetic and raises on disagreement.

The reference is the supplied JSON-defined sequential binary64 computation. It is
NOT an exact-real sum. Official/exported CatBoost implementations may use a different
reduction or conversion; observed score differences mean that unconditional
all-input agreement with EVERY CatBoost backend must not be inferred. All measured
original-backend labels agree on the retained corpus, but near ties can depend on
a backend's numerical definition. No backend-equivalence theorem is claimed.

## Why the fallback preserves the source

Validated integer feature codes convert exactly to the source feature values.
An exact binary32 border induces the same integer predicate as q > floor(border),
with out-of-domain borders clamped to constant predicates. Repeated predicates,
constant splits and depths through12 are retained. The route selects exactly the
original leaf. The flat and interned banks contain identical binary64 bits, including
signed zero. At every tree addition the previous accumulator and next operand
are the same as the reference. Induction preserves every rounded accumulator;
separate scale and bias operations and the first-index maximum complete the same
function. This argument assumes the admitted IEEE environment and compiler flags.
The compact branch additionally relies on the unchanged exact-rational certificate
argument; independent reconstruction checks the entire compact proof, not just a
source hash. The runtime does not invent a label when that certificate fails.

## Bounds, arithmetic and trusted boundary

The inherited source validator admits numeric oblivious trees only: bounded input
width through256, source trees through4096, depth through12, at most500,000 leaf
scalars, positive common scale, finite parameters and a conservative no-overflow
envelope. Compact accumulator constraints can reject otherwise valid source models.
The source scalar range, dyadic step and proof-work restrictions remain unchanged.
The native format permits up to64 classes,64MiB bytes and checked leaf-index
inventories. Each call is limited to65,536 rows and8million input elements; score
outputs have their own8million-cell cap. Input byte buffers must not be mutated
concurrently. Invalid late input is checked before returning any result.

The tested scope is little-endian Linux x86-64, binary64 nearest-even, gradual
underflow, no contraction or reassociation. Explicit AVX2 builds require compatible
hardware; portable is not Windows/ARM acceptance. The environment is checked on
load and each request. No correct-rounding assumption for exp is needed in this
exact tree fallback: its source operation set is additions and scale/bias.

Native structural checks are not the numerical proof. Only a VerifiedTotal object
constructed by full deterministic source reconstruction reaches the public Python
runtime. Private state tampering, forged native pointers, hostile executable
libraries and a malicious rewrite of the complete evidence are outside this
boundary. CRC/SHA256 identify bytes, not authors. The existing owning-resource
lease serializes close/calls and defers reentrant destruction. There is no global
prediction cache or cross-input reuse of answers.

## Storage and publication

Interning uses whole-vector binary64 bytes, with the first occurrence as canonical.
Every source leaf keeps an index. It neither prunes trees nor skips additions.
The proof, original bank, index and source JSON for verification all consume space;
this format is larger than the abstaining residual-only representation.

The optional JSONL runner accepts strict integer rows and stages a complete output
in a trusted destination directory. Hard-link create-if-absent publication never
replaces another file. Errors before publication do not expose a partial final
file, although a forced kill can leave a private partial file. Hard-link support
is required; no directory-fsync or crash-durability promise is made. Output records
contain class indices, not an unsupported interpretation of ground-truth labels.
