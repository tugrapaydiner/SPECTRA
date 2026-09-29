# Quantized prototype classifier contract

This is a new trained classifier, not a certificate of the old SVM's predictions.
All numerical claims refer to its own frozen parameters and computed scores.
No production SPECTRA interface or default is replaced by this experiment.

## Learned function

A model has P prototype centers c_j, positive local integer weights w_j and a dense
multiclass output head. Input q contains original uint8 feature codes in [0,D].
Prototype centers use Q subdivisions per raw-code unit. For each prototype:

    S_j = sum_f w_jf * (Q*q_f - c_jf)^2
    alpha = gamma / (U*(D*Q)^2)
    phi_j = H[S_j >> b] * L[S_j & (2^b-1)]
    H[k] = exp(-alpha * double(k*2^b))
    L[k] = exp(-alpha * double(k))
    score_c = ordered_sum_j(phi_j * head_jc) + bias_c

Weights are positive integers with sum_f(w_jf)=U*d for EVERY prototype. The maximum
signature is U*d*(D*Q)^2. A balanced power-of-two split gives approximately square-
root table storage. The loader caps the combined table at1,048,576 entries.
Tables, packed geometry, output coefficients and all other storage still count.
Prototype-specific feature weights do not imply a symmetric shared SVM kernel;
these are features in a discriminatively trained network.

The two-exp product is a deliberate numerical definition, not bitwise equality to
one exp(-alpha*S) call. Exact-feature head refitting uses this definition. Earlier
float32 training uses a straight-through quantized-geometry proxy, NOT the derivative
of exact integer projection/table indexing. Do not claim globally optimal discrete
training, an unbiased gradient, exact-real exponentials, or unchanged old models.

## Bounds and execution

Dimensions1..256, prototypes1..4096, classes2..128, maximum1..255, Q/U1..16; signatures
must stay below2^48 and table allocation must fit its cap. Center codes are uint16,
metric weights positive uint16, head and bias binary64. The mass bound controls
all nonnegative integer distance terms. SIMD32-bit accumulation is used only when
the whole signature is bounded by INT32_MAX; larger cases use the scalar uint64
path. Queries are not snapped or clipped. Invalid codes are rejected before output
writes. Floating model parameters must be finite and have a bounded output magnitude.

The original layout vectorizes input features. Packet layout transposes centers
and weights to evaluate eight independent prototypes at a time. Register layout
also retains class accumulators for6/10/26 classes; other counts use the generic
path. Every class still visits prototypes in original order. Strict compiler flags
forbid contraction and reassociation; no approximate exp or FMA is introduced by
these layouts. All three source implementations and the actual original binary
are retained for comparison. Extra packed geometry is additional memory, not free.

The tested scope is little-endian Linux x86-64, IEEE binary32/binary64,
round-to-nearest, gradual underflow, and the same deterministic system math library.
The runtime rejects non-nearest and unsupported SIMD control modes. System exp
accuracy across different platforms is not guaranteed by the C++ standard. The
independent observer recomputes both factors and ordered scores without candidate
headers or lookup tables; agreement is same-environment fidelity, not a real-number
proof or classification of the ground-truth label.

## Ownership and trust

The model is immutable after loading; each native call has private temporary
query/score storage. Python operations borrow the same owning native-resource lease
used by the existing project, serializing calls/close and deferring same-thread
closure until a live operation exits. Supplied input buffers stay exported for the
call. Callers must not mutate them concurrently. Returned labels are fresh lists.

SPPRO001 is a distinct bounded format with CRC and explicit class metadata. CRC
and SHA256 identify bytes, not authors. Native libraries and models must be trusted;
private pointer access is not a sandbox. Import never compiles or downloads.
No Windows/ARM, free-threaded Python, ASan, service tail, or production acceptance
is inherited from older SPECTRA versions.

## Comparators

FP64 ordered MLP and original SVC preserve their own fitted parameter conventions.
The stronger MLP controls execute all layers in single-thread OpenBLAS DGEMM/SGEMM
including normalization, scaling and fresh output handling. BLAS reduction order
and FP32 casting can alter numerical scores; all retained labels are checked, not
claimed universally identical. SPNF0001 stores the mechanical FP32 conversion
separately; there is no refitting after official evaluation.

The stronger SVC reuses the previously implemented finite/adaptive engine on the
same selected model with raw-code normalization charged. That engine retains its
old conditional approximation/library assumptions. The new prototype score
contract is separate. A control absent from this experiment (INT8 MLP, ProtoNN,
other compact networks or different tuning) has not been defeated.
