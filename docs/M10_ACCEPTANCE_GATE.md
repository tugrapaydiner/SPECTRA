# Milestone 10 Acceptance Gate — Faithful CPU Deployment

**Decision: PASS for the M10 deployment/fidelity gate.**

M10 establishes one real trained W1.58A8 SPECTRA configuration that can be exported to a self-contained versioned artifact, strict-loaded without the training checkpoint, and executed through a recursive CPU runtime whose ternary linear projections are actual C++ native calls. The remaining floating-point work is explicitly reported.

This pass is only a deployment-fidelity result. It does not change the separate M09 result: M09 remains incomplete because its learned-action practical-benefit gate failed.

## Accepted implementation evidence

```text
branch                 research/m10-cpu-deployment
accepted head          7f441c9b03c647fed2ec3c45d2544d4c8454bb90
Actions run            34142683006
job                     101807924641
artifact                m10-cpu-deployment-evidence
artifact id             10026561529
artifact ZIP SHA256     74cf5c06d2d06a9239724f97ff8ad5542854eec5857cd13d8ad7286a3a183359
artifact size           749,224 bytes
retention               14 days
```

All captured workflow exits were zero:

```text
compile    0
focused    0
experiment 0
evidence   0
fast       0
```

Tests:

```text
focused M10   8 passed, 1 warning
full fast     256 passed, 16 deselected, 1 warning
```

The warning is a test-only PyTorch warning caused by converting a requires-grad reference tensor to Python float in an assertion. It does not occur in the inference artifact/runtime and does not affect the numerical result.

## Trained source

The accepted source is a genuinely trained CPU FP32 W1.58A8 TRM:

```text
task                 generated validated 9x9 Sudoku
clues                30–35
train/val/test        384 / 96 / 128
data seed             20260910
model seed            10101
dim                   48
layers                1
heads                 4
n / T / N_sup         1 / 1 / 2
ternary               true
act8                  true
training steps        200
batch                 32
AdamW lr              1e-3
weight decay          0.01
quant warmup          50 steps
```

Accepted source checkpoint:

```text
SHA256        f14816e894a733c3a4df28d14e4e395d1a2b3ba4907cb986a2dcc0608b62ba3f
tensor SHA256 f915ce91f081d95ebd6a4d635921c450c1aab2f5999ee15c170b4accf33a8766
```

Every `FakeBitLinear.quant_strength` is exactly `1.0` after reload. Soft-ternary artifacts are rejected.

M10 does not condition its pass on Sudoku capability. On the 16 held-out puzzles used for deployment fidelity, the reference and deployed model both had zero semantic-valid solves; their partial metrics were identical. This is expected to remain a capability limitation, not hidden by the deployment result.

## Versioned deployment artifact

Accepted inference artifact:

```text
format            spectra.cpu_recursive
version           1
backend contract  packed_ternary_fp32_linear_v1
SHA256            e40f5d84a60ce21237a9a184c8459588b7603f991195c875ef8dcbf319a270e1
packed linears    7
FP tensors        11
```

Packed native linears:

```text
blocks.0.attn.q
blocks.0.attn.k
blocks.0.attn.v
blocks.0.attn.proj
blocks.0.ff.0
blocks.0.ff.2
out_head
```

FP32 tensors retained by the artifact:

```text
token_embed.weight
pos_encoder.row_embed.weight
pos_encoder.col_embed.weight
blocks.0.norm1.weight
blocks.0.norm2.weight
norm_y.weight
norm_z.weight
halt_head.weight
halt_head.bias
alpha_y
alpha_z
```

The artifact includes geometry, heads, recurrence schedule, precision semantics, A8 parameters, GELU/RMSNorm identity, tensor inventory/hashes, training/source provenance and exact packed linears. It does not require FP master ternary weights for inference.

The strict loader rejects unsupported format/version, unknown/incomplete inventories, inconsistent geometry, non-hard ternary state, reserved packed codes, invalid padding/scales, non-GELU FFN semantics, non-A8 recurrent configuration, and hash corruption.

## GELU mismatch resolved without substitution

The trained graph uses:

```text
FakeBitLinear -> GELU(approximate='none') -> FakeBitLinear
```

The historical fused integer FFN is not used by M10. No ReLU or other integer activation was substituted while claiming equivalence.

Instead, the two FFN ternary projections execute through the new packed native FP32 linear primitive and the original exact GELU remains PyTorch CPU FP32 between them.

Therefore:

```text
gelu_preserved                  true
historical_fused_int8_ffn_used  false
```

## Native backend identity

```text
backend          spectra_m10_native
runtime          spectra_cpu_recursive_v1
operator         packed_ternary_fp32_scalar_linear_v1
packed weights   2-bit ternary
input            FP32
accumulator      FP32
output           FP32
implementation   scalar C++ ordered accumulation
vectorized       false
compile flags    -O3 -std=c++20
```

This is a correctness-first mixed-precision backend. M10 makes no AVX2/SIMD claim for this dense operator and no speedup claim.

The native primitive consumes row-padded 2-bit ternary codes plus per-output-channel FP32 scales and optional FP32 bias. For each output it accumulates products in FP32 in increasing hidden-index order and then adds bias in FP32.

## Remaining floating-point work

The following work remains explicit outside the C++ ternary linear primitive:

```text
token embedding lookup                         FP32
row/column positional embedding lookup/add     FP32
RMSNorm                                         FP32
q/k/v reshape and transpose                     FP32
scaled dot-product attention + softmax           FP32
GELU(approximate='none')                        FP32
residual additions and learned alpha scaling     FP32
recurrent A8 scale/round/clamp/dequant            transient int8 + FP32 scale/state
halt head                                        FP32
final logits                                     FP32
argmax decode                                    integer ids
```

The model's A8 boundary is reproduced at the original recurrent y/z update locations only. Dynamic scale is per token:

```text
scale = max(abs(state), feature).clamp_min(1e-6) / 127
q = clamp(round(state / scale), -128, 127)
state = q * scale
```

No additional activation quantizer is inserted at native linear boundaries.

**Full integer inference is not claimed.**

## Recursive native integration

The artifact runtime reproduces the model recurrence and calls the native operator for every ternary projection. For the accepted `n=1`, `T=1`, `N_sup=2`, one forward has the architectural expectation:

```text
q/k/v/proj  4 calls each
ff.0        4 calls
ff.2        4 calls
out_head    2 calls
total       26 native linear calls
```

Observed on the 16-example held-out fidelity batch:

```text
native linear calls        26
expected native calls      26
call structure match       true
native input vectors       33,696
native scalar products     144,571,392
A8 recurrent calls         4
FP32 halt-head calls       2
decode calls               1
```

The single-puzzle profiler solve also recorded all 26 native linear calls, 2,106 native input vectors and 9,035,712 scalar ternary products.

## Frozen fidelity results

The tolerances were preregistered before these measurements:

```text
native linear    max <= 5e-5, mean <= 5e-6
SwapBlock        max <= 2e-4, mean <= 2e-5
recursive cycle  max <= 5e-4, mean <= 5e-5
full logits      max <= 1e-3, mean <= 1e-4
full decoded ids exact match required
per-step ids     exact match required
semantic decision exact match required
```

All seven packed weights reconstruct the reference hard-ternary effective weights exactly.

Observed native-linear maximum absolute errors:

| Layer | max abs | mean abs |
|---|---:|---:|
| `blocks.0.attn.q` | `2.6822e-7` | `4.0172e-8` |
| `blocks.0.attn.k` | `3.5763e-7` | `4.6307e-8` |
| `blocks.0.attn.v` | `3.5763e-7` | `3.6539e-8` |
| `blocks.0.attn.proj` | `1.1921e-7` | `1.4949e-8` |
| `blocks.0.ff.0` | `4.7684e-7` | `4.4058e-8` |
| `blocks.0.ff.2` | `4.7684e-7` | `5.9774e-8` |
| `out_head` | `1.4305e-6` | `1.0055e-7` |

Higher-level fidelity:

```text
SwapBlock max/mean    9.5367e-7 / 1.2429e-7
cycle y max/mean      4.7684e-7 / 6.2950e-8
cycle z max/mean      4.7684e-7 / 2.6730e-8
full logits max/mean  2.3842e-6 / 1.1235e-7
```

On the first 16 untouched test puzzles:

```text
final decoded IDs exact        true
supervision-step 1 exact       true
supervision-step 2 exact       true
semantic-validity decisions    exact
reference valid solves         0
native-runtime valid solves    0
```

Reference and deployment task metrics were identical:

```text
blank-cell accuracy    0.11183355
cell accuracy          0.47299382
exact reference match  0.0
semantic validity      0.0
```

Thus M10 demonstrates deployment fidelity, not reasoning-quality improvement.

## Cold versus warm timing

The final accepted run deliberately forced an empty experiment-private `TORCH_EXTENSIONS_DIR`, even though focused tests had compiled the operator earlier elsewhere. Evidence records:

```text
native build isolated             true
build directory existed before    false
empty before experiment import    true
```

Measured phases:

```text
training (excluded)              10.871530 s
export/pack/write                 0.017594 s
cold artifact load/validate       0.007918 s
cold native source compile/load  19.833225 s
one warm-up                       separate
warmed single-puzzle mean         0.018125 s
warmed single-puzzle median       0.018104 s
```

Five warmed samples:

```text
0.018178844
0.018097340
0.017989378
0.018104123
0.018256820
```

These timings are descriptive only. No speedup or target-hardware performance claim is made.

## Profiler evidence

One held-out trained puzzle was executed under the PyTorch CPU profiler with native calls enclosed by:

```text
spectra::dense_ternary_linear_fp32
```

Retained hashes:

```text
profiler_trace.json  0046311c80522371f1151d5e83f50ba8f4e405f7fd90a6ada9d36b53fc1ecade
profiler_table.txt   b57a1829562921ede7e72795f7b515854be4532e71788b8beb3d3e0c1b84bc27
profiler_record.json 2948e03468f462b48e25c1a283209d0b5670345f739482641680c31c7b21600c
```

The profiled deployment prediction exactly matched the corresponding reference prediction.

## bitnet.cpp boundary

A configured bitnet.cpp directory is no longer reported as a working inference backend.

M10 leaves:

```text
bitnet_cpp_supported = false
```

because no compatible SPECTRA artifact loader and recursive inference runtime has been integrated and tested against bitnet.cpp. `bitnet_cpp_root()` remains only a configuration-presence probe.

## Exact accepted execution

The workflow acceptance experiment used the isolated wrapper:

```bash
python scripts/m10_cpu_deployment_isolated.py --out .m10/experiment
python -m pytest tests/test_m10_cpu_deployment.py -q
python -m pytest -m 'not slow' -ra
```

For a normal local evidence directory:

```bash
python scripts/m10_cpu_deployment_isolated.py --out outputs/m10_cpu_deployment
```

The wrapper creates a fresh private Torch-extension build directory before importing/running the experiment so the reported cold native compile/load phase is not served from the focused-test build cache.

## Remaining limitations / unsupported claims

- The accepted dense native operator is scalar correctness-first, not optimized SIMD.
- The runtime is mixed precision and still relies on PyTorch CPU for attention, norm, GELU, embeddings, residuals and halt head.
- Only the fixed one-layer ternary+A8 TRM graph is accepted; unsupported modules/configurations are rejected.
- M10 validates the first 16 untouched test puzzles for full end-to-end fidelity, not every possible input or architecture.
- The deployed/reference model has zero strict solves on those 16 examples, so M10 is not reasoning-performance evidence.
- No energy, throughput superiority, cache-residency, bitnet.cpp, or full-integer claim is established.
- The historical fused INT8 FFN remains a different operator and is not part of this exact deployment path.
- M09 remains independently incomplete.

## M10 decision

**PASS.** A real trained hard-ternary+A8 checkpoint is exported into a self-contained strict artifact, loaded independently, and executed through a real recursive CPU runtime with 26 actual native packed-ternary calls per solve. Packed reconstruction, layer, block, cycle, full-logit, per-step decode, final decode and semantic-decision fidelity all satisfy the frozen gate. Remaining FP/A8 work and unsupported backends are visible rather than hidden.

**Stop here for M10.**
