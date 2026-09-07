# Milestone 10 Protocol — Faithful CPU Deployment for One Trained Configuration

**Scope:** one trained W1.58A8 TRM configuration, exported and executed through a real CPU deployment runtime with explicit mixed-precision boundaries.

This protocol is frozen before M10 fidelity measurements are generated.

## 1. Research/deployment question

Can one trained SPECTRA ternary recursive model be exported into a self-contained, versioned inference artifact, loaded independently of the training checkpoint, and run through a CPU runtime in which all ternary linear projections execute through the repository's native C++ extension while the remaining floating-point work is explicit and fidelity against the corresponding PyTorch reference is verified?

M10 is deployment/fidelity work. It does not change the M09 trained-action result or convert M09 into a passing milestone.

## 2. Fixed trained configuration

Task: generated validated 9×9 Sudoku, unique solutions, 30–35 clues.

```text
data seed          20260910
train              384 puzzles
validation          96 puzzles
held-out test      128 puzzles
model seed         10101
model              TRM
hidden dim         48
layers              1
attention heads     4
n                    1
T                    1
N_sup                2
alpha_y init        0.1
alpha_z init        0.1
max_grid_size       16
weights             ternary W1.58 forward
recurrent state     A8 fake quantization enabled
training steps      200
batch               32
optimizer           AdamW
learning rate       1e-3
weight decay        0.01
grad clip           1.0
quant warmup        first 50 steps
CPU training        FP32
```

The trained model is accepted for deployment only if every `FakeBitLinear.quant_strength` is exactly `1.0` after training/checkpoint reload. No soft-ternary export is supported.

The M10 gate is not conditioned on Sudoku accuracy. The model must be genuinely trained and the held-out deployment comparison must use data not used by optimizer steps.

## 3. Versioned deployment artifact

Artifact identity:

```text
format  = spectra.cpu_recursive
version = 1
backend_contract = packed_ternary_fp32_linear_v1
```

The artifact is self-contained for the declared runtime and records:

### Architecture / geometry

- model class `TRM`;
- `dim`, `num_tokens`, `seq_len`;
- `n_layers`, attention `heads`, FFN expansion ratio;
- `n`, `T`, `N_sup`;
- `alpha_y`, `alpha_z` values;
- `max_grid_size`;
- task height, width, box geometry;
- `ternary=true`, `act8=true`;
- GELU mode `approximate=none`;
- RMSNorm epsilon semantics (`eps=None`, therefore framework dtype default);
- A8 bits/qmin/qmax/eps.

### Packed ternary linears

Every `FakeBitLinear` used by recursive inference is exported as:

```text
2-bit row-padded ternary codes
per-output-channel FP32 scale
optional FP32 bias
in_features / out_features
packing code map and row-byte geometry
```

Required linears for the fixed one-layer configuration:

```text
blocks.0.attn.q
blocks.0.attn.k
blocks.0.attn.v
blocks.0.attn.proj
blocks.0.ff.0
blocks.0.ff.2
out_head
```

The artifact does not retain FP master weights for inference. Export proves that unpacked `(ternary_code * scale)` exactly equals the hard-ternary weight used by the reference model.

### Floating-point tensors

All non-ternary inference parameters required by the graph are stored as FP32 tensors:

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

The export contains an explicit tensor inventory, shapes, dtypes, and per-tensor SHA-256 hashes. Unknown/missing/extra inference parameters are rejected.

### Provenance

The artifact records source trained-checkpoint hash, source tensor-state hash, quant-strength state, training step/seed, data manifest hash/IDs, export git SHA, and exact precision/backend contract.

## 4. Supported and rejected graphs

M10 supports exactly the trained graph family needed by the fixed configuration:

- `TRM`;
- fully ternary `SwapBlock` attention implemented by `SelfAttention`;
- `FakeBitLinear` q/k/v/proj, two FFN projections, and output head;
- `nn.RMSNorm` with the trained weights;
- `nn.GELU(approximate='none')`;
- `FakeActQuant(bits=8)` recurrent y/z quantization;
- FP32 embedding/positional tables, residuals, normalization, attention, halting head.

Export rejects, rather than silently approximates:

- non-TRM models;
- `ternary=false`;
- any ternary layer with `quant_strength != 1.0`;
- dense `nn.MultiheadAttention` blocks;
- non-GELU FFNs or GELU approximation other than `none`;
- unsupported normalization/module substitutions;
- router-enabled inference;
- geometry inconsistent with `seq_len`/`max_grid_size`;
- malformed/reserved ternary packed codes;
- artifact versions/backend contracts other than the declared version.

## 5. GELU versus legacy fused FFN

The trained model's FFN is:

```text
ternary linear -> GELU(approximate='none') -> ternary linear
```

The historical fused integer FFN is **not** used in the faithful M10 runtime. It represents a separately evolved quantized operator and must not replace the trained GELU path while claiming exact equivalence.

M10 therefore uses the new dense packed-ternary FP32 native linear primitive for both FFN projections and executes the original GELU in FP32 between them.

No ReLU-for-GELU substitution is permitted.

## 6. CPU native primitive

Add one checked native operation:

```text
dense_ternary_linear_fp32(X, packed_weight, row_scale, bias, out_dim) -> FP32 Y
```

Contract:

- X: CPU contiguous FP32 `[vectors, in_features]`;
- packed weight: row-padded 2-bit ternary bytes;
- scale: CPU contiguous FP32 `[out_features]`;
- bias: CPU contiguous FP32 `[0]` or `[out_features]`;
- output: CPU FP32 `[vectors, out_features]`;
- accumulation: scalar FP32 in input-dimension order;
- each ternary product is `x * {-scale,0,+scale}`;
- bias is added in FP32 after accumulation;
- no activation requantization occurs inside this primitive.

The first accepted implementation is correctness-first. It does not claim the dense FP32 primitive is AVX2-vectorized merely because the extension binary was compiled with AVX2 enabled.

## 7. Exact mixed-precision runtime

### Native work

All seven ternary linear modules execute through `dense_ternary_linear_fp32`.

For a tensor `[B,L,D]`, the runtime flattens to `[B*L,D]`, performs one native call, and reshapes back.

### FP32 work retained outside native linears

The runtime explicitly keeps these operations in PyTorch CPU FP32:

- token embedding lookup;
- row/column positional embedding lookup/addition;
- RMSNorm;
- q/k/v reshaping/transposes;
- scaled dot-product attention score/matmul/softmax path;
- GELU (`approximate='none'`);
- residual additions;
- multiplication by learned `alpha_y/alpha_z`;
- recurrent A8 scale calculation, rounding/clamping and dequantization;
- halting-head dense linear;
- answer argmax.

This is a **mixed-precision CPU runtime**, not full integer inference.

### A8 activation-scale handling

A8 is reproduced exactly at the same graph locations as `FakeActQuant` in training/reference:

```text
scale = max(abs(state), feature_dim).clamp_min(1e-6) / 127
q = clamp(round(state / scale), -128, 127)
state = q * scale
```

Scale is dynamic per token and recomputed independently for each y/z state update. The scale itself and the dequantized state are FP32. No additional activation quantizer is inserted at native linear boundaries.

### Floating boundaries

- native packed ternary linear input: FP32;
- native accumulator/output: FP32;
- attention/GELU/norm/residual: FP32;
- recurrent A8 integer codes are transient; dequantized y/z are FP32 between recurrence calls;
- final logits: FP32;
- final answer: integer argmax token IDs.

## 8. Recursive forward semantics

The deployment runtime reproduces the trained TRM graph:

```text
x_emb = token_embed(x) + row_embed + col_embed
y = 0
z = 0

for supervision_step in 0..N_sup-1:
    for cycle in 0..T-1:
        repeat n times:
            update_z = f(x_emb + y + z)
            z = A8(RMSNorm(z + alpha_z * update_z))
        update_y = f(y + z)
        y = A8(RMSNorm(y + alpha_y * update_y))

    logits = native_out_head(y)
    halt_logit = FP32_halt_head(mean(y over tokens))

answer = argmax(final_logits)
```

There is no training-time detach distinction during inference because no gradient graph exists.

## 9. Fidelity gates — frozen before measurement

All comparisons use the exact trained hard-ternary reference after reload and CPU FP32 reference execution.

### Packed weight reconstruction

For every exported ternary layer:

```text
unpacked_code * exported_scale == reference hard-ternary weight
```

Required: `torch.equal` / exact FP32 tensor equality.

### Native linear

Across deterministic trained-layer inputs:

```text
max_abs_error <= 5e-5
mean_abs_error <= 5e-6
```

### Complete SwapBlock

```text
max_abs_error <= 2e-4
mean_abs_error <= 2e-5
```

### One recursive cycle

For both y and z:

```text
max_abs_error <= 5e-4
mean_abs_error <= 5e-5
```

### Full forward / solve

On the first 16 untouched test puzzles:

```text
final logits max_abs_error <= 1e-3
final logits mean_abs_error <= 1e-4
final decoded token IDs exact match = 100%
per-example semantic-validity decision exact match = 100%
```

Additionally compare each supervision step's decoded output exactly.

A larger error or an argmax/semantic mismatch fails M10; tolerances are not widened after observing results.

## 10. Actual native-call accounting

The runtime records:

- total native packed-linear calls;
- calls by layer name;
- native input vector count;
- native multiply dimensions;
- FP32 operation categories retained outside native calls;
- A8 quantization call count;
- halt-head calls;
- decode calls.

For the fixed architecture, the expected call structure is derived from `n`, `T`, `N_sup`, block count, and output-head evaluations and checked against the runtime record.

## 11. Cold versus warm timing

Timing is reported as separate phases:

1. training — excluded from deployment latency;
2. export/packing and artifact write;
3. cold artifact load/validation;
4. cold native extension compile/load;
5. one warm-up inference;
6. warmed inference repetitions.

No cold compile/packing time is folded into warmed solve latency.

Latency is descriptive only; M10 has no speedup claim or speed threshold.

## 12. Profiler and backend identity

One held-out trained-model solve attempt is captured with `torch.profiler` CPU tracing. Native calls are surrounded by a named profiler scope:

```text
spectra::dense_ternary_linear_fp32
```

Retained evidence includes:

- Chrome profiler trace JSON;
- profiler key-average table;
- runtime native-call accounting;
- host/CPU/compiler/PyTorch identity;
- extension backend identity (`avx2` or `scalar` build);
- dense M10 operator identity explicitly stating whether its implementation is scalar/vectorized;
- actual floating-point work declaration.

## 13. bitnet.cpp boundary

A configured `SPECTRA_BITNET_CPP` directory is **not** evidence of a working compatible backend.

M10 does not integrate bitnet.cpp. Backend reporting must therefore say:

```text
bitnet_cpp_supported = false
```

whether or not a directory is configured, with the reason that no compatible SPECTRA artifact loader/recursive runtime has been integrated and tested.

The historical directory probe may be exposed separately as configuration presence, never backend availability.

## 14. Acceptance gate

M10 passes only if:

1. a genuinely trained checkpoint is produced and hard-ternary state is verified;
2. the complete versioned deployment artifact exports and strict-loads independently;
3. packed reconstruction is exact;
4. the native FP32 packed-ternary primitive is actually called by recursive inference;
5. layer/block/cycle/full-forward fidelity satisfy the frozen tolerances;
6. held-out decoded outputs and semantic-validity decisions match the reference exactly for the declared sample;
7. backend report exposes native calls and remaining FP32/A8 work;
8. cold compile/load/export are separated from warmed inference;
9. one profiler trace from a held-out solve attempt is retained;
10. bitnet.cpp remains explicitly unsupported unless genuinely integrated/tested.

A configured directory, a successful export with no executable runtime, or isolated kernel tests without a recursive forward do not pass M10.
