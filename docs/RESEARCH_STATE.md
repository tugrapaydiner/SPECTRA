# SPECTRA Research State

This is the live milestone register. Complete historical registers are preserved rather than rewritten:

- M01–M04: [`RESEARCH_STATE_M01_M04.md`](RESEARCH_STATE_M01_M04.md)
- M05: [`RESEARCH_STATE_M05.md`](RESEARCH_STATE_M05.md)
- M06: [`RESEARCH_STATE_M06.md`](RESEARCH_STATE_M06.md)
- M07: [`RESEARCH_STATE_M07.md`](RESEARCH_STATE_M07.md)
- M08: [`RESEARCH_STATE_M08.md`](RESEARCH_STATE_M08.md)
- complete M09-era live register: [`RESEARCH_STATE_M09.md`](RESEARCH_STATE_M09.md)

## Milestone index

| Milestone | Scope | State |
|---|---|---|
| M01 | trustworthy baseline | accepted / merged; `40745dbe185c069aeee9eff3cf63dd411d9e17da` |
| M02 | native-kernel correctness/input contracts | accepted / merged; `e0781ec8b4e649ab4ccd48d4cd5f432a9b88d249` |
| M03 | task/data/evaluation contracts | accepted / merged; `01638b10777029fb28bb35229e374f0865c6e5d4` |
| M04 | reproducible training/checkpoint state | accepted / merged through PR #4; `370caf708755e1c68c59d5696778597f0290ea68` |
| M05 | checkpoint-backed evaluation | accepted / merged through PR #5; `1003c59e17dc17e652438317b7480c9e898379af` |
| M06 | controlled trained baseline | accepted / merged through PR #6; `385da5ef822dcb001c8193a2b8802fa292b31428`; gate clarification PR #7 `dbec23dd2077fc9f23e6a031b1b2b82c7c731de6` |
| M07 | grounded verifier training/evaluation | accepted / merged through PR #8; `d15d5578878a175442895867de08d98efb31e6da` |
| M08 | correct inspectable MCTS reference | accepted / merged through PR #9; `049ada6b01260662e1b3f037522fbe9a84d6d1c6` |
| M09 | trained search action mechanism | **INCOMPLETE**; negative result preserved/merged through PR #10; `b667204da950aa989df13bb846166c7b1c2d761b` |
| M10 | faithful CPU deployment for one trained configuration | **COMPLETE on `research/m10-cpu-deployment`; accepted implementation run `34142683006`; not merged at the time of this entry** |

M09 remains scientifically incomplete: no allowed action-policy development variant met its fixed practical-effect threshold and its final comparison stayed sealed. The user explicitly directed M10 as an orthogonal deployment/fidelity milestone. M10 acceptance therefore does not retroactively pass M09 or establish learned-search benefit.

---

# Milestone 10 — faithful CPU deployment

**Stage status: COMPLETE on `research/m10-cpu-deployment`.**

Authoritative preregistration: [`M10_PROTOCOL.md`](M10_PROTOCOL.md).
Accepted evidence/claims boundary: [`M10_ACCEPTANCE_GATE.md`](M10_ACCEPTANCE_GATE.md).

## Accepted trained configuration

One genuinely trained hard-ternary+A8 TRM is the deployment source:

```text
9x9 Sudoku, 30–35 clues
train / val / test  384 / 96 / 128
data seed            20260910
model seed           10101
dim                   48
layers                 1
heads                  4
n / T / N_sup          1 / 1 / 2
ternary                true
act8                   true
training steps         200
quant warmup            50
```

Source checkpoint:

```text
SHA256        f14816e894a733c3a4df28d14e4e395d1a2b3ba4907cb986a2dcc0608b62ba3f
tensor SHA256 f915ce91f081d95ebd6a4d635921c450c1aab2f5999ee15c170b4accf33a8766
```

All `FakeBitLinear.quant_strength` values are exactly `1.0` after reload. M10 rejects soft-ternary export.

## Versioned artifact

```text
format            spectra.cpu_recursive
version           1
backend contract  packed_ternary_fp32_linear_v1
artifact SHA256   e40f5d84a60ce21237a9a184c8459588b7603f991195c875ef8dcbf319a270e1
packed linears    7
FP tensors        11
```

The artifact is self-contained for the declared runtime: it contains packed hard-ternary q/k/v/proj, both FFN projections and output-head weights with scales/biases; embeddings, RMSNorm weights, halt-head parameters, alpha scalars; architecture/geometry; precision/A8/GELU semantics; and provenance/hash inventory.

The loader rejects unsupported or malformed configurations rather than silently substituting them.

## GELU/native-path decision

The trained FFN is:

```text
ternary linear -> GELU(approximate='none') -> ternary linear
```

The historical fused integer FFN is not used. No ReLU-for-GELU or other activation substitution is accepted.

M10 adds a separate correctness-first native primitive:

```text
packed_ternary_fp32_scalar_linear_v1
```

It consumes 2-bit ternary packed weights plus FP32 row scales/biases and uses FP32 inputs, ordered scalar FP32 accumulation, and FP32 outputs.

The implementation is explicitly:

```text
vectorized = false
performance claim = false
```

## Mixed-precision recursive runtime

All seven ternary linear modules execute through the C++ primitive:

```text
blocks.0.attn.q
blocks.0.attn.k
blocks.0.attn.v
blocks.0.attn.proj
blocks.0.ff.0
blocks.0.ff.2
out_head
```

The remaining graph is explicit CPU FP32/PyTorch work:

- token/positional embeddings;
- RMSNorm;
- attention reshape/matmul/softmax;
- exact GELU;
- residual arithmetic and alpha scaling;
- halt head;
- final logits/argmax.

Recurrent A8 is preserved at the trained y/z boundaries with dynamic per-token scale and transient int8 round/clamp followed by FP32 dequantized state.

M10 is mixed precision, **not** full integer inference.

## Actual native work

For the accepted architecture, one complete forward must issue 26 native linears. Observed on the 16-example held-out fidelity batch:

```text
native calls                 26
expected native calls        26
call structure match         true
q/k/v/proj calls              4 each
ff0/ff2 calls                 4 each
out_head calls                2
native input vectors      33,696
native scalar products 144,571,392
A8 calls                       4
FP32 halt-head calls           2
decode calls                   1
```

The profiled single-puzzle solve also executed all 26 native calls.

## Fidelity

All seven packed hard-ternary weights reconstruct the reference effective weights exactly.

Frozen tolerances:

```text
native linear    max 5e-5 / mean 5e-6
block            max 2e-4 / mean 2e-5
cycle            max 5e-4 / mean 5e-5
full logits      max 1e-3 / mean 1e-4
final IDs        exact
per-step IDs     exact
semantic decision exact
```

Observed:

```text
block max / mean        9.5367e-7 / 1.2429e-7
cycle y max / mean      4.7684e-7 / 6.2950e-8
cycle z max / mean      4.7684e-7 / 2.6730e-8
full logits max / mean  2.3842e-6 / 1.1235e-7
final decoded exact     true
step decoded exact      true / true
semantic decision exact true
```

On the 16 held-out examples both reference and deployment still had zero semantic-valid solves and exactly matching partial metrics. M10 is therefore fidelity evidence, not a reasoning-capability result.

## Cold/warm separation

The accepted run uses an empty experiment-private Torch extension build directory, independent of the earlier focused-test extension cache.

```text
training, excluded              10.871530 s
export/pack/write                0.017594 s
cold artifact load/validate      0.007918 s
cold native source compile/load 19.833225 s
warm-up                           separate
warmed solve mean                 0.018125 s
warmed solve median               0.018104 s
```

No speedup claim is made.

## Profiler / backend identity

Profiler scope:

```text
spectra::dense_ternary_linear_fp32
```

Accepted evidence hashes:

```text
profiler_trace.json  0046311c80522371f1151d5e83f50ba8f4e405f7fd90a6ada9d36b53fc1ecade
profiler_table.txt   b57a1829562921ede7e72795f7b515854be4532e71788b8beb3d3e0c1b84bc27
profiler_record.json 2948e03468f462b48e25c1a283209d0b5670345f739482641680c31c7b21600c
```

Backend identity:

```text
spectra_m10_native
packed_ternary_fp32_scalar_linear_v1
scalar C++ FP32 accumulation
vectorized false
```

## bitnet.cpp

`SPECTRA_BITNET_CPP` directory presence is configuration only. It no longer makes `is_bitnet_cpp_available()` true.

```text
bitnet_cpp_supported = false
```

No compatible SPECTRA artifact loader and recursive runtime has been integrated/tested against bitnet.cpp.

## Accepted execution

```text
head          7f441c9b03c647fed2ec3c45d2544d4c8454bb90
run           34142683006
job           101807924641
artifact      m10-cpu-deployment-evidence
artifact id   10026561529
ZIP SHA256    74cf5c06d2d06a9239724f97ff8ad5542854eec5857cd13d8ad7286a3a183359
size          749,224 bytes
```

```text
focused M10   8 passed, 1 test-only warning
full fast     256 passed, 16 deselected, 1 test-only warning
compile       0
focused       0
experiment    0
evidence      0
fast          0
```

## Remaining boundary

M10 does not establish:

- full integer inference;
- SIMD/AVX2 acceleration of the new dense operator;
- speedup, energy superiority or cache residency;
- bitnet.cpp compatibility;
- learned-search benefit;
- reasoning-quality improvement;
- broad architecture support beyond the fixed accepted graph.

## M10 decision

**PASS.** A trained model can be exported, strict-loaded and run through the declared recursive CPU backend with verified layer/block/cycle/full fidelity. Actual native calls and all remaining floating-point/A8 work are visible.

**Stop here for M10. Do not begin another milestone automatically.**
