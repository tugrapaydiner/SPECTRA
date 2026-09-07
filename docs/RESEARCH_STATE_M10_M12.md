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
| M10 | faithful CPU deployment for one trained configuration | accepted / merged through PR #11; `17c60819e2b5aace48b7b9994d0254989df25ab9` |
| M11 | real adaptive execution | accepted / merged through PR #12; `40ce9c65f97e3950bb927713d8e117ab6c4b6e1c` |
| M12 | grounded router/halter RL training path | **COMPLETE on `research/m12-router-halter-rl`; accepted run `34154069653` at `c54a719f8c30c958e58b4ecb32abb54a75de3cfd`; learned-control quality NEGATIVE / COLLAPSED** |

M09 remains scientifically incomplete: no allowed action-policy development variant met its fixed practical-effect threshold and its final comparison stayed sealed. M10 and M11 are orthogonal deployment/execution milestones. M12 establishes a grounded adaptive-policy training path but its bounded pilot did not show useful learned control. None of M10–M12 retroactively passes M09 or establishes learned-search benefit.

---

# Milestone 10 — faithful CPU deployment

**Stage status: COMPLETE and merged through PR #11.**

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

---

# Milestone 11 — real adaptive execution

**Stage status: COMPLETE and merged through PR #12.**

Authoritative preregistration: [`M11_PROTOCOL.md`](M11_PROTOCOL.md).
Accepted evidence/claims boundary: [`M11_ACCEPTANCE_GATE.md`](M11_ACCEPTANCE_GATE.md).

## Execution interface

Inference now has an explicit recurrent state and one-step primitive. `run_execution_step` performs exactly one deep-supervision step, updates `y/z`, emits logits/halt information and actual work counters, and leaves later steps uncomputed.

`run_with_halting` consumes this state online instead of first calling a full forward. Recorded stopping reasons are:

```text
policy_halt
budget_exhausted
model_exhausted
```

Focused contracts verify first-step policy exit, full model exhaustion, forced two-step budget exit, returned-state observability and agreement with the corresponding independently truncated dense reference.

## Real early exit

The retained batch-size-one four-step experiment forced a policy halt after supervision step 0:

```text
executed steps        1 vs 4 full
recursive cycles      1 vs 4
block applications    2 vs 8
Q/K/V vectors        32/32/32 vs 128/128/128
FFN vectors           32 vs 128
query-key pairs     2048 vs 8192
```

Mean descriptive wall timing:

```text
dense direct          9.6538 ms
first-step halt       2.2547 ms
```

The halt path therefore skipped the later three supervision steps and was about `76.6%` faster in this small controlled measurement. The halter decision itself cost about `0.0832 ms` in the retained step-0 run.

## Faithful active-token contract

Partial batch-size-one sparsity compacts only operations whose semantics can be preserved exactly for the accepted one-block graph:

- Q rows;
- active-query attention score/output work;
- attention output-projection rows;
- FFN rows;
- recurrent RMSNorm/A8 update rows.

K and V remain dense over the complete token sequence. Frozen tokens are not removed from attention context and their recurrent `y/z` state is copied forward unchanged.

At 50% active density over four steps:

```text
Q vectors              64 vs 128 dense
K vectors             128 vs 128 dense
V vectors             128 vs 128 dense
attention outputs      64 vs 128
FFN vectors            64 vs 128
query-key pairs      4096 vs 8192
active updates          64
frozen copies           64
```

Multi-block/batched partial masks fall back to dense execution where exact intermediate frozen-context semantics are not implemented. M11 does not claim whole-model linear scaling with active density.

## Empty-active and reactivation behavior

Empty active set:

```text
recursive cycles    0
block applications  0
Q/K/V work          0
FFN work            0
y/z                  unchanged exactly
```

The step boundary can still execute output/halting logic so the result remains observable.

Reactivation is explicit:

- `allow` lets a later router decision reactivate a token;
- `sticky` permanently freezes a token after its first frozen step.

Both behaviors are covered by focused tests.

## Native adaptive work

The M11 native path reuses M10's accepted packed-ternary FP32 C++ linear. Active pointwise rows are gathered into that existing primitive; K/V calls remain dense.

Four-step native work:

```text
                           full       50% active
native input vectors        832          576
native scalar products 1,583,104      927,744
native linear calls          52           52
Q vectors                   128           64
K vectors                   128          128
V vectors                   128          128
```

This is a `30.77%` reduction in native input vectors and `41.40%` reduction in native scalar products. Call count does not fall because the same layers are invoked with smaller row matrices.

Full-density native incremental output agrees with the M10 dense runtime. Partial native output agrees with the corresponding PyTorch adaptive reference within the accepted numerical boundary. Native empty-active tests additionally verify no recursive Q/K/V/FFN work and exact state preservation.

## Overhead and slower cases

Retained internal timing for the 50%-active router+halter path:

```text
core              8.5758 ms
router input      0.0892 ms
router            0.3037 ms
halter            0.3941 ms
decode            0.0075 ms
initialization     0.1358 ms
total              9.7164 ms
```

External mean wall timing:

```text
dense direct               9.6538 ms
full-density state         9.5655 ms
50%-active router          9.9150 ms
all-frozen first step      0.7331 ms
native dense               9.5699 ms
native full incremental    9.5787 ms
native 50%-active          9.4965 ms
```

The reference 50%-active adaptive path was about `2.7%` slower than dense despite lower arithmetic because control, router and gather/scatter overhead dominated this tiny workload. Native 50%-active was only about `0.86%` faster than native full incremental and essentially tied with simple native dense execution. M11 therefore establishes work reduction, not a universal sparse latency speedup.

## Descriptive output behavior

On 32 retained held-out 4x4 examples:

```text
dense structural score             0.330078
first-step halt score              0.332031
first-step minus dense            +0.001953
first-step/dense answer agreement   0.4375
50%-active score                    0.326172
50%-active minus dense             -0.003906
50%-active/dense answer agreement   0.3750
```

The early result happened to score slightly higher in this small descriptive sample, while exact output agreement remained low. The 50%-active execution was modestly worse. These are not learned-policy-quality claims; they are retained evidence that execution adaptation can change predictions and accuracy.

## Accepted execution

```text
head          c713f814c2382d5b2776d40160e3bcaef1e9b6b6
run           34151872552
job           101835666462
artifact      m11-adaptive-execution-evidence
artifact id   10029673915
ZIP SHA256    9f26142027ad8b7facdad756b6f59b4ef098f2254ba35ca53eb3821eb6637d39
size          29,653 bytes
```

```text
focused M11   13 passed, 1 deselected
full fast     263 passed, 16 deselected, 1 pre-existing test warning
compile       0
focused       0
experiment    0
evidence      0
fast          0
```

## Remaining boundary

M11 does not establish:

- learned router quality;
- learned halter quality;
- universal latency improvement;
- whole-model linear density scaling;
- multi-block token compaction;
- batched/ragged compaction throughput;
- exact-solve improvement;
- energy superiority;
- learned-search benefit.

## M11 decision

**PASS.** Earlier halting executes fewer real steps; sparse execution skips the declared active-query/pointwise operations while preserving dense K/V context; full-density, truncated-reference and native/reference equivalence contracts pass; and router/halter overhead is explicitly measured.

---

# Milestone 12 — grounded router/halter RL training path

**Stage status: COMPLETE on `research/m12-router-halter-rl`.**

Authoritative preregistration: [`M12_PROTOCOL.md`](M12_PROTOCOL.md).
Accepted evidence/claims boundary: [`M12_ACCEPTANCE_GATE.md`](M12_ACCEPTANCE_GATE.md).

## Real training path

M12 adds a real command that strictly loads a `spectra.training` reasoner checkpoint and a compatible `grounded_state_verifier` checkpoint, freezes both, and runs 120 actor-critic optimizer updates over the M11 step/state execution interface.

Trainable optimizer ownership is exactly:

```text
RLTokenRouter
HaltingPolicy
online LatentValueHead critic
```

The target critic, recursive core and grounded verifier are excluded from the optimizer. Every training update checks that those forbidden families receive no gradients. The target critic is gradient-free and changes only through explicit Polyak update.

## Episode and reward semantics

For each example the training path tracks transition validity, true termination and time-limit truncation separately.

Independent terminal correctness is the symbolic Sudoku validator. Exact success bypasses the learned halter. A voluntary unsolved halt is terminal; an unsolved final-horizon stop is truncation.

GAE uses per-example masks:

```text
bootstrap = valid * (1 - terminated)
trace     = valid * (1 - terminated) * (1 - truncated)
```

so truncation retains the final next-state bootstrap but ends the sampled trace, while a true terminal has no bootstrap.

Grounded potential shaping uses the same discount as the return:

```text
gamma * Phi(next_effective) - Phi(current)
```

with terminal next potential zero and real next potential retained for truncation.

The cost objective uses only declared logical proxies:

```text
lambda_step          0.01
lambda_active_token  0.02 * active_density
cost_kind            logical_step_token_proxy_v1
measured energy      not used
```

No step/token proxy is called joules or physical energy.

## Forced actions

The step-zero router action is forced all-active and receives zero actor credit. Exact-success termination and final-horizon stopping likewise do not manufacture halter decisions.

The historical `halting_episode` helper was corrected so its forced final stop contributes no fake REINFORCE log-probability.

Focused tests verify that changing a forced slot's synthetic log-probability does not change policy loss and that its gradient is exactly zero.

## Hand-verified rollout math

Focused M12 tests include hand-computed terminal, truncated and mixed-batch rollouts. They verify:

- true terminal bootstrap is zero;
- time-limit truncation retains the final critic bootstrap;
- truncation ends the GAE trace;
- invalid post-terminal batch slots contribute nothing;
- discounted potential shaping uses the declared terminal treatment;
- target critic remains gradient-free and updates only by Polyak.

Accepted focused result:

```text
22 passed, 1 deselected
```

## Strict pilot artifacts

```text
reasoner SHA256       495ad9f7ff1a30cb1cfc8e7cacaa994bb5ca24257287f83757d83b53b9a1c736
grounded verifier     af65d46981b8194b8d08660364de26fd79003ef552690c4b749f237d968bbf72
RL input artifact     da8285841c7cd16765744d539339664fa472a8504e99fa9f16bfefc1a07aa967
adaptive RL checkpoint ead326e0ee241008dedeed53beaa881e937fc445dcbb9330e958b438718ccffd
```

The RL input artifact contains 256 training and 96 held-out puzzle inputs and explicitly contains no solution targets.

The reasoner is a bounded mechanics fixture, not a strong solver:

```text
validation board accuracy  0.0
validation cell accuracy   0.169921875
```

The grounded verifier training states are strongly imbalanced (`24` positive / `616` negative). Its `0.942708` validation accuracy is therefore not treated as strong verifier evidence.

## Real update and ownership evidence

Before/after hashes changed for:

```text
router          true
halter          true
online critic   true
target critic   true
```

and remained exactly unchanged for:

```text
reasoner        false
verifier        false
```

The optimizer owned exactly 12 parameter tensors: the union of router, halter and online critic, with zero overlap with the target critic, verifier or recursive core. No forbidden gradients were observed.

The strict `spectra.adaptive_rl` v1 checkpoint reloaded exact router, halter, online-critic and target-critic tensor states before held-out evaluation.

## Bounded pilot training behavior

Mean over the final ten updates:

```text
task success          0.000000
realized steps        4.000000
active density        0.253516
router entropy        0.033420
halter entropy        0.008408
router policy loss   -0.069069
halter policy loss    0.000029
value loss            0.000081
total loss           -0.069418
```

The actors became very low entropy. The router moved toward near-minimal optional activity, while the halter did not learn useful early termination.

## Held-out comparison — negative policy result

All controllers use the same 96 held-out puzzles, exact symbolic success gate and maximum four-step horizon.

```text
                 success   structural score   steps   active density   total proxy
fixed depth      0 / 96       0.0378689        4.0         1.00           0.12
heuristic        0 / 96       0.0378689        4.0         1.00           0.12
learned RL       0 / 96       0.0348824        4.0         0.25           0.06
```

Every held-out example ended by budget truncation under all three controllers. The trained halter therefore produced no held-out early-exit benefit.

Learned RL versus fixed depth:

```text
success delta          0.000000
structural-score delta -0.0029865
proxy-cost delta       -0.0600000
```

The learned controller cut the declared logical compute proxy in half but slightly worsened the already-low structural score. Mean density `0.25` over a four-step episode with a forced fully active first step is consistent with router collapse toward freezing almost all optional later work.

**No learned-control superiority is established.**

## Accepted execution

```text
head          c54a719f8c30c958e58b4ecb32abb54a75de3cfd
run           34154069653
job           101842113858
artifact      m12-router-halter-rl-evidence
artifact id   10030392531
ZIP SHA256    8a23f819b00e47e60fd6d7c7f246fb4f362be305253749f2a9c044fdd1f7bdc4
size          346,998 bytes
```

```text
compile       0
focused       0
fixture       0
training      0
evidence      0
fast          0
full fast     272 passed, 16 deselected, 1 pre-existing test warning
```

The first end-to-end M12 attempt (`34153655193`) failed before any RL optimizer update because a Python boolean horizon flag was inverted with bitwise `~`, promoting the episode mask to integer. The accepted fix uses an explicit boolean horizon tensor; no reward, hyperparameter, baseline or success criterion changed after that failure.

## M12 decision

**PASS — grounded RL training path only.** A real training command updates the intended policies and critic, hand-verified rollout cases produce the declared targets, frozen checkpoint ownership is enforced, a strict adaptive-RL checkpoint is written/reloaded, and held-out behavior is compared with simple controls.

**Learned-control quality result: NEGATIVE / COLLAPSED.** The pilot did not produce useful halting, solved no held-out puzzles, and traded lower logical proxy cost for a slightly worse structural score.

M12 does not establish learned adaptive-control superiority, exact-solve improvement, latency speedup, measured-energy reduction, batched-compaction efficiency, broad generalization or learned-search benefit.

**Stop here for M12. Do not begin M13 automatically.**
