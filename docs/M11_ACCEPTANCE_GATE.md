# Milestone 11 Acceptance Gate — Real Adaptive Execution

**Decision: PASS.**

M11 establishes a real batch-size-one adaptive inference path. The accepted implementation executes one supervision step at a time, can stop before later recursion is computed, can skip a faithful subset of active-token work while preserving full key/value attention context, exposes stopping/state/work information, reuses the accepted M10 packed-ternary native primitive, and records the cost of the control machinery itself.

Authoritative preregistration: [`M11_PROTOCOL.md`](M11_PROTOCOL.md).

## Accepted execution

```text
branch        research/m11-adaptive-execution
head          c713f814c2382d5b2776d40160e3bcaef1e9b6b6
Actions run   34151872552
job           101835666462
artifact      m11-adaptive-execution-evidence
artifact id   10029673915
ZIP SHA256    9f26142027ad8b7facdad756b6f59b4ef098f2254ba35ca53eb3821eb6637d39
size          29,653 bytes
```

Retained evidence hashes:

```text
summary.json       ae74e5b8034ae9ce0c7d28dc1ba1d40dc630c6aec639be1f0d4caffeb5184492
measurement.json   ae74e5b8034ae9ce0c7d28dc1ba1d40dc630c6aec639be1f0d4caffeb5184492
score_rows.json    d081381e66cc164815d3269a3d9a81bb000bafda55e2f73fe984cc87d4e4463c
m11_cpu_artifact   0493170b628e9dc8b221e63211b06245309dc716b51901ac17d69d22a527ff7c
```

Execution gate:

```text
compile      0
focused      0
experiment   0
evidence     0
fast         0
focused      13 passed, 1 deselected
full fast    263 passed, 16 deselected, 1 warning
```

The warning is the pre-existing test-only M10 tensor-to-scalar warning; no M11 regression failed.

---

## 1. Explicit incremental state

`model.trm.TRM` now exposes an inference state containing the already-computed input embedding, recurrent `y/z`, next supervision-step index, active mask, freeze history, and cumulative work counters.

The execution primitive:

```text
run_execution_step(state, ...)
```

executes exactly one deep-supervision step. It does not materialize a later step. The ordinary dense `forward()` remains compatible and is implemented by repeatedly invoking the same step interface.

Full-density execution deliberately still passes through ordinary `nn.Module.__call__` for the block, preserving forward-hook and telemetry semantics. The first implementation attempt bypassed those hooks; the accepted implementation fixed this and the full telemetry regression suite passed.

## 2. Halting is now physical early exit

`model.halting.run_with_halting` no longer computes `model(x)` before selecting an intermediate answer. It now runs:

```text
initialize state
-> execute one supervision step
-> evaluate halter
-> return immediately if stopped
-> otherwise execute the next step
```

Observable stop reasons are:

```text
policy_halt
budget_exhausted
model_exhausted
```

The returned detailed result includes `halt_step`, `executed_steps`, final logits/answer, recurrent `y/z`, per-step halt probabilities, work counters and timing.

Focused tests verify:

- policy halt at step 0 executes exactly one step;
- a no-halt run reaches `model_exhausted` after all four steps;
- a forced two-step budget returns `budget_exhausted`;
- online early-exit state/logits agree with the independently truncated dense reference (bit-exact on the tested ternary path; fixed `2e-6` absolute tolerance for the FP MHA truncation test).

### Measured early exit

For the retained four-step batch-size-one experiment:

| Quantity | halt after step 0 | full no-halt |
|---|---:|---:|
| executed supervision steps | 1 | 4 |
| recursive cycles | 1 | 4 |
| block applications | 2 | 8 |
| Q vectors | 32 | 128 |
| K vectors | 32 | 128 |
| V vectors | 32 | 128 |
| attention output vectors | 32 | 128 |
| FFN vectors | 32 | 128 |
| query-key pairs | 2,048 | 8,192 |

Mean wall time was `2.2547 ms` for the forced policy halt versus `9.6538 ms` for the direct dense reference, a descriptive reduction of about `76.6%` on this small CI workload. This is evidence that later recursion was not computed; it is not a claim that every learned halter will choose an appropriate stopping point.

The measured step-0 halter itself cost `0.0832 ms` inside that retained run.

---

## 3. Faithful active-token execution

M11 intentionally does **not** remove frozen tokens from the transformer sequence.

For the accepted one-block batch-size-one sparse contract:

### Work compacted to active rows

- attention Q projection;
- attention score/output work for active query rows;
- attention output projection;
- FFN input/GELU/output rows;
- recurrent norm/A8 update rows.

### Work that stays dense

- embeddings/positions;
- the `norm1` states needed to construct full attention context;
- attention K projection for every token;
- attention V projection for every token;
- K/V reshape/context and normalization over the complete key set;
- router evaluation;
- halter pooling/head;
- output-head work needed at a returned supervision-step boundary.

Frozen token `y/z` values are copied forward unchanged. Frozen tokens remain available as K/V context, so active queries see the complete token set rather than a silently shortened attention problem.

Partial masks on graph shapes for which this exact contract is not implemented—most importantly multi-block sparse propagation—fall back to dense block execution rather than changing attention semantics.

## 4. Real sparse work reduction

With retained active density `0.5` for all four steps, the PyTorch/reference path recorded:

| Work | dense | 50% active |
|---|---:|---:|
| Q vectors | 128 | 64 |
| K vectors | 128 | **128** |
| V vectors | 128 | **128** |
| attention output vectors | 128 | 64 |
| FFN vectors | 128 | 64 |
| recurrent norm/A8 vectors | 128 | 64 |
| query-key pairs | 8,192 | 4,096 |

This is the intended contract: query/pointwise work falls with active density, while K/V stay dense to preserve context.

### Verified M10 native primitive

`deploy.m11_adaptive_runtime.AdaptiveCPURecursiveRuntime` gathers active pointwise rows and sends those rows through the same M10 `packed_ternary_fp32_scalar_linear_v1` primitive. It does not introduce a second arithmetic implementation.

Measured native work over the four-step run:

| Work | full density | 50% active | reduction |
|---|---:|---:|---:|
| native input vectors | 832 | 576 | `30.77%` |
| native scalar products | 1,583,104 | 927,744 | `41.40%` |
| native linear calls | 52 | 52 | `0%` |
| Q vectors | 128 | 64 | `50%` |
| K vectors | 128 | 128 | `0%` |
| V vectors | 128 | 128 | `0%` |

The call count remains 52 because the same layers are invoked; the matrices passed to the compactable projections contain fewer input rows. This is why actual input-vector/scalar-product counters, rather than call count alone, are the relevant evidence.

Native full-density incremental output agrees with the accepted M10 dense runtime. Native partial execution agrees with the corresponding PyTorch sparse reference within the existing M10 numerical boundary.

---

## 5. Empty-active / frozen-state semantics

The all-frozen path is explicit rather than an accidental zero-length tensor case.

When the active set is empty for a supervision step:

```text
recursive cycles       0
block applications     0
Q/K/V vectors          0 / 0 / 0
FFN vectors            0
recurrent norm/A8      0
state y/z              copied exactly
```

The step boundary can still run the output head and halter so that the caller can observe/return an answer and stop reason. The native edge test likewise verifies zero Q/K/V/FFN work and exact recurrent-state preservation; only the native output-head call remains to produce step logits.

## 6. Reactivation policy

Two explicit policies are tested:

- `allow`: a token frozen on one step may become active on a later router decision;
- `sticky`: once frozen, a token remains frozen for the rest of the solve.

`allow` is the default because it preserves the historical router's ability to reconsider a token. Tests verify that the previously inactive half can update under `allow`, while the same attempted reactivation produces an empty active set and unchanged state under `sticky`.

No implicit reactivation behavior is left to mask arithmetic.

---

## 7. Router/halter overhead and slower cases

Control overhead is measured separately rather than subtracted from the reported adaptive wall time.

For the retained 50%-active reference adaptive run:

```text
core              8.5758 ms
router input      0.0892 ms
router            0.3037 ms
halter            0.3941 ms
decode            0.0075 ms
initialization     0.1358 ms
total              9.7164 ms
```

Mean external wall timing over 12 repetitions:

```text
dense direct             9.6538 ms
full-density state       9.5655 ms
50%-active router        9.9150 ms
first-step halt          2.2547 ms
all-frozen first step    0.7331 ms
```

The 50%-active reference path was about **2.7% slower than dense direct** even though the intended arithmetic counters fell. At this small width/sequence length, gather/scatter, Python control flow, routing and halting can cost more than the skipped arithmetic saves.

Native timing showed the same narrow margin rather than a dramatic speedup:

```text
M10-style native dense       9.5699 ms
native full incremental      9.5787 ms
native 50%-active            9.4965 ms
```

The 50%-active native path was only about `0.86%` faster than native full incremental and remained essentially tied with the simple dense native path. M11 therefore makes **no general sparse speedup claim**.

---

## 8. Descriptive output-quality behavior

M11 is an execution milestone, not a trained router/halter-quality milestone. On 32 retained held-out 4x4 examples:

```text
dense structural score             0.330078
first-step halt score              0.332031
first-step halt - dense           +0.001953
first-step/dense answer agreement   0.4375
50%-active sparse score             0.326172
sparse - dense                     -0.003906
sparse/dense answer agreement       0.3750
```

The early score happened to be slightly higher in this tiny descriptive sample, but its exact answer agreed with dense on only 43.75% of cases. The 50%-active path was modestly worse on structural score and agreed with dense on only 37.5% of answers.

These results are deliberately retained because adaptive execution can change accuracy. They do not demonstrate learned adaptive policy quality.

---

## 9. What M11 establishes

M11 establishes:

- a real step/state inference interface;
- physical online early exit before later recursion is computed;
- observable stop reasons, steps, states, work and timing;
- faithful batch-size-one active-query sparsity with dense K/V context;
- exact frozen-state semantics plus explicit reactivation policy;
- empty-active core skipping;
- full-density and truncated-reference equivalence tests;
- adaptive execution on the verified M10 native primitive;
- measured reduction in real native input vectors/scalar products;
- explicit router and halter overhead;
- honest slower/accuracy-changing cases.

M11 does **not** establish:

- learned router quality;
- learned halter quality;
- whole-model linear scaling with active density;
- a universal latency speedup;
- exact-solve improvement;
- energy superiority;
- batched/ragged compaction throughput;
- multi-block faithful token compaction;
- learned-search benefit.

## M11 decision

**PASS.** The acceptance gate is satisfied: earlier halting executes fewer steps, sparse execution skips the declared operations while retaining dense K/V context, full-density/truncated/native equivalence contracts pass, and router/halter overhead is measured.

**Stop here for M11. Do not begin M12 automatically.**
