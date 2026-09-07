# Milestone 11 Protocol — Real Adaptive Execution

**Status: frozen before M11 empirical measurements.**

M11 is exclusively about making SPECTRA's adaptive-compute claims physically real at inference time. It does not change the accepted M10 deployment result and does not reopen M09 learned-search claims.

## Acceptance gate

M11 passes only if all of the following are demonstrated by executable tests and retained measurements:

1. **Real early exit:** an online batch-size-one solve can halt before later supervision steps are computed, and its work counters show fewer recursive cycles / block applications / linear input vectors than the corresponding full run.
2. **Faithful sparse execution:** a partial active-token mask skips the declared query/FFN work while preserving dense key/value context and frozen-token state.
3. **Equivalence:** full-density adaptive execution agrees with the ordinary dense reference; online stopping agrees with the corresponding explicitly truncated dense reference; native and reference adaptive paths agree within the existing M10 numerical boundary.
4. **Edge cases:** empty-active/all-frozen behavior, reactivation policy, and forced budget stops are explicit and tested.
5. **Measured overhead:** router and halter wall time are reported separately from core compute, together with dense/adaptive latency and real operation counts. M11 must report cases where adaptation is slower or changes accuracy rather than hiding them.

## Explicit step/state interface

Inference is refactored around a state that owns the already-computed input embedding and the recurrent answer/scratch states:

```text
ExecutionState
  x_emb              [1, L, D]
  y                  [1, L, D]
  z                  [1, L, D]
  next_step          integer in [0, N_sup]
  active_mask        [1, L, 1] bool/float
  ever_frozen        [1, L, 1] bool
```

`init_execution_state(x)` performs input embedding/position work exactly once.

`run_execution_step(state, ...)` performs **one deep-supervision step only** (all `T` cycles belonging to that step), emits logits/halt state, updates `next_step`, and returns an explicit work record. It must never precompute a later supervision step.

`forward()` remains the dense compatibility API and is implemented by repeatedly invoking the step interface.

## Online halting semantics

`run_with_halting` must not call `model(x)` first. It initializes state and then alternates:

```text
run one supervision step
compute halter decision
if halt -> return immediately
if forced step budget reached -> return immediately
otherwise continue
```

Observable stop reasons are:

- `policy_halt`
- `budget_exhausted`
- `model_exhausted`

The result records:

- zero-based `halt_step`;
- `executed_steps`;
- stop reason;
- final `y`, `z`, logits and answer;
- per-step halt probability;
- core/router/halter timing and work counters.

For batch size one, `executed_steps == halt_step + 1` for policy halts.

## Active-token semantics

M11 does **not** claim that every transformer operation scales linearly with active density.

For the faithful one-block path used by the accepted M10 graph, a partial active mask means:

### Operations that may be skipped for frozen query tokens

- attention `Q` projection rows;
- attention score/output work for frozen query rows;
- attention output-projection rows;
- FFN first linear rows;
- GELU rows;
- FFN second linear rows;
- recurrent RMSNorm/A8 output rows whose state is frozen.

### Operations that remain dense

- input embedding / positional encoding (once per solve);
- block `norm1` inputs needed to form dense K/V context;
- attention `K` projection for **all tokens**;
- attention `V` projection for **all tokens**;
- K/V reshape/transposition and the key/value context itself;
- router evaluation unless its implementation documents a separate sparse policy;
- halter pooling/head;
- final output decoding needed to return an answer.

Frozen tokens remain present as attention keys and values. They are never removed from the sequence, and attention normalization still spans the full token set. Therefore an active query sees the same frozen-token context as in the dense masked reference.

For multi-block models, faithful sparse propagation of inactive intermediate representations would require additional cached/intermediate semantics. M11 therefore **falls back to dense block execution** when the exact one-block sparse contract is unavailable rather than silently changing attention behavior.

## Frozen-state and reactivation policy

A frozen token means its recurrent `y` and `z` values are copied forward unchanged for that supervision step.

Two explicit policies are supported:

- `allow`: every supervision step may reactivate a token based on the new router mask;
- `sticky`: once a token has been frozen it remains frozen for the rest of the solve.

The default for M11 is `allow`, because it preserves the existing router's ability to reconsider tokens. The selected policy is recorded in every adaptive result.

If the active set is empty, the recursive core for that step is skipped completely and `y/z` are unchanged. The halter is still evaluated after the step boundary because global stopping remains observable.

## Native integration

The accepted M10 native primitive already computes a packed-ternary FP32 linear for an arbitrary matrix of input vectors. M11 reuses that verified primitive; it does not introduce a new arithmetic kernel.

For active-only pointwise projections, M11 gathers active token rows, calls the same M10 native linear on the smaller `[A, D]` matrix, and scatters the result back. Dense K/V projections call the unchanged M10 primitive on all `L` rows.

Native work accounting records actual input-vector counts and scalar-product counts per layer. Therefore a lower active density must reduce the declared Q/projection/FFN vector counts while K/V counts remain dense.

## Reference integration

The PyTorch reference uses the same sparse semantics. Full density bypasses the sparse decomposition and calls the ordinary dense block path, providing an exact regression anchor. Partial density is compared with a dense full-block calculation followed by the same recurrent-state freeze mask; active-token outputs must agree within a fixed floating-point tolerance.

## Work counters

At minimum M11 records:

```text
supervision_steps
recursive_cycles
block_applications
attention_q_vectors
attention_k_vectors
attention_v_vectors
attention_output_vectors
ffn_input_vectors
ffn_output_vectors
active_token_updates
frozen_token_copies
all_frozen_step_skips
router_calls
halter_calls
output_head_vectors
native_linear_calls
native_input_vectors
native_scalar_products
```

Counters report actual executed work, not theoretical estimates.

## Timing / overhead experiment

On the CI CPU and on any reproduced target, M11 measures batch-size-one runs after warm-up for at least these cases:

1. dense full-depth;
2. full-density step interface;
3. forced early halt after the first supervision step;
4. partial-density sparse execution;
5. all-frozen step;
6. router + halter adaptive loop.

Report separately:

- total wall time;
- core execution time;
- router time;
- halter time;
- decode time;
- saved core work relative to dense.

Timing is descriptive, not a required speedup claim. For small dimensions or mild sparsity, gather/scatter, Python control flow, router, and halter overhead may make the adaptive path slower even when arithmetic work falls. M11 must say so if observed.

## Accuracy boundary

M11 is an execution milestone, not a learned-routing-quality milestone. A forced early halt or aggressive router may return a less accurate intermediate answer. Measurements report any accuracy/structural-score delta, but M11 does not tune thresholds on the test examples to manufacture parity.

## Batched compaction

Batched ragged compaction is optional and out of scope for acceptance. The accepted path is batch size one. No batched throughput claim is permitted unless a later milestone implements and validates it explicitly.

## Claims boundary

A PASS establishes real batch-size-one adaptive execution with observable stopping, faithful active-query sparsity, verified native/reference integration, and measured overhead.

It does **not** establish:

- learned router quality;
- learned halter quality;
- end-to-end speedup on every workload;
- linear whole-model scaling with active density;
- exact-solve improvement;
- energy superiority;
- batched compaction efficiency;
- deep-search benefit.
