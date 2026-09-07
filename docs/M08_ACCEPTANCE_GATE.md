# Milestone 08 Acceptance Gate — Correct, Inspectable MCTS Reference

**Decision: PASS for the M08 search-semantics and accounting contract.**

M08 passes because the rollout-budget edge cases, finished-tree export, hand-checkable deterministic reference tree, virtual-loss cleanup invariants, actual-work accounting, deterministic tie semantics, horizon bounds, repeated-call reset behavior, and the tested serial/batched agreement boundary all pass.

This is **not** a claim that MCTS improves task accuracy, that the current verifier is reliable at arbitrary search depth, that leaf batching with `leaf_batch > 1` follows the serial trajectory, or that this is an optimal production search algorithm.

## Frozen semantics

Full preregistration: [`M08_PROTOCOL.md`](M08_PROTOCOL.md).

For positive search budgets, fixed problem context is `x`/`x_emb`; each native node stores:

```text
(y, z_codes, z_scale, depth, action_from_parent, path)
```

Search statistics are separate FP/integer state:

```text
prior P
visits N
value_sum W
q = W/N
```

Actions are codebook ids in ascending integer order. A transition dequantizes `z`, applies the selected residual action, advances the actual `(y,z)` state through exactly checkpoint `T` recursive cycles, optionally snaps through latent VQ, and requantizes `z` to INT8 storage.

Selection uses:

```text
q(child) + c_puct * prior(child) * sqrt(parent.visits + 1) / (1 + child.visits)
```

Children are action ordered, so exact ties deterministically choose the lowest action id.

Each rollout selects a leaf/horizon node, expands that selected node once if it is below `max_depth`, evaluates the **selected node itself**, and backs the real scalar evaluation to every node on that selected path. The newly materialized children are choices for later rollouts; they are not silently substituted as the current rollout's target.

For positive budgets, the chosen output node is the evaluated node with the highest observed value. Exact value ties retain the earliest evaluated node. Decoding is separate from search and is explicitly counted.

## Zero-search baseline

`n_rollouts = 0` has explicit non-MCTS semantics: run the ordinary greedy TRM forward, retain its final `(y,z)` state as the single inspectable root, and use zero codebook transitions and zero verifier evaluations.

It is not a decode of an all-zero search root and is not reported as an MCTS gain.

## Exact budget semantics

`n_rollouts` is an integer `>= 0`. For `n_rollouts > 0`, it means exactly the number of real selected-state verifier evaluations.

Initial root expansion is real work and is counted. Serial search performs one verifier API call per rollout. Batched search uses:

```text
current_batch = min(leaf_batch, remaining_rollouts)
```

until `remaining_rollouts == 0`.

The focused tests cover the requested edge cases with `leaf_batch=4`:

| Requested rollouts | Real evaluations | Batch sizes |
|---:|---:|---|
| 0 | 0 | `[]` |
| 1 | 1 | `[1]` |
| 3 (< batch) | 3 | `[3]` |
| 4 (exact batch) | 4 | `[4]` |
| 5 (non-divisible) | 5 | `[4,1]` |
| 8 (multiple exact batches) | 8 | `[4,4]` |

The old floor-division behavior (`max(1, n_rollouts // leaf_batch)`) is no longer used.

## Actual-work accounting

Each search resets and records one canonical work structure. The native search itself—not a separate wrapper—owns the counters used by tree export and checkpoint-backed evaluation.

Recorded fields include:

- requested and completed rollouts;
- initial expansion calls and initial expansion transition calls;
- all successful/attempted codebook transition calls entering `_step`;
- recursive-cycle calls performed by transitions;
- expansion calls;
- verifier API calls and successful verifier evaluations;
- selection edges traversed;
- leaf-batch count and actual batch sizes;
- virtual-loss applications and cleanups;
- greedy-forward calls;
- decode calls;
- maximum reached depth;
- evaluated action paths;
- best path/value;
- status/error metadata.

`eval/research_eval.py::CountingLatentNativeMCTS` now maps the canonical M08 work record to the historical M05 field names instead of maintaining independent counters. Checkpoint-backed realized settings explicitly record:

```text
mcts_rollout_semantics = real_leaf_state_evaluations
mcts_max_depth         = 32
initial_expansion_counted = true
```

This closes the duplicate-accounting drift exposed by the first M08 CI run.

## Hand-checkable reference tree

`eval/mcts_reference.py` is deliberately independent of torch, TRM, learned actions, and neural verifiers.

Frozen fixture:

```text
priors    = [0.5, 0.5]
c_puct    = 1.0
max_depth = 2
rollouts  = 4

value((0,))   = 0.2
value((1,))   = 0.8
value((1,0))  = 0.9
all other paths = 0.0
```

The checked result exactly matches the preregistered hand values:

| Path | Visits N | Value sum W | q |
|---|---:|---:|---:|
| root `()` | 4 | 2.8 | 0.7 |
| `(0,)` | 1 | 0.2 | 0.2 |
| `(1,)` | 3 | 2.6 | 0.8666666667 |
| `(1,0)` | 2 | 1.8 | 0.9 |
| `(1,1)` | 0 | 0.0 | 0.0 |

```text
chosen path          (1,0)
best value           0.9
expansion calls      3
transition calls     6
verifier evaluations 4
selection edges      6
max depth reached    2
```

Exact reproduction:

```bash
python scripts/run_m08_reference.py --out outputs/m08_reference_tree.json
```

## Virtual-loss invariants

Virtual loss exists only while selecting multiple leaves before a batched evaluation. It must never survive into exported `N`, `W`, or `q`.

M08 tests both major exception classes:

1. batched verifier evaluation raises after virtual penalties were installed;
2. a transition raises during expansion after virtual penalties were installed.

For both cases, the exception escapes, the partial tree remains inspectable with `status=error`, and:

```text
virtual_loss_applications == virtual_loss_cleanups
virtual_loss_outstanding  == 0
completed_rollouts        == 0
all exported visits/value_sum contain no temporary penalties
```

The transition-exception fixture also verifies that the failed transition attempt is retained as actual work rather than disappearing from accounting.

## Repeated-call and horizon behavior

Every call resets the search root, best-node metadata, and work counters. Tests verify two calls on the same controller produce different root objects and the second call's counters contain only the second budget.

`max_depth` must be a positive integer. Selection stops at the horizon and expansion cannot create children deeper than it. A `max_depth=1`, nine-rollout fixture stays at depth one, performs only the initial expansion, and still completes all nine state evaluations.

Invalid negative/non-integer budgets, invalid horizon, invalid PUCT values, non-positive leaf batch, negative/non-finite virtual loss, and invalid batch/input shapes are rejected rather than silently coerced.

## Serial versus batched claims

### `leaf_batch = 1`

Under equivalent deterministic model/verifier/action/settings, the focused test requires batch-one search to agree with serial on:

- chosen action path and decoded output;
- every exported node's visits/value sum/q;
- completed rollout count;
- expansion and transition counts;
- recursive-cycle count;
- verifier count/evaluation count;
- maximum depth.

This equivalence passed.

### `leaf_batch > 1`

No trajectory-equivalence claim is made. Multiple paths are scheduled under temporary virtual loss before the current batch's real values are known, so selection order and later tree shape can legitimately differ from serial MCTS.

The larger-batch tests require only the warranted invariants: exact real-evaluation budget, exact final partial batch, horizon/resource validity, deterministic execution for deterministic components, clean exported non-virtual statistics, actual-work accounting, and retained root/tree metadata.

## Accepted implementation evidence

Implementation head:

```text
6651d5088cce6cd82d431558470346cb03067d94
```

Accepted Actions run:

```text
run      34130972353
job      101770742147
artifact m08-mcts-reference-evidence
id       10022097839
ZIP SHA  a0624e47c496e795400864b037984fad47f28daaf7e91971f6bd8cbd06c2c251
size     6,137 bytes
retention 14 days
```

Test evidence:

```text
reference CLI              PASS
focused M08 contracts      18 passed in 0.08 s
M05/M07/native compatibility 31 passed, 1 deselected in 4.62 s
full fast suite            233 passed, 16 deselected in 52.59 s
compile/reference/focused/compat/fast exit codes = 0/0/0/0/0
```

Key retained hashes:

```text
reference_tree.json  f6f8cfe428f5964a54fabb408b7fc1db06d5a5c26ad1e23e6946f87a69bdef67
pytest_m08.txt        2ff158e0598d4e92245e9df28474438ab9988395c696e6ab3d61327663a2cf7b
pytest_compat.txt     8a22548867ed1cc9e9b72ec1b9aabba8fa9924bf803d5a48db8ff089f5a215e2
pytest_fast.txt       1c59b31237c1f7f6c8828aae44e05ebdf523821442c555462eed380c5a0f3fcb
```

## Pre-acceptance failure retained

Run `34130363528` was not accepted. It exposed two implementation plumbing defects while preserving evidence:

- direct `scripts/run_m08_reference.py` lacked the repository-root import bootstrap;
- M05's old `CountingLatentNativeMCTS` duplicated search accounting and its `_expand` override did not accept the new initial-expansion hook.

The new M08 tests themselves were already green (`17 passed`) in that failed run. The CLI import was repaired, and duplicate M05 counters were removed in favor of the canonical M08 work record. No preregistered rollout, selection, evaluation, backup, horizon, or batching semantics were changed in response to those failures.

## Remaining approximations / unsupported claims

- M08 proves inspectability/correctness for the tested reference semantics; it does not prove those semantics are the uniquely best MCTS design.
- Final node selection is highest **observed verifier value**, not visit-max robust-child selection. This is explicit and tested.
- Positive search evaluates the selected leaf itself after expansion; it does not automatically score newly created children in that rollout.
- Checkpoint-backed research evaluation currently fixes `max_depth=32`; it is explicit but not yet a user-swept `InferenceSetting` parameter.
- `leaf_batch>1` is a virtual-loss scheduling variant and can diverge algorithmically from serial trajectories.
- M07's grounded `(x,y,z)` verifier adapter still refuses the legacy batched z-only callback until a batch interface carries both `y` and `z` for each leaf. M08 does not erase that safety boundary.
- Existing MCTS `q=W/N` values remain bootstrapped search targets, not independent ground truth.
- M08 makes no task-accuracy, energy, scaling, or search-benefit claim.

## M08 decision

M08 closes the **correct, inspectable reference-search and accounting layer** for the tested semantics. Budget edge cases are exact, initial and subsequent work is visible, trees survive for inspection, temporary virtual statistics are cleaned even on tested exceptions, repeated calls reset state, horizon/tie rules are deterministic, and batch-one equivalence is distinguished from larger-batch scheduling invariants.

**Stop here for M08.**
