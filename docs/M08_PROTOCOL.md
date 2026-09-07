# Milestone 08 Protocol — Correct, Inspectable MCTS Reference

**Scope:** M08 changes only search semantics, accounting, inspection, and reference tests. It does not claim search improves task accuracy, energy, or reasoning quality.

## Acceptance gate

M08 passes only if all of the following are demonstrated by tests and retained CI evidence:

1. rollout-budget edge cases are exact for `0`, `1`, `< leaf_batch`, non-divisible, and exact-batch budgets;
2. finished trees and per-search work counters are exportable and repeated calls reset state;
3. a neural-independent deterministic reference tree has independently checkable visits, values, work counts, and chosen output;
4. virtual-loss penalties are fully cleaned up, including when batched evaluation raises;
5. batch size one agrees with the serial search under equivalent settings;
6. larger leaf batches are described only by tested invariants, not claimed to have identical trajectories to serial MCTS.

## Native search state

For `n_rollouts > 0`, the fixed input `x` and its embedding `x_emb` are external context. Each tree node stores:

```text
(y, z_codes, z_scale, depth, action_from_parent, path)
```

where `z_codes` is literal INT8 latent storage and `z_scale` dequantizes it. Search statistics are separate:

```text
prior P, visits N, value_sum W, q = W / N
```

`path` is the tuple of integer action ids from the root and is also the deterministic tie-break identity.

## Transition and action

Actions are integer codebook ids in ascending order:

```text
a in {0, ..., n_actions - 1}
```

For a node and action `a`:

1. dequantize `z`;
2. apply `LatentActionCodebook.apply_action(z, a)`;
3. starting from the node's actual `y`, run exactly checkpoint/model `T` calls to `TRM.recursive_cycle(x_emb, y, z)`;
4. optionally apply the configured latent-VQ snap;
5. quantize the resulting `z` back to INT8 storage;
6. create one child at `depth + 1` with the corresponding action id/path.

Every call to this transition is counted. Initial root expansion is real work and is included in transition/recursive-cycle counts.

## Selection

For child `c` of parent `p`:

```text
PUCT(c) = q(c) + c_puct * P(c) * sqrt(N(p) + 1) / (1 + N(c))
```

Children are stored in ascending action order. Python's stable first-maximum rule is made part of the contract: exact PUCT ties choose the lowest action id at that node.

Selection stops at the first node with no children or at `max_depth`.

## Expansion and evaluation timing

For each rollout:

1. select a leaf/horizon node;
2. if its depth is below `max_depth` and it has no children, expand it once;
3. evaluate **that selected node itself**, not one of its newly created children;
4. back up the real scalar value to every node on the selected path, including the selected node.

Thus expansion creates future choices, while the current rollout's target belongs to the state that was selected before expansion.

## Backup target

The search backup is the verifier output used for that selected state. It is an MCTS/search value, not independent ground truth. M07's independently grounded verifier target remains separate from MCTS backups.

For each completed rollout and every node on its path:

```text
N <- N + 1
W <- W + value
q <- W / N
```

Virtual loss is never an exported visit/value target.

## Horizon and termination

`max_depth` is a positive integer and is enforced during selection and expansion. No child deeper than `max_depth` may be created.

`n_rollouts` is an integer >= 0 and means **exactly the number of real leaf/state evaluations requested**.

For `n_rollouts > 0`, root expansion occurs once before the first selection and is counted as work. Search then executes exactly `n_rollouts` real evaluations unless an exception interrupts the call.

## Zero-search / greedy baseline

`n_rollouts = 0` is not a zero-state decode and does not construct an action tree. It is the explicit ordinary greedy TRM baseline:

```text
model.forward(x, height, width)
```

The final ordinary-forward `(y,z)` state is retained as a single root node so tree export remains defined. Work metadata records one greedy forward and zero codebook transitions / verifier calls. Decoding that node returns the ordinary greedy output.

This path is not called MCTS improvement.

## Final-answer selection

For positive search budgets, the returned node is the evaluated node with the highest observed verifier value. Exact value ties keep the earliest evaluated node, which is deterministic under the action-order and PUCT tie rules above.

`decode(node)` is separate and counted. Search itself does not decode candidate answers.

## Serial work accounting

The per-search record includes at least:

- requested/completed rollouts;
- initial expansion calls/transitions;
- total expansion calls;
- transition calls;
- recursive-cycle calls;
- verifier API calls;
- verifier examples/evaluations;
- selection edges traversed;
- greedy forward calls;
- decode calls;
- max depth reached;
- best path/value;
- status (`complete` or `error`).

For positive serial budgets, verifier API calls and verifier evaluations both equal the number of completed rollouts. Transition work can exceed rollout count because every expansion materializes all codebook actions.

## Batched scheduling semantics

`leaf_batch` must be a positive integer. Batched search consumes exactly:

```text
current_batch = min(leaf_batch, remaining_rollouts)
```

until no requested rollouts remain. Therefore no floor division or implicit minimum batch is permitted.

Within a batch, paths are selected sequentially. A temporary virtual loss is applied to each selected path before selecting the next path in that same batch. All temporary changes are removed before real backup. Cleanup occurs in `finally`, including if expansion or batched verifier evaluation raises.

After cleanup, each successful evaluated path receives exactly one real visit/value backup.

### Batch size one

`leaf_batch = 1` must agree with serial search under equivalent model/verifier/settings on:

- chosen path/output;
- exported visits and values;
- completed rollouts;
- expansion/transition counts;
- verifier-evaluation count;
- horizon.

### Larger batches

For `leaf_batch > 1`, scheduling occurs before the current batch's real values are known. Virtual loss can therefore cause a different selection order/tree trajectory from serial search. M08 does **not** claim trajectory equivalence.

Only invariants are required: exact rollout count, valid horizon, clean non-virtual exported statistics, deterministic behavior for fixed deterministic components, correct actual-work counters, and retained tree/root metadata.

## Repeated calls and exceptions

Every search call resets `root`, best-node metadata, and work counters before constructing the new search. The new root is retained as soon as it exists.

If a search raises, temporary virtual penalties must be cleaned before the exception escapes. The partially constructed tree and an `error` work record remain inspectable. A later call starts from a fresh root and fresh counters.

## Neural-independent reference tree

M08 includes a pure deterministic two-action reference MCTS with no torch model, latent tensors, or neural verifier. Fixture semantics:

```text
priors       = [0.5, 0.5]
c_puct       = 1.0
max_depth    = 2
rollouts     = 4
value(path):
  (0,)   -> 0.2
  (1,)   -> 0.8
  (1,0)  -> 0.9
  all other paths -> 0.0
```

Under the selection/evaluation/backup rules above, the expected completed result is frozen before neural M08 testing:

```text
root:   visits=4, W=2.8, q=0.7
(0,):   visits=1, W=0.2, q=0.2
(1,):   visits=3, W=2.6, q=0.8666666667
(1,0):  visits=2, W=1.8, q=0.9
(1,1):  visits=0, W=0.0, q=0.0
chosen path = (1,0)
expansion calls = 3
transition calls = 6
verifier evaluations = 4
```

These numbers are mathematical reference expectations, not generated from the neural implementation.

## Claims boundary

Passing M08 establishes that the tested search implementation has explicit semantics, exact budget handling, inspectable trees/work, clean virtual-loss accounting, and a reference agreement surface.

It does not establish that MCTS improves SPECTRA accuracy, that batched search is identical to serial for `leaf_batch > 1`, that verifier values are correct at arbitrary depth, or that the implementation is an optimal/production search algorithm.
