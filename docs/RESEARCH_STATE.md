# SPECTRA Research State

This is the live milestone register. Historical cumulative state is preserved rather than rewritten:

- M01–M04: [`RESEARCH_STATE_M01_M04.md`](RESEARCH_STATE_M01_M04.md)
- M05-era register: [`RESEARCH_STATE_M05.md`](RESEARCH_STATE_M05.md)
- M06-era register: [`RESEARCH_STATE_M06.md`](RESEARCH_STATE_M06.md)
- complete M07-era live register: [`RESEARCH_STATE_M07.md`](RESEARCH_STATE_M07.md)

The live register below records accepted milestone status and the latest tested boundary.

## Accepted milestone index

| Milestone | Scope | Accepted state |
|---|---|---|
| M01 | trustworthy baseline | merged; `40745dbe185c069aeee9eff3cf63dd411d9e17da` |
| M02 | native-kernel correctness/input contracts | merged; `e0781ec8b4e649ab4ccd48d4cd5f432a9b88d249` |
| M03 | task/data/evaluation contracts | merged; `01638b10777029fb28bb35229e374f0865c6e5d4` |
| M04 | reproducible training/checkpoint state | merged through PR #4; `370caf708755e1c68c59d5696778597f0290ea68` |
| M05 | checkpoint-backed evaluation | merged through PR #5; `1003c59e17dc17e652438317b7480c9e898379af` |
| M06 | controlled trained baseline | merged through PR #6; `385da5ef822dcb001c8193a2b8802fa292b31428`; evidence gate clarified through PR #7 `dbec23dd2077fc9f23e6a031b1b2b82c7c731de6` |
| M07 | grounded verifier training/evaluation | merged through PR #8; `d15d5578878a175442895867de08d98efb31e6da` |
| M08 | correct, inspectable MCTS reference | **COMPLETE ON `research/m08-mcts-reference`; accepted implementation run `34130972353`; not merged at the time of this entry** |

M06 progress is evidence-based rather than conditioned on beating a baseline. M07 establishes useful shallow meaning for its independently grounded one-cycle verifier target but explicitly does not establish deep-state verifier reliability or MCTS benefit.

---

## Milestone 08 — correct, inspectable MCTS reference

**Stage status:** COMPLETE ON `research/m08-mcts-reference`; implementation/evidence gate green; not merged at the time of this entry.

Full preregistration: [`M08_PROTOCOL.md`](M08_PROTOCOL.md). Accepted evidence and claims boundary: [`M08_ACCEPTANCE_GATE.md`](M08_ACCEPTANCE_GATE.md).

M08 changes search semantics, budget accounting, tree inspection, virtual-loss safety, and research-evaluation work provenance. It does not perform or claim a task-accuracy improvement experiment.

### Native search state

For positive search budgets, `x` and `x_emb` are fixed problem context. Each native tree node stores:

```text
(y, z_codes, z_scale, depth, action_from_parent, path)
```

where `z_codes` is literal INT8 latent storage and `z_scale` is the per-token dequantization scale. Search statistics are separate:

```text
prior P
visits N
value_sum W
q = W/N
```

`path` is the ordered action-id trajectory from the root and provides an inspectable deterministic node identity.

### Action and transition

Actions are integer codebook ids in ascending order:

```text
0 .. n_actions-1
```

For each action transition:

1. dequantize the node's `z`;
2. apply the codebook residual action;
3. preserve/use the node's actual `y`;
4. run exactly model/checkpoint `T` calls to `TRM.recursive_cycle(x_emb,y,z)`;
5. optionally apply configured latent VQ;
6. requantize the resulting `z` to INT8 storage;
7. create the child at `depth+1` with its action id/path.

Initial root expansion uses the same transition path and is counted as real work.

### Selection and deterministic ties

PUCT is:

```text
q(child) + c_puct * prior(child) * sqrt(parent.visits + 1) / (1 + child.visits)
```

Children are stored in ascending action order. Python's first-maximum behavior is part of the contract: exact PUCT ties choose the lowest action id.

Selection stops at an unexpanded node or at `max_depth`.

### Evaluation timing and backup

One serial rollout is:

1. select the leaf/horizon state;
2. if below `max_depth`, expand that selected node once;
3. evaluate the **selected state itself**;
4. update best-observed node using strict `>` (earliest evaluation wins exact value ties);
5. back the real scalar value up to every node on the selected path, including the selected node.

Real backup is:

```text
N <- N + 1
W <- W + value
q <- W/N
```

New children created by the expansion are future choices. They are not silently used as the current rollout's evaluation target.

MCTS `q=W/N` remains a bootstrapped search statistic/target, not independent ground truth.

### Horizon and final answer

`max_depth` must be a positive integer. No child can be expanded beyond that bound.

For positive budgets, search returns the **evaluated node with the highest observed verifier value**. This is intentionally not visit-max robust-child selection. `decode(node)` is separate from search and is counted.

### Explicit zero-search baseline

`n_rollouts = 0` is the ordinary greedy TRM baseline:

```text
model.forward(x, height, width)
```

The final greedy `(y,z)` becomes a one-node inspectable root. The zero-search record contains:

```text
greedy_forward_calls = 1
codebook transitions = 0
verifier evaluations = 0
```

It is not an all-zero-root decode and is not labeled an MCTS gain.

### Exact rollout-budget semantics

`n_rollouts` is an integer `>=0` and for positive search means exactly the number of real selected-state verifier evaluations.

Batched scheduling uses:

```text
current_batch = min(leaf_batch, remaining_rollouts)
```

until the budget is exhausted. The previous floor-division form `max(1, n_rollouts // leaf_batch)` is removed.

Focused batch-budget cases (`leaf_batch=4`):

| Requested | Completed real evaluations | Batch sizes |
|---:|---:|---|
| 0 | 0 | `[]` |
| 1 | 1 | `[1]` |
| 3 | 3 | `[3]` |
| 4 | 4 | `[4]` |
| 5 | 5 | `[4,1]` |
| 8 | 8 | `[4,4]` |

Thus small, less-than-batch, exact-batch, non-divisible, and multiple-exact-batch budgets have explicit tested semantics.

### Canonical actual-work record

`LatentNativeMCTS` now owns one per-search work record shared by serial search, batched search, tree export, and checkpoint-backed research evaluation. Recorded work includes:

- requested/completed rollouts;
- initial expansion calls/transitions;
- expansion calls;
- transition calls;
- recursive-cycle calls;
- latent-VQ calls;
- verifier API calls;
- successful verifier evaluations;
- selection edges;
- actual leaf-batch sizes;
- virtual-loss applications/cleanups;
- greedy-forward calls;
- decode calls;
- maximum depth reached;
- evaluated paths;
- best path/value;
- completion/error status.

`CountingLatentNativeMCTS` no longer maintains a second set of private counters. It maps this canonical record to the historical M05 research fields.

Checkpoint-backed learned-search provenance now explicitly includes:

```text
mcts_rollout_semantics    real_leaf_state_evaluations
mcts_max_depth            32
initial_expansion_counted true
```

### Inspectable tree export

After serial or batched search, `self.root` retains the finished tree. `export_tree()` emits stable, non-latent-heavy records with:

- path/depth/action-from-parent;
- prior;
- visits/value sum/q;
- child paths;
- `y`/INT8-z shape/dtype metadata;
- `y` norm;
- canonical work/statistics record.

Repeated calls reset root, best-node metadata and counters. A test verifies the second call receives a fresh root object and contains only the second call's budget/statistics.

On a raised search exception, the partial root/tree and an `error` work record remain inspectable.

### Neural-independent deterministic reference

`eval/mcts_reference.py` contains no torch/TRM/learned-verifier dependency.

Frozen fixture:

```text
priors       [0.5,0.5]
c_puct       1.0
max_depth    2
rollouts     4
value(0)     0.2
value(1)     0.8
value(1,0)   0.9
other paths  0.0
```

Preregistered and observed result:

| Path | N | W | q |
|---|---:|---:|---:|
| `()` | 4 | 2.8 | 0.7 |
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

This result is checked by `scripts/run_m08_reference.py` against literal expected values, not inferred from the neural search.

### Virtual-loss invariant

For `leaf_batch>1`, paths are selected sequentially under a temporary virtual penalty before batched real evaluation.

All temporary visit/value modifications are removed before real backup. M08 tests cleanup when:

1. batched verifier evaluation raises;
2. a transition raises during leaf expansion after virtual penalties were installed.

Both failure paths retain inspectable error trees and require:

```text
virtual_loss_applications == virtual_loss_cleanups
virtual_loss_outstanding  == 0
completed_rollouts        == 0
```

No temporary visit or value penalty remains in exported node statistics. The transition-exception fixture also confirms the failed transition attempt remains visible in actual-work accounting.

### Serial versus batched agreement boundary

#### `leaf_batch=1`

Under equivalent deterministic settings, M08 requires batch-one search to agree with serial on:

- chosen action path/output;
- all exported node N/W/q statistics;
- completed rollout count;
- expansion/transition/recursive-cycle counts;
- verifier counts;
- maximum depth.

This passed.

#### `leaf_batch>1`

No identical-trajectory claim is made. Several leaf paths are scheduled before the current batch's true values are available, and virtual loss changes within-batch selection. Therefore a larger batch can legitimately take a different trajectory from serial search.

Only tested invariants are claimed: exact rollout budget, correct final partial batch, clean non-virtual statistics, valid horizon/resource bounds, deterministic behavior for deterministic components, actual-work accounting, and retained tree metadata.

### Tests / accepted implementation evidence

Accepted implementation head:

```text
6651d5088cce6cd82d431558470346cb03067d94
```

Accepted run:

```text
Actions run          34130972353
job                  101770742147
artifact             m08-mcts-reference-evidence
artifact id          10022097839
artifact ZIP SHA256  a0624e47c496e795400864b037984fad47f28daaf7e91971f6bd8cbd06c2c251
artifact size        6,137 bytes
retention            14 days
```

Test results:

```text
reference CLI                PASS
focused M08                  18 passed in 0.08 s
native/M05/M07 compatibility 31 passed, 1 deselected in 4.62 s
full fast                    233 passed, 16 deselected in 52.59 s
all gate exit codes          0
```

Key retained hashes:

```text
reference_tree.json f6f8cfe428f5964a54fabb408b7fc1db06d5a5c26ad1e23e6946f87a69bdef67
pytest_m08.txt       2ff158e0598d4e92245e9df28474438ab9988395c696e6ab3d61327663a2cf7b
pytest_compat.txt    8a22548867ed1cc9e9b72ec1b9aabba8fa9924bf803d5a48db8ff089f5a215e2
pytest_fast.txt      1c59b31237c1f7f6c8828aae44e05ebdf523821442c555462eed380c5a0f3fcb
```

Exact commands:

```bash
python scripts/run_m08_reference.py --out outputs/m08_reference_tree.json
python -m pytest tests/test_m08_mcts_reference.py tests/test_m08_virtual_loss_exception.py -q
python -m pytest tests/test_latent_mcts_native.py tests/test_m05_checkpoint_eval.py tests/test_m07_grounded_verifier.py -m 'not slow' -q
python -m pytest -m 'not slow' -ra
```

### Pre-acceptance failure

Run `34130363528` is retained as failed process evidence, not acceptance evidence.

It exposed:

1. the direct reference CLI lacked a repository-root import bootstrap;
2. M05's legacy `CountingLatentNativeMCTS._expand` duplicate wrapper did not accept M08's explicit initial-expansion accounting hook.

The M08-focused semantics tests were already green in that failed run (`17 passed`). The CLI entry point was fixed and M05 duplicate counters were removed in favor of the canonical M08 search-work record. No selection, rollout, backup, horizon, final-answer, or batching acceptance rule was loosened after seeing the failure.

### Remaining approximations / unsupported claims

- The tested M08 semantics are an inspectable reference contract, not proof that they are the only or optimal MCTS semantics.
- Final answer selection is highest observed verifier value, not robust-child/visit-max selection.
- A rollout evaluates the selected leaf after expansion, not one of its newly created children.
- Checkpoint-backed learned-search evaluation currently fixes `max_depth=32`; it is explicit but not yet a user-facing sweep parameter.
- `leaf_batch>1` is a scheduling variant and can diverge from serial trajectories.
- M07's grounded `(x,y,z)` adapter still rejects the legacy batched z-only callback until batched leaves carry both state components. M08 preserves that boundary.
- MCTS backup values are bootstrapped and are not independent verifier truth.
- M08 does not establish task accuracy improvement, search benefit, energy savings, scaling laws, or target-hardware performance.

### M08 decision

M08 passes its acceptance gate: **budget edge cases, reference-tree results, tree export, virtual-loss cleanup (including tested exceptions), repeated-call reset, horizon/tie rules, actual-work accounting, and batch-one serial equivalence are all tested; claims for larger batches are limited to validated invariants.**

**Stop here for M08.**
