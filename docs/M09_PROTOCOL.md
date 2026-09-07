# Milestone 09 Protocol — Trained Search Actions

**Scope:** only a trained latent action mechanism and its fair held-out comparison.

This protocol is frozen before any M09 development or test comparison is generated.

## 1. Existing gap

`model/latent_action.py` currently exposes trainable `directions` and global `prior_logits`, but `LatentNativeMCTS.search*` executes under `torch.no_grad()`. Search therefore does not train either tensor. The current prior is also global: it does not depend on `x`, `y`, or `z`.

M09 does not treat parameter existence, a decreasing loss, or MCTS visit counts as evidence that an action mechanism learned.

## 2. Research question

Can an offline-trained, state-conditioned latent action mechanism use independently grounded frozen-reasoner trajectories to improve a small, equal-budget held-out search over an unguided search using the **same learned directions**?

The primary practical endpoint is the validated, reference-free Sudoku structural score:

```text
model.verifier.sudoku_score(puzzle, decoded_candidate, box=3)
```

No retained solution/reference answer is an input to action-target generation or the primary comparison.

## 3. Data / isolation

Generated validated 9x9 Sudoku, unique solutions, 30–35 clues.

```text
data seed             20260909
reasoner train        384 puzzles
reasoner validation    96 puzzles
untouched test         128 puzzles
```

The frozen action-trajectory source is partitioned before action fitting:

```text
action-fit puzzles     first 256 reasoner-train puzzles
action-development     remaining 128 reasoner-train puzzles
trajectory depths      0..3
untouched evaluation   all 128 test puzzles; root search only
```

The action-development puzzles are not used in optimizer steps or candidate selection. The untouched test inputs are not inspected for action choice, hyperparameters, revision decisions, or stopping.

The reasoner may have seen the action-development puzzles during its own fixed training schedule; they are therefore a **development** set, not an independent final set. Final acceptance uses the separate test split.

## 4. Frozen reasoner

Train the same small FP reference family used by M07:

```text
TRM FP32
dim=48, layers=1, heads=4
n=1, T=1, N_sup=2
200 AdamW steps
```

After checkpoint reload, every reasoner parameter is `requires_grad=False` and the model is kept in eval mode. A tensor-state hash is recorded before and after all action target generation/training.

Action fitting may backpropagate only through the action module. No MCTS execution is used as an optimizer step.

## 5. Target generation — train only

### 5.1 Candidate bank

Create `24` deterministic fixed candidate residual directions with seed `2901` in the 48-dimensional latent feature space. Each is L2-normalized. Action 0 is identity. Candidate residual scale is fixed at `0.5` before development results.

For every action-fit state `(x,y,z)` and each candidate direction `d_j`, independently evaluate:

```text
z_action = z + 0.5 * d_j
(y_next, z_next) = frozen_reasoner.recursive_cycle(x_emb, y, z_action)
utility_j = sudoku_score(x, argmax(out_head(y_next)), box=3)
```

Identity utility uses the same one-cycle continuation with no residual.

This table is generated without reference solutions and is frozen before policy fitting.

### 5.2 Three direction prototypes

Select three non-identity candidate directions by deterministic greedy **coverage** on action-fit utilities. Starting from identity, each slot chooses the remaining candidate that maximizes the mean per-state best utility obtainable from identity plus the selected candidates. Exact ties choose the lowest candidate index.

This is offline train-only target construction, not gradient descent.

### 5.3 State-conditioned action target

For each action-fit state, restrict the utility table to:

```text
identity + three selected prototypes
```

The hard action label is the highest-utility action, ties to lowest action id. The full four-action utility vector is also retained for the optional development-only revision below.

## 6. Trainable mechanism

M09 introduces a state-conditioned codebook whose search state is:

```text
(x, y, z)
```

It owns:

- three trainable residual `directions`,
- an input-token embedding,
- small `y` and `z` projections,
- a pooled MLP policy head producing four state-conditioned prior logits.

There is **no global learned prior** in this M09 class. Legacy `LatentActionCodebook.prior_logits` remains supported for old checkpoints.

The state-conditioned prior is:

```text
P(a | x,y,z) = softmax(policy_logits(x,y,z))
```

### 6.1 Optimizer ownership

One explicit AdamW optimizer owns all M09 action-module parameters. The reasoner is excluded.

Version-1 loss:

```text
L = CE(policy_logits, oracle_best_action)
  + 0.5 * MSE(trainable_directions, selected_prototype_directions)
```

Fixed budget:

```text
steps        400
batch         64 states
lr          2e-3
weight decay 0.01
grad clip     1.0
```

Required training evidence:

- direction gradient norm > 0 on an intended training step,
- policy-head gradient norm > 0,
- direction tensor changes after optimizer step,
- policy tensor changes after optimizer step,
- applying a changed learned direction changes the transition input/state,
- changing state-conditioned policy logits changes action priors/selection in a deterministic fixture.

## 7. Search integration / evaluation harness

`LatentNativeMCTS` may consume `priors_for_state(x,y,z)` when an action module exposes it; otherwise it uses the legacy global `priors()` path.

M09's controlled action-isolation comparison uses the M08 serial search semantics with:

```text
max_depth   = 1
n_rollouts  = 2
c_puct      = 1.5
```

The search evaluator is a transparent symbolic-oracle harness that decodes the selected state and returns `sudoku_score`. This intentionally isolates the action mechanism from M07 verifier error. It is **not** a production no-decode or MCTS-quality claim.

Positive-search final selection remains the M08 rule: highest observed evaluator value; exact value ties keep the earliest evaluated node.

## 8. Baselines

On every development/test puzzle compare:

1. **trained state-conditioned mechanism** — trained directions + state-conditioned priors;
2. **equal-budget unguided** — the exact same trained directions with uniform/global equal priors;
3. **fixed-random directions** — first three deterministic candidate-bank directions, frozen, uniform priors;
4. **identity only** — action 0 only.

Primary fairness comparison is (1) vs (2), because these share the exact direction set and differ only in learned state-conditioned guidance.

For (1), (2), and (3), per puzzle:

```text
initial expansion transitions = 4
real evaluator evaluations    = 2
max depth                     = 1
```

Identity is cheaper and its lower work is reported rather than hidden.

The state-conditioned policy's own forward pass is counted separately and is not represented as free compute.

## 9. Predeclared practical benefit / cost

### Development gate

Before touching the test split, version 1 must satisfy all of:

```text
mean score(trained) - mean score(unguided) >= 0.005
paired wins(trained > unguided) > paired losses
mean score(trained) >= mean score(fixed random)
mean score(trained) >= mean score(identity)
trained and unguided have identical transition/evaluator budgets
```

Additional diagnostics: policy top-1 accuracy, top-2 oracle-action recall, action-label entropy, identity-best rate, selected-prototype coverage gain, direction cosine fidelity, and per-action selection frequency.

### Cost bounds

```text
candidate bank                    24
selected nonidentity actions       3
target state-action equivalents <= 25,000
optimizer steps                 <= 400 per fitted version
state-conditioned parameters    < 50,000
search rollout budget             2
search depth                       1
```

Target-generation state-action equivalents and actual batched recursive-cycle calls are both retained separately from search cost.

## 10. One allowed development-only revision

If version 1 misses the development gate, retain it and diagnose:

- target quality / oracle advantage,
- state-conditioning accuracy and top-2 recall,
- exploration / action-frequency collapse,
- optimizer/gradient behavior,
- direction fidelity,
- search integration and realized work.

The only preregistered revision is **version 2 utility distillation** on the same frozen candidate/prototype table and the same optimizer budget. It replaces hard CE with a soft utility target:

```text
teacher(a|s) = softmax((utility_a - max_a utility_a) / 0.02)
L_policy = KL(teacher || learned_prior)
L_total  = L_policy + 0.5 * direction_MSE
```

No candidate direction, scale, search budget, development set, or practical threshold may be changed based on version-1 results.

If version 2 also misses development, M09 remains incomplete and the test split is not used to rescue it.

## 11. Untouched final evaluation / acceptance

Only the first fitted version to pass development is frozen and evaluated once on the untouched 128-puzzle test split.

M09 passes only if test evidence independently satisfies the same practical benefit:

```text
mean score(trained) - mean score(unguided) >= 0.005
paired wins > paired losses
mean score(trained) >= fixed-random
mean score(trained) >= identity
identical trained/unguided transition and evaluator-call budgets
```

Report the paired mean delta, median delta, wins/ties/losses, bootstrap 95% interval as a diagnostic, all baseline means, and all realized work. The interval is reported but is not an extra pass/fail condition.

An unsuccessful held-out comparison **does not pass M09**, even if gradients, parameters, training loss, or development metrics look good.

## 12. Checkpoint / provenance

The accepted action checkpoint must bind to the exact frozen reasoner SHA-256 and record:

- auxiliary format/version,
- class and architecture,
- state representation,
- state-conditioned/global prior semantics,
- direction scale/action count,
- optimizer/training steps and fitted version,
- candidate-bank seed/count,
- selected candidate indices,
- target utility-table hash,
- train/development/test ID hashes,
- reasoner checkpoint/tensor hashes,
- target-generation work,
- explicit `reference_target_used=false`.

Checkpoint loading must reject incompatible reasoner/task/dimension metadata.

## 13. Claims boundary

A passing M09 establishes only that this trained action mechanism improves the declared one-ply/two-evaluation symbolic-score search under the stated distribution and budget.

It does not establish:

- exact Sudoku solve improvement,
- M07 verifier-guided MCTS improvement,
- deep-tree search benefit,
- energy/latency superiority,
- large-batch equivalence,
- a self-improvement flywheel.
