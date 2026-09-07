# M09 Development Revision — Budget-Aligned Challenger Policy

**Frozen after the first M09 development run and before this revision is evaluated. The untouched test split remains unopened.**

## Observed development failure retained

The original protocol produced two legitimate negative development results:

```text
v1 hard CE:
  trained mean     0.07801253
  unguided mean   0.07886348
  mean delta      -0.00085095
  wins/ties/loss  11 / 110 / 23

v2 utility distillation:
  trained mean     0.08162555
  unguided mean   0.07886348
  mean delta      +0.00276206
  wins/ties/loss  38 / 94 / 12
```

Both retained the required equal transition/evaluator work. V2 improved the comparison and had a positive bootstrap interval, but it did **not** reach the predeclared `+0.005` mean-delta practical threshold. Therefore neither version passes M09 and the test split was not evaluated.

## Diagnosis motivating one new design

The failure is not explained by missing gradients or an inert tensor:

- train-only prototype coverage gain over identity: `+0.00507487`;
- v1 first-step direction/policy-head gradient norms: `0.01210` / `3.59748`;
- v2 first-step direction/policy-head gradient norms: `0.01210` / `1.03048`;
- direction cosine fidelity to selected prototypes: ~`0.99999`;
- v2 development top-1 action accuracy: `0.81597`;
- v2 development top-2 oracle-action recall: `0.89236`;
- equal search work was verified;
- applying a learned action materially changed the recursive transition.

Two search-integration mismatches remain.

First, on development trajectory states the oracle best action is identity `73.44%` of the time. But the two-evaluation one-ply M08 search already evaluates identity under the unguided tie-breaking schedule. Training the policy primarily to predict identity therefore spends most capacity on an action the search budget effectively provides for free. What matters is which **non-identity challenger** should receive the second evaluation.

Second, v1/v2 fitted equally over depths `0..3`, while the declared M09 acceptance comparison is a **root-only** (`depth=0`) one-ply search. The policy target distribution was therefore broader than the distribution on which the practical benefit is measured. Revision v3 reuses the already generated train-only utility table but fits its challenger policy only on the depth-0 rows. This is distribution alignment, not test tuning; the test split remains unopened.

## Revision v3: identity + state-conditioned challenger

No data split, reasoner, candidate bank, selected prototype directions, residual scale, development threshold, or untouched test set changes.

A new `BudgetAlignedChallengerCodebook` uses the same complete state `(x,y,z)` and the same three trainable prototype directions, but its policy head predicts only the three non-identity actions.

Policy-fitting rows are the existing `240` action-fit **depth-0** states only. Prototype selection remains the original all-depth train-only coverage selection, so no direction is reselected after observing development results.

Training target:

```text
challenger_target = argmax utility(action in {1,2,3})
```

Training loss:

```text
teacher = softmax((utility_nonidentity - max utility_nonidentity) / 0.02)
L_policy = KL(teacher || challenger_policy)
L_total = L_policy + 0.5 * direction_MSE
```

Optimizer and budget stay unchanged:

```text
AdamW
400 steps
batch 64
lr 2e-3
weight decay 0.01
grad clip 1.0
```

At search time the codebook converts its state-conditioned challenger choice into a sparse prior:

```text
P(identity)          = 0.5001
P(best challenger)   = 0.4999
P(other challengers) = 0
```

The v3 development/test harness additionally verifies the **actual evaluated paths** for every puzzle. A valid trained run must evaluate identity first and the predicted challenger second; the equal-budget unguided baseline must evaluate identity and the lowest-id challenger. If this scheduling invariant is not realized, v3 fails rather than attributing a result to the policy.

Both trained and unguided searches retain the accepted M08 settings:

```text
c_puct=1.5
max_depth=1
n_rollouts=2
```

Both still materialize four root transitions and perform exactly two evaluator calls. The trained method additionally records one state-conditioned policy forward. Thus the primary fairness budget remains unchanged.

## Revision development gate

Exactly the original practical gate remains:

```text
mean score(trained) - mean score(unguided) >= 0.005
paired wins > paired losses
trained mean >= fixed-random mean
trained mean >= identity mean
identical trained/unguided transition and evaluator budgets
```

No threshold is weakened because V1/V2 missed it.

If v3 misses this gate, M09 remains incomplete and the untouched test is still not evaluated. The next executable experiment must be preserved rather than adding another post-hoc variant.

If v3 passes, freeze/save/reload its checkpoint first, then evaluate the untouched 128-puzzle test split once using the original acceptance gate.
