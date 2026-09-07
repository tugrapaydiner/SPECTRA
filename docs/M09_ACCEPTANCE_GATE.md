# Milestone 09 Acceptance Gate — Trained Search Actions

**Decision: INCOMPLETE / FAIL for the M09 practical-benefit acceptance gate.**

M09 now has a real offline training path, explicit optimizer ownership, strict core-bound checkpointing, verified nonzero gradients and parameter movement, reference-free target generation, state-conditioned search integration, fair equal-budget development comparisons, and a preserved loadable development checkpoint.

It does **not** pass the user-defined milestone gate because none of the allowed trained variants demonstrated the predeclared practical benefit of at least `+0.005` mean held-out development score over the equal-budget unguided search using the same learned directions. The untouched 128-example test comparison was therefore never run.

A trainable tensor, decreasing training loss, compatible checkpoint, positive paired win count, or a development improvement below the declared threshold is not promoted to successful learned search.

## Existing mechanism traced

Before M09, `model/latent_action.py` exposed `LatentActionCodebook.directions` and global `prior_logits` as trainable parameters, but native MCTS executes under `torch.no_grad()`.

Therefore:

- MCTS search itself does not update `directions`;
- MCTS search itself does not update `prior_logits`;
- the historical prior is global rather than conditioned on `x`, `y`, or `z`;
- search visit/value statistics are not an optimizer path for those parameters.

M09 keeps that legacy checkpoint/interface available but does not cite it as a learned search policy.

## Predeclared practical benefit

Before development measurements, M09 fixed the primary comparison to the validated reference-free Sudoku structural score:

```text
model.verifier.sudoku_score(puzzle, decoded_candidate, box=3)
```

The trained mechanism had to beat an **equal-budget unguided search using the exact same learned directions** under:

```text
max_depth  = 1
n_rollouts = 2
c_puct     = 1.5
```

Development and final-test pass criteria were fixed as:

```text
mean score(trained) - mean score(unguided) >= 0.005
paired wins > paired losses
mean score(trained) >= mean score(fixed random)
mean score(trained) >= mean score(identity)
trained and unguided use identical transition/evaluator budgets
```

The `+0.005` threshold was never weakened after seeing results.

The symbolic evaluator deliberately decodes selected states and evaluates `sudoku_score` so M09 isolates action quality from M07 verifier error. This is an experiment harness, not a production no-decode MCTS claim.

## Data isolation

Generated validated 9×9 Sudoku, unique solutions, 30–35 clues:

```text
data seed             20260909
reasoner train        384 puzzles
reasoner validation    96 puzzles
untouched test         128 puzzles
action fit             240 train puzzles
action development     144 train puzzles
```

The test IDs were frozen in provenance, but **test inputs were not evaluated by the action experiment** because development never passed.

## Frozen reasoner and target grounding

The trajectory generator is a small FP TRM:

```text
dim=48
layers=1
heads=4
n=1
T=1
N_sup=2
200 optimizer steps
```

In the final retained run the reasoner checkpoint SHA-256 is:

```text
bb50b2c232602b97eb6bc566f83fb45bbba5819808ef732e0cf71fe0a56c7c42
```

The reasoner was frozen before action-target generation/training and remained unchanged during the action experiment.

Action targets do not consume reference solutions. For each frozen trajectory state, M09 evaluated identity plus 24 deterministic residual candidates by:

```text
z_action = z + 0.5 * direction
(y_next,z_next) = frozen_reasoner.recursive_cycle(x_emb,y,z_action)
utility = sudoku_score(x, argmax(out_head(y_next)), box=3)
```

Target provenance records:

```text
reference_target_used = false
candidate_bank_seed   = 2901
candidate_bank_count  = 24
candidate_scale       = 0.5
selected candidates   = [22,15,20]
```

Train-only target-generation work:

```text
states                         960
identity + candidate actions    25
state-action equivalents     24,000
batched recursive-cycle calls   375
batched decode/oracle calls     375
```

This remains under the predeclared `25,000` state-action-equivalent cap.

The frozen train utility-table/prototype identity is:

```text
57e823d041a866cbffc6d70ed532bafaef17b49e5d58850129f02d6bc6609708
```

The three selected prototypes have a train-only oracle coverage gain over identity of:

```text
+0.0050748661160469055
```

That shows the candidate set contains some measurable oracle opportunity. It does not show the learned policy can reliably harvest it under the declared search budget.

## Explicit optimizer ownership and parameter-update evidence

All M09 action variants use an explicit AdamW optimizer over exactly the action-module parameters. The frozen reasoner is excluded.

Fixed action-training budget:

```text
steps        400
batch         64
lr          2e-3
weight decay 0.01
grad clip     1.0
```

For the final v3 mechanism, first-step evidence is:

```text
direction grad norm       0.0120629566
policy-head grad norm     0.9345633984
direction step change     0.0239991453
policy-head step change   0.0277158991
```

Final movement:

```text
direction change L2     1.73448658
policy-head change L2   0.48307455
```

The trained directions remained nearly collinear with the selected prototypes:

```text
cosines = [0.99999744, 0.99999988, 0.99999839]
```

The v3 action module contains `10,163` parameters, below the predeclared `50,000` cap.

Focused tests also demonstrate that learned residuals change the actual recursive transition and state-conditioned priors change deterministic action selection. Legacy no-grad search is tested not to mutate its action tensors.

## Variant 1 — hard action classification

Method:

```text
P(a|x,y,z)
loss = CE(policy_logits, oracle_best_action)
     + 0.5 * direction_MSE
```

Development result:

| Metric | Result |
|---|---:|
| trained mean | `0.07801253` |
| unguided same-directions mean | `0.07886348` |
| fixed-random mean | `0.06509653` |
| identity mean | `0.06424805` |
| trained − unguided | **`-0.00085095`** |
| wins / ties / losses | `11 / 110 / 23` |

V1 failed the practical gate.

The failure was retained; M09 did not interpret its nonzero gradients or parameter updates as success.

## Variant 2 — utility distillation

The preregistered development-only revision kept the same candidate table, directions, scale, search budget, data, optimizer budget, and threshold. It replaced hard CE with:

```text
teacher(a|s) = softmax((utility_a - max utility) / 0.02)
loss_policy  = KL(teacher || learned_prior)
```

Development result:

| Metric | Result |
|---|---:|
| trained mean | `0.08162555` |
| unguided same-directions mean | `0.07886348` |
| trained − unguided | **`+0.00276206`** |
| bootstrap 95% mean-delta interval | `[0.00147387, 0.00411502]` |
| wins / ties / losses | `38 / 94 / 12` |
| policy top-1 action accuracy | `0.81597` |
| top-2 oracle-action recall | `0.89236` |

V2 improved over unguided and had more wins than losses, but its mean improvement remained below the fixed `+0.005` threshold. V2 therefore failed M09.

### V2 failure diagnosis

On development trajectory states the oracle-best action distribution was dominated by identity:

```text
identity  73.4375%
action 1  15.6250%
action 2   8.6806%
action 3   2.2569%
```

Yet the two-evaluation deterministic search already tends to evaluate identity. The full-action policy was spending much of its capacity predicting an action the search budget effectively supplied for free.

This motivated the documented v3 revision; the acceptance threshold was not altered.

## Variant 3 — budget-aligned root challenger

The v3 revision was frozen in `M09_DEVELOPMENT_REVISION.md` before v3 development results.

It makes two search-alignment changes while retaining the same data, reasoner, selected prototypes, scale, optimizer budget, search budget, and pass threshold:

1. policy fitting uses only the 240 **depth-0 train states**, because the declared acceptance search is root-only;
2. the policy predicts which of the three **non-identity challengers** deserves the second evaluation.

`BudgetAlignedChallengerCodebook` produces:

```text
P(identity)          = 0.5001
P(predicted challenger) = 0.4999
P(other challengers) = 0
```

The actual scheduling invariant was verified on every development puzzle:

```text
trained:  identity first, one predicted challenger second
unguided: identity first, action 1 second
```

Observed trained challenger schedules:

```text
identity -> action1 : 101 puzzles
identity -> action2 :  38 puzzles
identity -> action3 :   5 puzzles
```

V3 challenger top-1 accuracy on root development states was `0.68056`.

### V3 equal-budget work

Across 144 development puzzles:

```text
                         trained       unguided
root transitions             576            576
recursive-cycle calls        576            576
real evaluator calls         288            288
symbolic oracle calls        288            288
policy forwards              144              0
```

The additional policy-forward cost is recorded rather than treated as free.

### V3 development result

| Metric | Result |
|---|---:|
| trained mean | `0.07988069` |
| unguided same-directions mean | `0.07891949` |
| fixed-random mean | `0.06509653` |
| identity mean | `0.06424805` |
| trained − unguided | **`+0.00096121`** |
| bootstrap 95% mean-delta interval | `[0.00011661, 0.00183227]` |
| wins / ties / losses | `19 / 116 / 9` |

V3 passed the paired-win, fixed-random, identity, equal-work, and scheduling checks, but failed the predeclared practical effect-size threshold:

```text
0.00096121 < 0.005
```

No fourth post-hoc mechanism was added.

## Preserved compatible development checkpoint

Because parameter/checkpoint mechanics are independently valuable evidence even though the practical gate failed, the final retention run deterministically refits v3 from the already-frozen train-only table and saves a strict development-only checkpoint:

```text
file   development_action_policy.pt
class  BudgetAlignedChallengerCodebook
prior  identity_plus_state_challenger_v1
SHA256 9df824c450b794656711dd7e9be681c2d608c279819a9401ce1977d1362ea6c3
```

The checkpoint:

- strict-reloads successfully;
- binds to the exact retained frozen reasoner checkpoint/task/dimension/vocabulary/sequence metadata;
- stores optimizer state, trained-step count and fitted version;
- records target-table/candidate/data provenance;
- records `reference_target_used=false`;
- is explicitly marked `accepted_practical_benefit=false`;
- records `test_inputs_evaluated=false`.

This checkpoint proves a real compatible training/checkpoint path exists. It is **not an accepted learned-search checkpoint**.

## Untouched final comparison

**Not run.**

All three allowed development variants missed the required `+0.005` mean improvement. Consistent with the preregistration, M09 did not use the 128-example test split to rescue, tune, select, or advertise the mechanism.

Therefore the acceptance-gate requirement for independent held-out comparisons supporting the practical benefit is unsatisfied.

## Final retained execution evidence

Final retention run:

```text
Actions run        34139874650
job               101799222166
branch head        be41d3e0ecfeed431641ce206c0d6dd265ade6ed
focused contracts 15 passed in 1.22 s
full fast suite    248 passed, 16 deselected in 42.20 s
```

Captured exit state:

```text
compile     0
focused     0
experiment  3   # development benefit not demonstrated
preserve    0   # failed-development checkpoint saved/reloaded
 evidence    2   # intentionally nonzero: acceptance not satisfied
fast        0
```

The overall workflow is intentionally red. Turning it green by accepting an effect below `+0.005` would violate the milestone contract.

Final evidence artifact:

```text
artifact    m09-trained-actions-evidence
artifact id 10025514537
ZIP SHA256  cbb60c0bdbf8b558e8ac179ba7bbf88cfedb071e4a56761539fb52ddac9dacef
size        726,131 bytes
retention   14 days
```

Key retained hashes:

```text
development_action_policy.pt  9df824c450b794656711dd7e9be681c2d608c279819a9401ce1977d1362ea6c3
reasoner.pt                   bb50b2c232602b97eb6bc566f83fb45bbba5819808ef732e0cf71fe0a56c7c42
target_table.pt               a59be32c38e3b86e891750219f482b2c36cca9748f3225a4579646b04f521362
summary.json                  85ae23fe52d9eddd988116a3177ee9532b8233fe5ad715a4e4963fe7064b0597
target_generation.json        4bffdab94a3c651476f7a5939d2d5cf81d5d8fe04e1901dd0a6939660509cf58
v3_development.json           434862940a2bf8c089dc9501bc594c950416793cb975fddc45886d3b04a34e08
```

The earlier v1/v2 failed-development run is also retained separately:

```text
run         34138116310
job         101793708718
artifact id 10024865157
artifact ZIP SHA256 87d880f2258a8d67f541c865bf5924004b013c53f4811f0108cdb47449c926c1
```

## Next executable experiment

M09 stops here. The next experiment must be separately preregistered rather than introduced as another M09 rescue variant.

The retained executable direction is:

> Preserve the frozen candidate/prototype table and all v1/v2/v3 failures; test richer **root-state action-value supervision** or a **larger train-only direction basis** under a newly frozen cost/benefit contract before opening the still-untouched M09 test split.

Useful questions for that experiment include whether the current 3-direction prototype basis is the bottleneck, whether predicting per-action value/advantage is better aligned than classification/distillation, and whether the `+0.005` oracle coverage ceiling from the current prototype set is simply too close to the required practical threshold for robust policy extraction.

## M09 decision

M09 does **not** pass its acceptance gate.

What is established:

- a real action target-generation path;
- explicit optimizer ownership;
- nonzero gradients and parameter movement;
- state-conditioned priors that affect action selection;
- learned residuals that affect transitions;
- strict compatible checkpoint save/reload with optimizer/provenance state;
- fair equal-budget development comparisons against unguided, fixed-random and identity baselines;
- transparent failure analysis across three development variants.

What is **not** established:

- the predeclared practical benefit (`+0.005`) over equal-budget unguided search;
- an independent untouched-test benefit;
- successful learned MCTS/search;
- exact Sudoku solve improvement;
- M07 verifier-guided search improvement;
- deep-tree action benefit;
- energy/latency advantage.

**Stage status: INCOMPLETE. Do not advance past M09 under this gate.**
