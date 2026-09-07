# SPECTRA Research State

This is the live milestone register. Historical cumulative state is preserved rather than rewritten:

- M01–M04: [`RESEARCH_STATE_M01_M04.md`](RESEARCH_STATE_M01_M04.md)
- M05-era register: [`RESEARCH_STATE_M05.md`](RESEARCH_STATE_M05.md)
- M06-era register: [`RESEARCH_STATE_M06.md`](RESEARCH_STATE_M06.md)
- M07-era register: [`RESEARCH_STATE_M07.md`](RESEARCH_STATE_M07.md)
- complete M08-era live register: [`RESEARCH_STATE_M08.md`](RESEARCH_STATE_M08.md)

The live register below records accepted milestones and the current experimental boundary.

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
| M09 | trained search action mechanism | **INCOMPLETE on `research/m09-trained-actions`; practical-benefit gate failed; untouched test remains unopened** |

---

# Milestone 09 — trained search action mechanism

**Stage status: INCOMPLETE. Do not advance past M09 under the current acceptance gate.**

Authoritative protocol: [`M09_PROTOCOL.md`](M09_PROTOCOL.md).
Development revision and its preregistration boundary: [`M09_DEVELOPMENT_REVISION.md`](M09_DEVELOPMENT_REVISION.md).
Full result/claims boundary: [`M09_ACCEPTANCE_GATE.md`](M09_ACCEPTANCE_GATE.md).

## Why M09 was needed

The pre-M09 `LatentActionCodebook` contained trainable `directions` and global `prior_logits`, but native MCTS executes under `torch.no_grad()`. Search therefore did not itself provide a gradient path for those parameters, and the prior was not conditioned on problem/search state.

M09 does not infer learned behavior from parameter declarations or MCTS visits. It introduces a separate offline optimizer-owned training path grounded in frozen-reasoner trajectory utilities.

## Fixed practical endpoint

Before development measurements, M09 declared that a trained action mechanism must improve the validated, reference-free Sudoku structural score over an equal-budget unguided search that uses the **same learned directions**.

Search contract:

```text
max_depth  = 1
n_rollouts = 2
c_puct     = 1.5
```

Required development and final-test benefit:

```text
mean(trained) - mean(unguided) >= 0.005
paired wins > paired losses
trained >= fixed-random baseline
trained >= identity baseline
trained and unguided have identical transition/evaluator budgets
```

The `+0.005` threshold was not changed after any result.

## Data isolation

Validated generated 9×9 Sudoku, unique solutions, 30–35 clues:

```text
data seed          20260909
reasoner train     384
reasoner val        96
untouched test     128
action fit         240 train puzzles
action development 144 train puzzles
```

The test identity hash was frozen for provenance, but test inputs were not used for action selection, revision decisions, or final comparison because no development version passed.

## Frozen target construction

M09 trains a small FP reasoner only as the trajectory generator, then freezes it.

Final retained reasoner checkpoint:

```text
bb50b2c232602b97eb6bc566f83fb45bbba5819808ef732e0cf71fe0a56c7c42
```

Target bank:

```text
24 deterministic normalized residual directions
seed 2901
scale 0.5
action 0 = identity
```

For each frozen train trajectory state and candidate:

```text
z_action = z + scale * direction
(y_next,z_next) = frozen TRM one-cycle continuation
utility = sudoku_score(puzzle, argmax(out_head(y_next)), box=3)
```

No stored solution/reference answer is used in the action target.

Total train target-generation cost:

```text
960 states
25 actions including identity
24,000 state-action equivalents
375 batched recursive-cycle calls
375 batched decode/oracle calls
```

This is below the preregistered 25,000-equivalent cap.

Greedy train-only coverage selected candidate indices:

```text
[22,15,20]
```

Frozen utility-table identity:

```text
57e823d041a866cbffc6d70ed532bafaef17b49e5d58850129f02d6bc6609708
```

Selected-prototype oracle coverage improvement over identity:

```text
+0.0050748661160469055
```

This is an opportunity bound/diagnostic, not learned-policy benefit.

## Training mechanism and verified updates

M09 action mechanisms use the complete state:

```text
(x,y,z)
```

Optimizer ownership is explicit: AdamW owns exactly the action module, never the frozen reasoner.

Common training budget:

```text
400 steps
batch 64
lr 2e-3
weight decay 0.01
grad clip 1.0
```

For v3, first-step evidence:

```text
direction grad norm       0.0120629566
policy-head grad norm     0.9345633984
direction parameter move  0.0239991453
policy-head parameter move 0.0277158991
```

Final v3 parameter movement:

```text
directions L2 change    1.73448658
policy head L2 change   0.48307455
```

Action-module size:

```text
10,163 parameters
```

Focused tests verify:

- legacy no-grad search does not update its action tensors;
- intended M09 direction and policy parameters receive nonzero gradients;
- optimizer steps change those parameters;
- learned residual actions change the actual recursive transition;
- full-state priors change deterministic action selection;
- strict action checkpoints bind to the exact core and reject incompatible/leaky provenance.

## V1 — hard full-action classification

Development comparison:

```text
trained mean       0.07801253
unguided mean      0.07886348
fixed random       0.06509653
identity           0.06424805
trained-unguided  -0.00085095
wins/ties/losses  11 / 110 / 23
```

**V1 failed.**

## V2 — full-action utility distillation

The preregistered revision kept the data, directions, scale, target table, optimizer/search budget and threshold fixed and replaced hard CE with utility distillation.

Development comparison:

```text
trained mean       0.08162555
unguided mean      0.07886348
trained-unguided  +0.00276206
bootstrap 95%     [0.00147387, 0.00411502]
wins/ties/losses  38 / 94 / 12
policy top1       0.81597
top2 recall       0.89236
```

V2 improved over unguided and had more wins than losses, but:

```text
0.00276206 < 0.005
```

**V2 failed.**

Failure diagnosis showed the development all-depth oracle target was dominated by identity (`73.44%`). Since the two-evaluation search already evaluates identity under deterministic scheduling, full-action policy capacity was poorly aligned with the marginal second-evaluation decision.

## V3 — budget-aligned root challenger

A further development revision was frozen before v3 results. It did not change the candidate directions, scale, reasoner, data, search budget, development set, or practical threshold.

Changes:

1. policy fitting uses only the 240 depth-0 action-fit states because acceptance search is root-only;
2. the policy predicts only which non-identity direction deserves the second evaluation;
3. priors force deterministic scheduling of identity first and the selected challenger second.

Prior contract:

```text
identity            0.5001
predicted challenger 0.4999
other challengers   0
```

Development schedule was verified exactly:

```text
identity -> action1 101 puzzles
identity -> action2  38 puzzles
identity -> action3   5 puzzles
```

Equal work across 144 development puzzles:

```text
                       trained  unguided
transitions                576       576
recursive-cycle calls       576       576
real evaluations            288       288
oracle evaluations          288       288
policy forwards             144         0
```

Development comparison:

```text
trained mean       0.07988069
unguided mean      0.07891949
fixed random       0.06509653
identity           0.06424805
trained-unguided  +0.00096121
bootstrap 95%     [0.00011661, 0.00183227]
wins/ties/losses  19 / 116 / 9
```

V3 passed every preregistered condition except the minimum practical effect size:

```text
0.00096121 < 0.005
```

**V3 failed. No V4 was added.**

## Preserved development checkpoint

The failed v3 mechanism is nevertheless reproducibly preserved through the strict action checkpoint path:

```text
file      development_action_policy.pt
class     BudgetAlignedChallengerCodebook
version   v3_budget_aligned_challenger
prior     identity_plus_state_challenger_v1
SHA256    9df824c450b794656711dd7e9be681c2d608c279819a9401ce1977d1362ea6c3
```

The checkpoint stores optimizer state, training/provenance metadata and exact core compatibility and strict-reloads successfully.

Its metadata explicitly records:

```text
accepted_practical_benefit = false
reference_target_used      = false
test_inputs_evaluated      = false
```

This establishes a genuine training/checkpoint path but **not** a successful learned-search mechanism.

## Untouched test state

The 128-puzzle final action comparison was **not run**.

Because V1, V2, and V3 all failed the unchanged development effect-size requirement, opening the final test split would have violated the preregistered selection boundary.

Therefore the independent held-out-benefit component of the M09 acceptance gate remains unsatisfied.

## Execution evidence

Original V1/V2 development run:

```text
run          34138116310
job          101793708718
artifact id  10024865157
ZIP SHA256   87d880f2258a8d67f541c865bf5924004b013c53f4811f0108cdb47449c926c1
```

Final V3/checkpoint-retention run:

```text
run          34139874650
job          101799222166
head         be41d3e0ecfeed431641ce206c0d6dd265ade6ed
artifact id  10025514537
ZIP SHA256   cbb60c0bdbf8b558e8ac179ba7bbf88cfedb071e4a56761539fb52ddac9dacef
artifact size 726,131 bytes
```

Final retained test state:

```text
focused M09         15 passed in 1.22 s
full fast           248 passed, 16 deselected in 42.20 s
compile exit        0
focused exit        0
experiment exit     3   # benefit not demonstrated
checkpoint preserve 0
incomplete evidence 2   # intentionally nonzero: acceptance not satisfied
fast regression     0
```

The overall workflow is red by design because M09 does not pass. The retention/evidence validator independently printed:

```text
M09_INCOMPLETE_EVIDENCE_RETAINED=PASS
```

## Claims boundary

M09 establishes:

- explicit reference-free action target generation;
- explicit optimizer ownership;
- real action-policy training and parameter changes;
- state-conditioned action selection;
- actual transition effects from learned directions;
- strict core-compatible checkpoint save/load with optimizer/provenance state;
- controlled comparisons against equal-budget unguided, fixed-random and identity baselines;
- measured failure of three development variants.

M09 does **not** establish:

- the predeclared `+0.005` practical search benefit;
- independent untouched-test action benefit;
- successful learned MCTS/search;
- exact Sudoku solve improvement;
- M07-verifier-guided search improvement;
- deep-tree action benefit;
- energy/latency improvement.

## Next executable experiment

Do not add another rescue variant inside M09.

The preserved next experiment is to preregister a new action-learning study that keeps the still-untouched final test sealed and evaluates one of:

1. richer **root-state action-value / advantage supervision**, rather than only action classification/distillation; or
2. a **larger train-only direction basis**, because the present three-prototype oracle coverage gain (`+0.00507`) is itself almost equal to the required practical effect size and may leave too little margin for imperfect policy extraction.

That experiment requires a new fixed cost/benefit contract before development evaluation.

## M09 decision

**M09 is INCOMPLETE and does not pass its acceptance gate.**

The implementation/training/checkpoint portions are real and retained, but the required practical benefit was not demonstrated and the independent untouched-test comparison was correctly withheld.

**Stop here. Do not advance to M10 under this gate.**
