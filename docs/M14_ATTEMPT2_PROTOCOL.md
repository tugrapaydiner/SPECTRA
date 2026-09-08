# Milestone 14 Attempt 2 Protocol — Input-Conditioned Single-Stream Recurrence

**Status: preregistered after M14 Attempt 1 failed development, but before any Attempt 2 training, validation, development, timing, or confirmation result.**

This is a new adaptive-development attempt inside Milestone 14, not a rewrite of Attempt 1. Attempt 1 remains permanently classified **INCOMPLETE** and is preserved by Actions run `34159474745` at head `66acbc62f48b3e5b1e81080897b89562edfd8bd8`.

## 1. Why a second attempt is justified

Attempt 1 trained the declared FP recursive, W1.58A8 recursive, larger FP single-pass, dynamic-INT8 context baseline, grounded-verifier search variants, and one preregistered dim-64 reserve intervention over five training seeds.

The dim-64 reserve was not optimization-limited: validation semantic validity was about `0.838` at one recursive supervision step and final training/validation fits were strong. On the paired 512-example development set it was essentially quality-tied with the larger FP baseline (`candidate - baseline = -0.00508`, 95% bootstrap CI `[-0.0445, +0.0332]`) but was **slower**, with complete-solve median ratio `1.0943` and 95% CI `[1.0888, 1.0998]`.

The observed bottleneck is therefore structural inference work, not an obvious lack of hidden width. One ordinary TRM supervision step with `n=1` executes two shared-block applications: one input-conditioned `z` update and one `y` update. A literal `TRM(n=0)` is rejected as an intervention because its first inherited `y` update is `f(y+z)` and therefore does not see `x_emb` when recurrent state starts at zero.

Attempt 2 tests one falsifiable structural intervention: retain problem conditioning while reducing each recursive step to one block application.

## 2. Frozen primary claim and thresholds

The primary comparator remains the same credibly trained larger floating-point single-pass Transformer baseline (`System1Student`, dim 96, two blocks).

The primary task endpoint remains strict symbolic Sudoku semantic validity after identical given-cell preservation.

Attempt 2 must satisfy **the same M14 practical gate** on development and then independently on untouched confirmation. No threshold is changed from `docs/M14_PROTOCOL.md`.

### Path A — quality superiority at comparable cost

- semantic-validity difference `candidate - baseline >= +0.03`;
- paired hierarchical-bootstrap 95% CI lower bound `> 0`;
- complete-solve median latency ratio `candidate / baseline <= 1.15`;
- bootstrap 95% upper bound on latency ratio `<= 1.20`.

### Path B — quality/cost trade-off

- semantic-validity difference `candidate - baseline >= -0.02`;
- paired hierarchical-bootstrap 95% CI lower bound `>= -0.03`;
- complete-solve median latency ratio `candidate / baseline <= 0.65`;
- bootstrap 95% upper bound on latency ratio `<= 0.75`.

Attempt 1's miss does not weaken these values.

## 3. Sequential-selection accounting

Attempt 2 is explicitly informed by Attempt 1's development evidence. Therefore:

- Attempt 1 is counted as a prior adaptive development attempt.
- The M14 train/validation/development hierarchy may be reused for further development.
- Confirmation seed `2026091402` was never generated or opened by Attempt 1 and remains eligible as the first independent confirmation set.
- Confirmation is generated only after Attempt 2 passes the development gate and the candidate/baseline checkpoints and inference budget are frozen and hashed.
- If confirmation seed `2026091402` is opened and fails, it becomes development evidence for any later change. A later M14 claim would then require reserve confirmation seed `2026091403` or another preregistered independent set.
- No favorable-seed filtering is allowed; all five training seeds remain part of every primary effect.

The statistical interpretation must state that the development procedure is adaptive across at least two attempts. The final confirmation claim, if reached, is still one independent confirmation of a candidate selected after adaptive development; it is not evidence for a universal scaling law.

## 4. Data hierarchy

Attempt 2 uses the exact M14 task contract and development hierarchy:

```text
task                 generated unique 4x4 Sudoku
box                  2
clues                 uniform 6..10
augmentation          false
solution method       random_backtracking
train / validation / development   4096 / 512 / 512
development seed      2026091401
```

Only train rows receive gradients. Validation selects the Attempt 2 inference operating point. Development is used exactly once for the selected candidate in this attempt.

If development passes, confirmation uses:

```text
confirmation seed     2026091402
confirmation rows     1024
```

with explicit fingerprint overlap audit against the development hierarchy.

## 5. Attempt 2 candidate family

The only new learned candidate architecture is an **input-conditioned single-stream recursive model**.

```text
name                 InputConditionedSingleStreamTRM
hidden dim           96
blocks               1 shared transformer block
heads                4
precision            FP32
training N_sup       4
T                    1
recurrent streams    y only; z is a compatibility zero state and is never updated
```

One recursive supervision step is defined as:

```text
h_k       = x_emb + y_k
u_k       = f_theta(h_k)                  # exactly one shared-block application
y_{k+1}   = RMSNorm(y_k + alpha_y * u_k)
answer_k  = out_head(y_{k+1})
```

This differs deliberately from inherited `TRM(n=0)`: every step sees `x_emb`, including the first step.

The same block parameters are reused across all four training supervision steps. The model is still trained as a recurrent shared-weight reasoner even when validation later chooses a one-step inference operating point.

The candidate must remain smaller in trainable parameter count than the unchanged two-block dim-96 single-pass baseline. This is checked before training results are accepted.

No ternary version, learned router, learned halter, search action retraining, LR sweep, data-size sweep, or hidden-width sweep is permitted in Attempt 2.

## 6. Training budgets

Candidate and primary FP baseline are trained from scratch in the Attempt 2 run on exactly the same train manifest and five seeds:

```text
seeds               1401, 2402, 3403, 4404, 5405
optimizer           AdamW
learning rate       1e-3
weight decay        0.01
batch size          64
steps               1800 per model per seed
grad clip           1.0
```

The single-pass baseline remains:

```text
System1Student
dim                 96
blocks               2
heads                4
confidence head      frozen / excluded from trainable count
```

The baseline is not weakened, shortened, quantized for the primary comparison, or selected by seed.

Training cost is reported separately as wall time, optimizer steps, examples sampled, trainable parameters, and block applications.

## 7. Training objective

The baseline retains blank-cell cross entropy.

The single-stream recurrent candidate retains the original M14 recursive objective so the intervention isolates execution structure rather than also changing supervision weighting:

```text
four blank-cell CE losses
weights [0.1, 0.2, 0.3, 0.4] / sum
```

No early stopping or best-checkpoint selection is permitted. Final-step checkpoints are used.

## 8. Validation-only inference-budget selection

Each trained single-stream checkpoint is evaluated on validation at:

```text
N_sup in {1,2,3,4}
```

No checkpoint is retrained for an inference budget.

For each operating point, validation records semantic validity, exact reference match, blank-cell accuracy, and complete-solve latency. The same larger FP baseline is measured on the same examples and seeds.

Selection is deterministic and claim-aware:

1. Compute validation point quality difference and latency ratio versus the larger FP baseline.
2. **Path-B point-eligible** means quality difference `>= -0.02` and latency ratio `<= 0.65`.
3. **Path-A point-eligible** means quality difference `>= +0.03` and latency ratio `<= 1.15`.
4. If any Path-B point is eligible, choose the one with highest semantic validity; ties: lower latency, then lower `N_sup`.
5. Otherwise, if any Path-A point is eligible, choose the one with highest semantic validity; ties: lower latency, then lower `N_sup`.
6. Otherwise choose the highest-semantic-validity point with latency ratio `<= 1.15`; if none exists, choose the lowest-latency point. This fallback is diagnostic only and does not relax the development gate.

Uncertainty is **not** assessed on validation for selection. The unchanged hierarchical-bootstrap rule is applied only on development and confirmation.

## 9. Context baselines

Attempt 2 also reports, when supported:

- dynamic INT8 conversion of the trained larger single-pass baseline, clearly labelled as partial/conventional dynamic quantization where attention remains floating point;
- deterministic exact Sudoku backtracking as a task-specific symbolic contextual reference.

Neither can replace the FP primary comparator.

Attempt 1 remains the retained source for the original dim-48/dim-64 recursive, W1.58A8, and grounded-verifier search comparisons. Attempt 2 does not rerun those families because the intervention question is specifically structural cost reduction versus the same primary baseline.

## 10. Complete-solve cost

Latency measurement remains batch-size one and wraps the complete solve:

- model recurrent/single-pass computation;
- output head and argmax;
- given-cell preservation;
- symbolic semantic-validity validation.

No target solution is used inside inference.

For the single-stream model, actual per-step block applications are counted and retained. A one-step operating point must execute exactly one shared transformer block per solve before output decoding.

CPU-package energy is measured only if M13's validated counter layer reports a complete valid package measurement; otherwise energy stays unavailable/null and never becomes zero. CPU package is not GPU or whole-system energy.

## 11. Uncertainty

Development and confirmation use the unchanged paired hierarchical bootstrap:

- resample the five training seeds with replacement;
- within each sampled seed, resample paired examples with replacement;
- at least 5000 replicates;
- bootstrap seed `2026091499` on development and `2026091500` on confirmation;
- percentile 95% intervals.

Raw per-example task outcomes and measured costs are retained.

## 12. Decision hierarchy

1. Train all five single-stream candidate checkpoints and all five unchanged larger FP baseline checkpoints.
2. Select exactly one single-stream inference budget on validation using Section 8.
3. Evaluate that frozen operating point once on development against the baseline.
4. If development misses the unchanged practical gate, Attempt 2 is **INCOMPLETE** and confirmation remains sealed.
5. If development passes, hash/freeze candidate checkpoints, baseline checkpoints, architecture id, and inference budget.
6. Only then generate/open confirmation seed `2026091402` and run one final paired comparison.
7. M14 completes only if the same unchanged gate passes on confirmation.

A valid miss is retained evidence. No threshold, baseline, seed set, LR, training step count, or validation rule may be changed to convert a miss into a pass.

## 13. Required provenance

Attempt 2 must retain:

- this protocol commit SHA proving it predates Attempt 2 results;
- reference to Attempt 1 run/artifact and incomplete status;
- exact train/validation/development manifest;
- all ten training curves/checkpoints and training-cost records;
- validation raw rows and deterministic selection record;
- development raw rows, effect, uncertainty, and decision;
- complete-solve timing rows and energy availability record;
- actual single-stream block-application counts;
- conventional INT8 audit and symbolic context rows;
- confirmation freeze manifest and independent confirmation manifest only after a dev pass;
- result table and Pareto plot generated from raw rows;
- updated M14/state documentation.

**Acceptance remains exactly the user-specified M14 acceptance gate: an independently confirmed, predeclared practical improvement over credible trained baselines.**