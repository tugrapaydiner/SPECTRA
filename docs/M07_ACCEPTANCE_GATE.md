# Milestone 07 Acceptance Gate — Grounded Verifier

**Decision: PASS for the M07 grounded-verifier evidence contract.**

This is a pass because M07 produced a trained verifier checkpoint, documented independently grounded target construction, and held-out evidence that the verifier score has useful meaning. The pass does **not** claim eventual-solve probability, MCTS improvement, search superiority, calibrated epistemic uncertainty, deep-state generalization, or a circular self-verification result.

## What the verifier predicts

M07 target id:

```text
sudoku_one_cycle_improvement_v1
```

State representation:

```text
search_state_xyz_v1 = (x, y, z)
```

For a frozen reasoner state `s=(x,y,z)`:

1. decode the current answer accumulator: `a = argmax(out_head(y))`,
2. compute the validated symbolic structural score `Q_before = sudoku_score(x, a, box=3)`,
3. apply exactly one deterministic frozen `TRM.recursive_cycle` with no action, noise, router, MCTS backup, or search,
4. decode the resulting `y'` and compute `Q_after = sudoku_score(x, argmax(out_head(y')), box=3)`,
5. label the state positive iff `Q_after > Q_before + 1e-6`.

The trained sigmoid score therefore estimates the probability of this **one-cycle structural-improvement event on the declared trajectory distribution**. It is not interpreted as probability that the puzzle will eventually be solved.

The oracle scorer consumes only the puzzle and decoded candidate. Held-out reference solutions are not inputs to label construction. The accepted run records:

```text
reference_target_used = false
bootstrapped          = false
oracle                = model.verifier.sudoku_score
```

## Why the verifier uses `(x,y,z)`

The recursive transition is not a function of `z` alone:

```text
update_z = f(x_emb + y + z)
update_y = f(y + z)
```

The prior latent verifier `E(x,z)` therefore omits part of the state that determines continuation dynamics. M07 introduces `search_state_xyz_v1` rather than assuming `z` is sufficient.

### State-aliasing stress test

M07 held `x,z` fixed and substituted a different **real** `y` from another depth of the same puzzle trajectory, then recomputed the independent oracle label.

Across 256 counterfactual pairs:

```text
label changed       = 127 / 256
label-change rate   = 0.49609375
```

This is a counterfactual sufficiency audit, not a claim that exact floating-point `z` collisions naturally occur. It demonstrates that for this target, the omitted `y` can change the continuation label while `x,z` are held fixed.

### z-only learned ablation

The same training/evaluation protocol was also run with `y` omitted as a diagnostic ablation.

Held-out shallow ROC AUC:

```text
full (x,y,z) = 0.88160735
z-only       = 0.88585556
full - z     = -0.00424820
```

So M07 does **not** claim that adding `y` improves average shallow predictive ranking in this pilot. The z-only ablation is essentially tied and slightly higher by ROC AUC. The full representation is retained because it matches the actual transition state and because the counterfactual aliasing audit shows `y` can be causally relevant to the target, not because this run establishes an empirical shallow-ranking advantage.

## Frozen reasoner contract

M07 first trained a small FP recursive reasoner, saved it in the existing versioned SPECTRA training-state format, loaded the recorded evaluation identity, then froze it before trajectory generation or verifier training.

Accepted reasoner checkpoint:

```text
SHA-256 = d23ac0c629658257c96d50b71b4da42243094768d2352fb9bffb5f01e3be1625
```

Frozen tensor-state hash before and after verifier training/evaluation:

```text
04d97ee472b983375a8450457d6820ab6154e073503ec1c1fc31efb1477aaec1
```

The hashes were identical, all reasoner parameters had `requires_grad=false`, and the reasoner remained in evaluation mode. No reasoner optimizer exists during verifier training.

The reasoner's own final validation board accuracy was `0.0` and cell accuracy was approximately `0.19039`. M07 acceptance does not depend on the reasoner solving Sudoku; it depends on whether the verifier has independently grounded meaning for the declared continuation event.

## Data and target balance

Generated validated 9×9 Sudoku, 30–35 clues, unique solutions.

```text
data seed  = 20260907
train      = 256 puzzles
validation = 64 puzzles
test       = 96 puzzles
```

Trajectory subsets:

```text
verifier train puzzles = 192
validation puzzles     = 48
held-out test puzzles  = 64
training depths        = 0..3
deep stress depths     = 4..7
```

Independent label counts:

| Split / states | Positive | Negative | Total |
|---|---:|---:|---:|
| train | 245 | 523 | 768 |
| validation | 55 | 137 | 192 |
| held-out depths 0–3 | 83 | 173 | 256 |
| held-out depths 4–7 | 38 | 218 | 256 |

The target is not one-class on train or held-out evaluation.

## Trained verifier

Accepted verifier:

```text
class                EnsembleGroundedStateVerifier
members              3
member seeds          7101, 7202, 7303
dim                  48
layers               1
heads                4
z activation boundary A8 fake quantization
state representation search_state_xyz_v1
training steps/member 300
optimizer             AdamW
learning rate         2e-3
weight decay          0.01
batch size            64 states
```

Accepted checkpoint SHA-256:

```text
ce68e1bc9b6d7dff52bdcdba41861c0ed676a00b861afbf5e3be822f2cb1e151
```

The checkpoint reuses the existing `spectra.learned_auxiliary` v1 format, records `kind=grounded_state_verifier`, and is strictly bound to the exact frozen reasoner checkpoint SHA/task/dimension/vocabulary/sequence contract.

A z-only diagnostic checkpoint is separately typed and cannot load as an accepted M07 grounded verifier.

## Held-out meaning

Primary held-out set: 256 real frozen-reasoner states from 64 unseen puzzles at depths 0–3.

| Metric | Result |
|---|---:|
| ROC AUC | **0.88160735** |
| Average precision | **0.86070040** |
| Brier score | **0.07669393** |
| ECE, 10 bins | **0.05412407** |
| Accuracy at 0.5 | **0.91796875** |
| False-acceptance rate at 0.5 | **0.015625** |
| False-positive rate | **0.00578035** |
| TP / FP / TN / FN | `63 / 1 / 172 / 20` |

This clears the preregistered useful-meaning floor (`ROC AUC >= 0.60`) with both classes present. Only one negative state was falsely accepted at threshold 0.5 on this held-out shallow set.

Calibration metrics are reported because the output is interpreted as a probability of the declared one-cycle event. They are not extrapolated to eventual task success.

## Perturbation stress tests

Deterministic held-out perturbations were used only for evaluation, with labels independently recomputed from the frozen reasoner + symbolic oracle.

| Evaluation set | ROC AUC | AP | Brier | ECE(10) | False acceptance |
|---|---:|---:|---:|---:|---:|
| real depths 0–3 | 0.88161 | 0.86070 | 0.07669 | 0.05412 | 0.015625 |
| perturbed `y` | 0.90229 | 0.89236 | 0.06789 | 0.02380 | 0.015625 |
| perturbed `z` | 0.90010 | 0.88049 | 0.05993 | 0.03535 | 0.015625 |

The verifier remained useful under these small deterministic perturbations in this pilot.

## Deeper-state failure

Generalization to unseen depths 4–7 was poor:

```text
ROC AUC           = 0.53331721
average precision = 0.21949778
Brier              = 0.13113642
ECE(10)            = 0.07440630
TP / FP / TN / FN  = 0 / 0 / 218 / 38
```

At threshold 0.5 the verifier accepted no deeper states at all. The superficially high `0.85156` accuracy is class-imbalance accuracy and is not evidence of useful deep ranking. M07 therefore **does not establish deep-search-state generalization**. A future search experiment must either remain inside the validated state-depth regime or independently establish verifier meaning on its deeper state distribution.

## Ensemble disagreement

Member disagreement is retained as a heuristic and its relationship to held-out error was measured rather than assumed.

Primary shallow held-out states:

```text
Spearman(disagreement, absolute probability error) = 0.863625
mean absolute error, low-disagreement half         = 0.041644
mean absolute error, high-disagreement half        = 0.214714
```

Deeper states still showed a positive relationship (`Spearman ≈ 0.69008`), but deep ranking itself was near random.

This evidence supports disagreement as an **error-correlated heuristic in this experiment**, not as a calibrated uncertainty estimator. M07 does not enable a nonzero MCTS uncertainty penalty on the basis of this result.

## MCTS integration and circularity boundary

`GroundedLatentNativeMCTS` passes both `node.y` and `node.latent()` to a grounded verifier. It refuses to silently use the legacy z-only batched path because that path lacks corresponding `y` states.

Existing per-node MCTS backups are explicitly typed:

```text
mcts_bootstrap_value_v1
independent_ground_truth = false
```

Those backups can remain useful later as self-training signals, but they did not train or validate the M07 acceptance checkpoint. M07's success evidence comes from `sudoku_one_cycle_improvement_v1`, independently recomputed from the puzzle, frozen continuation, and symbolic oracle.

## Leakage and provenance tests

The focused M07 contract suite verifies:

- target-construction API has no reference-answer argument,
- held-out labels are binary and reference-free,
- unfrozen reasoners are rejected,
- reasoner tensor state is unchanged across verifier training,
- grounded verifier requires full `(x,y,z)` input,
- non-binary targets are rejected,
- checkpoint is bound to exact core checkpoint identity,
- leaky (`reference_target_used=true`) or circular (`bootstrapped=true`) grounding metadata is rejected,
- z-only diagnostic cannot masquerade as accepted M07,
- grounded MCTS passes both `y` and `z`,
- legacy incomplete batched state is rejected,
- MCTS backups are marked bootstrapped rather than ground truth.

Accepted focused suite:

```text
11 passed in 0.10 s
```

Full fast regression after the accepted experiment:

```text
215 passed, 16 deselected in 62.44 s
```

## Exact commands

Primary reproduction:

```bash
python scripts/train_grounded_verifier.py --out outputs/m07_grounded_verifier
```

Focused M07 contracts:

```bash
python -m pytest tests/test_m07_grounded_verifier.py -q
```

Full fast regression:

```bash
python -m pytest -m 'not slow' -ra
```

## Accepted evidence

Authoritative accepted execution:

```text
branch experimental head = b1b2d1cb53984ba2938e2156c4e8cf031fc67b6f
Actions run             = 34128295166
job                     = 101762049774
artifact                = m07-grounded-verifier-evidence
artifact id             = 10021097338
artifact ZIP SHA-256    = 0a7a0888cf866fb89495791d80ce070dda8c0732bd181a281abf99e670d0633f
artifact size           = 1,397,215 bytes
retention               = 14 days
```

Key file hashes:

```text
grounded_verifier.pt  ce68e1bc9b6d7dff52bdcdba41861c0ed676a00b861afbf5e3be822f2cb1e151
reasoner.pt           d23ac0c629658257c96d50b71b4da42243094768d2352fb9bffb5f01e3be1625
summary.json          25ac3123cf1366161c60495e61da1f28e589b4b6d619f0e3b6a5db7aeed05480
metrics.json          3190c54be68b87201de2524188b0b86c7bdb727efb58752c407590e12e52dc63
target_provenance.json c96e0feb52b4baf467e7782747dd68a80ca2995e428b208bbf267ff568ebb40d
freeze_audit.json     1425590c51c4362c550440c87f7f1cd567c4ba0f8aae3dfe9cdf1f0f5564e426
aliasing_audit.json   7e76b7f12a7a8a449ce497f72155e2f64e5ca92193969ebf7671ce80376285e9
z_only_ablation.pt    626e0b7f5af1f15466271f4e25ca00550fb45a9a9e75d3083af2cf652c488bb4
```

The artifact also retains six verifier-learning curves, held-out/deep/perturbed per-state predictions, depth metrics, exact commands, environment metadata, data manifest, reasoner-training evidence, test output, raw logs, and per-file SHA-256 records.

## M07 decision

M07 establishes a **grounded verifier with useful held-out meaning for the declared shallow one-cycle continuation target**, without using held-out reference answers and without using MCTS backups as independent truth.

The milestone also records real limitations:

- the target is one-cycle structural improvement, not eventual solve,
- full `(x,y,z)` is representation-correct, but did not beat the z-only ablation on shallow natural-state ROC AUC,
- deep-state ranking at depths 4–7 collapsed close to random,
- ensemble disagreement is error-correlated but remains a heuristic.

These limitations do not invalidate the M07 acceptance gate; they define the evidence boundary for whatever experiment follows.

**Stop here for M07. Do not turn the grounded verifier result into an MCTS-success claim inside this milestone.**
