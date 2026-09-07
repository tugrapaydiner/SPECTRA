# SPECTRA Research State

This is the live milestone register. Historical cumulative state is preserved rather than rewritten:

- M01–M04: [`RESEARCH_STATE_M01_M04.md`](RESEARCH_STATE_M01_M04.md)
- complete M05-era register: [`RESEARCH_STATE_M05.md`](RESEARCH_STATE_M05.md)
- complete pre-M07/M06-era live register: [`RESEARCH_STATE_M06.md`](RESEARCH_STATE_M06.md)

The live register below records accepted milestone status and the latest experimental boundary.

## Accepted milestone index

| Milestone | Scope | Accepted state |
|---|---|---|
| M01 | trustworthy baseline | merged; `40745dbe185c069aeee9eff3cf63dd411d9e17da` |
| M02 | native-kernel correctness/input contracts | merged; `e0781ec8b4e649ab4ccd48d4cd5f432a9b88d249` |
| M03 | task/data/evaluation contracts | merged; `01638b10777029fb28bb35229e374f0865c6e5d4` |
| M04 | reproducible training/checkpoint state | merged through PR #4; `370caf708755e1c68c59d5696778597f0290ea68` |
| M05 | checkpoint-backed evaluation | merged through PR #5; `1003c59e17dc17e652438317b7480c9e898379af` |
| M06 | controlled trained baseline | merged through PR #6; `385da5ef822dcb001c8193a2b8802fa292b31428`; evidence gate clarified through PR #7 `dbec23dd2077fc9f23e6a031b1b2b82c7c731de6` |
| M07 | grounded verifier training/evaluation | **COMPLETE ON `research/m07-grounded-verifier`; accepted run `34128295166`; not merged at the time of this entry** |

M06 progress is evidence-based, not conditioned on beating a baseline. Its authoritative acceptance definition is [`M06_ACCEPTANCE_GATE.md`](M06_ACCEPTANCE_GATE.md).

---

## Milestone 07 — grounded verifier training and evaluation

**Stage status:** COMPLETE ON `research/m07-grounded-verifier`; accepted execution/evidence gate green; not merged at the time of this entry.

Full preregistration: [`M07_PROTOCOL.md`](M07_PROTOCOL.md).  Accepted evidence and claims boundary: [`M07_ACCEPTANCE_GATE.md`](M07_ACCEPTANCE_GATE.md).

### M07 target

M07 does not train a verifier to predict eventual Sudoku solve probability.

The independently grounded target is:

```text
sudoku_one_cycle_improvement_v1
```

For frozen search state `s=(x,y,z)`:

```text
Q_before = sudoku_score(x, argmax(out_head(y)), box=3)
(y',z')  = frozen_TRM.recursive_cycle(x,y,z)
Q_after  = sudoku_score(x, argmax(out_head(y')), box=3)
label     = 1[Q_after > Q_before + 1e-6]
```

Continuation policy is exactly one deterministic frozen reasoner cycle with no search action, noise, router, MCTS backup, or test reference answer.

The score therefore means: estimated probability that this declared continuation improves the validated symbolic Sudoku structural score on the declared state distribution.

Grounding metadata records:

```text
oracle                = model.verifier.sudoku_score
reference_target_used = false
bootstrapped           = false
```

Held-out reference solutions are not passed to label construction.

### Search-state representation

Version:

```text
search_state_xyz_v1 = (x,y,z)
```

This replaces the unverified assumption that `(x,z)` alone is sufficient. The reasoner transition consumes both states:

```text
update_z = f(x_emb + y + z)
update_y = f(y + z)
```

A counterfactual state-sufficiency audit held `x,z` fixed and substituted another real `y` from the same puzzle trajectory. The independent continuation label changed in:

```text
127 / 256 pairs = 0.49609375
```

This is not evidence of naturally occurring exact-z collisions; it is direct evidence that omitted `y` can change the target under fixed `x,z`.

A separately typed z-only learned ablation was trained on the same rows/protocol. On shallow held-out states it was essentially tied/slightly better:

```text
full xyz ROC AUC = 0.88160735
z-only ROC AUC   = 0.88585556
delta            = -0.00424820
```

Therefore M07 does **not** claim that `y` improves average shallow ranking. The full representation is retained because it matches the actual transition state and avoids a known state-sufficiency omission.

### Frozen reasoner

M07 trains one small FP reasoner only as a trajectory generator, saves it through the existing versioned SPECTRA training checkpoint path, reloads its recorded evaluation identity, and then freezes it before verifier data generation/training.

Reasoner checkpoint SHA-256:

```text
d23ac0c629658257c96d50b71b4da42243094768d2352fb9bffb5f01e3be1625
```

Tensor-state hash before/after verifier training:

```text
04d97ee472b983375a8450457d6820ab6154e073503ec1c1fc31efb1477aaec1
```

The hashes were identical; all reasoner parameters were `requires_grad=false`; the reasoner stayed in eval mode.

The reasoner's own final validation board accuracy was 0.0 and cell accuracy about 0.19039. Reasoner task success is not an M07 acceptance condition.

### Data and independent labels

Validated generated 9×9 Sudoku, unique solutions, 30–35 clues.

```text
data seed   20260907
train       256 puzzles
validation   64 puzzles
test         96 puzzles
```

Verifier trajectory subsets:

```text
train puzzles       192
validation puzzles   48
held-out puzzles     64
training depths     0..3
deep stress depths  4..7
```

Label counts:

| State split | Positive | Negative | Total |
|---|---:|---:|---:|
| train | 245 | 523 | 768 |
| validation | 55 | 137 | 192 |
| held-out depths 0–3 | 83 | 173 | 256 |
| held-out depths 4–7 | 38 | 218 | 256 |

The target was non-degenerate on training and held-out states.

### Trained grounded verifier checkpoint

Architecture:

```text
EnsembleGroundedStateVerifier
members      3
member seeds 7101,7202,7303
dim          48
layers       1
heads        4
z boundary   A8 fake quantization
training     300 AdamW steps/member
lr           2e-3
weight decay 0.01
batch        64 states
```

Accepted checkpoint SHA-256:

```text
ce68e1bc9b6d7dff52bdcdba41861c0ed676a00b861afbf5e3be822f2cb1e151
```

The checkpoint reuses `spectra.learned_auxiliary` v1, has kind `grounded_state_verifier`, records target/representation/provenance, and is strictly compatible with the exact frozen reasoner checkpoint SHA/task/dim/vocabulary/sequence metadata.

The z-only ablation is saved under a separate diagnostic format and cannot masquerade as the accepted grounded checkpoint.

### Primary held-out evidence

256 real states from 64 unseen puzzles, depths 0–3:

| Metric | Result |
|---|---:|
| ROC AUC | **0.88160735** |
| Average precision | **0.86070040** |
| Brier score | **0.07669393** |
| ECE (10 bins) | **0.05412407** |
| Accuracy @ 0.5 | **0.91796875** |
| False-acceptance rate @ 0.5 | **0.015625** |
| False-positive rate | **0.00578035** |
| TP / FP / TN / FN | `63 / 1 / 172 / 20` |

This clears the preregistered useful-meaning floor (`ROC AUC >= 0.60`) without a one-class shortcut. The probability calibration metrics refer only to the declared one-cycle event.

### Perturbation evidence

Labels were independently recomputed after deterministic held-out perturbations:

| Set | ROC AUC | AP | Brier | ECE | False acceptance |
|---|---:|---:|---:|---:|---:|
| real depth 0–3 | 0.88161 | 0.86070 | 0.07669 | 0.05412 | 0.015625 |
| perturbed y | 0.90229 | 0.89236 | 0.06789 | 0.02380 | 0.015625 |
| perturbed z | 0.90010 | 0.88049 | 0.05993 | 0.03535 | 0.015625 |

The verifier remained useful under these small perturbations in this experiment.

### Deeper-state failure

On unseen depths 4–7, verifier meaning degraded sharply:

```text
ROC AUC           0.53331721
average precision 0.21949778
Brier             0.13113642
ECE(10)           0.07440630
TP/FP/TN/FN       0/0/218/38
```

At threshold 0.5 the verifier accepted no deeper states. The resulting high raw accuracy is driven by class imbalance and is not useful deep ranking evidence.

**M07 does not establish verifier generalization to deeper search states.** Any later search experiment using depths outside the validated regime must independently establish verifier meaning on that state distribution or explicitly report the extrapolation.

### Ensemble disagreement

Disagreement was treated as a heuristic and tested against actual held-out error.

Shallow held-out states:

```text
Spearman(disagreement, |prob-label|) = 0.863625
mean |error| low-disagreement half   = 0.041644
mean |error| high-disagreement half  = 0.214714
```

Deeper states: Spearman about 0.69008, but ranking performance itself was near random.

Thus disagreement is error-correlated in this run, but M07 does not promote it to calibrated epistemic uncertainty and does not enable a nonzero MCTS uncertainty penalty.

### Circularity boundary

M07 independent labels did not use MCTS backups.

Existing MCTS `q=W/N` process targets are now explicitly identified as:

```text
mcts_bootstrap_value_v1
independent_ground_truth = false
```

They remain available as later self-training targets, but cannot be cited as independent M07 verifier truth.

`GroundedLatentNativeMCTS` passes both `node.y` and `node.latent()` to a grounded verifier. It refuses the legacy batched z-only path until batched leaves carry both components.

### Tests and reproducibility

Focused M07 contracts:

```text
11 passed in 0.10 s
```

Full fast suite:

```text
215 passed, 16 deselected in 62.44 s
```

Exact commands:

```bash
python scripts/train_grounded_verifier.py --out outputs/m07_grounded_verifier
python -m pytest tests/test_m07_grounded_verifier.py -q
python -m pytest -m 'not slow' -ra
```

The focused tests enforce target provenance, absence of a reference-answer target argument, frozen reasoner state, exact core binding, binary grounded labels, full-state MCTS input, rejection of incomplete z-only grounded search, and explicit bootstrapped MCTS provenance.

### Accepted evidence

```text
experimental head    b1b2d1cb53984ba2938e2156c4e8cf031fc67b6f
Actions run          34128295166
job                 101762049774
artifact            m07-grounded-verifier-evidence
artifact id         10021097338
artifact ZIP SHA256 0a7a0888cf866fb89495791d80ce070dda8c0732bd181a281abf99e670d0633f
artifact size       1,397,215 bytes
retention           14 days
```

Key hashes:

```text
grounded_verifier.pt   ce68e1bc9b6d7dff52bdcdba41861c0ed676a00b861afbf5e3be822f2cb1e151
reasoner.pt            d23ac0c629658257c96d50b71b4da42243094768d2352fb9bffb5f01e3be1625
summary.json           25ac3123cf1366161c60495e61da1f28e589b4b6d619f0e3b6a5db7aeed05480
metrics.json           3190c54be68b87201de2524188b0b86c7bdb727efb58752c407590e12e52dc63
target_provenance.json c96e0feb52b4baf467e7782747dd68a80ca2995e428b208bbf267ff568ebb40d
freeze_audit.json      1425590c51c4362c550440c87f7f1cd567c4ba0f8aae3dfe9cdf1f0f5564e426
aliasing_audit.json    7e76b7f12a7a8a449ce497f72155e2f64e5ca92193969ebf7671ce80376285e9
z_only_ablation.pt     626e0b7f5af1f15466271f4e25ca00550fb45a9a9e75d3083af2cf652c488bb4
```

The evidence artifact retains the reasoner/verifier checkpoints, six verifier learning curves, held-out/deep/perturbed per-state predictions, data manifest, depth metrics, aliasing/freeze audits, target provenance, exact commands, environment, test logs and per-file hashes.

### M07 decision

M07 passes its acceptance gate: **a trained verifier checkpoint exists, target construction is independently grounded and documented, and held-out shallow-state evidence shows useful score meaning without a circular success claim.**

Claims that remain unsupported:

- eventual solve probability,
- MCTS improvement from this verifier,
- deep-state verifier generalization,
- superiority of full xyz over z-only on shallow natural trajectories,
- calibrated epistemic uncertainty from ensemble disagreement,
- search/energy/scaling gains.

The deep-state failure and z-only tie are retained as constraints for future experiment design, not as reasons to rewrite the M07 target after seeing results.

**Stop here for M07.**
