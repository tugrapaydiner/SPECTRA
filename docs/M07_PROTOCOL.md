# Milestone 07 Protocol — Grounded Search-State Verifier

**Status:** preregistered before any M07 verifier metric is inspected.

M07 adds one independently grounded verifier path. It does not claim MCTS benefit, self-improvement, search superiority, or eventual Sudoku success.

## Acceptance gate

M07 passes only if the repository contains:

1. a **trained verifier checkpoint** bound to an exact frozen reasoner checkpoint,
2. documented target construction with independent oracle provenance,
3. held-out evidence that the verifier score has useful meaning,
4. explicit state-representation/version metadata and leakage checks,
5. a clear diagnosis when ranking, false acceptance, calibration, aliasing, perturbation, or depth generalization is weak.

A circular result in which MCTS backs up verifier scores and those same backups are then treated as independent truth does **not** satisfy this gate.

## Target definition

M07 does **not** interpret the verifier output as probability of eventual solve.

The target is:

```text
sudoku_one_cycle_improvement_v1
```

For a frozen reasoner search state `s=(x,y,z)`, let:

- `decode(y) = argmax(out_head(y))`,
- `Q(x,y) = sudoku_score(x, decode(y), box=3)`, using the validated symbolic Sudoku scorer,
- `C_theta(x,y,z)` be **one deterministic `TRM.recursive_cycle`** under the frozen reasoner, with no search action, noise, router, MCTS backup, test target, or reference solution.

If `C_theta` returns `(y',z')`, the binary label is:

```text
label(s) = 1[ Q(x,y') > Q(x,y) + 1e-6 ]
```

The verifier output is a sigmoid probability estimating the frequency of this **one-cycle structural improvement event** on the declared state distribution.

This target is intentionally modest. A positive prediction means “one frozen deterministic continuation cycle is likely to improve the independently computed Sudoku structural score.” It does **not** mean “this state will eventually solve the puzzle.”

### Oracle grounding

`model.verifier.sudoku_score` uses only the puzzle and decoded candidate. It checks row/column/box validity fractions and clue preservation. It does **not** read the retained reference solution.

Held-out M07 label generation receives only held-out puzzle inputs plus frozen reasoner states. The held-out target arrays are not passed to the target-construction function.

## Search-state representation

Representation id:

```text
search_state_xyz_v1
```

Inputs are:

- `x`: puzzle token ids,
- `y`: the full FP answer accumulator used by `TRM.recursive_cycle`,
- `z`: the latent scratch state, using the same dequantized state boundary native MCTS supplies to a verifier.

The verifier embeds `x`, projects `y`, projects fake-quantized `z`, adds spatial position information, then scores the pooled state.

### Why `(x,z)` is not assumed sufficient

`TRM.recursive_cycle` computes both:

```text
update_z = f(x_emb + y + z)
update_y = f(y + z)
```

so future dynamics depend on both `y` and `z`. The old `LatentEnergyVerifier(x,z)` therefore omits part of the transition state.

M07 must report two aliasing diagnostics:

1. a **same-`x,z`, changed-`y` counterfactual audit**: hold `x,z` fixed, substitute another real `y` from the same puzzle trajectory, and measure how often the independently recomputed label changes;
2. a **z-only learned ablation** trained on the exact same rows/protocol, compared with the full `(x,y,z)` verifier on held-out ranking/error metrics.

These diagnostics investigate the omission; they do not claim naturally occurring exact float `z` collisions.

## Frozen reasoner / coordinated updates

M07 trains one small FP recursive reasoner only to generate trajectories, using the existing versioned `spectra.training` checkpoint format. After its checkpoint is written it is loaded in evaluation identity, switched to `eval()`, and frozen.

No reasoner optimizer exists during verifier training. A deterministic tensor-state hash is recorded before trajectory generation and after verifier training; any change fails M07.

The verifier checkpoint is bound to the exact reasoner checkpoint SHA-256 through the existing learned-auxiliary compatibility contract.

## Data

Flagship task: validated generated 9×9 Sudoku (`box=3`, unique solution, 30–35 clues).

Frozen split:

```text
data seed       20260907
train           256 puzzles
validation       64 puzzles
test             96 puzzles
```

Reasoner training may use train targets. Verifier trajectory labels use puzzle inputs and the oracle score only.

Verifier trajectory subsets:

- training: first 192 train puzzles,
- validation: first 48 validation puzzles,
- held-out: first 64 test puzzles.

Training states use real frozen-reasoner depths `0..3`. Held-out evaluation also generates deeper states `4..7` without retraining.

## Reasoner training

Small FP reference:

```text
TRM dim=48
n_layers=1
heads=4
n=1
T=1
N_sup=2
FP32
AdamW lr=1e-3, wd=0.01
batch=32
200 steps
```

The reasoner is a frozen trajectory generator for M07. Its task accuracy is recorded but is not an M07 acceptance condition.

## Verifier architecture

`GroundedStateVerifier`:

```text
num_tokens=10
dim=48
n_layers=1
heads=4
act_bits=8
state representation=search_state_xyz_v1
```

A three-member deep ensemble is trained with independent initialization/shuffle seeds. Its mean sigmoid probability is the primary verifier score.

A matching z-only ablation has the same verifier capacity except the `y` projection is disabled. It is diagnostic only and is not a valid native-search state representation.

## Verifier optimization

- loss: binary cross entropy with logits,
- optimizer: AdamW,
- learning rate: `2e-3`,
- weight decay: `0.01`,
- batch size: `64` states,
- verifier steps: `300` per member,
- no held-out threshold tuning,
- fixed decision threshold for false acceptance: `0.5`.

Class counts are retained. If either training or held-out labels contain only one class, M07 must stop and report the target as empirically unusable rather than manufacture a ranking metric.

## Held-out metrics

Primary meaning tests:

- ROC AUC for the independently grounded binary target,
- average precision,
- Brier score,
- expected calibration error (10 equal-width probability bins),
- false-acceptance rate = false positives / all predicted positives at threshold 0.5,
- false-positive rate = false positives / all actual negatives,
- accuracy and class balance.

Useful meaning is established only if the full-state verifier is materially above random ranking and the result is not explained by one-class leakage. The exact observed values are reported; no metric is fabricated or clipped into a pass.

### Deeper / perturbed states

Report the same metrics separately for:

- in-distribution real depths `0..3`,
- deeper real depths `4..7`,
- `y`-perturbed held-out states,
- `z`-perturbed held-out states.

Perturbations are deterministic from a recorded RNG seed and are used for evaluation only.

## Ensemble disagreement

Ensemble standard deviation is a **heuristic**, not a calibrated uncertainty claim.

M07 measures its relationship to error using:

- Spearman correlation between ensemble disagreement and absolute probability error,
- mean error in low- vs high-disagreement halves.

M07 does not enable a nonzero MCTS uncertainty penalty on the basis of disagreement alone.

## Native MCTS integration

`LatentNativeMCTS` must use `value_state(x,y,z,...)` when the loaded verifier provides it. Legacy z-only verifier support remains for historical checkpoints, but the grounded M07 path must pass `node.y` and `node.latent()`.

## Bootstrapped PRM boundary

Existing `LatentNativeMCTS.prm_targets()` values are explicitly labelled:

```text
mcts_bootstrap_value_v1
```

They are self-bootstrapped search targets, **not independent ground truth**. Later PRM training may combine them with M07 grounding, but M07 acceptance relies only on the independently computed oracle-improvement labels above.

## Checkpoint metadata

The trained verifier auxiliary must record:

- `kind=grounded_state_verifier`,
- exact core checkpoint SHA-256,
- task/dim/vocabulary/sequence compatibility,
- architecture and ensemble member count,
- `state_representation=search_state_xyz_v1`,
- `target_id=sudoku_one_cycle_improvement_v1`,
- continuation policy and margin,
- `oracle=model.verifier.sudoku_score`,
- `reference_target_used=false`,
- train/validation/test puzzle-ID hashes,
- trained steps and member seeds,
- disagreement interpretation=`heuristic`.

## Evidence retained

- frozen reasoner checkpoint,
- trained grounded verifier checkpoint,
- z-only ablation checkpoint,
- reasoner/verifier learning curves,
- trajectory/label provenance manifest,
- held-out per-state predictions/labels,
- depth/perturbation metrics,
- aliasing audit,
- frozen-reasoner hash audit,
- environment metadata,
- exact commands,
- focused tests and full fast regression output,
- final M07 summary and evidence hashes.

## Stop conditions

Stop and report rather than weakening the target if:

- reasoner checkpoint/freeze contract fails,
- held-out target construction receives or reads reference answers,
- label provenance is missing,
- train or held-out target has only one class,
- verifier checkpoint cannot be strictly restored against the frozen reasoner,
- non-finite training occurs,
- representation metadata disagrees with the model,
- evidence files required by the gate are missing.
