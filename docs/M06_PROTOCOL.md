# Milestone 06 Protocol — Controlled Trained Baseline

**Status:** preregistered before any M06 baseline accuracy is inspected.

M06 is deliberately narrow: one controlled trained pilot on one validated task. It does not add MCTS, routing, halting policy training, PRM training, distillation, scaling-law claims, or target-hardware performance claims.

## Research question

On the same fixed 9×9 Sudoku distribution and training/evaluation protocol, does a small floating-point recursive TRM learn measurable held-out task structure, how much of that learning survives a matched W1.58A8 recursive model at full declared quantization, and how do both compare with a deliberately larger but single-pass feed-forward baseline?

This is a pilot, not a claim that recursion or ternary computation is superior. Zero exact solves is an admissible result.

## Flagship task

- task: validated `sudoku` pipeline
- board: 9×9 (`box=3`)
- clues: uniformly generated within the existing `30..35` configured range for this experiment
- unique solution required: yes
- generator: existing randomized backtracking path
- augmentation: existing validated Sudoku augmentation path
- official benchmark claim: none; this is the repository's generated Sudoku distribution

4×4 Sudoku is excluded from the main pilot. It may be used only for a diagnostic overfit/minimal-configuration check if the 9×9 optimization signal fails.

## Primary and secondary metrics

**Primary metric:** strict Sudoku semantic solve rate on the frozen test split (`semantic_validity`).

Also report, without promoting them to the primary endpoint:

- exact reference match
- blank-cell accuracy (cells that must be inferred)
- all-cell accuracy
- train/validation loss curves

No nonzero exact or semantic solve rate is assumed.

## Frozen data protocol

One split is generated once and reused by every architecture and model seed:

- data seed: `20260907`
- train: `384`
- validation: `96`
- test: `128`
- cross-split group/exact-overlap audit must remain zero

The test split is never used for model selection, step selection, hyperparameter changes, or failure diagnosis.

## Models

### A. Floating-point recursive reference

`TRM`:

- `dim=48`
- `n_layers=1`
- `heads=4`
- `n=1`
- `T=1`
- `N_sup=2`
- FP32 parameters/activations

### B. Matched ternary recursive model

Exactly the same TRM architecture as A, except:

- ternary weights enabled
- recurrent activation fake-quantization enabled (`W1.58A8` training path)
- quantization strength ramps linearly from 0 to 1 during the first quarter of the main training steps
- evaluation is invalid unless every `FakeBitLinear.quant_strength` is exactly `1.0` at checkpoint/evaluation time

### C. Larger single-pass baseline

`System1Student`:

- `dim=96`
- `n_layers=2`
- `heads=4`
- one feed-forward pass, no recursion
- FP32

The experiment must record exact trainable parameter counts and fail before main training if C is not larger than A or if A and B do not have identical trainable parameter counts.

## Optimization protocol

Common settings, fixed before results:

- optimizer: AdamW
- learning rate: `1e-3`
- weight decay: `0.01`
- batch size: `32`
- gradient clipping: global norm `1.0`
- no architecture-specific LR search
- no early stopping on validation accuracy
- final-step checkpoint is evaluated

Recursive A/B use the repository's existing deep-supervision loss (token CE at each supervision step plus the existing halting/improvement terms). Single-pass C uses token cross-entropy because it has no recursive supervision or halting head. This objective difference is architectural and must be reported as a limitation; optimizer, batch, data, step count, and seed protocol remain fixed.

Ternary B alone has the preregistered quantization-strength warmup required by its numerical representation. No other model-specific optimizer tuning is allowed.

## Timed resource probe and training budget

Before main training, run exactly `5` optimizer steps for each architecture on the 9×9 training split with the same batch size. Probe models are discarded and probe accuracy is not inspected.

Recorded main-training wall-clock budget on the runner: **1080 seconds total** across all main model fits. Target main step count is **200 optimizer steps per model**.

The seed/step decision uses timing only, never accuracy:

1. Let `s_max` be the slowest measured seconds/step across A/B/C.
2. If `6 * 200 * s_max <= 1080`, run two model seeds (`1101`, `2202`) for all three models at 200 steps each.
3. Else if `3 * 200 * s_max <= 1080`, run one seed (`1101`) for all three models at 200 steps each and label M06 single-seed.
4. Else choose one-seed steps as `floor((1080 / (3*s_max)) / 10) * 10`, capped at 200.
5. If that value is below 40 steps, stop the main experiment and report compute infeasibility rather than changing architectures or task difficulty.

Evaluation budget is the complete frozen 128-example test split once per final checkpoint, batched, plus validation snapshots for the learning curves. No test-time search is allowed in M06.

## Stop conditions

A model fit stops and is recorded as a failure if any of these occur:

- non-finite loss
- non-finite gradient norm
- runner/main wall-clock budget exceeded
- checkpoint serialization failure
- ternary final quantization strength differs from 1.0
- data-overlap audit fails
- preregistered parameter-count relation fails

There is no accuracy-based early stopping.

## Learning-failure localization

Exact-solve rate equal to zero is **not** by itself an optimization failure.

A fit is classified as an optimization-learning failure only if its final-window training loss does not improve by at least 5% relative to its initial-window training loss, or gradients are zero/non-finite. For a failing architecture only, M06 then records:

1. one-batch gradient inspection,
2. an 8-example overfit check,
3. a minimal 4×4 Sudoku configuration as pipeline debugging only.

No new mechanism may be added in response to a failed learning check inside M06.

## Evidence to preserve

The workflow artifact must include:

- this preregistration and git commit identity
- runner/environment metadata
- frozen data manifest and split IDs
- timing probe and derived seed/step decision
- exact model parameter counts
- per-model/per-seed checkpoints
- JSONL learning curves
- per-example test predictions with stable IDs
- aggregate test metrics
- gradient norms and failure records
- ternary per-layer quantization strengths and ternary statistics
- diagnostic outputs if triggered
- final M06 summary

The state log is updated only after observing the retained evidence.
