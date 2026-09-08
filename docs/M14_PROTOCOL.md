# M14 preregistration: trained accuracy versus complete-solve latency

Status: frozen before M14 dataset generation, model fitting or accuracy inspection.
Parent source: `b9e2decf19dcc772bfcef3a78615c2f4c8b350c0` (M13 merged).
Machine-readable declaration: `config/m14_comparison.json`.

## Hypothesis and acceptance

On fresh generated, uniquely solvable **9×9 Sudoku with 30–35 requested clues**,
a small trained ternary recursive model improves strict semantic solve rate by
at least **5 percentage points** against BOTH a trained larger FP32 single-pass
model and its supported conventional dynamic-INT8 deployment, while its mean
complete-solve latency is at most **1.05 times** each baseline's latency.
No accuracy loss is permitted in exchange for lower cost. Lower cell loss alone
does not satisfy this hypothesis. The FP recursive model is a required control.

Acceptance requires the lower confidence bound for each accuracy difference
to exceed +0.05, and the upper confidence bound of each latency ratio to be at
most 1.05, on independent confirmation data after a passing development gate.
For both accuracy and cost, use the more conservative of (a) a paired two-way
bootstrap resampling training seeds and examples independently while preserving
all system pairings, and (b) a paired Student-t bound across the three seed-level
means (log ratios for cost). Use one-sided alpha 0.025 per baseline, Bonferroni
across the two baseline comparisons. This is an intersection-union gate: all
accuracy and cost requirements must hold. Three training seeds remain a small
pilot; bootstrap/t bounds cannot promise coverage under arbitrary dependence.
If all successes are identical, report the degenerate empirical interval and
do not mistake it for absence of population uncertainty; also report a Wilson
interval over per-example outcomes for each seed separately.

Energy counters are probed. This host exposes no GPU or powercap CPU-package
counter. **Latency is the predeclared cost endpoint**, not a proxy labelled
as measured joules. Energy is null with a reason; no energy-superiority claim.

## Data and evaluation hierarchy

- New data seed: 2026091401; existing M03 generator and independent RNG streams.
- Train: 1,536; tuning: 192; development gate: 192; confirmation: 512.
- Build M03 train/validation/test sizes 1536/384/512, then assign the first
  192 validation records to tuning and the remainder to development.
- Require zero exact pair, input and unaugmented group overlap across all four
  splits, unique IDs, and no within-split duplicate inputs. Persist inputs,
  solutions, stable IDs, group IDs, hashes, actual clue counts and RNG metadata.
- The existing grouping protects exact bases and their attached augmentations;
  it is not a proof against every possible Sudoku symmetry equivalence.
- Earlier milestone test sets are prior development knowledge. The fresh M14
  confirmation partition must not be loaded by fitting, tuning or development.
  Generation and hashing are allowed; no confirmation predictions or statistics
  are computed unless a frozen development candidate passes.
- A manifest read and overlap audit do not evaluate a model. Save the confirmation
  arrays in their own file and log each model-evaluation split access.
- If confirmation fails, retain it and stop: this protocol permits ONE final
  confirmation attempt. Future changes need a new protocol and new independent
  confirmation set; this one becomes development evidence.

This is a synthetic Sudoku distribution, not ARC, a language benchmark, or a
claim about official benchmarks. No task is replaced with easier 4×4 puzzles.

## Families, training and tuning

Common: AdamW, weight decay .01, batch 32, norm clip 1, 600 optimizer updates,
seeds 1401/1402/1403, learning rates .001/.003, CPU FP32 master parameters.
Every family receives both learning rates and every seed. Minibatch indices
are paired by seed across families and learning rates. No favorable seed is
selected. Report all 18 initial fits, curves, checkpoints, failures and costs.

- FP recursive: existing TRM, dim 48, one block, four heads, n=1, T=1,
  N_sup=4, max_grid_size=16, ten tokens and 81 positions.
- Ternary recursive: same geometry; existing FakeBitLinear + A8 recurrent-state
  quantization. Quantization strength ramps to 1 by step 150; checkpoint and
  evaluation enforce exactly 1 in every layer. FP32 arithmetic remains in the
  reference fake-quantized path. Bias/attention implementation differences are
  reported with actual parameter counts, not hidden behind parameter matching.
- Larger single-pass: existing System1Student, dim 96, two blocks, four heads.
  Its untrained confidence head is frozen and is not used for routing.
- Dynamic INT8: convert supported FFN and output/confidence Linear projections
  of the trained single-pass model with PyTorch's x86 dynamic quantization.
  MultiheadAttention, embeddings and norms remain FP32. Record converted names
  and weight coverage; this is **mixed INT8/FP32**, not an entirely INT8 model.
  Conversion failure is a missing required baseline, never an automatic pass.
- Symbolic reference: existing `data.sudoku.solve`, supplied only the puzzle,
  followed by independent Sudoku semantic validation. This is a contextual
  solver with hard-coded constraint and backtracking inductive bias, no training.

Use token CE for single-pass and the average of token CE over the recursive
supervision outputs. Freeze unused recursive halt heads. This common answer
objective avoids training one family with auxiliary halting losses that are
irrelevant to this fixed-budget comparison. Deep supervision and state detach
semantics still differ from a single-pass computation and are disclosed.

Checkpoint/tuning snapshots occur at steps 200 and 600. Rank each family's
(learning rate, checkpoint step, inference depth) by mean tuning semantic solve
rate across ALL three seeds, then mean blank-cell accuracy, then lower measured
latency, then stable configuration order. Recursion inference depths 1/2/4 are
an inference-budget sweep of those fixed weights, not additional training runs.
Choose INT8 separately on the same tuning partition/grid, without extra fits.
Do not describe a few points as a scaling law. Parameter size is fixed by family;
training-step and inference-depth sweeps remain separate columns in artifacts.

For ternary deployment, evaluate the reference PyTorch fake-quantized graph and
the existing M10 packed scalar CPU runtime when its native compiler works.
Export/load/build is separately timed; all recurrent conversions are inside the
warm solve. Require numerical/answer fidelity on tuning examples. A failing
native lane is retained and excluded from candidate selection with its reason.
Select the faster eligible ternary backend using tuning cost only.

M09 learned action experiments were negative and used a different 4×4 reasoner;
M12 learned control collapsed. There is no validated compatible trained 9×9
action/verifier/controller checkpoint here. Do not plug random auxiliaries into
the primary comparison or transplant incompatible weights. Record the exclusion.

## Complete-solve measurement and budget language

Batch size one, CPU inference_mode, two intra-op threads and one inter-op thread.
Use 8 distinct tuning warm-up examples. Measure 48 distinct tuning examples for
selection costs; evaluate every development example in three shuffled rounds.
Randomize system order within each example/round to reduce order/drift bias.
Freeze the selected configuration, checkpoints and budget thresholds before
development and again before the sole permitted confirmation evaluation.

Start the clock before converting the raw NumPy puzzle to a CPU input tensor.
Include the entire model, actual recurrence, unused heads that the ordinary
forward still executes, output conversion/argmax, any declared decoder and an
independent semantic check. Stop before serializing/logging the result.
Report first solve/model load/conversion separately from steady-state latency.
Training and tuning time are reported separately from inference.

Use paired per-example medians over the three timing rounds for analysis;
retain every raw timing row. Repeated timing rounds are not independent tasks.
Report mean, median and p95, seed-level solve and blank-cell metrics, all actual
per-example costs, and available aggregate package-energy windows separately.
No zero-joule substitution, synthetic costs, idle padding or cost truncation.

For an explicitly **iso-latency** label, the measured mean ratio must be in
[1/1.05, 1.05]. A cheaper point outside this range is `lower_cost_unmatched`;
an expensive point is `higher_cost_unmatched`. The permitted no-more-cost
hypothesis is distinct from equality of cost. Frozen budget caps come from the
selected baseline tuning latency times 1.05; actual development/confirmation
cost ratios and uncertainty must still pass. A slow point remains unmatched.

## Bounded failure investigation

Timing-only preflight (discarded models, no accuracy inspected) observed roughly
.173/.118/.085 seconds per step for FP/ternary/single-pass at N_sup=4 on this host.
Set a **2,700-second cumulative fitting budget** for initial and corrective fits,
with a 600-step fixed per-fit target. Do not shorten completed arms for parity or
convert interrupted runs into final candidates. Preserve partial checkpoints,
optimizer/minibatch RNG states, elapsed time, errors and completed curves.
Stop on nonfinite loss/gradients, invalid data, checkpoint failure or exhausted
budget. No paid hardware or remote training provisioning.

If the initial development gate fails, investigate clue copying, blank-cell
errors, invalid digit emissions, constraint violations and actual cost breakdown.
One predeclared falsifiable intervention is permitted: train **all three**
families with blank-cell-only CE at each selected family learning rate, the same
three seeds and 600-step target. Clamp given clues and exclude token zero at
decode for every neural family, including INT8. Keep the same primary target.
Prediction: blank accuracy should improve if easy clue reconstruction is
dominating the objective; counterevidence is no improvement or regression.
This changes objective and decoder together; separate decode-only measurements
on initial checkpoints identify the immediate decoder contribution. It is a
development intervention, not a clean causal proof of either component alone.

Intervention configuration selection uses tuning only; the development gate
has already informed this attempt and is labelled reused development evidence.
No further tuning or new mechanism is permitted inside this bounded pilot.
Either passing development attempt may unlock the independent final comparison.
Failure/inconclusive evidence does not complete milestone 14 and cannot unlock
milestones 15–20 as though the required improvement had been established.

## Provenance and reproduction

Before execution, commit this declaration, configuration and experiment code.
Use content-bound run IDs, relative paths, source and config SHA256s, exact
data/checkpoint hashes, environment/backend metadata, per-example predictions,
learning curves and an append-only split-access/attempt ledger. Hash the complete
bundle inventory after the run. Never overwrite a completed experiment directory.
Generate tables and Pareto plots solely from retained raw rows; record their
input hashes. Retain all selected and unselected trained checkpoints, including
failed/incomplete fits. Do not report M14 COMPLETE unless the gate above passes.
