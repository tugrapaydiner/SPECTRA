# Pre-acquisition integer-metric transfer — Satellite

This separate confirmation attempt follows the completed local Letter/Pendigits
learning study. That earlier result is retained: Letter3911→3928/4000 and
Pendigits3435→3432/3498 (uniform versus learned metric). The original0.5-point
quality gate therefore fails, and is NOT relaxed or replaced by this extension.
Those datasets were already consumed. No satellite model prediction has yet been
made, and the local official-data download failed. This protocol precedes the
successful acquisition and all satellite fitting/selection/evaluation.

## Data and scope

Acquire only official UCI Statlog (Landsat Satellite), id146, original sat.trn and
sat.tst partitions. Keep all36 original integer features and original labels.
Use the original8-bit0..255 domain. Preserve original archive/metadata/source hashes
and CC BY4.0 attribution. No downloaded executable, pickle, teacher or pretrained
model. This is a small historical benchmark, not a live satellite application or
geographic/time-disjoint deployment: the source lacks recoverable spatial IDs.
Exact duplicate feature rows must stay together within selection partitions;
report any original train/test repeated feature vectors separately, not silently
remove them. Reading test shape/domain for validation is not fitting or selection.

## Locked learner and equal tuning

Use the existing positive diagonal NCA-style learner:512 stratified anchors and
2048 gallery rows at most, no self matches, beta set by median same-label third
neighbor distance, continuous weights .5..4 with sum36, .02 identity regularizer,
SLSQP100 iterations, no further algorithm or objective changes from this dataset.

The native table limit is4,194,304 entries. To retain the full8-bit36-feature
schema within that limit, the uniform control has36 unit weights. Nonuniform
learned/variance arms use an explicit positive integer budget64: allocate one
unit to every dimension, then distribute28 units in proportion to nonnegative
continuous weights using largest remainders with lower-index ties. The trained
kernel divides by mean integer weight so overall scale remains comparable.
This projection is fixed before data acquisition, not selected on test accuracy.
The emitted model itself uses those integer weights; no inference substitution.

Three training-only group-stratified five-fold first splits, seeds611/977/1543.
Identical fit/validation rows for uniform, inverse-variance and learned arms.
Each arm has exactly nine C/gamma choices: C1/10/100 and gamma2/8/32. Select pooled
validation accuracy, then fewer supports, then lexicographic grid order. Refit each
selected arm once on all original training rows (metric seed20260928). No test
predictions until every selected model and control has been written and hashed.

Fit fixed additional controls: LinearSVC(C10,dual='auto',max_iter5000,seed20260928),
MLP(128,128,ReLU,Adam,lr.002,batch128,alpha1e-4,max_iter240,early_stopping,10% internal
validation,patience20,seed20260928), and QuadraticDiscriminantAnalysis(reg_param.01).
No test-based retraining/reselection. All controls and warnings remain in results.
This is not guaranteed equal training cost; report all actual CPU/wall costs.
CPU only, one numerical thread. Kernel training remains quadratic, not scalable.

## Evaluation and decision

Open the original test once after the model lock. Report every arm's accuracy,
macro-F1, confusion matrix, support/parameter count, storage and paired differences.
Descriptive row and exact-feature-group uncertainty is not a proof of spatial
independence. No world-first claim: NCA/metric learning and integer kernels are
established techniques; the question is whether learning improves this supported
compiled model's useful accuracy/cost tradeoff.

Target gate: learned metric at least0.5 percentage points better than independently
tuned uniform and at most1.25x its complete-call cost; compare every simpler
control before calling it a useful application winner. A failed quality result
stays failed regardless of speed. Both original tasks and this new task remain
visible. Do not tune the algorithm from this test even if it fails.

Verify all native labels against precomputed sklearn; independently compare actual
pair margins with unchanged original FP64 RBF execution on repeated integer
coordinates. Same deterministic exp/environment is assumed; no cross-library
exact-real guarantee. Record native full-call timings and kernel construction
separately. Preparation, additional table memory and raw feature-extraction limits
remain explicit. No production default, release or earlier evidence is replaced.
