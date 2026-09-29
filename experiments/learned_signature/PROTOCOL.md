# Learned finite-signature classifiers: development protocol

Base: e9e32743149bd15482f2144cbe90b32800da7781 (finite-kernel experiment).
This continuation changes the fitted classifier, not merely execution of old answers.
No existing production source or previously frozen benchmark model is replaced.

## Primary question
Can supervised diagonal metric learning and/or a learned nonnegative mixture of
radial distance scales improve classification while retaining one integer distance
and one stored kernel-table lookup per evaluated support vector?

This combines established metric/multiple-kernel learning with finite-domain
execution; no priority, general-intelligence, or novelty claim is preregistered.
The stored table defines the new finite-domain model. It is not a claim of exact
fidelity to a separately rounded continuous-input RBF model.

## Data and exclusion boundary
Use the original archived UCI Letter and Pendigits bytes. Letter: first 16000 rows
for development/final fitting, last 4000 for final descriptive evaluation.
Pendigits: original training file for development/final fitting and original test
file for final descriptive evaluation. Those test sets have been consumed by past
SPECTRA experiments: they are NOT fresh confirmation, even though no test model
calls or tuning will occur here until the final candidate freeze.

Prior incomplete conversation progress mentioned two unidentified new holdouts;
no recovered artifact identifies them. Neither is assumed available or evaluated.
Fresh Landsat/Semeion acquisition attempts failed (DNS); keep the failure receipt.
Do not substitute fabricated fresh-test status or recover earlier unsaved results
from prose. All work below is a new reproducible development experiment.

## Fixed development splits and candidates
Training-only stratified 80/20 partitions, seeds 1401, 2402, 3403. On each partition,
cap fitting to 5000 stratified training examples; validation uses all held-out
training rows. All learner families share exactly the same fitting/validation rows.
Raw integer features are retained, scaled only by their known domain maximum for
baseline hyperparameters. No feature standardization learned from evaluation rows.

1. Tuned ordinary isotropic RBF SVC, C in {1,10,100}, gamma multiplier in
   {0.125,0.25,0.5,1,2,4,8} relative to training-only sklearn gamma='scale'.
2. Supervised diagonal metric: up to 2048 fitting anchors, nearest same/different
   class points selected from at most 4096 fitting candidates, self-pairs excluded.
   Three rounds of triplet re-mining; positive trace-normalized weights optimized
   by a smooth triplet loss plus fixed log-weight regularization. Quantize to
   integer weights in 1..8 (mean near 4), then run the SAME 21-cell C/gamma search.
3. Nonnegative radial mixture learned by centered kernel-target alignment from at
   most 1024 fitting examples. Seven radial scales. Fit a nonnegative quadratic
   alignment combination; add a small diagonal ridge, normalize coefficients.
   Tune the SAME three C values and seven global scale multipliers. Compare with
   a uniform-mixture control. Metric and mixture combination is an optional,
   separately identified development arm, not an assumed gain.
4. LinearSVC C in {0.1,1,10}, and a fixed two-hidden-layer 64-wide MLP with
   training-only early stopping, max 300 epochs, three fixed seeds as controls.

Selection within a family is highest validation accuracy, then fewer supports,
then earlier grid index. Record every candidate, warning, and CPU/wall time.
The mixture/metric controls and all failed variants remain. CPU only, one numeric
thread per fit. Up to two independent processes may share this 5-CPU environment;
formal timing runs alone on one pinned core. No pretrained or teacher model.

## Final fits and evaluation
Select one configuration per family by mean three-split validation performance;
fit selected candidates on the complete official development portion. At most
one final model per task/family, with all final weights and table hashes frozen
before either test set is evaluated. No refitting after test results.

Primary learned-quality gate: candidate validation improves over tuned isotropic
RBF on average on both tasks; final descriptive accuracy increases on both tasks.
A result failing any part remains unpromoted. Report paired correctness changes,
bootstrap intervals for fixed models, exact paired tests, per-class results,
parameter/support counts, total fitting cost and complete prediction latency.
A second selected-family result is secondary, not a changed primary gate.

## Native execution and measurement
Define supported integer input domains explicitly. One immutable serialized model
contains integer weights, support vectors, ordered OvO terms, and the learned
binary64 distance table. All radial components are folded at export time. New
outputs may differ from previous models because learning is intentionally changed.
The native executor must match its own independent stored-table interpreter and
sklearn precomputed-kernel predictions; validate scalar and SIMD integer signatures,
empty/invalid inputs, multiclass tie rules, and isolated deployment.
Compare learned and baseline classifiers in the SAME new runtime and compare
against existing finite/ordinary executors where numerical interfaces apply.
Include all support, table, coefficient, setup and memory costs. No multiplying
same-model speedups and new-model accuracy gains into one unsupported claim.

## Interpretation
Changes in classification accuracy are narrow learned-capability results, not
language reasoning or general intelligence. Test reuse and missing fresh acquisition
preclude an independent-generalization claim. No self-assigned high-90s grade,
upstream submission, release publication, or default promotion follows automatically.
