# One additional learning diagnosis before official evaluation

The completed432-fit development selection rejects the primary full NCA
hypothesis: pooled validation accuracy is lower than the diagonal control on
Letter (93.9833 vs95.7333%), Pendigits(99.6441 vs99.6886%) and Satellite
(92.0331 vs92.3337%). No new official test prediction has been made.
Do not promote full NCA or hide this full grid.

Hypothesis: its global neighborhood-probability objective does not preserve the
fine within-class correlations that matter for the SVM. Try one simpler locally
estimated metric: covariance of differences to the three nearest SAME-LABEL
training neighbors. Its inverse-square-root estimates directions of locally
small variation, with the same0.05 mean-eigenvalue floor and trace normalization
as the already tested global whitening control. The integer projection and
factorized kernel are unchanged. This is related to established local metric
learning, not a claimed novel general learning algorithm.

Anchor cap2048, gallery cap6000 (all fitting rows below that cap), common seed,
self matches excluded. A second no-label local-neighbor covariance arm uses
identical sample caps and estimator, ignoring labels in neighbor selection.
This separates feature-distribution whitening from a supervised benefit.
No optimizer, projection coefficient scale or regularization search is added.

Run the identical pilot split/caps/C10/two-gamma choices for BOTH new arms on
all three tasks. Then one equal12-choice, three-split selection for both arms,
retaining the entire previous432-cell matrix. No test-based selection. Freeze
all final arms and selected models before the one exposed-partition evaluation.
The full NCA primary remains a failed hypothesis; any local-metric success is
explicitly SECONDARY/adaptive development and cannot retroactively pass it.
