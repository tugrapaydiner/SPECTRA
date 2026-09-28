# Learned finite-signature kernels: accuracy-focused protocol, 2026-09-28

Production ancestor b9d9f90; prior local finite backend e9e3274 remains separately
preserved. This continuation changes trained functions, unlike prior same-model
execution work. No new release, universal superiority or generic-intelligence claim.

Question: can a learned nonnegative mixture of radial kernels improve recognition
while finite-domain compilation avoids per-basis inference evaluation? Multiple
kernel learning and metric learning are established techniques; no world-first
status is assumed. An absent improvement must remain an absent improvement.

Development uses Letter first16000 and Pendigits training7494 with original
fit/validation indices. Their previous test partitions were already consumed;
later re-evaluation is development-transfer evidence, never fresh confirmation.
Initial controls: equally searched single-scale squared-distance RBF and L1
Laplacian SVMs, uniform mixtures and training-only nonnegative kernel mixtures.
Four scales .25,1,4,16 divided by training-subset median distance; C1/10/100.
At most1536 training rows for centered-alignment kernel-weight learning. Pairwise
one-v-one fits may replace a giant Gram matrix to bound memory. Preserve every
pair and deterministic class-order voting. Include linear and small-MLP controls
for task-level context. All exploratory changes/results remain logged.

Fresh Semeion confirmation: obtain the official UCI178 archive only. Fixed
stratified50% development/50% test, random_state20260928. Within development,
80% fit/20% validation with random_state20260929. Exact duplicate inputs must remain
in one partition when possible; report a grouped split deviation before model fits
if duplicates require it. Writer identities are not supplied, so no writer-disjoint
claim. The supplied binary features are not pretrained representations.

Freeze the method family/search after old-task development but before any Semeion
test-model calls. Select using its development validation, save all candidate
weights/checkpoints and predictions, then hash the selected models and final
selection record before opening its test predictions. No post-test refitting,
reselection or further experiments on test errors. Report all frozen controls.

Numerical scope: canonical integer inputs are rescaled by a power-of-two
denominator. A nonnegative mixture is positive semidefinite. The compiled table
must retain the chosen scalar kernel's evaluation order; finite-signature identity
is not exact-real exp or true-label certainty. Out-of-domain data must fall back to
the SAME learned model, not a previous different classifier. Accuracy and runtime
fidelity are separately measured. Charge supports, table/model memory, preparation,
transformation, every fallback and fresh outputs. Model training is CPU-only.

References: Rakotomamonjy et al., SimpleMKL (JMLR2008); Cortes et al., Algorithms
for Learning Kernels Based on Centered Alignment (JMLR2012). Official Semeion data:
https://archive.ics.uci.edu/dataset/178/semeion+handwritten+digit
