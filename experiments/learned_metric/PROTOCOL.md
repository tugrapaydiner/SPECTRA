# Label-trained integer metrics — prospective continuation

Parent: e9e32743149bd15482f2144cbe90b32800da7781 (finite-kernel experiment).
Question: can a label-trained distance geometry improve classification while
retaining finite integer signatures and precomputed RBF kernel execution?

## Data and independence

Use the original UCI Letter first 16000/last 4000 split and the official Pendigits
7494/3498 split from the previously acquired, hash-recorded archives. These test
sets have ALREADY been consumed in prior SPECTRA development. An interrupted
continuation also reported a Letter result without a recoverable model. Therefore
no result in this continuation is labelled fresh independent confirmation. Do not
use those old aggregate claims as numerical evidence. No evaluation labels are
used in this continuation's learner, hyperparameter selection or candidate revision.
Test evaluation happens once after saved final models and selection records.
Preserve duplicate-feature counts and a separately reported nonoverlap stratum.
A fresh Satellite download attempt failed; it is not part of the accepted panel.

## Mechanism and controls

Primary: a global nonnegative diagonal distance learned from nearby same-class and
different-class examples, followed by deterministic bounded integer quantization.
All coordinate values remain original integer codes. Signature S=sum_j w_j(q_j-r_j)^2.
Kernel exp(-gamma*S/(D^2*mean(w))) remains an RBF on a fixed diagonal embedding;
nonnegative weights preserve positive semidefiniteness in real arithmetic.
Retain w_j>=0 and sum w_j=2*d (at least one positive weight). No query-dependent
weights, label lookup, test-driven threshold, ensemble, teacher or pretrained model.
A same-geometry full-precision integer-signature reference must check native output.

Compare (a) uniform weights, (b) inverse training-feature-variance integer weights,
(c) learned integer weights. All use identical training examples and the same
nine C/gamma choices: C=[1,10,100], gamma=[0.5,2,8]. No refit/extra data advantage
for the learned arm. Keep the old frozen SVM as a historical, not matched, control.
Also fit a fixed small MLP and linear control without evaluation-set selection.

## Development and fitting budget

Three fixed stratified training-only splits with seeds611/977/1543. Each uses80%
for fitting and20% validation; cap fitting at6000 stratified examples and validation
at2000 for the selection sweep, identically across all arms. Learn the metric only
on that fitting subset. Use a fixed bounded neighbor/triplet learner (implementation
and all development changes recorded); first learner pilot uses training/validation
only. Choose each arm's C/gamma by aggregate validation accuracy, then fewer support
vectors, then fixed grid order. Refit each selected arm once on ALL original training
rows, refitting its metric from those rows. All final models frozen before opening
either evaluation set. Full-matrix training uses disk-backed bounded blocks and
one model at a time; do not claim this is scalable linear-time SVM training.
Use CPU only, one numerical thread; no GPU. Record actual fitting/selection costs.

## Evaluation and inference

Report complete accuracy, confusion/error pairs, macro-F1, per-class behavior,
model weights, support counts, serialized bytes and runtime/working-set tradeoffs.
A paired input bootstrap and discordant-case exact test are descriptive: repeated
benchmark development and correlated samples prevent fresh-confirmation claims.
Primary useful-model condition: at least0.5 percentage-point improvement over the
matched tuned uniform control and no more than1.25x complete-call cost on one task;
report both tasks even if one fails. No automatic high-grade or broad intelligence
claim follows. Interpretation is classification capability only.

Compare new compiled metric execution with scalar/exhaustive identical-model
execution, sklearn precomputed execution with K preparation included, and the old
finite executor where applicable. Use all evaluation rows, batches1/32/256,
seven randomized whole-job repetitions, one pinned core, no concurrent training.
Count input validation, query distance/table work, label return, all kernel/pair
work. Exclude loading/preparation and record these separately. No latency number
from another host/run is used as its measured comparison.

## Preservation and delivery

All original730 files remain unchanged. New models have a distinct magic/format
and cannot be silently loaded as ordinary RBF artifacts. Bounds, structural tests,
label parity, scalar/AVX2 comparison and installed-independent smoke are required.
No default/release replacement. Keep failed hypotheses and all raw outcomes.
Metric learning and quantization are established techniques, not a first-invention
claim. Prior art: Weinberger/Saul JMLR2009; Cortes/Mohri/Rostamizadeh JMLR2012.
