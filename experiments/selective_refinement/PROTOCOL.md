# Calibrated selective refinement — locked before new fitting/evaluation

Parent: c5af3032e1f9dd8908e922836dd7c06d4222b876. The earlier head-conditioning
pilot is retained in experiments/conditioned_heads: it changed no official-test
prediction and gives no basis for a large conditioning/accuracy claim. Do not
promote its optimizer simply because it converges differently.

## Question and architecture
A 256-prototype classifier is the cheap first stage. Its top-two class-score gap
determines whether to return the cheap label or run a separately trained SVC using
the existing finite/adaptive executor. Both are actual trained classifiers, not
teacher labels or distillation. A single native call performs routing, necessary
normalization, optional stronger inference and ordered fresh output. This is a
conditional-compute SYSTEM with both models stored, not a single compressed SVM,
a better fixed-budget individual network, or an invented general cascade idea.

## Independent data roles within already exposed public tasks
Use original training partitions of Letter, Pendigits, Satellite and OptDigits.
First separate exact-feature groups into model-development80% / calibration20%
using first split of StratifiedGroupKFold(5,shuffle=True,random_state=20260930).
No calibration group enters model/geometry/scaler/early-stop fitting. Inside model
development use a second grouped first-of-five split (seed611) for selection.
Selection fit cap6000 / validation2000, identical roles for all candidate families.
No refit after calibration. All official tests were previously consumed; no new
independent population/held-out discovery claim is permitted.

Fast model: P256, local integer metrics, quarter4/units4,200 epochs, regularization
1e-6, no affine/variable-width/normalization/averaging. Select gamma2/8 on Letter
and Pendigits,8/32 on Satellite,.125/.5/2 on OptDigits. SVC: C1/10/100 by gamma.5/2/8,
OptDigits gamma.125/.5/2. MLP: StandardScaler fitted inside fitting role, ReLU
128x128/256x256, alpha1e-5/1e-3, Adam .002, early stopping with internal10%, max400,
patience30. Choice by validation correct count then listed grid order. Refit each
selected family exactly once on all model-development rows, seed20260930. Counts,
regularization and training costs differ; no equal-compute claim. Final fast/SVC/
MLP artifacts are hashed before calibration; original full-training old models
are not silently compared as if they had the same data.

## Fixed calibration rule
Gap thresholds:0,.25,.5,1,2,3,4,6,8,12,16,24,32; also always-strong fallback.
For a threshold, count calibration rows where accepting the cheap answer would
make an error that the stronger model does not make. Calculate a one-sided exact
binomial upper confidence limit with delta=.05/13. Choose the lowest threshold
whose upper bound is <=.01. Thus primary epsilon1% is an added-harm bound relative
to stronger inference, NOT an absolute1% error claim. Also retain epsilon.005/.02
as declared sensitivity cases using the same simultaneous bounds, never choose
between them from test performance. An always-strong policy has identically zero
added harm and does not need a sampled confidence bound.

Under iid calibration/target pairs independent of trained models, Bonferroni gives
simultaneous finite-sample control and R(cascade)-R(strong)<=P(accepted harmful
cheap prediction). These benchmark writer/scene shifts and repeated research
exposure do NOT establish that iid premise. Report calibration bounds and observed
test harm separately, with no claimed distribution-shift safety/per-query proof.
Reference: Angelopoulos et al., Learn then Test, arXiv2110.01052 (2021/2022).

## Comparison and decision
Freeze calibrated policies before opening any official test in this run. Compare
always-cheap, always-strong, primary cascade, declared strict/loose policies, an
input-hash confidence-blind route with calibration-matched accept rate, and the
selected FP32 OpenBLAS MLP. All model-specific reference outputs, fixes/regressions,
route fractions, observed harmful acceptance, accuracy and macro-F1 are retained.
Target engineering/application gate: >=.5-point accuracy above the cheap model and
<=.5x always-strong mean complete-call cost on at least two tasks, and <=1 point
observed accuracy loss against always-strong. This gate is not statistical proof
or superiority over every possible classifier. Memory includes BOTH models.

Seven randomized complete jobs, batches1/32/256, all test rows, one pinned CPU.
Include validation, routing, packing, normalization, numerical evaluation and new
labels. Exclude loading, calibration and original image/sensor feature extraction;
report fitting/calibration/preparation and memory separately. No model or threshold
revision after official evaluation. No historical timing values substituted.

## Reproducibility repair
Only canonical source and newly generated evidence are used. Source manifests
bind the exact executing verifier, runtime and imported local files. Preserve all
actual new deployed model bytes, split indices, calibration outcomes and source
in the delivered artifact, not just a summary PASS. The old missing local audit
copy/model evidence means the previous mismatch remains unclassified, not waived.
Native scope remains Linux x86-64; no inherited Windows/ARM acceptance or release.
