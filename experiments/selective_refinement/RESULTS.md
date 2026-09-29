# Selective refinement: complete new evidence, September 29, 2026

## Decision

One native conditional-compute system now runs a compact classifier first and
invokes a selected SVC only for uncertain inputs. On Letter it corrects44 net
errors relative to the cheap stage, reaching3897/4000 versus3896/4000 for the
always-strong model at21.35% of its measured batch32 cost. This is a useful observed
tradeoff, not statistically established superiority over the stronger classifier.
The overall two-task accuracy/cost gate FAILS. Optical improves accuracy but not
speed; Satellite loses one answer versus the cheap stage; Pendigits performs no
refinement and exceeds the nominal calibration harm budget on its official test.
No experimental model, policy or backend is promoted to a production default.

## Scope and recovered evidence

PR40 parent c5af3032e1f9dd8908e922836dd7c06d4222b876 and tree
442f7c7d799b093b3c6dceb383e0e18fbd640206 were reconstructed from the actual CI archive
and real Git bundle. Their source identity was verified, including the canonical
old audit.py. The missing previous local auditor and old24 trained prototype
models were NOT recovered. Therefore the earlier source mismatch remains
unclassified and its full scientific audit was NOT rerun. New results are a new
experiment with actual saved artifacts, not a continuation of unrecoverable fitted
models or an inferred old PASS. The prior official summary values remain historical.

The new source-bound audit runner hashes and executes the SAME auditor bytes and
records that digest in its result. It refuses a different source before execution.
A separate closure manifest checks every source/data file. These prevent accidental
source/result mismatches but are neither signatures nor proofs of trustworthy
clocks, statistical independence, or malicious wholesale evidence replacement.

## Rejected first direction: output-head conditioning

A training-only pilot tested centered diagonal and Cholesky preconditioning of the
same ridge-softmax objective. The original regularization was transformed correctly
and the coefficients folded back without increasing inference size. Raw150/raw300
iterations and warm continuations were retained, so changing optimizer coordinates
was not compared only against an artificially short baseline. Across four tasks
there was no convincing consistent validation gain. Letter remained1372/1500 and
optical753/765 throughout. The few Satellite differences were not promoted. This
pilot consumed70.928 CPU seconds, did not open a new official test, and is retained
with every objective/gradient/timing outcome. The new system below does not use
this optimizer or claim its accuracy improvements.

## New model selection and independent roles

For each original training partition, exact-feature groups are split into80%
model-development and20% calibration using first-of-five grouped stratification
(seed20260930). A second grouped development split(seed611) chooses hyperparameters.
No calibration row/group enters geometry, scaler, head, SVC, or MLP fitting, and
no model is refitted after calibration. All12 final models are frozen before
calibration and all policies before official evaluation. This avoids fitting-role
leakage; it does NOT by itself establish the IID sampling needed by a probability
guarantee. All four official tests were previously exposed in SPECTRA research.

The cheap model is256 local integer prototypes with200 epochs and the parent
quantized-geometry training. The stronger model is a separately fitted RBF SVC
using the prior finite/adaptive executor. The MLP receives StandardScaler, selected
128x128/256x256 layers, regularization and a mechanically exported FP32 OpenBLAS
implementation. All families share roles, not equal training cost or parameter
counts. The complete fixed selection has61 fits and12 final refits. Final models
use all DEVELOPMENT rows, not the calibration rows. Comparisons to prior models
fitted on100% of training data would not be matched, so none is used as a new speed
baseline. Earlier higher-accuracy models remain: this is not best-ever accuracy.

## Calibration: added harm, not truth certification

A fixed grid of13 top-two-score-gap thresholds is tested using a one-sided exact
binomial upper bound with .05/13 failure probability. The primary added-harm budget
is1%; .5% and2% are fixed sensitivity policies, not replacements picked after tests.
Harm means an accepted cheap answer is wrong while the stronger answer is right.
The relative risk identity is R(cascade)-R(strong)=harm-benefit<=harm. Under fixed
models and independent IID calibration/target sampling, simultaneous bounds justify
threshold selection over this fixed grid. CONTRACT.md states the argument and all
assumptions. It is not a1% TOTAL-error guarantee or individual correctness proof.

The grouped/stratified benchmark and its contributor/scene differences do not
establish those IID conditions. Specifically, Pendigits calibration has one
harmful shortcut among1499 rows and accepts everything, whereas official-test
harm is59/3498=1.6867%, above the nominal1% budget. After12 beneficial shortcuts,
the relative accuracy loss is47/3498=1.3436 percentage points. This observation is
retained and invalidates an unconditional or shift-robust safety claim. The rule
was not retuned on that test and the stricter policy is not substituted as primary.

| Task | Primary gap threshold | Calibration cheap acceptance | Primary test cheap acceptance | Test harmful shortcuts |
|---|---:|---:|---:|---:|
|letter|2.0|2948/3200|3721/4000|14/4000|
|pendigits|0.0|1499/1499|3498/3498|59/3498|
|satellite|6.0|641/887|1399/2000|0/2000|
|optdigits|4.0|712/765|1578/1797|1/1797|

## All frozen test results

| Task | Cheap | Always strong | Primary refinement | Blind routing | Strict sensitivity | Loose sensitivity | Selected FP32 MLP |
|---|---:|---:|---:|---:|---:|---:|---:|
|letter /4000|3853|3896|3897|3854|3896|3876|3849|
|pendigits /3498|3383|3430|3383|3383|3408|3383|3404|
|satellite /2000|1812|1805|1811|1813|1805|1809|1808|
|optdigits /1797|1731|1760|1760|1734|1760|1753|1729|

Letter fixes63 cheap-stage errors and introduces19, net+44/4000 (+1.1 points,
147 errors become103). Only279/4000 rows invoke the stronger model. The primary
improves44 net answers while the confidence-blind control improves one, although
the hash control has only one fixed seed and test acceptance is not forced to
match the primary. The results do not prove universal confidence ranking quality.

Optical fixes39 and introduces10, net+29/1797; its prototype stage is nearly as
expensive as the SVM, so conditional execution provides no speed win. Satellite
fixes62 but introduces63, net-1. Pendigits' primary accepts all rows and makes no
accuracy gain. Historical Letter3928/3929 and Pendigits3444 are still higher than
this pair's outputs. The system is not a stronger fixed-budget individual model;
it pays for a second model when uncertain and stores both at all times.

## Complete native timing

The fixed588-job matrix uses all four test sets, seven randomized repetitions and
batches1/32/256. Raw uint8 code input to fresh labels includes validation, routing,
required normalization/scaling, kernels/layers, crossing the native interface and
allocation. Loading, table preparation, calibration and original image/sensor
feature extraction are excluded. The same selected SVM uses the existing optimized
finite/adaptive path, not the old full-FP64 slow baseline. The MLP uses one-thread
OpenBLAS SGEMM, not Python layer dispatch. Each arm matches its own frozen outputs.
Both model stages are loaded for the fast-only and strong-only benchmark controls;
standalone component memory is measured separately. All route session objects are
prepared before each task's timing loop. No outlier is removed or cell retimed.

Host: Intel Xeon Platinum8272CL @2.60GHz, Linux x86-64, CPython3.13.5, GCC strict
noncontracted AVX2 build, one pinned CPU, one numerical thread. These are observed
whole-job costs on this host, not production P95, concurrency or other-device
latency. The model pair and splits differ from previous experiments; no cross-host
historical timer is substituted. All1,660,365 repeated outputs concern11,295
underlying rows, not that many independent samples.

| Task | Cheap us/row | Strong us/row | Primary us/row | FP32 MLP us/row | Primary/strong | Fixed quality/cost gate |
|---|---:|---:|---:|---:|---:|---|
|letter|5.090171|43.964596|9.386934|11.720683|0.213511|PASS|
|pendigits|4.798205|11.087076|3.475444|7.929926|0.313468|FAIL|
|satellite|5.281513|15.498709|10.931385|8.080923|0.705309|FAIL|
|optdigits|8.205383|8.619837|8.722280|3.186594|1.011885|FAIL|

The headline Letter cost is4.68x lower than always strong (78.65% less time), but
1.84x the cheap-only model. Letter batch1 has a smaller2.94x speedup; do not describe
a batched result as isolated request latency. Its seven paired primary/strong
ratios span0.167–0.272. Optical's paired ratios span0.721–1.582, so its1.012 median
ratio is not precise evidence of a small regression or gain. Even identical-answer
fast/accept-all paths show material timing variability on this host. Large Letter
savings are the supported claim, not every small percentage in the matrix.

The gate requires >=0.5 points over cheap, <=0.5x strong time and <=1 point quality
loss relative to strong on TWO tasks. Only Letter passes; the overall gate remains
FAILED. Sensitivity settings and the blind control are not retrospectively promoted.

## Storage and actual fresh-process memory

No model compression is claimed. Both original trained components, tables, packed
representations and worker storage remain. Model-file sums exclude OS/native shared
libraries; counted native state is not peak RSS and excludes allocator metadata.

| Task | Cheap file bytes | Strong file bytes | Combined files | Counted pair state | Cheap-only peak KiB | Strong-only peak KiB | Pair peak KiB |
|---|---:|---:|---:|---:|---|---|---|
|letter|69976|2215512|2285488|2721592|16984–16992|23928–23964|24128–24276|
|pendigits|37032|203472|240504|1718948|16872–16936|18524–18532|18848–18888|
|satellite|49280|332472|381752|796468|17088–17128|17700–17752|18216–18256|
|optdigits|86184|500960|587144|1107640|17024–17044|18516–18556|18856–18892|

There are36 fixed fresh-process memory/setup observations (three per task/component).
Each loads the actual model(s), validates one predicted row, reads its OWN VmHWM,
and imports no numerical Python library. Shared-library mappings differ across
standalone components. Filesystem caches may be warm. These are not maximum-batch
or cold-cache memory results. The pair's Letter footprint is about23.6MiB, not the
small file size of the prototype stage, and it retains the full SVM dependency.

## Validation and completeness

Seventy unique NEW automated tests cover native routing, invalid buffers, closure,
calibration, the conditioned objective, policy/model binding and exact-source
receipts. Twenty-five core refinement tests also pass on the final portable and
AVX2 UBSan libraries. The parent's68 prototype contracts pass against the final
refinement library. These repeated builds are not extra independent test cases.
An earlier parent-test command was interrupted by the shell time limit; its log
is retained and the whole68-case test run was repeated, not counted as partial
success. A packaging-script syntax mistake was fixed before any kit was generated;
it changed no fitted model, scientific result or timed implementation.

The separate scalar observer reproduces168,950 prototype class scores bit-for-bit.
Every new native strong-model output matches its fitted sklearn classifier, and
all FP32 MLP outputs match an independently expressed float32 forward pass with
zero changed labels relative to the selected original MLP. This is same-environment
fidelity, not proof that the ground-truth label is correct.

The standalone auditor independently reconstructs61 choices,12 final model/role
inventories,52 calibration threshold rows, binomial bounds by CDF inversion,
all routing outcomes/harms, exported prototype and FP32 bytes, and588 timing cells.
Its PASS explicitly embeds the FAILED two-task gate and Pendigits budget violation.
Sixteen corrupted disposable copies are rejected. The audit is run through the
exact-source binder, and final sealed artifacts are re-extracted and checked.
This cannot authenticate clocks or validate IID assumptions from hashes.

Both supplied offline builds replay67,770 policy/control predictions and an
additional11,295 chunk-boundary checks over the same11,295 inputs. They load no
numerical Python framework. Input arrays, actual model bytes and policy hashes
are included this time, not just summary receipts. Scientific replay and claimed
CI acceptance refer to this new source, never the unrecovered old local auditor.

The new selection/final fitting used507.373 CPU seconds (~8.46 minutes), the failed
conditioning pilot70.928 seconds, and calibration3.570 seconds. Data recovery,
imports, compilation, evaluation, benchmarks, tests, audit and packaging are
additional. No GPU, pretrained teacher, label API or distillation was used. All
four datasets were already exposed; no fresh outside-user reproduction or world-
first result is claimed. No Windows/ARM, ASan, full historical suite, production
wheel or release acceptance is inferred. The prior numerical runtime assumptions
remain applicable to the strong stage.

## Next scientific boundary

This is a conditional-compute application improvement, not a new general learning
algorithm or general intelligence. The unresolved issue is reliable routing under
source shifts and the cost of carrying both models. The current experiment provides
counterevidence against treating random training-role calibration as a deployment
risk certificate. Future policies need independent, source-appropriate calibration
and a fresh evaluation boundary, not tuning these now-open thresholds to erase
Pendigits. Always-strong remains an explicit policy when no admissible shortcut
exists. Neither this result nor additional tests justify an automatic high score.

Primary references: Learn-then-Test, https://arxiv.org/abs/2110.01052 ; ProtoNN,
https://proceedings.mlr.press/v70/gupta17a.html . Those techniques are established
prior art, not inventions claimed here. Dataset sources and CC BY4.0 attribution
are in DATA_SOURCES.md. No third-party upstream submission occurred.
