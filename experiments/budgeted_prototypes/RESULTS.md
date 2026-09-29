# Budgeted discriminative prototypes — September 29, 2026

## Decision

The original two-task learning gate FAILS. The local-prototype model improves
Letter by1.4 percentage points over the selected fixed-prototype control at roughly
tied cost, but gains only0.4002 on Pendigits, loses0.05 on Satellite and gains0.1113
on the newly opened optical-digit test. Do not turn a one-task success into a
transferable learner or a default replacement.

There is a narrower useful deployment result. On Satellite,256 local prototypes
score1817/2000 (90.85%) versus1790/2000 (89.5%) for the selected MLP, with2.315 versus
4.065us/row using the stronger ordinary FP32 OpenBLAS MLP control. Serialized model
sizes are49,280 versus307,588 bytes. However, prepared native storage is290,552
versus307,840 bytes, so the6.24x FILE-size difference is NOT a6.24x RAM reduction.
All tasks, stronger competitors, failed learning gates and extra table costs remain.

## What changes and what is established

This changes model family, not only execution of an old SVM. Class-balanced k-means
initializes a fixed prototype budget. Fitting labels train a multiclass head and,
depending on the arm, prototype locations and prototype-local diagonal metrics.
Centers use quarter-grid codes; positive integer metric weights retain fixed mass.
Training uses a straight-through float32 quantized-geometry proxy. After integer
geometry is frozen, the head is refitted on the exact two-table product features.
There is no teacher, pretrained network, distillation, label lookup or ensemble.

A dedicated SPPRO001 format and framework-free C++ executor preserve the new
function's ordered scores. The prototype count does not grow automatically with
more training examples. The multiclass head evaluates every prototype, rather than
retaining thousands of support vectors and changing one global metric. This is
related to RBF networks and ProtoNN, not an invented general prototype-learning idea.
A direct competitive ProtoNN implementation and INT8 MLP are NOT measured here.
See CONTRACT.md for arithmetic, integer caps, ownership and unsupported platforms.

## Training-only selection and official-test boundaries

Six pilot panels retain60 fits, including rejected affine heads, adaptive widths,
normalized responses, finer quantization and late-iterate averaging. These did not
use new official-test predictions. The locked full matrix has248 fits: three
prototype families, SVC and scaled MLP over four tasks and two duplicate-grouped
training splits. Fixed/centers/local each initially receive P256/512/1024 and two
gamma values. Head penalty, epochs and quantization are common across these arms.

Optical fixed prototypes reached a lower-gamma boundary, exposing an unfair width
range. A protocol committed before test opening added36 equal-opportunity cells:
gamma0.125/0.5 for all optical families/budgets/splits. The original248 cells remain.
The stronger fixed optical control then achieved99.08% validation, slightly above
local98.95%; the initial apparent four-point advantage was not promoted. This is
284 total selection fits, not a test-driven correction. Final source/model locks
precede parsing the optical official test and all new-model official predictions.

The MLP is not an untuned Python strawman. Fit-only StandardScaler plus128x128 or
256x256 hidden layers and alpha1e-5/1e-3 are selected using the same outer splits.
Its own early-stopping validation is inside the training role. SVC receives a
nine-choice C/gamma grid; linear C10 is a fixed lower bound. Different families have
different search/compute budgets and selected parameter counts. Do not claim equal
training cost or fixed equal final capacity merely because data partitions match.

Twenty-four classifiers (six per task) are refitted once on all original training
rows. Four later FP32 MLP files are mechanical conversions, NOT four newly trained
models. Every selected model and control is hashed before official evaluation.

Letter, Pendigits and Satellite are previously consumed research benchmarks.
OptDigits is newly acquired/test-opened in this continuation, not a new dataset
in the literature. UCI describes30 training contributors and13 different test
contributors, but the feature files do not contain per-row writer IDs. No raw-image
feature extraction or real deployment is tested. Letter's380 exact development/test
feature overlaps remain, with a separate nonoverlap report. Exact-feature grouping
inside selection cannot establish writer/font/geographic independence.

## All official correct counts

| Model | Letter /4000 | Pendigits /3498 | Satellite /2000 | OptDigits /1797 |
|---|---:|---:|---:|---:|
| Fixed prototypes |3832|3415|1818|1749|
| Moving centers |3858|3434|1820|1756|
| Local centers and metrics |3888|3429|1817|1751|
| Selected SVC |3911|3435|1814|1763|
| Selected scaled MLP |3856|3384|1790|1737|
| Fixed linear |2787|3145|1632|1701|

The FP32 MLP retains every original MLP prediction on all11,295 rows. Its score
arithmetic is not bitwise equivalent to float64. Local-versus-fixed Letter fixes85
errors and introduces29:168 errors become112,33.3% fewer. Only Letter reaches the
predeclared0.5-point quality condition. Local versus moving centers is worse on
Pendigits, Satellite and OptDigits. All descriptive paired statistics are retained;
no multiple-comparison-adjusted universal or writer-independent significance claim.

Prior best exposed results are not erased: learned/interaction Letter3928/3929
exceeds this local model3888; older Pendigits3444 exceeds3429; prior Satellite1836
exceeds1817. The new model is not a best-ever accuracy improvement. Its value must
come from a useful quality/resource operating point. Current Satellite SVC1814 is
three below local1817, too small to claim established accuracy superiority and not
the strongest historical classifier. No unseen competitor is declared defeated.

## Three complete native comparisons

Every warm job starts with original uint8 feature buffers and ends with new labels.
Input checks, normalization/scaling, all kernels/layers, packing and allocation are
charged. Loading, table preparation, compilation, fitting and original image/sensor
feature extraction are excluded. One pinned AMD EPYC9V74 core, Python3.13.5,
GCC14.2, strict noncontracted prototype arithmetic; OpenBLAS is fixed to one thread.
These are whole-dataset jobs, not individual-service latency or production tails.

Run1 retains1176 cells and ordered native MLP/original SVC controls. Review then
added batch OpenBLAS float64 MLP and the prior finite/adaptive SVM engine on the
SAME selected models: run2 retains1344 cells. Review then added mechanical FP32
MLP weights and SGEMM, leaving all classifiers and prototype implementations fixed:
run3 retains1428 cells. Each full matrix includes every prior arm. No favorable
cells are pooled, selectively retimed or substituted. Final claims use run3.

FP32 conversion was locked before checking its outputs. An independent float32
forward pass and all three batch scopes preserve every original MLP label. The
same original fitted scaler, layers and class order remain, with normal roundoff.
The shared OpenBLAS binary/config and build receipts are recorded. This is not an
INT8 or quantization-aware trained MLP, and does not cover every efficient neural
implementation. The finite SVC baseline retains its previously documented conditional
numerical bounds. Every timed output matches its own frozen model.

### Final batch32 costs in microseconds per row

| Task | Fixed prototypes | Centers | Local | Same selected SVC, finite | FP64 BLAS MLP | FP32 BLAS MLP |
|---|---:|---:|---:|---:|---:|---:|
|Letter|7.6918|8.1829|7.6867|21.7546|5.4109|4.0607|
|Pendigits|5.4240|2.9240|1.6142|2.1541|1.7403|1.3849|
|Satellite|7.9743|4.0695|2.3148|7.8239|5.4197|4.0653|
|OptDigits|9.4628|11.3994|11.0519|3.9088|5.6336|4.2618|

Satellite local is1.756x faster than FP32 MLP with27 more correct labels. Pendigits
local is16.6% slower than FP32 MLP but45 more correct; it is faster/smaller than the
current SVC at six fewer correct. Letter is1.89x slower than FP32 MLP for32 more
correct labels and below SVC quality. Optical local loses clearly to the accelerated
SVC in both speed and accuracy. Do not present the four tasks as universal domination.
Final run contains4,032,315 repeated prediction checks over11,295 underlying rows.

## Same-function layout ablation

Original execution vectorized input coordinates. The packet layout transposes
model geometry to evaluate eight independent prototypes; register accumulation
keeps6/10/26 output classes live while preserving each class's prototype order.
All twelve prototype full-score arrays match original/packet/register bit-for-bit.
This is not a learning gain or a new invention of SIMD. The actual old source and
binary remain the control, not a artificially slowed reconstructed baseline.

Final equal-model register/original cost ratio is0.68788045:31.21% lower cost, about
1.45x speed. All12 model medians improve; the original >=1.10x/no>1.10-regression
layout gate passes. That separate engineering gate cannot convert the FAILED
two-task learning gate into success. Scalar/direct-exp diagnostics and all packet
results remain. Direct exp happened to change no retained labels, but it computes
a different rounding function and is not guaranteed equivalent on other inputs.

## Storage, preparation and process memory

| Task | Local model bytes | SVC bytes | FP32 MLP bytes | Local prepared bytes | FP32 MLP prepared bytes |
|---|---:|---:|---:|---:|---:|
|Letter|278860|2595456|307564|352232|307760|
|Pendigits|37032|143272|80124|106360|80368|
|Satellite|49280|407208|307588|290552|307840|
|OptDigits|344232|548888|340604|627032|340848|

Counts include all prototype integer coordinates/local weights, head coefficients,
biases and metadata; prepared counts add tables and transposed banks. Model bytes
are not process RSS or total deployment size. The Satellite6.24x smaller file gives
only about5.6% smaller counted prepared native state. Optical's local model is NOT
smaller than the FP32 MLP even on disk, and its prepared state is substantially larger.

Seventy-two fresh processes cover all24 original deployments; twelve more cover
FP32 MLP. Each reads its own Linux VmHWM, creates the full tables/model, validates
one prediction and imports no numerical Python framework. Whole-process ranges:

| Task | Local peak KiB | FP32 MLP peak KiB | Local median setup ms | FP32 MLP median setup ms |
|---|---:|---:|---:|---:|
|Letter|17344-17348|18416-18424|4.968|6.883|
|Pendigits|16592-16596|17948-17984|2.572|4.751|
|Satellite|16800|18416-18580|2.980|6.499|
|OptDigits|17788-17792|18548-18676|6.447|7.044|

Library/interpreter mappings are part of process totals. Filesystem caches may be
warm. Three samples per deployment do not prove universal memory/initialization
advantages, and one-row memory is not a maximum batch-memory test. External BLAS
is not a hidden dependency of prototype inference; it is a stronger neural baseline.

## Training cost and validation

The60 pilots,284 selection fits and24 final fits total3891.872 recorded CPU seconds
(about64.86 CPU minutes). A separate training-only Nyström diagnostic used55.323
CPU seconds; imports, acquisition, compiler, tests, audits, measurements and packaging
are additional. Up to three independent training workers ran concurrently, each
with one numerical thread. No GPU was used. The prototype training uses bounded
N-by-P features, not an N-by-N SVM Gram matrix; this is not linear-memory complexity
for the entire comparison, which still includes ordinary SVC training.

Six of twelve final exact-feature head fits reached the150-iteration cap. Their
recorded lack of convergence remains; no test-driven extension or retry is used.
Fixed epoch training and approximate straight-through gradients do not establish
optimal prototype locations. Failed pilot variants and all model files remain.

All68 unique focused tests pass locally for original/packet/register, portable,
and AVX2 UBSan paths; repeated builds are not extra unique tests. The independent
scalar observer matches506,850 original prototype class scores bit-for-bit across
12 models. All67,770 model/input decisions across24 selected models preserve their
own reference outputs. Four converted MLP deployments add no new training examples.
The final standard-library auditor verifies284 selection cells,24 model locks,
three complete timing grids, independent FP32 byte conversion and the failed gate.
Sixteen deliberately corrupted disposable evidence copies are rejected by that
final auditor. It does not authenticate clocks, unrecorded research behavior,
statistical independence or a malicious replacement of the entire evidence set.

Dedicated GitHub CI tests the prototype contracts. Exact-final-head acceptance is
recorded separately; older core CI is not relabelled as final validation. No manual
full historical SPECTRA suite, Windows/ARM, ASan, free-threaded runtime, external
researcher reproduction or live application is claimed. Source changes are additive
on the actual748-file interaction branch. Other previously delivered unmerged
research versions remain separate; this source tree is not their silent replacement.

## Interpretation

The new architecture establishes a fixed inference budget and an actually trained
local classifier with a useful observed Satellite operating point. It does not
establish a broadly better learner, general reasoning, first invention or superiority
to all compact baselines. The fresh-to-this-continuation optical test is a negative
transfer result. Keep every model experimental and preserve earlier stronger SVMs.

Primary references:
- Gupta et al., ProtoNN, ICML2017: https://proceedings.mlr.press/v70/gupta17a.html
- Alpaydin and Kaynak, UCI OptDigits1998, DOI10.24432/C50P49:
  https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits
- Other official datasets: UCI Letter59, Pendigits81, Statlog Satellite146.
Dataset attribution/license is retained separately from SPECTRA's implementation
license. No upstream-maintainer or third-party publication was made by this work.
