# Budgeted learned prototypes — completed result, September 29, 2026

## Decision

The original TWO-TASK LEARNING GATE FAILS. Letter improves by1.4 percentage points
against the selected fixed-prototype control at roughly tied cost, but Pendigits
improves only0.4002, Satellite loses0.05, and the newly opened optical test improves
only0.1113. The threshold remains at least0.5 points on two tasks with at most1.25x
native latency. A separate same-function execution-layout gate passes; it cannot
convert the learning failure into success. No model or runtime becomes the default.

A narrower deployment result is useful: Satellite's256 local prototypes obtain
1817/2000 (90.85%) versus1790/2000 (89.5%) for the selected MLP, at2.315 versus4.065
microseconds per row against the stronger ordinary FP32 OpenBLAS implementation.
The model files are49,280 versus307,588 bytes. Prepared native state is290,552
versus307,840 bytes: the6.24x FILE-size ratio is NOT a6.24x RAM reduction.
Other tasks retain unfavorable comparisons, including clear optical transfer failure.

## Architecture and limits

This is a newly trained fixed-budget classifier, not faster execution of an old
SVM. Fitting-only class-balanced k-means initializes a fixed number of prototypes.
Labels then train a multiclass head and, in different arms, prototype locations
and prototype-local diagonal metrics. Centers use quarter-grid codes; positive
integer feature weights have mass4*d per prototype. Training uses a float32
straight-through quantized-geometry proxy, not exact table derivatives. The final
integer geometry is frozen before a float64 head refit on exact product-table
features. There is no teacher, pretrained network, distillation or ensemble.

The SPPRO001 executor computes bounded integer distances and a product of two
exponential-table values. That product defines this model's numerical function;
it is NOT bitwise equality to a single direct exp or the prior SVM. The multiclass
head visits prototypes in original order. See CONTRACT.md for integer/float bounds,
validation and owning-resource leases. ProtoNN and RBF networks are established
prior art. No direct ProtoNN implementation, INT8 MLP or every compact architecture
is measured; no first-invention or general-intelligence claim is made.

## Selection and data exposure

Sixty training-only pilot fits retain unsuccessful affine heads, variable widths,
normalization, finer quantization and averaging. Initial complete selection has248
fits over fixed/centers/local prototypes, SVC and properly scaled MLP across four
tasks and two duplicate-grouped training splits. Prototype choices are256/512/1024
centers and two gamma values with common training settings. SVC has a nine-choice
C/gamma grid; the MLP selects128x128/256x256 and alpha1e-5/1e-3 after fit-only
StandardScaler, with early stopping inside the fitting role. Linear C10 is fixed.

The optical fixed control reached the lowest gamma, exposing a width-range
handicap. Before opening any official test, an explicit amendment added36 cells:
gamma0.125/0.5 for every optical prototype family, budget and split. The complete
248+36=284 cells remain. The corrected fixed control achieved99.08% validation,
slightly above local98.95%, so the initial apparent four-point gain was not used.

Each selected family is refitted once on all original training rows:24 trained
classifiers, not28. Four later FP32 files mechanically cast the same selected MLP
and scaler without training. Search opportunities and actual CPU cost differ across
model families. Final capacity is separately selected: comparisons are not all
same-parameter-budget comparisons. Letter's three prototype families do share the
same final1024 centers/gamma8; Pendigits and Satellite local models use256.

All24 final artifacts are locked before their official predictions and before the
OptDigits test is parsed. Letter, Pendigits and Satellite were already exposed by
earlier SPECTRA research. OptDigits is new to this continuation, not new in the
literature; its source describes30 training contributors and13 different test
contributors, but no per-row writer IDs are available. No raw-image or sensor
feature-extraction cost, live deployment or broad writer/geographic independence
is established. Letter's380 exact train/test feature overlaps and the nonoverlap
stratum are retained. No model was retuned after these test predictions.

## All official correct counts

| Classifier | Letter /4000 | Pendigits /3498 | Satellite /2000 | OptDigits /1797 |
|---|---:|---:|---:|---:|
| Fixed prototypes |3832|3415|1818|1749|
| Moving centers |3858|3434|1820|1756|
| Local centers and metrics |3888|3429|1817|1751|
| Selected SVC |3911|3435|1814|1763|
| Selected scaled MLP |3856|3384|1790|1737|
| Fixed linear |2787|3145|1632|1701|

The FP32 conversion preserves every selected MLP prediction on11,295 rows, though
scores are not bitwise float64. Letter local versus fixed fixes85 errors and creates
29:168 errors become112,33.3% fewer. Only Letter passes the task-specific learning
condition. Local versus moving centers is worse on the other three tasks. Paired
statistics are descriptive, not multiple-testing-adjusted or writer-cluster claims.

Earlier best exposed models remain stronger in accuracy: Letter3928/3929 versus
new local3888, Pendigits3444 versus3429, Satellite1836 versus1817. The three-example
advantage over the current Satellite SVC1814 is not established accuracy superiority,
and that SVC is not the strongest historical model. No unseen competitor is defeated.

## Complete native comparison, strengthened twice

Warm jobs start with original uint8 feature codes and end with fresh labels.
Validation, normalization/scaling, kernels/layers, packing and allocation are timed.
Fitting, loading, compilation, table preparation and original feature extraction
are excluded. One pinned AMD EPYC9V74 core, CPython3.13.5 and GCC14.2; the prototype
uses strict noncontracted arithmetic, and each OpenBLAS control uses one thread.
Costs derive from whole-dataset jobs, not individual-service latency distributions.

Run1 has1176 cells with ordinary native MLP and SVC. Run2 has1344 cells, adding
batch OpenBLAS float64 MLP and the prior finite/adaptive engine for the SAME SVC.
Run3 has1428 cells, adding ordinary FP32 MLP/SGEMM. Every previous arm remains in
each complete rerun. Models and candidate arithmetic do not change. No favorable
cells are pooled, selectively rerun or substituted. Final claims use run3 only.

FP32 conversion is locked before its output checks and matches a separate float32
forward calculation's labels. BLAS reduction order may differ from ordered loops.
The old finite/adaptive SVC keeps its earlier conditional numerical assumptions.
No candidate borrows another model's expected labels. All model-specific timed
outputs match. External OpenBLAS binary/config/hash and all source snapshots remain.

### Final batch32 cost: microseconds per row

| Task | Fixed | Centers | Local | Same SVC, finite | FP64 BLAS MLP | FP32 BLAS MLP |
|---|---:|---:|---:|---:|---:|---:|
|Letter|7.6918|8.1829|7.6867|21.7546|5.4109|4.0607|
|Pendigits|5.4240|2.9240|1.6142|2.1541|1.7403|1.3849|
|Satellite|7.9743|4.0695|2.3148|7.8239|5.4197|4.0653|
|OptDigits|9.4628|11.3994|11.0519|3.9088|5.6336|4.2618|

Satellite local is1.756x faster than FP32 MLP with27 more correct predictions.
Pendigits local is16.6% slower than FP32 MLP but45 more correct; it is faster than
current SVC at six fewer correct. Letter is1.89x slower than FP32 MLP for32 more
correct, and less accurate than SVC. Optical local loses to accelerated SVC in both
speed and accuracy. Final timing checks4,032,315 repeated predictions over11,295
underlying rows; repetitions do not enlarge the independent evaluation set.

## Exact-order layout ablation

The original implementation vectorizes coordinates. Packet layout transposes
model geometry to process eight independent prototypes, and register accumulation
keeps6/10/26 class scores live while preserving original prototype order. Other
classes or wide signatures retain generic/scalar paths. Every full score across
12 prototype models matches original/packet/register bit-for-bit. SIMD itself is
not new research. The actual original source and binary remain the baseline.

Final register/original equal-model cost ratio is0.68788045:31.21% lower cost or
about1.45x speed. All12 model medians improve. The separate >=1.10x/no>1.10-regression
layout gate passes. Scalar, direct-exp and packet diagnostics remain. Direct exp
changed no retained label but is a different rounding function, not guaranteed
identical on other inputs. Additional packed model storage is explicitly counted.

## Actual file and prepared-state inventories

The byte counts below come from the frozen deployed files and runtime receipts,
not from a size estimate or labels-excluded coefficient calculation.

| Task | Local file bytes | SVC file bytes | FP32 MLP file bytes | Local prepared bytes | FP32 MLP prepared bytes |
|---|---:|---:|---:|---:|---:|
|Letter|278872|2594680|307564|352232|307760|
|Pendigits|37032|143272|80124|106360|80368|
|Satellite|49280|407256|307588|290552|307840|
|OptDigits|344232|548848|340604|627032|340848|

The earlier draft's Letter/SVC file-byte entries were transcribed incorrectly and
are corrected here; frozen files, model hashes, measurements and claims do not
change. Prepared counts add tables and duplicate/transposed banks; they are not
process RSS. Optical local is larger than FP32 MLP even on disk. Satellite's6.24x
file advantage corresponds to only about5.6% less counted prepared native state.

Seventy-two fresh processes cover24 original deployments; twelve more cover FP32
MLP. Each builds the full prepared model/table, validates one row and reads its own
Linux VmHWM without importing numerical Python frameworks. Filesystem caches may
be warm. Three samples are not worst-case batch/service or cold-filesystem evidence.

| Task | Local peak KiB | FP32 MLP peak KiB | Local setup median ms | FP32 setup median ms |
|---|---:|---:|---:|---:|
|Letter|17344-17348|18416-18424|4.968|6.883|
|Pendigits|16592-16596|17948-17984|2.572|4.751|
|Satellite|16800|18416-18580|2.980|6.499|
|OptDigits|17788-17792|18548-18676|6.447|7.044|

## Cost, validation and preservation

The60 pilots,284 selection fits and24 final fits total3891.872 recorded CPU seconds
(64.86 CPU minutes); a separate training-only Nyström diagnostic used55.323 seconds.
Imports, acquisition, compilation, tests, audits, timing and packaging are additional.
Up to three independent workers ran concurrently, one numerical thread each. No GPU.
Prototype fitting uses N-by-P features; the comparison's ordinary SVC still has its
own training costs. Six of twelve final exact-feature heads hit the150-iteration
limit; no convergence or globally optimal quantized geometry is claimed.

All68 unique focused tests pass locally for original/packet/register, portable and
AVX2 UBSan builds. The independent scalar observer matches506,850 prototype class
scores across12 models. All67,770 decisions over24 selected models match their own
references. These counts overlap across builds and observations, not independent
samples. No historical full suite, Windows/ARM, ASan, free-threaded runtime, external
researcher reproduction, production use or general reasoning capability is claimed.

The final standard-library auditor verifies all284 selection cells,24 artifact
locks, three timing matrices, official optical rows, labels/confusions, mechanical
FP32 bytes and failed gate. Sixteen corrupted disposable copies are rejected.
Hashes do not authenticate clocks, researcher behavior, independence or a hostile
wholesale replacement. Exact-final-head CI and extracted-kit checks are recorded
separately, not inferred from earlier core success.

All changes are additive on the actual748-file interaction branch. Other historical
or unmerged research deliveries remain separate. No default, release, model or
prior negative result is replaced. The useful conclusion is a bounded learned
architecture with a specific Satellite tradeoff, not a broadly superior learner.

Primary references:
- Gupta et al., ProtoNN, ICML2017: https://proceedings.mlr.press/v70/gupta17a.html
- Alpaydin and Kaynak, OptDigits1998, DOI10.24432/C50P49:
  https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits
- UCI Letter59, Pendigits81, Statlog Satellite146; dataset attribution and CC BY4.0
  terms are separate from implementation licensing. No upstream submission occurred.
