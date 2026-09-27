# Native-to-native result — September 27, 2026

**Native-speed admission: INCOMPLETE. Task-usefulness admission: FAIL.**
The larger SVM executes much faster than native LIBSVM, but a much cheaper linear
model has slightly higher observed accuracy. Generated C beats SPECTRA on five of
six retained small models. These results narrow the claim, not raise a hiring grade.

## Source and controls

Protocol0ca7b08 was committed before fitting or opening HAR test predictions.
Amendment66d0e79 preceded the second complete matched-ISA experiment. Implementation
5857afe adds ten experiment files; all were checked against their locally tested
blob identities. Runtime/native model execution in main remains unchanged.
The earlier local986613a stream-codec work remains a separate preserved candidate.

Upstream LIBSVM3.37 source commit6b907139084abf2da4d6d3cb10dc3b7eaffa2fbb is unmodified.
m2cgen0.10.0 generates unmodified C expressions. Model values, class ordering and
binary sign conventions are checked separately. The fixed linear control is
compiled from its own original coefficients with strict ordered arithmetic.

## Final matched-AVX2 comparison

AMD EPYC9V74, one pinned CPU core, Linux/CPython3.13.5. Seven whole-job repetitions,
all test rows, chunks1/32/256, fixed randomized order. The table is batch32 median
whole-job cost divided by rows, in microseconds. Validation/conversion, one symmetric
Python/native call per chunk and fresh label lists are included. Loading, compilation,
old raw-row preprocessing and original HAR sensor feature extraction are excluded.
This is not complete-file or sensor-application timing.

| Model | LIBSVM AVX2 | Generated C AVX2 | SPECTRA AVX2 default | SPECTRA AVX2 binary_stream |
|---|---:|---:|---:|---:|
| Wine |1.056|0.770|1.011|1.047|
| WDBC |2.571|2.172|1.574|1.242|
| Chess |38.010|7.640|10.021|8.552|
| Penguins |0.716|0.504|0.778|0.773|
| Titanic |6.359|3.270|5.403|4.021|
| Zoo |1.999|0.787|1.419|1.463|
| HAR |615.398|generation timeout|74.412|72.663|

HAR default SPECTRA is **8.270x** faster than native LIBSVM. Its same-build exhaustive
path is175.135us, so default is2.354x faster than that control. The portable default
is222.965us and is retained separately. No per-model best-profile promotion occurs.
The native linear classifier is **3.079us**, **24.167x faster than SPECTRA's default**.

The first complete run used portable external competitors; it remains in the
archive, not the final headline. The stronger full rerun adds -mavx2 to both external
competitors while preserving -O3 -fno-fast-math -ffp-contract=off and the same sources.
Both runs have1,176 timing cells and734,832 repeated predictions each, derived from
4,374 underlying model/input pairs. They are never pooled or selected per case.
All observations and batch sizes remain. Seven repetitions are not production-tail
or cross-host guarantees. No model loading/import-overhead advantage is timed.

## New held-out workload and quality

The six old controls are unchanged seed101 models (1,427 pairs). HAR uses its original
7,352 training/2,947 test rows,561 supplied features and disjoint21/9 training/test
subjects. SVC(C=10,gamma='scale') and LinearSVC(C=1,dual='auto',max_iter=10000,
random_state=20260927) were fixed before test evaluation. No GPU, pretrained model,
added scaling or post-test fitting. HAR has1,222 support vectors and a5,533,424-byte
SPECTRA model. The linear control has3,372 learned values (26,976 FP64 payload bytes).
These are not RSS measurements or compression of the same model.

| Fixed classifier | Correct /2,947 | Accuracy |
|---|---:|---:|
| RBF SVC |2,835|96.20%|
| LinearSVC |2,849|96.67%|

The linear point advantage is0.475 percentage points; a descriptive bootstrap over
nine test subjects gives95% interval[-0.486,+1.571] points. It includes zero: no
statistical accuracy-superiority claim. The point-based task gate fails because
linear is faster with higher observed accuracy. Per-subject outcomes are retained.
Fit-call CPU time is0.466s for SVC and1.207s for LinearSVC on this host, excluding
imports, acquisition, code generation, compilation, tests and evaluation.

## Missing comparator and validation boundaries

All six smaller generated models compile. HAR m2cgen reaches the predeclared120s
GENERATION limit (recorded120.036s); there is no generated C artifact and no HAR C
compiler attempt. This is missing, not defeated or inherently impossible. The
native admission cannot pass while that comparison is incomplete. Build limits
use4GiB Linux address space, not a claimed RSS result. Initial tool-orchestration
interruptions remain separate from the accepted script-controlled timeout.

All4,374 SVM labels agree with sklearn under native LIBSVM and all SPECTRA profiles;
all1,427 smaller generated-C labels also agree. The linear native control preserves
all2,947 linear predictions. Margin values need not be bitwise: the largest recorded
LIBSVM/sklearn absolute difference is1.219e-12. No changed class is excused by tolerance.

Twenty-five adapter tests pass on the initial and matched-AVX2 LIBSVM builds,
including six generated-C sign/vote tests. Independent standard-library audits
recompute exact-value model exports, complete grids, source/input/binary identities
and summaries for both runs; fourteen corrupted copies are rejected. The first44
and second47 measured-source snapshots remain separately bound. These are internal
checking implementations, not outside reproduction. The unchanged449-file historical
preservation audit passes. No fresh historical full-suite, Windows/ARM, sanitizer,
installed-wheel or production-usage result is claimed in this round.

## Decision and reproduction

Keep this HAR SVM as a same-function runtime case, not an efficient activity model
recommendation. Do not hide the linear control or retune this opened test split.
The next nonlinear-quality claim needs training/validation justification and fresh
subject/group-disjoint confirmation. No runtime/default/release is changed here.

[README.md](README.md) provides reconstruction/build/evaluation/audit commands.
The separate full report and evidence retain both experiments, source snapshots,
all frozen models and outputs, generated sources, build limits, initial failures,
quality uncertainty and test receipts. Pickles are trusted benchmark-only artifacts;
unknown pickles and supplied native libraries are not safe deployment inputs.

LIBSVM retains its COPYRIGHT. UCI HAR is CC BY4.0 (Anguita, Ghio, Oneto, Parra and
Reyes-Ortiz); original metadata/README and hashes are retained. MIT does not relicense
upstream data. WDBC is a software fixture, not clinical evidence.

Primary sources:
- https://www.csie.ntu.edu.tw/~cjlin/libsvm/
- https://github.com/BayesWitnesses/m2cgen
- https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
