# Larger-model HAR result — September 27, 2026

## Decision

Do not promote either RBF dependency-packing prototype. The primary C10 model's
training-only probe became slower, with unchanged predictions and work counts.
The original native runtime transfers to the larger workload, but the fixed
LinearSVC control has a better observed quality/cost combination. The optional
`spectra.linear` adapter deploys that already fitted control without a numerical
framework. It is ordinary linear inference, not a new learning algorithm.

Protocol commit9e8c168 precedes official data acquisition and fitting. Test subjects
are disjoint from training subjects. The linear deployment extension is explicitly
post-test engineering; there is no test-driven refit or new model selection.

## Fixed model quality

UCI HAR dataset240, 7352 training examples/21 subjects, 2947 test examples/9 subjects,
561 supplied features, six classes. Subject identities are not features. Input
signal filtering/windowing/feature extraction is outside all inference timers.

| Frozen model | Test correct | Accuracy | Deployment parameter/model bytes |
|---|---:|---:|---:|
| RBF SVC C1 |2801/2947|95.0458%|10088592-byte SRT model|
| Primary RBF SVC C10 |2835/2947|96.1995%|5533424-byte SRT model|
| LinearSVC C1 |2849/2947|96.6746%|26976 raw binary64 parameter bytes|

The linear model contains3372 coefficients/biases. Its generated C++ is75948 bytes
and the measured Linux library40200 bytes; metadata/Python/runtime are additional.
All three original fits took3.892 CPU seconds; imports, compilation and testing are
separate. No GPU, pretrained weights, search grid or extra scaling was used.

The linear model gets41 cases right where primary RBF fails, and27 wrong where RBF
is right: a14-case /0.4751-point difference. A descriptive9-subject cluster bootstrap
interval is[-0.4773,+1.5717] percentage points. The point estimate does NOT prove
population accuracy superiority. Correlated windows are not treated as independent
people. This is not a state-of-the-art HAR accuracy claim or a production deployment.

## Runtime measurements on one Intel Xeon Platinum8573C core

Primary frozen evaluation:256 evenly spaced test inputs, batches1/32, seven matched
randomized rounds,126 timing cells. Every full-call output is checked; loading and
feature extraction are excluded. At batch32 the existing C10 `beretta_cert` path
uses101.735 microseconds/row versus378.616 for sklearn. Existing `lazy` is slightly
faster at99.192. These are old-runtime transfer results, not a new packing speedup.

| Same fixed linear model | Batch1 us/row | Batch32 us/row |
|---|---:|---:|
| New compiled ordered evaluator |13.714|1.592|
| sklearn LinearSVC |83.487|3.008|
| Specialized NumPy validation/matmul/argmax |13.629|0.839|

**NumPy wins warm batches** and is effectively tied at batch1. The native adapter
must be justified by its deployment boundary, not a claim of superior matrix math.
These secondary timings use42 cells and the same fixed test subset. All2947 native
labels match the original LinearSVC;17682 scores match a separately decoded ordered
scalar interpreter bit-for-bit. BLAS score differences reach2.71e-14, so arbitrary
boundary/cross-BLAS equivalence is not claimed. No SVM vote certificate is supplied.

## Complete feature-file application

All five backends share the same strict JSONL reader,128-row packing, record writer,
flush/fsync and create-if-absent publication. Each process reads all2947 original
feature rows and produces checked complete output. Three fresh processes per arm.
Parent time includes interpreter startup/import/loading; processing is also shown.

| Backend | Median full process seconds | Processing seconds | Peak self VmHWM KiB |
|---|---:|---:|---:|
| Native fixed linear |0.903|0.812|16188–16360|
| Specialized NumPy-only linear |1.165|0.973|26972|
| sklearn linear |2.436|0.807|124696–125092|
| Native primary RBF |1.473|1.252|33116–33180|
| sklearn primary RBF |4.473|2.417|134672–134692|

NumPy reads inert coefficients directly; it does not import sklearn through a
pickle. All accepted children use isolated startup without the local NumPy-preloading
startup hook. Framework arms explicitly add their library directory. Startup is not
subtracted from the full-process column. Parsing dominates processing, and the
sklearn linear processing-only figure is slightly lower than native. Three runs
are descriptive, not a production latency distribution or strong statistical claim.
One final parent receipt was lost to a tool timeout; the completed unreceipted
output is retained and only that missing timing was rerun. Other14 runs are unchanged.

## Validation and remaining limits

Local full suite:2199 passed, one Windows-only skip,16 historical slow exclusions,
two prior warnings; includes41 new linear tests. Actual installed sdist-built wheel:
53 existing SVM checks plus16 separate linear checks,147 source-member matches.
Independent record auditor checks168 timing cells and15 full outputs (44205 repeated
predictions). Ten corrupted copies are rejected; those probes are not in the2199.
Original449-file history audit passes. No native SVM numerical source is changed.

The new model-specific compiler interface is covered by native Windows/ARM CI using
fixed synthetic/compatibility models. This does not imply the full561-feature HAR
model, its latency or every test score was replayed on those hosts. Linux HAR
reproduction and cross-platform API acceptance remain different receipts. No new
security certification, cross-platform bitwise-libm claim or outside-user adoption.

Raw data, all models, predictions, timings, rejected patches, initial failures and
identities are supplied in the evidence package. See README.md for the layout and
DEVIATIONS.md for chronology. Failed packing gates remain failed; no high-90s grade
is assigned and no existing release is replaced.

Dataset: Reyes-Ortiz, Anguita, Ghio, Oneto and Parra, UCI Human Activity Recognition
Using Smartphones, DOI10.24432/C54S4K, CC BY4.0. The original documentation and source
hashes are retained. Source: https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
