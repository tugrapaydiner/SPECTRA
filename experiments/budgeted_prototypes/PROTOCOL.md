# Budgeted discriminative integer prototypes — September 29, 2026

Base a118f571f96d4c26e29eaba4839941e9ce14fd05. The previous full feature metric failed its three-task learning comparison; no result/default is replaced. Change the model family: learn a bounded set of prototype locations and optionally prototype-local diagonal weights directly from class labels, rather than keep thousands of SVM support vectors and change one global metric.

## Candidate and numerical definition

A fixed prototype budget P, class-balanced k-means initialization on fitting rows only, quarter-grid prototype coordinates, positive integer local feature weights with fixed per-prototype mass, and a dense multiclass output head. Cross-entropy trains labels only: no pretrained network, teacher predictions, distillation, or test-driven feature engineering. The primary candidate jointly learns centers and local weights with quantization-aware forward values. Fixed centers and moving-centers-only models are ablations. The exact quantized kernel features and output head are refitted using fitting labels before final export. Kernel arithmetic is explicitly defined by exact bounded integer signatures and the compact product of two exponential tables; it is not a bitwise replacement of an old libm/SVM model. All learned coefficients, prototype metadata, tables and preparation costs count.

This is related to existing RBF networks and ProtoNN (Gupta et al., ICML 2017, https://proceedings.mlr.press/v70/gupta17a.html), not a claim to have invented discriminative prototypes. Any contribution must come from an observed accuracy/resource frontier or a defensible quantized implementation, not a familiar idea renamed.

## Development pilot

First use only original TRAIN partitions of Letter, Pendigits, Satellite. These official tests were already consumed historically and remain closed for new-model selection. Group duplicate features; first grouped split seed611; cap fitting6000 and validation2000. Pilot P256, initial gamma2/8 (Satellite8/32), up to120 optimization epochs; all fixed/moving/local arms retained. Training algorithm revisions may follow this pilot only and must be documented. Before full selection freeze prototype budgets, optimizer/regularization, epoch selection, equally explicit MLP/SVM controls and the complete candidate matrix. No candidate replacement after test predictions. CPU only, one numerical thread per run, no GPU. Record all fitting time, memory and intermediate failures.

## Confirmation boundary

Acquire official UCI Optical Recognition of Handwritten Digits (id80), original optdigits.tra/optdigits.tes, without model execution. The source describes disjoint training/test contributors, but individual writer IDs are not supplied in the tabular files. No new test predictions until every model family, selection grid and selected final artifact is locked. Check prior repository mentions; if the partition is discovered to be previously consumed, disclose that and do not call it fresh. Acquisition and schema checks do not authorize fitting on test. Preserve all original rows and report exact duplicate overlap. No claim that historical benchmark data are a real deployment.

## Scientific decision

Primary joint-learning test: at least0.5 percentage points above the matched fixed-prototype control on two tasks, no more than1.25x its complete-call native latency. Separately report whether the new model improves ANY actual accuracy/latency/storage frontier against the strongest retained SVM and a properly selected small MLP; passing the internal ablation is insufficient to call the method broadly superior. No retrospective threshold changes, hidden model selection, statistical significance claims from repeatedly exposed tests, or invented speedup versus a missing competitor.

Use grouped training-only model selection and final full-training refits; save hashes before opening official tests once. Report all correct counts, macro-F1, confusion matrices, parameters/prototypes, model bytes, tables, memory, cold preparation and timed raw-code-to-fresh-label calls. Compare each backend to its OWN frozen classifier output; approximation is a new model, not a certificate of the old classifier. Include independent scalar decoding, deliberately wrong-input/model tests and code-generation/source receipts. New native target scope is Linux x86-64; do not inherit older Windows/ARM acceptance.
