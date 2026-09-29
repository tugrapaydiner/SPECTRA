# Compact tree certificates: native continuation protocol

Continue the uploaded exact-rational checkpoint (certificate_oracle.py SHA256
704556048f58844787a89c2079710abbce949cc7f5d427e821fcb25f64f5c6a5).
Its real CatBoost model files were not recovered from available archives, Library
or repository branches. Do NOT reuse its earlier 5150/5151 report as evidence.
This is a NEW explicitly identified execution study, not recovery of those weights.
Base repository commit8c2458a0f64208d5a645ba5098de8d0f00511043 is unchanged.

## New fixed source panel, before fitting

Use original training partitions of UCI Letter, Pendigits, Satellite and OptDigits
already held in previous delivery archives. All four test sets have been exposed
by past SPECTRA research; no new holdout or independent accuracy confirmation.
Train one numeric oblivious CatBoost1.2.8 multiclass model per task:256 iterations,
depth6, learning_rate0.08,l2_leaf_reg3,bootstrap_type=No,random_strength0,
random_seed20260929,thread_count1,MultiClass,allow_writing_files=False. Raw uint8
features, no feature selection or normalization. No evaluation-based early stopping,
parameter tuning, pretrained teacher or GPU. Hash all four CBM/JSON/CPP exports
before computing their test predictions. These settings define compatibility
workloads, not competitive model-family tuning or a new best-accuracy assertion.

## Compilation and numerical scope

Start from the exact-rational checkpoint. Compile both signed8- and16-bit class
contrasts with a power-of-two step and outward-rounded source-arithmetic and
quantization bounds. The deployable binary must bind its original JSON digest,
compiler policy and integer bounds. Numerical claim is source-model class-index
agreement under the explicit binary64 summation/positive-scale/bias contract,
not the true label, every CatBoost backend or signatures authenticating the model.
Rows contain exact bounded integer features; no snapping. Native result is either
a certified class index or UNRESOLVED. Never count a raw quantized guess as exact.

Required correctness: rerun all checkpoint tests; match native output/status/work
with the Python oracle on exhaustive small domains and real samples; check every
accepted real decision against native CatBoost and exported original source.
Retain forced wrong-quantization cases, ties, cancellation, byte corruption,
invalid data and ownership. No hidden confidence or test-derived thresholds.

Implement compact contiguous integer leaves, reusable numeric predicates and
optional exact safe early exit. Preserve a simple end-only baseline and original
FP64 evaluator. Both8/16bit, checkpoint0/16 and any additional model-only bound
refinement must remain visible. A correction to the certificate requires a proof
and new tests before promotion, not tolerance relaxation from model outcomes.

## Fair complete-call comparison and storage

Compare native full precision, original CatBoost C++ export and the official
CatBoost1.2.8 CPU model shared library when obtainable. Include required uint8-to-
float conversion, validation, model traversal and fresh result arrays. A prepared
Pool-only timing must not be compared with raw-input end-to-end calls. Include
all fallback work in hybrid results; retain abstention rate for compact-only use.
Original model sidecars, libraries, integer metadata and suffix bounds are storage,
not free. Report compact-only versus complete-fallback footprint separately.

All frozen models, batches1/32/256, seven shuffled whole-dataset repetitions, one
pinned CPU/thread; no concurrent fitting or compilation during timing. Preparation,
source verification and compilation are separate costs. Timing scripts and model
identities locked before the full matrix; no selective cell reruns. No universal
speedup or storage claim versus an absent baseline. No source-default/release
replacement, high-90s grade or first-invention claim. Tree quantization and exact
pruning have prior art; this study must earn its value through measured results.
