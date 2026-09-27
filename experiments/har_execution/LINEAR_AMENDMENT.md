# Post-test deployment follow-up, with the original comparison retained

The initial runtime packing hypothesis failed on training-only probes and was not
promoted. The runtime, model hashes and original evaluation were then frozen before
opening HAR test predictions. That completed evaluation found the predeclared
LinearSVC control more accurate and cheaper than either RBF model. The original
results are retained without refitting or relabelling the packing gate as passed.

This secondary implementation is explicitly **after observing those test results**:
export the already fixed LinearSVC into a model-specific compiled binary64 linear
evaluator. No new learned parameters, fitting, change of test rows, sample selection,
label remapping or hyperparameter change is permitted. This is standard linear
inference/code generation, not an original learning algorithm or a decision proof.
The compiled evaluator uses ordered feature summation; agreement with arbitrary
BLAS implementations is a measured obligation, not a universal bitwise claim.

Before its first held-out timing, bind the final module and generated source,
parameter/library hashes and exact test files in LINEAR_LOCK.json. Compare the same
256 deterministic test rows, batch sizes1/32 and seven randomized whole-job rounds.
Every call returns a fresh label list. Include sklearn LinearSVC.predict and a
manually specialized NumPy matrix multiplication/argmax control, so the comparison
does not depend only on general-estimator validation overhead. Independently
recompute all ordered scores using scalar Python and preserved parameter bytes.
Do not discard slower cases or tune/rebuild from the resulting timing cells.

Separately measure complete bounded JSONL file processing for primary RBF/native,
RBF/sklearn, compiled-linear and linear/sklearn using the SAME strict parser,
chunk limit, record writer and no-overwrite file publication. Charge cold subprocess
startup/import/model loading in parent wall time and report inner processing and
peak self VmHWM separately. Three fixed rounds, all2947 original test rows; no
parallel benchmark work. Kernel/feature-extraction times are not assigned to the
whole application. HAR input features are already signal-processed by the dataset.
These are internal replay applications, not an outside user's deployed pipeline.
