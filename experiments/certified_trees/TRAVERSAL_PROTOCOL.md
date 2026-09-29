# Same-function traversal refinement after the first complete native comparison

The four source models, exact rational compiler, integer leaves and certificate
bounds are frozen. The complete first1092-cell matrix is retained. It shows the
initial generic row-major certificate runtime is SLOWER than the actual official
CatBoost CPU evaluator. Its source, binaries and observations must not be erased
or replaced by a deliberately weakened reconstruction.

Investigate tree-major tiles of32 input rows and geometry-only specialization of
common output widths and depth6. A tile retains per-row integer accumulators and
precomputed predicates, then reuses each tree's leaf data across its rows before
moving to the next tree. This changes memory access and loop organization, not
quantization, confidence, trained weights, error bounds or evaluated tree order
within a row. End-only certification remains identical; the existing checkpoint16
path is retained as a separate ablation. Tail rows and arbitrary supported class
counts/depths must remain correct. All added prepared/scratch bytes are counted.

Implement reference equivalence before timing: both precisions, complete scores
and unresolved outcomes, synthetic exhaustive inputs, scalar/AVX2, corrupt models,
source-library fallback and original compact-sidecar reuse. No compiler tolerance,
certificate margin or model setting may be relaxed in response to measurements.
This is established cache/loop optimization, not a claimed new theory.

Rerun the ENTIRE previous13-arm matrix with the same four frozen models, batch
sizes1/32/256 and seven repetitions, plus the retained original q16 end-only and
original q16-plus-official-CatBoost fallback controls. New fixed seed2026092918.
All official and exported-C++ controls remain. No cross-run timing cells are pooled.
The final result must name any remaining native-CatBoost regressions and distinguish
compact-only abstention from full-coverage fallback cost. No new accuracy gain,
production default, release, outside reproduction or first-invention claim.
