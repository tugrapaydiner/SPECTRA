# Prototype-lane execution: model-frozen implementation comparison

All24 selected classifiers and their quality results are frozen. No new fitting,
model selection, label-driven pruning or evaluation-set accuracy change is allowed.
The native baseline currently vectorizes across input features and then reduces
each prototype's integer distance. Investigate two fixed model-only layouts:
(1) eight independent prototypes in SIMD lanes, with transposed center/weight
blocks; (2) the same distance organization with register-resident output accumulators
for6/10/26 classes, preserving every class's original prototype summation order.
Other class counts keep the generic implementation. Wide integer signatures or
portable builds keep checked scalar distances. All output values, not only labels,
must match the original numerical definition. SIMD is established technology,
not a novelty claim. Additional packed model storage is counted explicitly.

Preserve the original runtime source and actual original library, not a recomputed
slower stand-in. Compare original, packet and register builds for ALL three fitted
prototype families on ALL four tasks, plus scalar/direct-exp diagnostics, the same
native SVM, the scaled native MLP and linear control. Batches1/32/256, seven fixed
shuffled whole-dataset repetitions, one pinned core and no concurrent training.
No selective timing reruns. The proposed register layout is primary, packet is an
ablation. Report every model regression; require >=1.10x geometric improvement
across the12 prototype models and no model >1.10x latency before recommending it.
This systems gate is separate from the unchanged failed/successful learning gate.
Use the SAME recommended backend for local and fixed prototypes when reporting
the primary quality/cost comparison. Do not multiply unrelated speedup factors.
