# Native tree-certificate recovery and execution study

Continue from inspected f816bb8bc6a6dd9bbb1fe63604f81d7b7752e3ba and the uploaded
exact-rational checkpoint. Only the three protocols are present on the published
tree branch; its older empirical native workspace and fitted models were not
found among mounted files or targeted Library searches. This is a NEW recorded
reproduction, not recovery of the missing original bytes or previous timing cells.
Do not label the old reported 5150/5151 count as verified by this work.

Use the same publicly locked source recipe: CatBoost1.2.8, four original TRAIN
partitions (Letter, Pendigits, Satellite, OptDigits), 256 depth6 oblivious trees,
learning_rate .08, l2_leaf_reg3, bootstrap_type No, random_strength0,
random_seed20260929, thread_count1, MultiClass, allow_writing_files false. No
selection, early stopping or test-driven change. Save all four CBM/JSON/CPP models
and hashes before official-test predictions. Tests are already exposed historical
benchmarks; these fits create compatibility workloads, not new generalization.

Preserve the checkpoint oracle verbatim. Build a separate bounded packed native
format with exact uint8 predicates, signed8/16-bit contrast leaves, outward source
roundoff/quantization bounds, and optional safe early exit. Verify packed contents
against source by deterministic exact-rational reconstruction. A structural loader
alone is not a numerical proof; deployment requires verified, trusted model bytes.
Use compact-only CERTIFIED/UNRESOLVED outputs, never a confident unproved guess.

Implement shared-route refinement: evaluate8-bit, then16-bit for unresolved rows
without repeating tree routing, then invoke the unmodified official CatBoost C
API for any remaining unresolved rows. One native owning pipeline holds both
compact models and the original model. A simpler16-bit-first path is a control.
All fallback computation, original model storage and library dependencies count.
No quantized answer may be substituted for an unresolved original class.

Compare complete raw-uint8-to-fresh-index calls against official CPU C API1.2.8
and1.2.10 when obtainable, exported original C++, simple scalar/tiled compact
8/16-bit, safe checkpoints0/16 and the fused refinement paths. Separate
compact-only abstention from full coverage. All four source models, batches1/32/256,
seven shuffled whole-dataset repetitions on one pinned core, no concurrent fitting.
Freeze exact arm/source/model inventory before timing; preserve all cells and
failures. Do not claim novelty, increased intelligence, fresh accuracy confirmation,
universal CatBoost equivalence or a speedup against a missing native comparator.

Rerun checkpoint tests, exhaustive synthetic domains, unsafe8-bit examples,
8-to16 refinement, source-score fidelity, strict malformed model/input cases,
late invalid-output atomicity, caller-buffer ownership, close/reentry and sanitizer
checks. Report full accuracy only descriptively for the newly fitted sources.
Final deliverables must include actual frozen models, raw outcomes, the independent
oracle, source hashes and executable replay. No overwrite of historical records,
production default, main merge or release publication.
