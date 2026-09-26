# Baseline-strength amendment — before final-test model calls

The original protocol at 1b1eb7c711263021dfb9f59413aaa347b8090839 is unchanged: keep all50 originally planned neural fits and both four-cell SVC selections. No final-test model prediction has yet been made.

Validation results and measured training cost show that a30-epoch MLP is a potentially weak task-level comparator on Letter. Add a SECONDARY stronger MLP control with the exact same16-96-96-C architecture, initialization seed and optimizer:480 epochs, batch128, AdamW lr0.002 weight_decay0.0001, no learning-rate tuning, highest validation accuracy checkpoint with earliest tie. Five original seeds on both tasks. This is16times the original number of MLP epochs, roughly the observed CPU cost of30 epochs of the query models; actual CPU/wall totals must be reported rather than calling costs exactly equal.

These10 additional fits are controls, not new SPECTRA candidates or retrospective replacements. Keep the original30-epoch MLP outcomes. All10 stronger MLP checkpoints must be frozen before either final-test model evaluation, even if they hurt the architectural story. No refit, architecture selection or tuning on test results.

Provide the stronger MLP with the same optimized native and ONNX execution paths as the original MLP. Include it in accuracy, same-function runtime and measured quality/cost-frontier comparisons. Do not claim a useful advantage when an at-least-as-accurate, faster small MLP exists. The original point gate remains reported separately from this stronger baseline comparison.

Training may run concurrently on separate pinned cores, one thread per fit; no training is allowed during formal timing. Concurrency does not reduce total reported CPU training work. The code, costs, all validation curves, final/selected checkpoints and source identities must be retained. This amendment is a pre-test addition of a stronger control, not proof of success or a new benchmark result.