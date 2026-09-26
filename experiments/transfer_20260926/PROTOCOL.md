# Independent transfer admission — frozen before data acquisition

This experiment follows the locally delivered output-aware source da012d31b84c1eb40248aa8f60e39f3cb289e9d5, not an alteration to the published v0.7.1 release. This branch initially acquires public data/dependencies only. No new result is claimed by this protocol.

## Questions

Does the existing immutable-context mechanism transfer beyond the adaptively reused 360 optical-digit examples? Can several evolving query rows improve quality without adding trainable values? Does the same-function runtime advantage survive a working ONNX Runtime comparator?

## Data and one-time test boundaries

Use UCI Pendigits (id81), its provided training/test files, preserving the writer-independent test partition. Use UCI Letter Recognition (id59), first16000 rows for development and last4000 for final testing as documented by UCI. Both CC BY4.0 sources must be attributed. No optical-digits confirmation will be claimed. Within each training portion use a fixed stratified 80/20 fitting/validation split, random_state20260926. No final-test evaluation until all neural checkpoints, SVC selection, export paths and timing subset are frozen. No retraining or model selection from final-test results. Test rows may be downloaded/parsed to establish schema and data integrity, not used for fitting or selection.

## Models and budget

Float32 CPU, no pretrained weights, no GPU. Five seeds1401,2402,3403,4404,5405. 16 real-valued features reshaped into eight pairs; Pendigits scale by100 and Letter by15. Fixed32-wide, four-head, one-block, two-round SPECTRA models. Primary candidate: four query rows initialized by contiguous-group means of input embeddings, all sharing the same immutable key/value context and existing block weights; average query outputs before the linear head. No extra learned query parameters. Controls: prior one-query/first-token initialization, one-query/group-mean initialization, corrected full-state architecture with bounded gates and branch normalization, and a two-hidden-layer ReLU MLP(16,96,96,classes). The one-query/group-mean arm distinguishes seeding from query-count changes.

All neural arms use30 epochs, batch128, AdamW lr0.002 weight_decay0.0001. Save all validation scores, choose the highest validation accuracy checkpoint (earliest tie); no adaptive training budget. The SVC control selects C in{1,10} and gamma in{scale,0.1} on the same fitting/validation portions; no extra refit. Failures/collapses stay in counts. All models selected before test evaluation.

## Runtime

Same-model checks compare native AVX2 (where supported), ONNX Runtime CPU all graph optimizations single-thread/sequential, and eager reference; applicable XLA/Inductor controls remain explicitly separate if not remeasured. Use both pinned ORT1.23.2 and current available1.30.0 when obtainable; do not describe failed installation as a performance win. Numerical check is unchanged: abs(error)<=0.001+0.0002*abs(reference), same class decisions. Include I/O preparation scope, output copying, completed calls, compiler/session costs separately. No changes to old exact contracts.

Final end-to-end callable timing: first256 test rows of each new dataset, all five seeds, nine repeats randomized by a fixed seed, all actual predictions validated. Record both prepared-input and full API scope if provided. A fresh output is returned. Do not multiply architectural and same-model speedups. Batch-throughput may be a distinct secondary measure.

## Acceptance

Report all architectures and baselines, every seed, dataset-specific accuracy and paired differences. Descriptive two-way seed/input bootstrap intervals and fixed-seed input intervals must be distinguished; only two tasks cannot establish population generality. Useful-workload admission requires candidate accuracy no more than1 percentage point below the strongest tested task-level control and at least1.5x lower complete-call latency; require a one-sided95% noninferiority interval for a statistical fixed-margin claim. Missing baselines, seed failures, negative results, hardware regressions and missed gates remain explicit. No automatic90/100 or general-reasoning claim.

## Preservation

Keep historical source/evidence byte-identical; new models belong to a separate optional experiment. Acquisition receipt records public URLs, byte lengths and SHA256 hashes. All subsequent source changes and deviations must be logged, before observing the affected test results. Training, compilation and measurement costs are separate. Never replace prior release bytes or change main from this branch.
