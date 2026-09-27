# Larger-model activity-classification execution — prospective protocol

Base main f98167b44610158d107164aee295656015a0ad89, tree32312589e398114cedbae75110b2c211b003bf6b. This is a CPU systems/application experiment, not a new neural reasoning claim. Freeze this before acquiring or making predictions on the new task.

## Workload and attribution

UCI Human Activity Recognition Using Smartphones, dataset240, DOI10.24432/C54S4K, CC BY4.0, Reyes-Ortiz, Anguita, Ghio, Oneto and Parra. Use the supplied561-dimensional feature vectors and original subject-separated train/test partitions. Preserve row order and every test example. Subject identities are used only to check partition separation and stratify reporting, never as classifier features. The supplied signal processing/feature extraction is NOT included in inference timings; this is classification from extracted features, not raw-sensor end-to-end timing or clinical/safety validation.

https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones

## Fixed models and development boundary

Train two fixed RBF SVCs on the supplied training portion: C1 and C10, gamma='scale', no probability output, no break_ties, cache_size128MiB, binary64 supplied features with no additional scaling. C10 is the predeclared primary; C1 is a robustness control, not chosen by test scores. Include LinearSVC(C1,dual='auto',max_iter20000,random_state0) as a task-level accuracy/cost control. No GPU, external learned weights, tuning grid or test-selected checkpoints. Record training CPU/wall costs and all warnings.

Only training examples may inform performance diagnosis or implementation selection. Any implementation candidates and failed pilots are retained. Lock source, model hashes, feature precision, evaluation schedule and binary builds before opening test predictions. Test labels never select a runtime, architecture or hyperparameter.

## Proposed systems investigation

Profile current native vs scikit-learn LIBSVM before implementing. Investigate support-vector memory organization or measured application overhead only when the training probe shows headroom. Any layout optimization must preserve each input's binary64 feature and coefficient reduction order, system exp and exact vote rule. No approximate kernels, new learned values or uncharged duplicate matrix. Preserve the old path as an explicit baseline. Account for preparation, extra memory and break-even requests. If the proposal fails, retain the failure rather than expanding tuning from the test set.

## Formal scopes after locking

All test rows get fidelity and per-subject quality results. Timed prepared-input probes use fixed256 evenly spaced test-row indices, batch1 and32, seven randomized matched rounds. Return fresh original labels and check every output. Compare strongest existing native execution schedules and the new path with the same arithmetic, plus scikit-learn's actual SVC.predict on matched binary64 inputs. Separately time complete JSONL file-to-file application work, including parsing, output serialization/publication and process startup where stated. Do not assign warm-kernel ratios to this scope. Include linear control accuracy/model footprint/latency rather than imply an RBF system wins every tradeoff.

A material same-native improvement requires primary C10 batch32 ratio<=0.90, with complete fidelity and reported regressions on C1 and batch1. This is an engineering point gate, not evidence of statistically universal superiority. If a compact linear model is preferable at relevant quality, state that result. One new task and two models are not broad production adoption.

Measure fresh-process model preparation and whole-process peak memory separately; keep transient preparation cost visible. Numerical comparators use unchanged tolerances/contracts. Repeated times are not additional independent data; uncertainty conditional on frozen inputs/host must be labelled. Independent result audits should bind source/model/data bytes and original labels. No final release/version overwrite or unperformed cross-platform claim.
