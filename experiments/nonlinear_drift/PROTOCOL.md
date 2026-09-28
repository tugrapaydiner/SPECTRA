# Time-separated nonlinear model admission and compact landmark execution

Frozen before acquiring/evaluating the new data. Purpose: determine whether the application needs nonlinear computation and whether a compact representation can retain that quality, not tune a SVM until it looks preferable. Existing HAR/handwriting results and compiler-index pilot failures remain. CPU only, no pretrained weights, no teacher, no third-party submission or commercial/safety deployment claim.

## Dataset and temporal boundary

Official UCI Gas Sensor Array Drift, id224, 128 supplied features, six classes, ten ordered acquisition batches. Train ONLY batches1-4; validation ONLY5-6; final-test ONLY7-10. Do not randomly mix chronological batches. Original features and labels remain; no sensor/feature selection from final data. The official page contains both legacy research-only language and a CC BY4.0 badge: preserve attribution and use conservatively as a research fixture, not evidence of permission for commercial deployment. This is not a gas-alarm or health-safety validation.

One StandardScaler fitted on batches1-4, reused unchanged everywhere. Select by pooled validation balanced accuracy, then validation accuracy, then predefined candidate order. All model/transform files and the selection manifest must be frozen before the first final-test prediction. Do not refit using validation, select seeds from final outcomes, adjust preprocessing from test data, or search for a more favorable partition. Report all chronological test-batch outcomes and all planned model-family controls even when unfavorable.

## Fixed low-compute controls

LinearSVC: C in[0.01,0.1,1,10,100], dual='auto', max_iter20000.
RBF SVC: C in[1,10,100], gamma in[0.1,1,10]/128, cache_size256MB.
MLP: hidden(64,64), ReLU/Adam, batch128, lr0.001, max_iter300, tol0, n_iter_no_change301, alpha in[0.0001,0.01], seeds101/202/303. No early stopping on final data.
ExtraTrees:100 trees, max_features in['sqrt',1.0], seeds101/202/303, n_jobs1; no depth or leaf tuning.
Nystrom+LinearSVC: components in[64,256], gamma in[0.1,1,10]/128, C in[1,10], seed20260928, same linear solver. These are established methods used as strong alternatives, not new learning-theory claims.

The compact candidate uses a shared nonlinear landmark bank. Offline collapse the learned Nystrom normalization and linear classifier into one landmark-to-class matrix; evaluate each Gaussian feature once and accumulate all output classes. This removes the dense M-by-M normalization from deployment. It preserves the real-valued function algebraically, NOT bitwise FP arithmetic: preserve every observed class decision, measure full score errors, and keep the unfused original reference. No kernel approximation is silently substituted for the original SVC; this is a separately trained model family.

## Quality and execution gates

Report final pooled accuracy, balanced accuracy, every acquisition batch, model storage and recorded training time. A nonlinear-need claim requires the selected nonlinear family's pooled accuracy to exceed the selected linear control by at least3 percentage points and be better on at least3 of4 chronological test batches. Group/bootstrap uncertainty has only4 test blocks; no broad statistical population claim.

Compact-candidate admission requires final accuracy within1 percentage point of the selected full SVC and at least2x lower matched native complete-call latency; report whether other families are better too. Include the independent linear and small MLP/forest controls rather than calling a SVM-only win task superiority. Parameter choice uses validation only. A failed gate remains failed.

Same-function execution controls: old SPECTRA SVC, unmodified native LIBSVM, generated C when bounded generation/build succeeds, and compact candidate's unfused mathematical reference. Bounded generation/build failures are missing, not defeated. Match feature inputs, fresh labels and one call per batch. End-to-end preprocessing and model preparation costs must be separately named. Do not multiply speedups from older hosts/experiments.

Use all final-test rows, batch sizes1/32/256, seven shuffled full-job repetitions where feasible, one pinned core with no concurrent numerical work. Retain all timing cells and failures. Source/model hashes bind the run. Numerical fidelity probes and timed execution remain separate from accuracy selection. No post-test retraining or accuracy tuning. Prior datasets remain consumed development data.
