# Chronological gas-sensor admission: frozen before acquisition or fitting

Base: merged main b9d9f90d28a3e7071080be38555cd6ca09afcccf. This is a research-only application-proxy experiment, not a gas alarm, commercial recommendation, new sensor system, or a grading milestone. PR36's compiler review remains separate. No runtime defaults or historical evidence change.

## Question and data

Does a nonlinear model earn its execution cost on measurements taken after its training/calibration period? Use official UCI Gas Sensor Array Drift dataset224, DOI10.24432/C5RP6W. Freeze chronological roles: batches1-5 fitting, batch6 validation/model selection, batches7-10 final evaluation. Never randomly mix time batches. Fit StandardScaler only on1-5. No concentration, batch ID, labels, future statistics, test-time adaptation or future-label updates enter inputs. Use all128 supplied features; acquisition and raw resistance-to-feature extraction are outside this experiment. The ten-batch order comes from the source documentation.

The current UCI page displays CC BY4.0 but its descriptive text retains a research-only/noncommercial restriction. Preserve both notices and attribution; this work is research only and does not resolve the conflicting permissions. Do not redistribute source data/model artifacts as a commercially licensed product. Download the official archive with size/SHA256 receipts; parse development batches first. Final batches may be checked for archive integrity without model predictions before freeze.

## Fixed CPU model panel

One numerical thread per fit; no GPU/pretrained parameters. Fit each arm on batches1-5 only; no refit on batch6 after selection. StandardScaler identical across every arm. Select by validation balanced accuracy (mean recall of all six classes), then validation accuracy, then first declared configuration. Preserve warnings, failures, every selection score and fit CPU/wall cost.

- LinearSVC: C in [0.01,0.1,1,10,100], dual='auto', max_iter10000, tol1e-5, random_state101.
- Exact RBF SVC: C in [1,10,100], gamma in [0.1/128,1/128,10/128], tol1e-4, cache_size256MiB, probability=False, break_ties=False.
- MLPClassifier: hidden layers(64,) or(128,64); seeds101/202/303; Adam learning_rate_init0.001, alpha0.0001, batch_size128, max_iter300, tol0, n_iter_no_change301, early_stopping=False. Select architecture by mean validation balanced accuracy across seeds, then mean accuracy; primary deployment seed101. All three selected-architecture seeds remain in final results.
- Nyström RBF features plus LinearSVC: components64 or256; gamma in [0.1/128,1/128,10/128]; C in[0.1,1,10]; mapping seeds101/202/303; linear settings as above. Reuse each fitted mapping only within its declared C grid. Select configuration by mean validation balanced accuracy across seeds; primary deployment seed101. This is an established approximate-kernel baseline, not a claimed new learning method.

All classes use default unweighted training. No fit is silently extended, model enlarged or alternative preprocessing added after evaluation. Failed fits stay recorded. Tiny controlled fixtures are separate from this model panel.

## Freeze and final evaluation

Before any final model calls, write a manifest of selected model/scaler bytes, validation predictions, configuration and source identities. Evaluate all selected families (including negative controls) on all7-10 rows with no subsequent tuning. Report accuracy, balanced accuracy, each batch and class, exact feature duplicates crossing partitions and all selected seeds. Primary aggregate weights batches equally; also report pooled rows. Descriptive intervals resample four time batches and must not be called broad population guarantees. No selective removal of difficult batches.

Research application-admission target, not a safety requirement: nonlinear deployment should reach at least90% mean batch-balanced accuracy and exceed the better linear/MLP control by3 percentage points. These are explicitly chosen research targets, not requirements elicited from an external adopter. The Nyström alternative must be included in the quality/cost comparison; a dominating cheaper approximate model prevents claiming the full SVM is preferable. Report both validation and final dispositions; failure is an informative completed result, not permission to retune the final data.

## Execution investigation

Compare the fixed full SVM through SPECTRA exhaustive/default and unmodified native LIBSVM using the same normalized values. Attempt original m2cgen generation within120s and compilation within180s, at most4GiB process address space. A failed comparator remains missing, not defeated. Use matched strict AVX2 flags where hardware supports them; portable scope separately recorded.

Also implement minimal native linear/MLP and Nyström baselines so framework overhead cannot determine the task-level winner. Candidate Nyström execution may fold its final linear transformation into kernel-output coefficients (normalization.T @ linear.coef_.T); this is ordinary algebra, not a new approximation method. Floating-point order changes require numerical checks and exact observed labels; do not claim bitwise equivalence for folding. Keep an unfused mapped-feature control. Emit original parameters and a source-bound compact native artifact with no benchmark fitting inside timers.

Primary latency: batch1 and32 over a fixed first-up-to512 rows from each final batch, nine shuffled whole-job repetitions. All predictions checked after every job. Time input conversion/preprocessing separately and explicitly; report prepared-input native costs, complete supported feature-to-label costs, export/build/loading cost, artifact bytes and break-even arithmetic where useful. No production-tail or general hardware claim. If new runtime paths fail fidelity, retain failures and do not quote them as accepted speedups.

## Delivery

Keep code, all model choices, raw timings, negative outcomes, tests and reproduction instructions. Commit to a focused branch from actual main; no force push, release publication or third-party upstream submission. No new grade follows automatically from completion.
