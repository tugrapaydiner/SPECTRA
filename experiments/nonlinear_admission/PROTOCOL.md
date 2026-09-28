# Nonlinear usefulness before runtime promotion

Frozen before acquiring or fitting the new task data. Base: main b9d9f90d28a3e7071080be38555cd6ca09afcccf. This is an additive experiment, not a new classifier architecture or production release.

## Two predetermined tasks

ISOLET (UCI54): supplied acoustic features, original isolet1+2+3+4 development file and speaker-separated isolet5 final test. Use a fixed stratified 80/20 development split (random_state20260928); this internal validation may share speakers. Do not infer individual speaker IDs from row numbers. Final test remains the official separate speaker group.

Gas Sensor Array Drift (UCI224): train on batches1..6, validate on7..8, final test on9..10. No random mixing across chronological batches. Report each final batch and the pooled result; two future batches do not establish broad temporal generality. These supplied features exclude original waveform/sensor feature extraction. Research-only provenance wording and the current UCI licence metadata must both be preserved; no commercial safety-use claim.

Both tasks and all outcomes stay in the report, including when linear/MLP/tree controls win. No further task is substituted because of results. Downloading/parsing schema is permitted; no final-test prediction or test-driven fitting before the final model freeze.

## Models, fixed selection, and compute

All models receive a StandardScaler fitted only on their fitting data. LinearSVC C in[0.01,0.1,1,10], dual=auto,max_iter=10000,random_state101. RBF SVC C in[1,10],gamma multiplier in[0.25,1,4]/feature_count, probability=False,break_ties=False,cache_size128MiB. HistGradientBoosting max_leaf_nodes in[15,31],max_iter200,learning_rate0.1,l2_regularization0.1,early_stopping=False,random_state101. MLPClassifier one ReLU hidden layer width64 or256,three seeds101/202/303,Adam,batch128,lr0.001,alpha0.0001,max_iter300,early_stopping=True,validation_fraction0.1,n_iter_no_change20. Record every warning, iteration count, fit CPU/wall time and failure. Each fit has a120s process wall cap and4GiB address-space cap; one numerical CPU thread, no GPU/pretrained weights.

Select each deterministic family's highest validation accuracy, first declared configuration breaks ties. Select MLP width by mean validation accuracy across all three seeds; do not select a seed. Final refit uses all ISOLET development data or gas batches1..8, with scaler refitted there, using frozen choices. Freeze all selected models and three MLP seeds before opening either final test. No post-test refit, tuning, model replacement or threshold change.

Point admission for an SVM-led application claim: at least1.0 percentage point validation advantage over BOTH the best linear control and mean selected-width MLP; confirm the same point margin on final test, also compare the tree control. Report macro-F1 and every MLP seed. This is an explicit research screen, not an externally mandated user requirement. Statistical improvement claims need suitable uncertainty; never call row bootstrap independent-speaker or independent-batch inference. A failed gate remains failed.

## Execution and decision record

Whatever the quality gate, execute the selected SVM with SPECTRA default/exhaustive and unmodified native LIBSVM on matched FP64 standardized inputs. Use identical strict compiler flags and same batch sizes1/32/256; all final inputs, seven randomized complete-job repetitions,one pinned core. Validate every timed label; internal margin differences do not excuse a changed class. Charge checking, interface calls and fresh output labels. Separately report StandardScaler-inclusive cost and preparation; no multiplication of speedups. Exported model and executable bytes are recorded, not called process RSS.

Provide optimized native linear and selected-width MLP controls where feasible, with their original parameters and explicit arithmetic/fidelity checks. A framework baseline is not relabelled as the best native baseline. Keep m2cgen generated-C comparison with120s generation/180s compile caps; failures remain missing comparators, not defeated methods. Model-specific generated code and fixed arrays may cost more preparation. Compare total preparation+N*request cost only when both measurements exist.

## Delivery and stopping rule

Complete both task evaluations, at least same-SVM native execution and correctness checks, and retain the full result even if no SVM application is admitted. New code/tests, model/data/source identities, selection/freeze chronology, raw measurements and all failures must be delivered. Keep existing runtime, earlier research and PR36 unchanged. No invented production deployment, upstream acceptance, high-90s grade, model-improvement claim or benchmark universality.
