# Learned finite kernels — prospective task-quality experiment

September 28, 2026. Base: merged SPECTRA b9d9f90d28a3e7071080be38555cd6ca09afcccf. The finite-domain executor in local e9e3274 motivates this separate learning experiment. No production defaults or existing evidence change.

## Question

Can a trained integer distance metric and/or a trained multiscale radial table improve classification, while retaining packed integer distances and one kernel lookup per support vector? This changes the learned function. It is not another fidelity-only speedup, evidence of general intelligence, or a claim to invent metric learning/multiple kernels.

## Fresh data and boundaries

Acquire official UCI Semeion (178, 1593 binary 256-feature handwritten digits) and Car Evaluation (19, 1728 categorical examples). Attribute their CC BY4.0 provenance. Car labels originate in a constructed decision model, not measured vehicle safety. Semeion lacks usable writer IDs in its supplied rows; do not claim writer-independent generalization.

Group identical feature rows before splitting; conflicting group labels abort rather than silently remove data. Stratified 50% held-out groups, random_state=20260928. Save exact indices and all input hashes. Holdout labels may be parsed for stratification but no model predictions or performance analysis on that holdout occur until all selected models, runtime implementation and training records are frozen. Within development, stratified group-disjoint 75/25 fit/validation splits with seeds101,202,303. These use the same outer holdout and overlap; they are not three independent test datasets. No test-based tuning, reselection or refit.

Semeion uses its original binary pixels and sixteen fixed4x4 spatial groups. Car uses one-hot categories from its documented schema, six original-attribute groups. No learned vocabulary from test observations. Store the input encoder and report conversion cost separately from prepared-feature inference.

## Models and fixed optimization budgets

Primary: integer-metric RBF SVM. Control: tuned isotropic RBF SVM. Each receives32 identical gamma/C candidate slots: gamma multipliers[0.125,0.25,0.5,1,2,4,8,16] divided by its fitting-set median positive distance; C[0.1,1,10,100]. Highest validation accuracy, then fewer supports, then earlier candidate wins. No refit on combined fit+validation.

Learn group distance weights using a diagonal, leave-one-out neighbor objective (NCA-style):120 Adam steps, learning rate0.05, full fitting-set comparisons, exclude self, inverse temperature8/median original distance, normalized positive group weights,0.01 mean-squared log-weight regularization. Project once after the fixed final step: clip(round(2*w),0,7), at least one nonzero weight. This projection defines the actual trained metric; no approximation is substituted silently at evaluation. Record all losses, continuous/projected weights and fitting CPU time. Single numerical thread, no GPU/pretrained/teacher API. Metric optimization capped at60 CPU seconds per split; over-budget is a recorded failure, not silently resumed selection.

Secondary factorial controls: uniform and nonnegative centered-alignment-learned mixtures of the same eight radial scales, on both isotropic and learned integer metrics. Mix weights use fitting data only, positive normalization; regularize their8x8 alignment system with fixed1e-8 diagonal stabilization. Each mixture fits four C values only. Report every family separately; do not replace a failed primary by the best test-scoring secondary.

Task-level controls: LinearSVC C[0.1,1,10,100]; small MLP hidden layers[(32,),(64,),(32,32),(64,64)] crossed with alpha[0.0001,0.01], Adam max_iter300, random_state=inner seed, no test-based early stopping; DecisionTree max_depth[4,8,16,None]; ExtraTrees200 trees,n_jobs1,default depth. Select on the same validation data. Report warnings, actual time and parameter/support storage, not fictitious identical cost. No training-time claim from the number of candidate slots alone.

## Implementation and mathematical contract

Compile T[S]=sum_l beta_l*exp(-gamma_l*S) once; weights beta_l are nonnegative. Nonnegative integer feature weights produce S=sum_j w_j*(x_j-z_j)^2. Binary coordinates permit XOR/popcount with up to3 weight bitplanes. Test membership exactly; reject unsupported inputs rather than snapping. The serialized table bytes define the deployed new model, not a claim that a new classifier reproduces the old RBF predictions. Preserve coefficient order and deterministic first-class tie handling. Independent integer-distance/ordered-score replay must match the new reference. Keep baseline and candidate on the SAME native executor for fair accuracy/latency/storage comparison.

## Acceptance and reporting

Freeze all six kernel families and external controls for both datasets and all three splits before opening either outer holdout. Primary success requires>=2 percentage-point mean accuracy gain over the tuned isotropic control on at least one task, with a paired two-way seed/input95% interval excluding zero, and native batch32 latency and model bytes each<=1.25x baseline. These thresholds are prospective local decisions, not population guarantees. Also report whether a stronger MLP/tree/linear control dominates the candidate. Unmet thresholds, additional training cost and regressions remain visible.

Native timing: all test rows, batch sizes1/32/256, nine shuffled whole-job repeats, one pinned core. Validate every prediction. Charge conversion, native call, fresh labels; report encoder separately. Cold loading/model bytes separate; no multiplication by previous speedups. Test compile/runtime correctness before opening holdout. No broad certification, Windows/ARM, outside reproduction or world-first claim without evidence.

Prior art: Goldberger et al., Neighbourhood Components Analysis, NIPS2004; Cortes, Mohri, Rostamizadeh, Algorithms for Learning Kernels Based on Centered Alignment, JMLR2012. The intended contribution is measured quality/cost of deployment-compatible learning, not those established ideas themselves.
