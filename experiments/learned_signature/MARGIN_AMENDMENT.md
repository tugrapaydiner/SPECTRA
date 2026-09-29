# Margin-trained mixture: prospective secondary arm

The first complete three-split experiment failed the primary validation gate.
Metric/alignment and nested pair experts are retained as negative controls. No
final evaluation model call has been made. Original final fits may finish while
this new training-only experiment runs; they will not be refitted from results.

Add one secondary family whose five nonnegative radial mixture coefficients are
learned by minimizing the sum of binary SVM dual optima across class pairs,
under a simplex constraint. Start from the uniform mixture. Use at most 2048
stratified fitting examples, seed+71; never outer validation examples. Eight
projected-gradient iterations, initial normalized step0.5, at most four halving
attempts per iteration, accepting only nonincreased training objective (relative
numerical slack1e-8). No claim of exact optimizer convergence. All internal fits,
objective trajectories and time count toward its budget.

For fixed dual variables, the gradient for component m is
-0.5*sum_pair(alpha_y.T @ K_m @ alpha_y). The implementation must check this
fixed-dual gradient against finite differences and retain ordinary SVC controls.
This is established multiple-kernel learning, not a claimed new learning theory.
The same21 C/global-gamma grid, three outer fitting/validation splits, input
values and5000-example outer-fitting cap apply. One final setting is chosen from
mean outer validation and refit once on all training data. The learned components
are folded into one authoritative distance table before inference.

The primary gate stays failed; this arm is secondary and development-adaptive.
Its purpose is to test a stronger task-related objective, not to select against
final test results. Do not evaluate either final test partition until ALL models,
including this arm, are fixed and hashed. Keep all failed line searches, warnings,
fits and settings. Original test partitions remain historically consumed and are
not fresh confirmation even with this current-turn boundary.

Reference: Rakotomamonjy et al., SimpleMKL, JMLR2008.
https://www.jmlr.org/papers/v9/rakotomamonjy08a.html
