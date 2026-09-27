# Optional compiled preprocessing — scope frozen before measurement

Base: PR30 119200fe649e03d335577aa3f64efb6bc66d2b7a, tree
03fae0afd7981322a76d794e29ae5073b3154a99. Date: 2026-09-27.

Keep all eighteen frozen pipelines and consumed raw inputs from the previous
six-task study. No training, selection, scheduler change or new-accuracy claim.
Evaluate an optional CPython extension for the SAME impute/subtract/divide and
one-hot operations. Input structure validation is shared with the reference;
non-built-in scalar objects may fall back to that exact reference implementation.
Keep the GIL while traversing Python objects. This is not a parallel preprocessing
claim, new learning algorithm, abi3 binary or zero-copy raw-input guarantee.

Before formal timing: bitwise transformed-feature fidelity on every retained row,
class fidelity, errors/caps, subclass fallback, fresh-output lifetime, model binding,
and installed-wheel compilation without numerical frameworks must pass. Keep the
same input/output scopes in controls: canonical Python rows to fresh label lists.
The specialized block-vectorized NumPy control uses the same SVM library; fitted
sklearn provides a secondary context. Imports/loading/compilation/preprocessing
plan preparation excluded from warm time. All raw row validation and conversion
inside it. No reciprocal substitution, fast-math or altered SVM arithmetic.

Formal run: all 18 models, 31 randomized repetitions seeded 20260927, first retained
row and first min(128,N) rows. All calls checked. Report per-model distributions and
per-task geometric-mean ratios; no timing-row dropping or retraining. Primary
promotion gate: >=1.20x lower complete-batch cost than the existing Python pipeline
on each task, no changed features or predictions. Specialized NumPy comparisons
are separately reported, not a requirement silently adjusted after measurements.
Binary tasks and slower rows remain. No production-P95 or cross-machine claim.
