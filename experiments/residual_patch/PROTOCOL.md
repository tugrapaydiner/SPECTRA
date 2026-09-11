# Residual-patch overhaul: development protocol

Base: `e73dc05379b38ab2689e0e3054f6eee74e0290c0`, tree `50601f3b4feb9784a2389f95d54ab7276eaa95d7`.

This branch preserves every existing file and default. New code lives under `experiments/residual_patch/`; it may replace the grid-transformer/search assumptions inside this experiment, not in the historical implementation. The experiment is a development pilot, not independent confirmation, a novelty certificate, or a hiring claim.

## Hypothesis

A compact native-executed model conditioned on exact current constraint residuals can choose coordinated variable-flip repairs that improve complete SAT witness-finding performance after accounting for feature extraction, inference, initialization, repair and checking. The learned component must beat nonlearned controls over the same candidate generator; otherwise the learned contribution is not established.

## Initial experiment design

- Deterministic non-tautological 3-CNF generation. Uniform and community-structured distributions are separate strata of ONE task. Planted and unplanted inputs, if used, are explicitly distinguished. No solver-based filtering or favorable-instance replacement.
- Training, validation and development seeds have disjoint declared ranges. No fresh confirmation is opened in this pilot. Known generator-level transformations and order-normalized duplicate identities are checked; this does not claim complete SAT isomorphism exclusion.
- Native stochastic local search with independently checked complete Boolean witnesses. A negative budget-limited search outcome means unknown, never certified UNSAT.
- A localized pool of single-variable and simultaneous two-variable flips exposes exact make/break and interaction features. Atomic patch semantics are tested against direct full-formula evaluation, including duplicate literals, tautologies, empty clauses and unused variables.
- Train a small candidate scorer on training-only counterfactual continuations under a fixed native continuation policy and horizon. Keep all training seeds and labels; retain collection, training and evaluation costs separately. The target must describe its actual budgeted event rather than claiming general solvability.
- Include native single-flip stochastic-search, random-patch, deterministic exact-residual patch and input-only/coarse-feedback ablations. All search arms use the same native substrate. External CDCL configurations are additional context when the native dependency is available; absence is disclosed, never replaced by an invented number.
- Separate fixed-work diagnostics from actual paired wall-clock evaluations. Complete warm solve timing includes formula conversion/construction, initialization, native search/features/model calls and final independent checking. Native library build and model load are separately reported cold-start costs. Deadline overruns are recorded and do not receive within-budget credit.
- Freeze numeric configuration and model hashes before reading the development evaluation. Use multiple training/search seeds, retain every planned arm/input/round/budget observation, and group analysis by independent formula and model seed rather than treating timing rounds as new samples.

## Promotion and preservation

A substantial capability target is at least +10 percentage points of within-budget witness success versus the strongest preselected control, or 2x lower complete cost at a predeclared matched-quality target. A small pilot cannot certify general impact even if its point estimate passes. Any later change to the source, candidate inventory, data distribution, budget or target is a separately recorded adaptive follow-up, not a rewrite of this attempt.

Retain failed experiments and their source. Correctness tests passing is separate from the scientific gate. Do not merge experimental model defaults into main based only on infrastructure success. Keep the legacy implementation usable and unchanged.

Related work already includes probSAT, NLocalSAT, learned local-search sampling oracles and neural large-neighborhood repair. Correlated/state-conditioned repair alone is not a defensible broad novelty claim. The experiment must establish a precise useful distinction and measured benefit.
