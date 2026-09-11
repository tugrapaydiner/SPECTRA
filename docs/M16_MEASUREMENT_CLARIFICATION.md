# M16 measurement clarification

Recorded before validation/development evaluation and before confirmation generation. Auxiliary target fitting has completed; no candidate or threshold is selected using holdout results.

Quality is evaluated on every puzzle. Primary timing uses the first min(128, N) generated puzzles per surface, without inspecting difficulty, outcomes or timing values. Those puzzles receive three randomized interleaved rounds; remaining puzzles receive one quality round whose timing remains a secondary observation. Four separate validation puzzles warm each solver. Report the sampling scope and do not describe three-round timing as covering every confirmation puzzle.

The new controller does not evaluate a learned proxy while executing its baseline trajectory. It checks each baseline decode, retaining valid answers; proxy evaluations begin only if baseline execution exhausts without success. This preserves the preregistered baseline-first algorithm and charges all executed work. Embedding/solver construction is included in the outer complete-solve timer; the internal work ledger covers transitions, policy, value, decode and check callbacks and explicitly excludes the preceding embedding.

The first-hit head is queried at min(3, max_depth - node_depth) remaining identity cycles. Each query has an explicit target contract. Horizon zero denotes current validity; positive horizons name the identity-continuation policy and their actual horizon. A single fixed-horizon score is not silently relabeled as current correctness or optimal search value.

The first-hit MCTS control uses horizon-specific continuation values for exploration/backups and time-zero validity probability for final-answer selection. Both are read from the same categorical prediction, without a second network call. It does not select an unfinished state merely because that state has high future-solve probability. A future-horizon score used for immediate answer selection is retained only as an explicitly mismatched fixed-pool diagnostic.

FP32 fixed-pool computation uses batched execution. It shares the baseline transition graph but does not assume floating-point bit identity across batch sizes. Compare decoded answers/validity to B=1 execution and disclose numerical differences. This does not change the input identities, target definitions, pool paths, fitting procedure, closed-loop budgets or acceptance thresholds.
