# M16 refinement — matched targets, frozen pools and native integration

Frozen before any new M16 data generation, auxiliary fitting, development score or confirmation result. The original M16 protocol remains authoritative. No acceptance threshold is relaxed.

## Source and ancestry

Use exactly the two accepted M14 dimension-64 FP checkpoints (seeds 1401/2402), and the three retained M15 action directions plus identity. Verify the published M14/M15 archive hashes and checkpoint tensor hashes. Reconstruct each old dataset from its recorded generator recipe, verify every reconstructed example against the stored pair fingerprint and ID, then add input-only hashes. Reconstruction failure is fatal; it does not authorize a new unlabeled substitute. Include all five M14/M15 manifest surfaces and all splits in the forbidden-input index. Original artifacts remain unchanged.

The data seeds, sizes, clue ranges, exclusion rules and confirmation order are those in M16_PROTOCOL.md. Exact separation is not symmetry/OOD separation. All generated rows retain their actual input and target, content hashes, rejection counts and generation recipe.

## Fixed-pool target experiment

Use FP32 recurrent states and these twelve fixed action paths, in this order:

```
0; 00; 000; 0000; 1; 2; 3; 10; 20; 30; 1000; 2000
```

Action 0 is identity; 1–3 add the exact M15 direction at its recorded scale. Cache shared path prefixes, retaining actual transition counts. The first four states are exactly the ordinary unquantized baseline trajectory. Reuse is for causal candidate-selection analysis, not an online speedup claim.

All newly fitted targets use the SAME decoder: argmax followed by immutable-given restoration. This deliberately differs from the legacy M15 verifier's raw-decode improvement labels; the legacy ensemble is a contextual arm, not the matched-target causal comparator.

Fit four separate one-member GroundedStateVerifier backbones per core, with matched initial backbone parameters, the same sampled state-index schedule and the same 300-step AdamW recipe (batch 64, lr 0.002, weight decay 0.01, gradient clip 1). No architecture or learning-rate sweep. Targets:

1. One identity-cycle structural improvement, binary cross entropy.
2. Current exact validity, binary cross entropy.
3. Current absolute structural score, soft-target binary cross entropy.
4. First exact-valid decode under frozen identity continuation: categorical time 0, 1, 2, 3, or not solved within 3; cross entropy. Cumulative probabilities describe success within the requested horizon under THAT continuation policy, not optimal search or an arbitrary future policy.

The categorical head shares the same backbone but has five outputs; disclose its extra parameters. All heads are trained only on fit-set fixed-pool states. Predictions, label prevalence, Brier/calibration diagnostics and per-puzzle pool selection are retained. No temperature/calibration parameter is fitted on confirmation. Stable first-in-pool ties apply to every selector.

Current-validity probability is a final-selection target; continuation probability is an exploration heuristic. Neither is an exact certificate. Exact terminal checks dominate learned scores whenever used.

## Closed-loop controls

Retain the original semantic exit K4 and fixed depth 4. Run original M15 learned-prior and equal-direction uniform-prior MCTS (12 evaluations, maximum depth 4), and original task-specific terminal guard. Compare new target values without changing the MCTS code where possible; distinguish these controls from the new budgeted best-first controller.

The preselected new controller first executes the K4 semantic-exit baseline. If it succeeds it returns immediately. Otherwise it conducts checked best-first exploration, guided by the first-solve distribution under identity continuation. The complete run is capped at 20 transitions, 24 decodes/checks, 20 value calls, 20 policy calls and maximum tree depth 4. All baseline work counts against these caps. A valid incumbent cannot be replaced by an invalid proposal. Do not call this MCTS or treat its budget as identical to twelve MCTS evaluations.

No wall deadline is imposed in the primary quality experiment; actual wall/process CPU time and work are retained. Cooperative deadline mechanics are separately tested. Report paired quality differences and seed-specific results. Bootstrap puzzles jointly across the two fixed cores; do not pretend two cores establish a training-seed population confidence interval.

## Native CPU comparisons

The new native adapter leaves the M10 attention, exact GELU, RMSNorm, residuals, quantization boundaries and recurrence unchanged. Its immutable handle validates weights at load time. The AVX2 path vectorizes output channels while preserving hidden-index FP32 accumulation order, with FMA contraction disabled. It retains an extra transposed int8 layout: report those bytes and do not claim packed-only execution or cache residency.

Compare checked scalar, validated scalar and validated AVX2 on identical inputs; compare complete M10 deployed solves and optimized dense-FP linear execution of the same graph. The M10 checkpoint is a fidelity workload, not a positive task-capability claim. Compare original M14 semantic exit versus the native-checker adapter on exact same checkpoints and examples; also give the single-pass baseline the same native checker. Include exact symbolic/finite-solution filtering context for 4x4. Measure full decode/check/control costs, randomized paired order, mean/median/p95 and all raw rounds. A win against our slow scalar reference is not a win against the strongest external method.

## Freeze and outcome

Before confirmation generation, hash the implementation, recipe, trained heads, ancestral inventory and fit/validation/development manifests into an exclusive-create freeze record. Confirmation refuses altered code/checkpoints or existing outputs. Implementation changes after opening confirmation require a disclosed new attempt and a new confirmation seed, never silent reuse.

Record success AND failure. The previously defined 20% mean-cost / matched-quality gate is unchanged. Native fidelity, target alignment, harder-task generalization and research importance remain separate gates. No guaranteed hiring, L7 certification or automatic 100/100 claim is made.
