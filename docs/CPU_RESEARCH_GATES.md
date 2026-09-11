# CPU-only research gates: from a reliable instrument to an important result

These are proposed future acceptance criteria, not completed milestones, guarantees of publication, or an official hiring/leveling rubric. Do not change existing M14-M17 results to make a new gate pass. Freeze the new protocol, code, model-selection allowance, CPU budget, seeds and baseline tuning budget before confirmation. A failed gate is an informative stopping/pivot decision for that hypothesis, not a reason to keep reopening its test data.

## Current bottleneck

The strongest positive capability result is the small FP dual-stream model with exact Sudoku validity termination, not the entire advertised ternary/learned-halting/MCTS stack. M14 confirms +15.1367 percentage points against an FP single-pass network at comparable median latency, but the candidate's mean and p95 latency ratios are 1.18383 and 2.51147. The exact symbolic comparator is faster and perfect. See [M14 acceptance](M14_ACCEPTANCE_GATE.md).

M16 improves a faithful native-checker implementation, while M17 does not establish the two-family claim. Its maze fixed pools contain a valid candidate in only 38 of 256 model-example pools. No ranking rule confined to those pools can solve more than 14.84375% of them. The decoder also makes every nonempty invalid path score 0.75, so that structural target does not grade progress between such failures. See [fixed-pool replay](FIXED_POOL_REPLAY.md). The [symmetry audit](SYMMETRY_AUDIT.md) now identifies a separate structural-transfer limitation in consumed Sudoku data.

The priority is therefore not another control head or another kernel chart. It is to choose and demonstrate one useful closed-loop effect that survives appropriate baselines.

## Research thesis to test

**A verifier-safe, budget-conditioned allocation rule spends scarce CPU time on continuations with genuine solve probability, rather than on proxy-score increases, and improves the complete-solve quality/latency frontier.**

Use immutable input/core identities, a semantically verified incumbent, and an explicit accounting of transition, copying, decode, checking and value-model time. Start with FP32. Introduce quantization only after the decision rule earns its cost. Semantic safety protects a valid answer; it cannot certify that the search policy is efficient or that an invalid candidate is useful.

For frozen policy pi and remaining budget b, the relevant value is

    V_pi(s,b) = Pr(a semantically valid answer is found within b | state s, policy pi).

For an action a with measured cost c(a), compare continuation values at b-c(a), including the option to continue the incumbent path, restart, switch to a classical solver, or stop. The original probability of an improvement in a structural proxy is not generally this value. Keep the policy, budget definition, hardware scope and terminal label in the checkpoint contract. This is a testable design direction, not a novelty claim by itself.

## Gate 1 — trustworthy and relevant problem families

Construct a new structurally disjoint Sudoku pilot using the input-orbit policy and full artifact-ancestry exclusions. Use it to validate the protocol, not as the sole flagship benchmark. Add a genuinely distinct family and a workload whose deployer benefits from the proposed resource tradeoff. Maze is a useful negative control but must not be repeatedly tuned against its consumed development results and relabeled as fresh evidence.

Require zero exact and declared-equivalence overlap across all consumed ancestors; independent semantic validators; separate development and once-opened confirmation; at least five model seeds for a final model-level claim; and uncertainty resampling over problem groups and model seeds, not timing repetitions. A leave-generator or leave-difficulty-family-out confirmation is needed for the corresponding transfer claim. Define each equivalence relation explicitly; the new Sudoku4 canonicalizer is not a generic deduplicator.

Baseline set: the strongest affordable tuned FP one-pass and recursive models, unlearned continuation, equal-budget random/restart allocation, semantic early exit, the training-only orbit lookup where applicable, and a task-appropriate classical solver. Charge data generation, preprocessing and lookup setup separately from serving, and report amortization assumptions. A learned method need not beat a classical solver on every research pilot, but it needs at least one clearly justified useful regime before making a deployment claim.

## Gate 2 — prove candidate headroom before training a better ranker

On frozen development inputs, retain each generated candidate and its incremental measured cost. Define C_i(B) as the candidates reachable by the chosen generator within a stated budget, and compute the diagnostic oracle envelope

    U(B) = mean_i max_{a in C_i(B)} semantic_validity(a).

U(B) upper-bounds selection accuracy for that fixed pool; it is not an attainable runtime policy and its oracle computation is not free. If U(B) minus the strongest ordinary baseline is below the minimum worthwhile effect, ranking cannot deliver that effect without changing candidate generation. The present maze pool cap is a concrete example.

Proposed advancement rule: on two development families, require at least 5 percentage points of oracle headroom at the intended complete-solve budget, with positive group-aware uncertainty bounds. Also demonstrate that proposed actions create new successful trajectories compared with equal-cost extra recurrence/restarts. Otherwise change the transition/action mechanism or select a more relevant task; do not add another value head to an exhausted pool.

## Gate 3 — a closed-loop effect, not just a selection statistic

Train budget-conditioned terminal/continuation labels from frozen, reference-free execution. Validate calibration and ranking on disjoint development states, but make actual complete-solve outcomes the primary endpoint. Include head-inference and defensive state-copying costs. Compare the same frozen generator with learned, uniform, random and oracle allocation; use ablations that isolate candidate coverage from selector quality.

Proposed final quality gate: at equal predeclared wall-clock and memory envelopes, gain at least 5 percentage points over the strongest applicable practical baseline, with a positive paired group/model-aware 95% interval, on two genuinely distinct families. Alternatively predeclare a speed gate: at matched quality within a 1-percentage-point noninferiority allowance, achieve at least 25% lower mean latency, with the latency-ratio interval upper bound below 0.85 and p95 no worse than 1.10 times baseline. Select one path before confirmation, not whichever passes afterward. These thresholds are design choices and can be revised only before a new study is frozen.

Report the whole frontier, cold start, warm B=1 median/mean/p95, deadline overruns, failed solves, model state, packed bytes and process RSS. A soft deadline checked between atomic transitions is not a hard real-time guarantee; measure its overshoot explicitly. Do not silently discard slow or failed instances.

## Gate 4 — the claimed deployment stack must carry the win

Only after the FP method passes, export and test the actual trained low-bit system. Compare identical inputs and checkpoints where applicable; report numerical disagreement and semantic quality loss, not only layer tolerances. Keep scalar/native differential tests, odd-size/overflow cases, malformed artifact handling and immutable shared weight handles.

Proposed deployment gate: no more than a predeclared 1-percentage-point quality loss and at least a 25% measured mean complete-solve latency reduction against the appropriate deployed FP reference, without a p95 regression, on two independent CPU hosts. Show a workload region where the memory/latency tradeoff is useful. Count all mixed-precision fallbacks and conversion traffic. A precomputed K-input GEMV benchmark is not sequential recurrent inference. Do not attribute the FP success to ternary weights or learned halting until those exact arms pass.

No GPU training, GPU inference or GPU profiler is required by this plan. CPU-only checkpoints and small controlled models are the default. Physical energy claims require real, valid package counters or a defined external meter; otherwise report null. Packed size is not evidence of L3 residency. Cache claims require direct controlled hardware measurements and an explicit working-set scope.

## Gate 5 — external reproducibility and importance

Release one command that obtains hash-bound data/checkpoints, runs the declared CPU environment, replays the primary outcome and regenerates tables from raw rows. Keep all negative attempts and exact source trees. Require another person on an independent host to reproduce the primary scientific conclusion, with numerical/hardware scope documented rather than hidden.

Then demonstrate use beyond this repository: for example, an independently maintained solver adopting the allocation rule, a downstream CPU workload with measured benefit, or external experiments exposing and confirming the mechanism. CI replication by the author is valuable but is not external adoption. This is where a sound small experiment becomes evidence of consequential research engineering.

## Novelty and career boundary

Tiny recursive models and low-bit CPU inference are established prior work, not sufficient novelty claims: see Jolicoeur-Martineau, [Less is More: Recursive Reasoning with Tiny Networks](https://arxiv.org/abs/2510.04871), and the official [BitNet](https://github.com/microsoft/BitNet) and [T-MAC](https://github.com/microsoft/T-MAC) repositories. A further targeted literature audit of the eventual successful decision rule is required before claiming first-of-kind work. This document does not assert exhaustive novelty.

OpenAI's public [RL/Reasoning role](https://openai.com/careers/research-engineerresearch-scientist-rlreasoning-san-francisco/) emphasizes ownership, controlled experiments and trustworthy conclusions. Its [Codex systems role](https://openai.com/careers/performance-and-systems-engineer-codex-san-francisco/) emphasizes whole-system, user-relevant performance improvements. Neither supplies an official repository-to-L7 score conversion. Passing these research gates would strengthen evidence of those abilities; it does not establish a hiring level, compensation, an interview exemption or guaranteed inbound offers.
