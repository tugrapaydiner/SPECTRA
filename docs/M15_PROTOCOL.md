# Milestone 15 Protocol — Mechanism-Focused Ablations

**Status: preregistered before any M15 auxiliary training, M15 development result, or M15 confirmation result.**

M15 is deliberately bounded. It does not reopen the M14 acceptance criterion and it does not attempt a new scaling-law claim. The goal is to explain one important failure mechanism around latent search and to remove/qualify unsupported theory language.

## 1. Starting point

M14 established a narrow positive result for a trained dim-64 FP dual-stream recursive TRM plus reference-free Sudoku semantic-validity termination on generated unique 4×4 Sudoku. The accepted M14 system does **not** use the M12 learned halter, M09 learned actions, M07 grounded verifier, latent VQ, or native MCTS.

M07 established a grounded learned verifier whose target is:

`P(one deterministic frozen-reasoner cycle improves reference-free Sudoku structural score | x,y,z)`.

M07 also recorded that its shallow ranking was useful but its deeper-state ranking collapsed near random. M09 established a real learned action-policy path but no practical search benefit.

M15 asks why learned verifier-guided latent search can regress even when the reasoner itself is strong.

## 2. Primary mechanism question

### Preferred explanation H1 — target/value mismatch

The grounded verifier predicts **one-cycle improvability**, but MCTS consumes the probability as if it were an **absolute state value**. A high-quality or already-valid state can have low probability of further improvement because it has little room left to improve. Search that maximizes the proxy can therefore prefer a worse-but-improvable state over a better/terminal state.

This is a concrete Goodhart-style proxy mismatch, not a claim that the search intentionally hacks the verifier.

### Competing explanation H2 — deep distribution shift / verifier misranking

Search-generated states move outside the verifier's shallow training distribution. The verifier may lose ranking quality with depth even if its target were appropriate. M07's depths-4..7 failure makes this independently plausible.

### Competing explanation H3 — INT8 trajectory distortion

Repeated latent quantization/requantization may move the search trajectory enough that verifier ranking or final answers degrade. If FP32-state search avoids the failures while INT8-state search exhibits them, precision is a primary cause.

### Competing explanation H4 — action/policy quality

The learned or selected action directions may simply generate poor states. If an independent symbolic absolute-quality evaluator using the *same action set and search work* regresses similarly, verifier target/misranking is not the principal bottleneck.

## 3. Disconfirmation rule

H1 is **disconfirmed as the primary explanation** if the retained development evidence shows any of the following:

1. absolute symbolic quality and verifier score remain positively rank-aligned across the deeper search-state regime, with no solved-state inversion; or
2. an oracle absolute-quality evaluator using the same action set/work regresses by a similar amount to learned-verifier search; or
3. the preregistered terminal-validity guard does not materially reduce search-induced regressions; or
4. FP32-state search removes most regressions while INT8 search does not, making H3 the stronger explanation.

No threshold will be lowered after seeing M15 development results.

## 4. Prior-work / novelty boundary frozen before results

M15 does **not** claim novelty for any of these broad ideas:

- learned verifiers/reranking: Cobbe et al., *Training Verifiers to Solve Math Word Problems* (2021), arXiv:2110.14168;
- process-level learned verification: Lightman et al., *Let's Verify Step by Step* (2023), arXiv:2305.20050;
- proxy-reward overoptimization / Goodhart effects under stronger optimization: Gao, Schulman & Hilton, *Scaling Laws for Reward Model Overoptimization* (ICML 2023);
- ensemble/conservative uncertainty objectives mitigating reward-model overoptimization: Coste et al., *Reward Model Ensembles Help Mitigate Overoptimization* (ICLR 2024), arXiv:2310.02743;
- generic quantization-induced accuracy/calibration changes: established PTQ/QAT literature, and recent task-specific discrimination/calibration studies.

Therefore the maximum M15 contribution is a **system-specific controlled mechanism diagnosis**: interaction between recursive search depth, the declared one-cycle-improvement verifier target, search-state precision, action guidance, and task-specific terminal verification in SPECTRA. If the evidence is not clean, M15 will report an inconclusive mechanism rather than invent novelty.

## 5. Frozen core checkpoints

M15 does not retrain the primary recursive core for the mechanism experiment. It imports exact M14 checkpoint fixtures and verifies both file and tensor-state identity before use.

Primary accepted-family checkpoints from final M14 evidence run `34280670349` / artifact `10077794916`:

- `fp_recursive_dim64_seed1401.pt`
  - file SHA-256 `d5d4769726e3e45822e84a947dd4e8cb007a35374a8a40ae61a786c4c2ba55a4`
  - tensor-state SHA-256 `4cf4c93ec9d3bd688850394685924cb23d0762a8716eb3e2f42e42befd249f08`
- `fp_recursive_dim64_seed2402.pt`
  - file SHA-256 `b43f111af13bc8f7b667c9b7e97558b2b9743f522945193a7a625db77ea194ff`
  - tensor-state SHA-256 `00a84312a4fba02e31b1954bddffe10b5933e01eaaae4226eb068490a3f97c0d`

The detailed mechanism experiment uses both seeds. No favorable seed filtering is permitted.

A separate precision-context pair from the original M14 controlled experiment is retained for one matched-architecture FP-vs-ternary diagnostic only:

- dim-48 FP seed1401 file SHA-256 `be4353c20ec2239f000de2a15d446e8a62ce17f35eebfcac45eed09840585349`
- dim-48 ternary+A8 seed1401 file SHA-256 `c502b5f5e3b3b96520da3c023a51d789fc4403347c512f8c6fa3a81624703d5a`

The ternary pair is contextual; it cannot replace the accepted M14 FP core in the primary mechanism conclusion.

## 6. Data hierarchy

M15 uses new generated 4×4 unique Sudoku data. It does not use M14 confirmation for method selection.

### Development hierarchy

- seed: `2026091501`
- train / validation / development: `2048 / 256 / 256`
- clue range: uniform integer `6..10`
- box: `2`
- augmentation: false
- solution generation: randomized backtracking
- exact/group overlap: must be zero across splits

Only train data may fit M15 auxiliary models. Validation may choose the fixed uncertainty coefficient from the preregistered set below. Development is used for mechanism analysis and the predeclared intervention decision.

### Frozen confirmation

- seed: `2026091502`
- examples: `512`
- same generator contract as development
- generated only after the M15 code, auxiliary-training recipe, action directions, verifier strengths, search budgets, guard semantics, and validation-selected uncertainty coefficient are frozen
- explicit fingerprint-overlap audit against the complete M15 development hierarchy

Confirmation is analysis only: no subsequent M15 method change may use it while retaining a confirmation label.

### Prespecified difficulty / distribution-shift audit

M03 established clue count as explicit persisted Sudoku difficulty metadata and explicitly warned that split construction does not prove matched population difficulty. M15 uses that established metadata axis to define a **new M15 shift**, not to claim M03 already measured it:

- in-distribution slices: clues `6..10`, reported by exact clue count;
- difficult in-distribution slice: clues `6..7` and examples not semantically valid after the first ordinary recurrent step;
- lower-clue shift: seed `2026091503`, `256` unique examples, clues `4..5`, evaluated only after the M15 intervention is frozen.

The lower-clue set is an M15 OOD/difficulty stress test, not an official Sudoku benchmark.

## 7. Auxiliary verifier training

For each primary core seed, freeze the core before any auxiliary target generation or training.

Train-only verifier states are generated from the first 384 train puzzles at depths `0..3`. Labels are the existing independent, reference-free `sudoku_one_cycle_improvement_v1` event. Reference solutions are not inputs to the label function.

Two strengths are fixed:

### Weak verifier

- one `GroundedStateVerifier`
- dim matches core (`64`)
- 1 block, 4 heads, A8 fake-quantized z input
- optimizer AdamW
- steps `60`
- batch `64`
- lr `2e-3`
- weight decay `0.01`

### Strong verifier

- `EnsembleGroundedStateVerifier`, 3 members
- same architecture per member
- member seeds derived deterministically from core seed
- steps/member `180`
- batch `64`
- lr `2e-3`
- weight decay `0.01`

Verifier fitting records loss curves, parameter ownership, frozen-core hashes, shallow/deep AUC/AP/Brier/ECE, disagreement/error correlation, and per-depth absolute-quality/proxy relationships.

## 8. Train-only action mechanism

For each primary core seed:

- candidate direction bank: `16` unit-norm directions, deterministic seed `2026091511 + core_seed`;
- action scale: `0.5`;
- select exactly `3` directions on train-only trajectory utility using identity plus candidate one-cycle reference-free symbolic structural score;
- fit one `StateConditionedLatentActionCodebook` with utility-distillation supervision;
- fit states: depths `0..2` from a disjoint train-only puzzle slice;
- policy steps `300`, batch `64`, AdamW lr `2e-3`, weight decay `0.01`;
- no reference solution enters action utility or policy training.

Unguided search uses **the exact same three learned/selected directions and action scale** but a uniform/global prior. Learned search differs only by the state-conditioned policy prior. Transition/evaluator budgets are identical; policy-forward overhead is measured separately and the pair is called measured-cost-matched only if complete-solve median latency is within ±5%.

## 9. Search/value implementations

Primary learned-verifier search is serial B=1 MCTS over complete `(x,y,z)` state with deterministic PUCT semantics.

Fixed search budget for the main comparison:

- actions: identity + 3 directions
- rollouts: `12`
- c_puct: `1.5`
- max depth: `4`
- same action directions for unguided, learned-prior, guarded, and oracle-value diagnostics

Depth ablation: max depth `{1,2,4}` with the same 12 rollouts.

### Value variants

1. **learned proxy** — strong grounded verifier mean probability of one-cycle improvement;
2. **weak learned proxy** — weak verifier probability;
3. **uncertainty LCB diagnostic** — ensemble `mean - beta * std`, `beta in {0.0, 0.5, 1.0, 2.0}` selected on validation by lowest search-induced-regression rate, ties by higher semantic success then lower latency;
4. **oracle absolute-quality diagnostic** — decoded reference-free symbolic Sudoku structural score; this is a diagnostic and has extra decode/oracle cost, so it is not presented as latency-matched to the learned verifier;
5. **terminal-validity guard (primary intervention)** — before treating a leaf as nonterminal, decode it, restore immutable givens, and run the independent Sudoku validity check. A valid leaf gets terminal value `1.0` and is not expanded further; invalid leaves use the strong learned proxy. The guard never consumes the reference solution.

The terminal guard is preregistered before M15 results. It is task-specific and is not claimed as generic learned halting.

## 10. No-search and routing/halting ablations

The accepted M14 semantic-exit solver is the primary no-search system context.

On the same core/data, report:

- fixed ordinary recurrence through four supervision steps, no early exit;
- M14 semantic-validity early exit with max K=4;
- learned M12 router/halter is **not** substituted into the M15 primary mechanism because M12 already found learned control collapsed and it is not part of the accepted M14 system.

This isolates the useful task-specific halting mechanism without turning M15 into another RL rescue attempt.

## 11. Precision ablations

### Search-state storage precision

For the same FP64 core/action/verifier:

- current native-search-style per-token symmetric INT8 z storage/requantization;
- FP32 z-state reference search with otherwise identical transitions, priors, rollout/depth budget, and value function.

The comparison records per-depth z divergence, answer agreement, symbolic score, search-induced regressions, and task success. It does not call the pair hardware-cost-equivalent.

### Weight/activation precision context

The frozen dim48 FP and ternary+A8 seed1401 pair is evaluated under ordinary recurrence at depths `1..4` on the same M15 development examples. It is a matched-architecture precision context check only; no scaling law or generic ternary conclusion is permitted.

## 12. VQ claim correction and trajectory measurement

The current `model/latent_vq.py` prose incorrectly implies that finite codebook membership/idempotent projection bounds error relative to an unquantized trajectory at arbitrary depth. M15 will remove that claim.

A finite codebook only bounds the state to the codebook and gives an observed/local projection error relative to the *pre-projection state*. Without assumptions on transition contraction, codebook coverage of the reference trajectory, and assignment stability, it does **not** bound divergence from the unquantized trajectory.

M15 fits a small VQ codebook using train-only FP64 trajectory z states (no test states) and reports on development/confirmation at depths `1..8`:

- `||z_variant - z_fp||` mean/p95 trajectory divergence;
- observed local projection error;
- unique code utilization fraction and code perplexity;
- decoded answer agreement with FP trajectory;
- strict Sudoku semantic success;
- comparison against simple INT8 requantized trajectory.

No statement that VQ preserves rank, isometry, topology, OOD detection, or reasoning is permitted from these measurements.

## 13. Primary mechanism metrics

For each search-evaluated leaf retain:

- core seed, example id, clue count, difficult-slice flag;
- path/depth/action;
- proxy value and ensemble disagreement;
- independent symbolic absolute structural score;
- one-cycle improvement label where defined;
- semantic validity;
- INT8/FP32/VQ precision label;
- work counters and complete-solve latency.

Primary summaries:

- verifier ROC AUC/AP for its declared improvement target by depth;
- Spearman(proxy value, absolute symbolic score) by depth;
- solved-vs-unsolved proxy inversion: mean proxy(valid) minus mean proxy(invalid);
- search semantic success and mean symbolic score;
- search-induced regression rate: no-search succeeds but search fails;
- search-induced rescue rate: no-search fails but search succeeds;
- proxy-exploitation rows: selected state has higher proxy but lower symbolic score than an available evaluated alternative;
- terminal-guard prevention rate among unguarded regressions;
- latency mean/median/p95 and realized transitions/verifier/policy/semantic-check counts separately.

## 14. Development support rule for H1

H1 is supported as an important effect only if development shows all of:

1. unguarded strong-verifier learned search at depth 4 has at least `5 percentage points` search-induced regression relative to the accepted no-search semantic-exit system **or** at least `10%` of its selected outputs are proxy-exploitation cases;
2. oracle absolute-quality search with the same action set/work improves semantic success or mean symbolic score materially over learned-proxy search, with at least `50%` fewer search-induced regressions;
3. the terminal-validity guard prevents at least `50%` of unguarded search-induced regressions and does not reduce overall semantic success by more than `1 percentage point`;
4. H3 does not dominate: switching INT8 search-state storage to FP32 does not by itself prevent at least `75%` of the unguarded regressions.

If these conditions fail, the mechanism conclusion is `INCONCLUSIVE` or another competing explanation is preferred. The acceptance gate for M15 does not require H1 specifically; it requires at least one important controlled effect with competing hypotheses and failure cases.

## 15. Frozen confirmation rule

After development analysis is complete:

- freeze auxiliary verifier/action state hashes, selected `beta`, action directions, search budgets, guard implementation identity, VQ state, and all analysis thresholds;
- then generate confirmation seed `2026091502` and lower-clue shift seed `2026091503`;
- run the same fixed comparisons once;
- no post-confirmation method change may be described as confirmed M15 evidence.

Because only two primary core seeds are used, M15 is a bounded pilot. Uncertainty will be reported with paired example bootstrap and seed-stratified summaries, but no broad population or scaling claim is permitted.

## 16. Required negative/failure evidence

M15 must preserve and report:

- difficult low-clue examples;
- cases where search breaks a no-search success;
- cases where search rescues a no-search failure;
- examples where proxy value rises while symbolic quality falls;
- learned-vs-unguided failures;
- any uncertainty-penalty failure;
- lower-clue shift behavior;
- VQ trajectories with large divergence or answer disagreement;
- any result contradicting H1.

## 17. Unsupported theory cleanup

M15 will update current prose so that:

- VQ finite-state/idempotence is not described as an arbitrary-depth bound relative to an unquantized trajectory;
- ensemble disagreement is not asserted to identify OOD states without calibration/evidence;
- no regularizer is said to guarantee rank, isometry, topology, preserved reasoning, or OOD detection absent explicit assumptions and evidence;
- synthetic death-trap tests remain synthetic mechanics tests, not real-trajectory theorems.

## 18. Acceptance

M15 passes only if:

- at least one important effect has a controlled explanation with explicit competing hypotheses and a disconfirming result;
- the prioritized ablations are reproducible with frozen checkpoint/data/budget controls;
- difficult examples and search-induced regressions are retained;
- VQ trajectory divergence/utilization/agreement/task metrics are measured rather than inferred from codebook boundedness;
- unsupported theoretical claims are removed or qualified;
- frozen confirmation is not used for subsequent M15 method selection.

A negative or mixed result is acceptable if the mechanism is still controlled and informative. M15 does not require a new task-accuracy win.

**Stop after M15. Do not begin M16 automatically.**