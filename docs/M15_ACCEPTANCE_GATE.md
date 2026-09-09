# Milestone 15 Acceptance Gate — Mechanism-Focused Ablations

**Decision: PASS for a bounded mechanism contribution.**

M15 does **not** establish a generic latent-search advantage. It explains one important failure mechanism in the existing SPECTRA search stack and preserves counterevidence, difficult examples, distribution-shift failures, and the limits of the intervention.

## Mechanism question

The preferred preregistered explanation was **target/value mismatch**. The grounded verifier is trained to estimate whether **one additional frozen-reasoner cycle improves reference-free Sudoku structural score**. Native MCTS, however, had been using that probability as though it were an **absolute state value**. Those are not the same objective: a solved or high-quality state can have low probability of further improvement precisely because little improvement remains available.

Competing explanations were:

- **H2 — depth/distribution shift:** deeper search leaves the verifier's validated shallow state distribution and ranking collapses;
- **H3 — INT8 trajectory distortion:** repeated INT8 latent storage/requantization causes the regressions;
- **H4 — poor action/policy quality:** the available directions themselves are bad, so a better value function would not repair search.

H1 was explicitly disconfirmable before results: it would lose if proxy and absolute quality stayed positively aligned, oracle absolute-quality search regressed similarly, the terminal-validity guard failed to reduce regressions, or FP32 search-state storage removed most regressions.

## Frozen sources and data hierarchy

M15 uses the exact accepted M14 dim-64 FP recursive checkpoints for seeds `1401` and `2402` from M14 final artifact `10077794916` / run `34280670349`. The artifact, serialized checkpoint hashes, and tensor-state hashes are checked before any M15 auxiliary fitting. A failed earlier attempt to reproduce the source by retraining stopped before any M15 data generation or result and is retained as provenance rather than relabelled as a scientific attempt.

Development hierarchy:

```text
seed                         2026091501
train / validation / dev     2048 / 256 / 256
Sudoku                       unique 4x4, box=2
clues                        uniform 6..10
core seeds                   1401, 2402
```

Only training data fits auxiliary verifiers/actions/VQ. Validation selects `beta` from the frozen grid `{0, 0.5, 1, 2}`. Development evaluates the mechanism and intervention. Only after the development rule passes are auxiliary hashes/configuration frozen and confirmation seed `2026091502` generated. Confirmation contains 512 examples and has zero fingerprint overlap with the development hierarchy. Lower-clue shift seed `2026091503` contains 256 examples with 4–5 clues and also has zero overlap.

## Controlled ablations

The retained experiment includes:

- no-search fixed four-step recurrence;
- the accepted M14 semantic-validity early exit;
- learned-verifier search at depths 1, 2 and 4;
- equal-action/equal-rollout unguided search using the **same three selected directions**;
- weak versus strong verifier;
- validation-selected ensemble LCB (`beta=2.0`);
- task-specific terminal-validity guard;
- oracle absolute symbolic-quality value diagnostic;
- INT8 versus FP32 search-state storage on the same exact FP cores;
- VQ32 versus simple INT8 requantization versus unquantized FP trajectories through depth 8.

The optional matched dim-48 FP-versus-ternary context pair was omitted because the exact accepted context state was not present in the primary accepted M14 artifact. The mandatory precision test — FP32 versus INT8 search-state storage on the exact accepted dim-64 FP cores — remained intact. No substitute checkpoint was introduced.

M12 learned routing/halting is not reused as a rescue mechanism: M12 already established a negative/collapsed learned-control result, and it is not part of the accepted M14 system.

## Development result

On two core seeds × 256 development examples, the preregistered H1 rule passed:

```text
unguarded learned-search regressions     85 / 512
oracle regression reduction              100.0%
terminal-guard prevention                 91.76%
FP32-state prevention                      0.0%
H1 development decision                   PASS
```

The intervention and all analysis thresholds were then frozen before confirmation.

## Independent confirmation

Across two exact M14 core seeds × 512 untouched confirmation examples (`n=1024` paired rows):

| Configuration | Semantic success | Regressions vs M14 semantic-exit | Proxy exploitation | Median latency |
|---|---:|---:|---:|---:|
| M14 semantic exit, no search | **0.99121** | — | 0 | **1.238 ms** |
| Learned strong verifier, d1 INT8 | 0.85059 | 144 / 1024 | 6.74% | 25.039 ms |
| Learned strong verifier, d2 INT8 | 0.84863 | **146 / 1024** | 13.18% | 36.958 ms |
| Learned strong verifier, d4 INT8 | 0.84863 | **146 / 1024** | **13.28%** | 61.093 ms |
| Equal-action unguided d4 INT8 | 0.84863 | **146 / 1024** | 13.18% | 59.240 ms |
| Weak verifier d4 INT8 | 0.86133 | 133 / 1024 | 9.28% | 47.108 ms |
| LCB beta=2 d4 INT8 | 0.87109 | 123 / 1024 | 6.35% | 61.336 ms |
| Terminal-validity guard d4 INT8 | **0.96973** | **22 / 1024** | 1.17% | 5.300 ms |
| Oracle absolute-quality d4 INT8 | **0.99121** | **2 / 1024** | 0 | 28.293 ms |
| Learned strong d4 FP32 state | 0.84863 | **146 / 1024** | 13.18% | 59.131 ms |

Confirmation H1 diagnostics:

```text
unguarded regressions                     146
oracle regression reduction               98.63%
terminal-guard prevention                  84.93%
FP32-state prevention                       0.00%
H1 confirmation decision                   PASS
```

This is strong counterfactual evidence against H3 as the primary explanation: replacing INT8 state storage with FP32 left the selected answers, symbolic score, semantic success, and all 146 regressions unchanged. It also weakens H4 as the principal explanation: learned and unguided priors over the same action directions end with the same task result, while replacing the learned proxy value with independent absolute symbolic quality nearly eliminates regressions.

## Why this is target/value mismatch rather than only depth collapse

The verifier can still rank **its declared one-cycle-improvement target** while being negatively related to **absolute task quality**.

On confirmation search leaves:

```text
depth 1:
  one-cycle-improvement ROC AUC             0.7536
  Spearman(proxy, absolute symbolic score) -0.2524
  mean proxy(valid) - mean proxy(invalid)  -0.1649

depth 2:
  one-cycle-improvement ROC AUC             0.8054
  Spearman(proxy, absolute symbolic score) -0.2004
  mean proxy(valid) - mean proxy(invalid)  -0.0490
```

Thus the learned model is not simply becoming random at depth 1–2. It retains useful discrimination for the event it was trained on, yet that event is the wrong scalar objective for choosing the best current state. M15 therefore marks **H1 target/value mismatch confirmed** and **H2 deep-ranking collapse not confirmed as the primary mechanism**.

A representative retained failure has a no-search symbolic score of `1.0` and an evaluated search state with score `1.0`, but learned-proxy search selects a different state with score `0.8125` because that state receives the larger proxy value. These are recorded as proxy-exploitation/search-regression examples, not anthropomorphic claims that the model intentionally hacks a reward.

## Distribution-shift stress test

On the preregistered 4–5 clue shift (`n=512` paired rows):

```text
M14 semantic exit success                 0.85938
learned strong d4 INT8 success            0.32031
learned strong regressions                276 / 512 = 53.91%
proxy exploitation                        58.59%
learned strong d4 FP32 success            0.32031
LCB beta=2 success                        0.34375
terminal guard success                    0.70508
oracle absolute-quality search success    0.87109
oracle regressions                         8 / 512
oracle rescues                            14 / 512
```

The shift amplifies the same mechanism. The guard helps substantially but does not restore the accepted no-search quality under shift; it is a bounded task-specific mitigation, not a general solution.

## VQ claim correction

The old `model/latent_vq.py` prose incorrectly inferred a depth-independent bound relative to the unquantized trajectory from finite codebook membership and idempotent projection. M15 removes that claim.

A finite codebook bounds the projected state to the codebook and makes local projection error measurable. It does **not**, without transition/contraction/coverage/assignment assumptions, bound divergence from the counterfactual unquantized recurrent trajectory or guarantee preserved topology, rank, isometry, calibration, OOD detection, answers, or reasoning.

The retained depth-1..8 trajectory experiment demonstrates why that distinction matters. Simple INT8 requantization stays close to FP (`~0.05` mean per-token L2 divergence) with essentially complete answer agreement in this pilot. The train-only VQ32 codebook has much larger divergence (`~5–7` mean per-token L2 early), limited code utilization, and answer/task disagreement that grows with depth. At depth 8:

```text
core seed 1401:
  VQ answer agreement      0.7773
  VQ semantic success      0.8184
  FP semantic success      0.8535

core seed 2402:
  VQ answer agreement      0.9609
  VQ semantic success      0.9629
  FP semantic success      0.9961
```

These are measurements, not a universal statement that VQ is harmful. They directly reject the previous proof-style claim that snapping alone bounds deviation from an unquantized reasoning trajectory at arbitrary depth.

## Stability / uncertainty theory boundary

M15 also qualifies two other theoretical overreaches:

- ensemble member standard deviation is an empirical disagreement statistic; without calibration evidence it does not identify OOD states or guarantee conservative search avoids proxy overoptimization;
- VICReg-style variance/covariance penalties and sampled Jacobian norm penalties are training objectives/diagnostics, not proofs that the recurrent representation remains full-rank or dynamically isometric. Such guarantees require explicit assumptions and direct spectral/rank evidence.

## Prior-work / novelty boundary

Relevant prior work was checked before interpreting M15. Learned answer/process verifiers, reward/proxy overoptimization under stronger optimization, ensemble/conservative reward-model objectives, VQ-VAE quantization, anti-collapse representation regularization, and dynamical isometry are established ideas. M15 therefore makes **no broad novelty claim** for verifier-guided search, Goodhart effects, uncertainty penalties, VQ, VICReg, or dynamical-isometry regularization.

The bounded contribution is narrower: a controlled diagnosis, in this SPECTRA recursive-search system, showing that a verifier trained for *one-step improvability* can remain competent on that target while being anti-aligned with *absolute state quality*, and that this target/value mismatch explains most observed search-induced regressions better than INT8 state distortion or action-prior choice.

## Provenance

Accepted M15 execution:

```text
branch          research/m15-mechanism-ablations
head            18f3693dd022aa3a83e1d6d2e39a1cc209790a4a
run             34287134895
job             102265201074
artifact        m15-mechanism-ablation-evidence
artifact id     10080887785
ZIP SHA256      c49e5c06e0f2822088cb9ce807fcc6bc37da6236962953ccc8c594be01c55b2e
size            10,624,283 bytes
focused         47 passed
full fast       326 passed, 16 deselected, 1 pre-existing M10 warning
```

The compact Git-retained result/provenance surface is under `results/m15/`; full raw leaf/search/trajectory rows and auxiliary checkpoints remain in the Actions artifact.

## M15 decision

**PASS — bounded mechanism-focused contribution.** The evidence explains an important failure with competing hypotheses, direct counterevidence and retained failure cases. It does not turn learned latent search into a positive capability result. The accepted M14 semantic-exit system remains substantially better and cheaper than the failed learned-proxy search on this task.

**Stop here for M15. Do not begin M16 automatically.**
