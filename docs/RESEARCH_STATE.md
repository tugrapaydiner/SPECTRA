# SPECTRA Research State

## Milestone index

| Milestone | State |
|---|---|
| M01–M08 | accepted / merged |
| M09 | **INCOMPLETE**; trained-search benefit target missed |
| M10–M11 | accepted / merged |
| M12 | training path accepted; learned control **NEGATIVE / COLLAPSED** |
| M13 | accepted / merged |
| M14 | **COMPLETE**; independently confirmed task-specific quality-superiority gate |
| M15 | **COMPLETE**; bounded mechanism explanation for learned-verifier search regressions |

# Milestone 15 — mechanism-focused ablations

**Status: COMPLETE on `research/m15-mechanism-ablations`.**

Authoritative files: [`M15_PROTOCOL.md`](M15_PROTOCOL.md), [`M15_PROTOCOL_CLARIFICATION.md`](M15_PROTOCOL_CLARIFICATION.md), [`M15_SOURCE_ARTIFACT_AMENDMENT.md`](M15_SOURCE_ARTIFACT_AMENDMENT.md), [`M15_PRIOR_WORK.md`](M15_PRIOR_WORK.md), and [`M15_ACCEPTANCE_GATE.md`](M15_ACCEPTANCE_GATE.md). Compact evidence is under [`../results/m15/`](../results/m15/).

M15 uses exact accepted M14 dim-64 FP core checkpoints for seeds 1401/2402. Development selected only predeclared controls; confirmation seed `2026091502` was generated only after the mechanism rule passed and had zero fingerprint overlap with development.

## Controlled explanation

Preferred explanation: **target/value mismatch**. The grounded verifier estimates whether one more frozen-reasoner cycle improves reference-free structural score. Native search had used this probability as an absolute state value. On confirmation, the verifier still ranked its declared one-cycle target (ROC AUC 0.754 at search depth 1; 0.805 at depth 2) while its score was negatively correlated with absolute symbolic quality (Spearman -0.252 and -0.200).

The unguarded learned strong-verifier search produced **146/1024** regressions relative to the accepted M14 semantic-exit system. Replacing the proxy with independent absolute symbolic quality reduced regressions by **98.63%**; a reference-free task-specific terminal-validity guard prevented **84.93%**; changing INT8 search-state storage to FP32 prevented **0%**. Equal-action unguided search matched the learned-prior task result. These controls support target/value mismatch over INT8 distortion or action-prior choice as the primary observed mechanism.

The 4–5 clue shift amplifies the failure: accepted semantic exit succeeds on 0.8594, learned d4 search on 0.3203 with 276/512 regressions and 58.59% proxy-exploitation events. The guard improves to 0.7051 but does not restore the accepted system; oracle absolute-quality search reaches 0.8711.

## VQ / theory boundary

M15 removes the old claim that finite codebook snapping or local projection error bounds divergence from an unquantized recurrent trajectory at arbitrary depth. Depth-1..8 measurements show simple INT8 requantization stays close to FP in this pilot, while train-only VQ32 can diverge materially, use a limited code subset, and lose answer/task agreement with depth. This is a bounded measurement, not a universal anti-VQ conclusion.

Ensemble disagreement is an empirical heuristic, not an OOD certificate. VICReg-style variance/covariance and sampled Jacobian penalties are training objectives/diagnostics, not proofs of rank preservation or dynamical isometry.

Accepted scientific run: `34287134895`, job `102265201074`, artifact `10080887785`, ZIP SHA256 `c49e5c06e0f2822088cb9ce807fcc6bc37da6236962953ccc8c594be01c55b2e`; 47 focused tests and 326 fast tests passed (16 deselected, one pre-existing M10 warning).

**M15 passes as a bounded mechanism contribution, not as learned-search superiority. Stop here; do not begin M16 automatically.**
