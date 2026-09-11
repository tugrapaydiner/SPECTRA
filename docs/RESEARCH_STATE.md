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
| M15 | **COMPLETE** bounded mechanism result; ancestral-overlap correction retained in M16 |
| M16 | Implemented and evidence retained; native-checker speedup and evaluator-target controls |
| M17 | Execution complete; **TWO_FAMILY_CLAIM_NOT_ESTABLISHED** |

## Current integration boundary

M16/M17 implementation is consolidated on `research/verifier-aligned-cpu-search`.
The native-checker path improves the historical complete-solve implementation on
its retained small Sudoku comparison, not the performance of an arbitrary solver.
M17 confirms a fixed-pool evaluator-target effect on harder Sudoku but fails its
maze development gate; maze confirmation remains unopened. No generic learned
search, cross-task superiority, physical-energy, or frontier-engineer-level claim
is made. See [integration review](INTEGRATION_REVIEW.md).

The M15 within-stage overlap statement below is historical: M16's ancestor-wide
retrospective audit found one M15 confirmation puzzle present in M14 training.
The sensitivity analysis excluding that puzzle preserves the broad mismatch
conclusion, but the original cross-stage independence claim must not be reused.
The accepted M14/M15 records have not been rewritten to hide this correction.

`python scripts/verify_retained_results.py --out outputs/retained-audit.json`
verifies pinned M16/M17 archives, 1,728 manifest rows and 11,520 stored
complete-solve answer records, then reproduces retained summaries without opening
new confirmation data. Timing repetitions are not independent examples. Fixed-pool
labels are reaggregated rather than independently inferred by this audit.

## Independent fixed-pool inference replay

`python scripts/verify_fixed_pool_replay.py --out outputs/fixed-pool-replay`
reconstructs all previously published M17 pools from frozen checkpoints rather
than only reaggregating their stored labels. The CLI uses an explicit historical
AVX2 CPU profile, records dispatch settings, and requires the six exact pool-tensor
hashes. It checks 16,384 candidates, their 16,384 continuations and 49,152 head
scores. Already-published Sudoku confirmation is replayed without generation or
selection; the unused maze confirmation remains unopened.

On these maze pools, all 3,707 nonempty invalid candidates have structural score
0.75. For restored maze answers the score equals
`0.5 + 0.25 * nonempty_path + 0.25 * exact_validity`; it does not grade progress
among nonempty invalid paths. This narrows interpretation of the failed maze
experiment, without changing its result. See [replay details](FIXED_POOL_REPLAY.md).

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

**Historical M15 acceptance: a bounded mechanism contribution, not learned-search superiority. Later M16/M17 work and its limitations are summarized above.**
