# Independent transfer — completed local result, 2026-09-26

The protocol at `1b1eb7c711263021dfb9f59413aaa347b8090839` and stronger-MLP amendment at `2e76985c59a1a7842ad826d48c92fcf9ad6097e8` preceded the final-test model calls. Acquisition workflow `36259003288` succeeded; its public dataset/wheel archive was downloaded and hash-checked. The new implementation and raw results are in the accompanying ChatGPT delivery, not uploaded to this branch or cloud-CI certified.

**Decision: do not promote the four-query architecture.** Native same-function execution passes, but the primary useful-workload gate fails on both held-out tasks. The stronger MLP is more accurate and faster on both; native SVC also dominates on Pendigits under the measured accuracy/latency scope. No hiring score, new release, main merge or default-model change follows.

## Final-test accuracy

Neural values average five fixed seeds, not an ensemble. SVC is one selected model per task. All fitting and validation choices precede test opening; no post-test refitting or reselection.

| Model | Pendigits, 3498 rows | Letter, 4000 rows |
|---|---:|---:|
| Prior first-token query | 96.8897% | 92.8100% |
| One mean-seeded query | 97.3814% | 92.4450% |
| Four queries, primary | 97.1527% | 92.4750% |
| Full-state recurrent control | 96.8439% | 92.5900% |
| MLP, original30 epochs | 96.3636% | 83.4250% |
| MLP, pre-test480-epoch control | 97.7416% | 94.3400% |
| RBF SVC | 98.4563% | 95.7750% |

The recurrent arms have13388/13916 parameters; MLP has11914/13466. Sixty neural fits consumed2711.298 CPU seconds including epoch validation. The four-query fits used636.410 seconds and the stronger MLP637.597 seconds in aggregate, not exact equality per task. Compilation, benchmark/verification and SVC costs are separate. No GPU or pretrained weights.

Pendigits retains its writer-independent partition. Letter follows first16000 development/last4000 test;380 test feature vectors repeat development features. All rows remain primary and the3620 nonoverlap stratum is separately reported. These are two related recognition tasks, not broad generalization or full-development-refit benchmark scores.

## Same-function runtime

Intel Xeon Platinum8370C, one pinned core, Python3.13.5, Torch2.10.0+cpu. ORT CPU uses all graph optimizations, sequential execution and one intra/inter-op thread. Input preprocessing/session creation are excluded; state reset, context preparation, full inference, fresh output and class decoding are included. Every timed output is checked.

| Task | Four-query native AVX2 | Same model ORT1.30.0 | Speedup |
|---|---:|---:|---:|
| Pendigits | 53.803 us | 306.456 us | 5.696x |
| Letter | 54.637 us | 312.633 us | 5.722x |

Paired fixed-model input-bootstrap ratio intervals: [0.174344,0.176804] and [0.173567,0.175988]. Zero case-median regressions;11/1280 and8/1280 empirical case-P95 regressions. Nine repeats do not establish production-tail guarantees. Primary matrix460800 cells; separate ORT1.23.2/native matrix276480 cells gives5.426x/5.357x. Do not pool separate runs or multiply older XLA/Inductor results from different models/hosts. XLA and Inductor were not remeasured here.

The stronger native MLP takes23.175/24.918 us, versus53.803/54.637 us for the primary. Native SVC takes47.642/386.737 us. Thus runtime improvement does not establish task superiority, and both one-percentage-point quality gates remain failed. Descriptive two-way seed/input intervals and all per-seed outcomes are retained.

## Verification and preserved source

All224940 neural model/test pairs match eager class predictions under native and ORT1.30; maximum absolute logit errors0.0000610352/0.0000401139 under the unchanged nonbitwise tolerance. An independent interpreter checks37490 primary pairs and74980 recurrent steps. All7498 native SVC decisions match scikit-learn. Separate standard-library auditors verify original dataset bytes/labels, complete timing grids, source identities and reported aggregates; this is not outside replication.

Current checks:1647 historical fast tests pass,16 slow excluded,two historical warnings;50 layout tests;77 transfer contracts. ASan/UBSan processes check37490 primary predictions and reject16 malformed model files and3 invalid input streams. The separate vector-math helper is not sanitizer-instrumented. Extracted SDK portable andAVX2 builds each check992 examples over62 models without importing a numerical framework. Counts overlap. All692 prior output-aware source files, including the596 mapped historical originals, remain byte-identical.

## Actual local delivery

Local commit **26b6c64db01190372f15316e7071a357e5e2759f**, tree **94d09be361b1835befcfda17fad55f06484b34c9**,722 tracked files. The bundle requires original main9fd6682345b34e992b3cab7e55750e9ef4b04e87. It is a review branch, not an instruction to overwrite current main. Old PR27/28/29 merges remain intact; no new release or settings change occurred.

| ChatGPT-delivered file (not a GitHub release asset) | Bytes | SHA256 |
|---|---:|---|
| SPECTRA_transfer_project.zip | 51702376 | d60803e90da007cf5709ae3cf664923fb93e73b66b7029a2089c236b56da2912 |
| SPECTRA_transfer_sdk.zip | 3736967 | f0aa3ed7ebd143a2b858c3d781d5edd2bd33a05d778a27edb4ded1a1322ee097 |
| SPECTRA_transfer_evidence.zip | 90384114 | 3cde647d3dc395689122b8cc50bffd7a3e63190f0b5cb01527a8bad6e46ac6ae |
| SPECTRA_TRANSFER_REPORT.md | 18798 | 7fd098286ef65fedf94707c3884652fb6aafddd89595f075dfcb450702e7332e |

The evidence retains original public data with CC BY4.0 attribution, all selected/final checkpoints, validation curves, raw observations, initial failed attempts, build receipts, audits, source inventory, patch and Git bundle. Rebuildable compiler caches and numerical-framework wheel bytes are explicitly excluded. The SDK supplies tested Linux native binaries. No Windows/ARM acceptance, energy/peak-RSS result, new frontier capability or external reproduction is claimed.
