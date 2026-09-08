# M14 retained evidence — Attempt 5

Milestone 14 completed on the preregistered **Attempt 5** dual-stream FP64 recursive model with task-specific, reference-free Sudoku semantic early exit.

Authoritative run: `34171128736` at head `12f911b1abfa9710c7aa14a3f4ba2e691f84ac1c`. Artifact ID `10035946017`; artifact ZIP SHA-256 `0306f64efb48c264aa7b387ff6b449def011fb64ad02aa74eb03ee3b4e58e314`.

## Confirmed primary result

- candidate: `dual_stream_exit_k4`, original dual-stream FP64 TRM, semantic-validity early exit
- primary baseline: trained FP32 `System1Student`, dim 96, two blocks
- five training seeds: `1401, 2402, 3403, 4404, 5405`
- independent confirmation: 1,024 examples, seed `2026091402`, zero fingerprint overlap with the 5,120 development-hierarchy examples
- strict semantic-validity difference: **+0.1513671875**
- hierarchical-bootstrap 95% CI: **[+0.1361328125, +0.169140625]**
- preregistered paired median complete-solve latency ratio: **0.9842012359971253**
- latency-ratio 95% CI: **[0.9808512693098508, 0.9873359073434289]**
- decision: **Path A / quality superiority PASS**

## Cost-distribution disclosure

The primary gate was preregistered on **median** complete-solve latency. It did not require matched mean or p95 latency. The raw confirmation rows show:

- candidate mean / median / p95: **1.466297 / 1.215085 / 3.206238 ms**
- baseline mean / median / p95: **1.238605 / 1.234590 / 1.276632 ms**
- mean latency ratio: **1.18383×**
- p95 latency ratio: **2.51147×**
- candidate mean executed supervision steps: **1.2484**
- candidate cases requiring more than one step: **16.80%**

Therefore the defensible claim is **higher solve accuracy at comparable median latency**, not equal average compute, equal tail latency, universal speedup, or iso-energy. See `latency_tail_audit.json`.

## Raw evidence

The exact raw JSONL rows remain in the accepted Actions artifact (`10035946017`, ZIP SHA-256 `0306f64efb48c264aa7b387ff6b449def011fb64ad02aa74eb03ee3b4e58e314`). The accepted confirmation raw-row SHA-256 is `52c8e8e8ecfb632c4858634f98e40c51f03b17d7ba3dccff4d6ccbb1cb3f80ea`. `result_table.csv`, `result_table.md`, and the original renderer provenance are retained here. `pareto.svg` is a deterministic Git-retained regeneration computed from the same accepted confirmation raw rows; `git_figure_provenance.json` binds it to the raw-row hash.

Physical CPU-package energy was unavailable on the accepted host (`no_package_domain`) and remains null. No GPU or whole-system energy claim is made.

The task-specific semantic stop uses only the predicted board, puzzle givens, and independent Sudoku validity; it never sees the reference solution. This is a hybrid inductive bias specific to Sudoku and does not establish generic learned halting or cross-task transfer.

This Git-retained evidence surface is part of the final M14 docs-inclusive validation head; it does not alter the preregistered experiment or accepted result.
