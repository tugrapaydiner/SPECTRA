# M14 measured development results

Generated from independently checked saved predictions and raw complete-solve timing rows.

Timing rounds are reduced to per-example medians. Three training seeds share paired examples; they do not multiply the unique test sample count. The symbolic solver is untrained.

## initial

| System | Solve rate | Blank accuracy | Mean solve ms | p95 ms |
|---|---:|---:|---:|---:|
| Small FP32 recursive | 0.00% | 39.61% | 1.896 | 2.145 |
| Larger mixed INT8 single-pass | 0.00% | 56.49% | 1.477 | 1.711 |
| Larger FP32 single-pass | 0.00% | 56.48% | 1.363 | 1.583 |
| Symbolic MRV reference | 100.00% | 100.00% | 0.428 | 0.570 |
| Small ternary · PyTorch | 0.00% | 32.77% | 3.641 | 4.224 |

## decode only

| System | Solve rate | Blank accuracy | Mean solve ms | p95 ms |
|---|---:|---:|---:|---:|
| Small FP32 recursive | 0.00% | 39.61% | 1.764 | 2.029 |
| Larger mixed INT8 single-pass | 0.00% | 56.49% | 1.364 | 1.685 |
| Larger FP32 single-pass | 0.00% | 56.48% | 1.245 | 1.507 |
| Small ternary · PyTorch | 0.00% | 32.77% | 3.290 | 3.996 |

## blank only

| System | Solve rate | Blank accuracy | Mean solve ms | p95 ms |
|---|---:|---:|---:|---:|
| Small FP32 recursive | 0.00% | 34.47% | 1.050 | 1.235 |
| Larger mixed INT8 single-pass | 0.00% | 55.03% | 1.339 | 1.630 |
| Larger FP32 single-pass | 0.00% | 55.05% | 1.223 | 1.493 |
| Symbolic MRV reference | 100.00% | 100.00% | 0.442 | 0.604 |
| Small ternary · PyTorch | 0.00% | 31.33% | 3.311 | 4.013 |

Energy per solve: unavailable; these are latency measurements.

Inspect `per_seed` in `summary.json` for individual successes, Wilson intervals and training-seed variation.
