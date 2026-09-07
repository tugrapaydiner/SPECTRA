# SPECTRA Claims Ledger

Purpose: separate what the repository currently **implements**, what its tests actually establish, what has a reproducible measurement artifact, and what remains a **hypothesis**. Passing a unit test upgrades only the domain actually exercised by that test.

Detailed execution history and exact milestone evidence live in [`docs/RESEARCH_STATE.md`](RESEARCH_STATE.md).

## Status vocabulary

- **implemented** — a concrete code path exists, but current evidence is insufficient for a broader validation claim.
- **unit-tested** — tests exercise the stated behavior on the documented/tested domain and pass in the recorded milestone gate.
- **measured** — a retained/inspectable measurement artifact grounds the claim, subject to its host/methodology assumptions.
- **hypothesis** — the prose statement is broader than current evidence or its required trained checkpoint/raw measurement/ground truth is absent.

## Claim map

| README / project claim | Code / evidence path | Boundary / assumptions | Status |
|---|---|---|---|
| Recursive TRM-style reasoning with shared recurrent computation and deep supervision is implemented. | `model/trm.py`, `train/losses.py`, `tests/test_recursion.py`, `tests/test_smoke.py`; M01/M02 fast gates | Establishes forward/backward/update mechanics, not task-level reasoning quality. | **unit-tested** |
| Spectral controls/stability machinery exists. | `tests/test_spectral.py`, `tests/test_stability.py` | Numerical invariants on tested configurations; not an arbitrary-depth theorem. | **unit-tested** |
| W1.58 ternary-weight and INT8 activation machinery is implemented. | `model/bitlinear.py`, `model/fake_quant.py`, export/packing paths, native tests | Representation/mechanics only; not end-to-end trained accuracy at target precision. | **unit-tested** |
| Native AVX2 packed and decoded ternary dot products support the full INT8 activation range without the `INT8_MIN * -1` byte-negation error. | `deploy/cpp_sparse_kernel/spectra_kernel.cpp`, `tests/test_kernel.py`, `tests/test_kernel_contract.py`, M02 run `34075718044` | M02 independently reproduced M01 scalar `+1` vs AVX2 `-1`, then verified scalar/AVX2 `+1`. All 256 INT8 values × ternary signs are checked against a wide-integer reference. | **unit-tested** |
| Native packed rows support non-multiple-of-four hidden widths under an explicit row-padding contract. | `deploy/pack_ternary.py::pack_ternary_rows`, checked C ABI, `tests/test_kernel_contract.py` | Contract is `ceil(hidden/4)` bytes per row, independent zero padding, reserved code `11` rejected. Representative widths `1,3,4,31,32,33,63,64,65` are tested; not literally every legal width. | **unit-tested** |
| Native dot accumulators are protected from mathematical INT32 overflow within the public contract. | checked C ABI + `deploy/torch_kernel.py`; M02 state log | Width is limited to `floor(INT32_MAX/128)=16,777,215`. The bound is enforced analytically; the maximum-size tensor is not allocated in CI. | **implemented** |
| Public/native extension inputs fail loudly for invalid CPU device, dtype, rank/shape, contiguity, lengths, shifts, multipliers, packed codes/padding, or active indices. | `deploy/torch_kernel.py`, `deploy/cpp_sparse_kernel/extension.cpp`, checked C ABI, `tests/test_kernel_contract.py` | Raw C pointers cannot prove allocation/device truth beyond their typed pointer and explicit length arguments; Python/PyTorch boundaries enforce CPU/device/contiguity. | **unit-tested** |
| SPECTRA has a real scalar native fallback when AVX2 is unavailable. | `deploy/torch_kernel.py`, root/native `setup.py`, `CMakeLists.txt`; M02 forced `SPECTRA_AVX2=OFF` build | M02 proves forced scalar build on x86 and AVX2 build on the CI x86 host. Non-x86/ARM and Windows execution are not tested. | **unit-tested** |
| The optional PyTorch C++ extension builds in the current resolved environment. | `setup.py`, `deploy/cpp_sparse_kernel/extension.cpp`, M02 run `34075718044` | PyTorch 2.14.0+cpu, C++20, AVX2 host in recorded run. Backend build errors are surfaced instead of silently accepted. | **unit-tested** |
| Sparse native execution skips inactive rows and defines zero/duplicate active-index behavior. | checked sparse kernel + `tests/test_kernel_contract.py` | Zero active indices: no writes. Frozen raw-C rows untouched. Duplicate indices: deterministic recompute/overwrite, no accumulation. | **unit-tested** |
| Weight-stationary kernel can decode a weight row once and reuse it over precomputed `K` input vectors. | `scripts/bench_kernel.py`, native weight-stationary path, `tests/test_kernel.py` | Inputs are precomputed; this does not prove end-to-end causal recursive reuse. | **implemented** |
| Published `K`-reuse throughput/amortization numbers are re-certified by M02. | README + benchmark scripts | M02 only ran a small correctness-regression microbenchmark, not the target-hardware publication benchmark. | **hypothesis** |
| Published AVX2/SIMD, sparse, cache-residency, GOP/s, and physical-performance headline numbers remain established after the correctness fix. | README, `scripts/bench_kernel.py`, `scripts/bench_sparse.py`, `scripts/bench_cache.py` | M02 found a correctness fix has a measured CI microbenchmark cost (~6.1% packed, ~18.3% decoded vs buggy M01 path). Target-hardware headline figures were not re-run. | **hypothesis** |
| A small packed model footprint / ternary storage advantage is implemented. | `deploy/pack_ternary.py`, export tests | Exact headline MB depends on the concrete model/export artifact. | **unit-tested** |
| Latent MCTS can search without decoding every latent node. | `eval/latent_mcts.py`, `tests/test_latent_mcts_native.py` | Plumbing behavior on tested configs; does not establish task accuracy. | **unit-tested** |
| Ensemble/uncertainty/LCB verifier mechanics are implemented for search. | `model/energy.py`, `eval/latent_mcts.py`, verifier/MCTS tests | Mechanics, not calibration against true task outcomes. | **unit-tested** |
| VQ snapping projects latents to a codebook and can prevent blow-up in the repository's constructed recurrence. | `model/latent_vq.py`, `tests/test_death_traps.py` | Constructed 30-step synthetic expansive recurrence. | **unit-tested** |
| VQ guarantees bounded task-relevant error/topology for arbitrarily deep real SPECTRA trajectories. | `model/latent_vq.py`, README | No trained real-trajectory proof/test at README strength. | **hypothesis** |
| Adaptive halting selects an earlier generated answer according to a policy/budget. | `model/halting.py`, `tests/test_halting.py` | Current helper first executes full recursion and then postprocesses the selected step. | **unit-tested** |
| Current halting saves real recursive compute/wall time/energy through execution-time early exit. | `model/halting.py` | Not established: the current helper has already run the recursion before selecting the output. | **hypothesis** |
| Parameter/scaling evaluation demonstrates a trained SPECTRA scaling law. | `scripts/eval_scaling_laws.py`, `eval/scaling.py`, `tests/test_scaling_laws.py` | Inspected path constructs fresh models and does not load/train checkpoints before accuracy evaluation. | **hypothesis** |
| Null-hypothesis comparisons establish a scientific advantage over a System-1 baseline. | scaling/null-hypothesis paths, `tests/test_null_hypothesis.py` | Fresh/random model construction tests plumbing, not learned comparative capability. | **hypothesis** |
| Verifier backup values can be propagated and distilled mechanically. | `eval/latent_mcts.py`, `train/distill.py`, `tests/test_prm.py` | Backup/distillation mechanics only. | **unit-tested** |
| Verifier backups are grounded estimates of true task quality and drive real self-improvement. | verifier/MCTS/PRM paths | Current tests use verifier/search-produced targets without external task grounding. | **hypothesis** |
| RAPL energy measurement code degrades safely when counters are unavailable. | `eval/edge_energy.py`, `tests/test_energy_measurement.py`, M01 audit | M01 host had no readable counters and returned `None` as designed. | **unit-tested** |
| README Joule/energy frontier is measured by M01/M02 CI. | README + energy harness | Neither milestone had readable RAPL counters/target-device energy execution. | **hypothesis** |
| “Every figure is measured on physical hardware / nothing is illustrative” is independently auditable from the current commit. | benchmark + render scripts | Complete raw measurement/provenance needed to regenerate every headline figure is still not established. | **hypothesis** |

## Current evidence boundary

After M02, it is defensible to state that the native kernel has an explicit full-INT8 numerical contract, representative vector-tail widths are validated, malformed native inputs fail visibly, and both AVX2 and scalar x86 builds are real. It is **not** defensible to use M02 as evidence for the broader README performance, energy, trained-scaling, end-to-end reuse, arbitrary-depth VQ, compute-saving halting, or grounded self-improvement claims.
