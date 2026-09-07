# Milestone 02 Acceptance Gate

**Decision: PASS**

Milestone 02 is accepted on `research/m02-native-contract` for native-kernel correctness and input contracts only. This page records the strict acceptance criteria and the executable evidence for them.

## Accepted evidence

- Accepted code/test head: `e230f7136a5c4098b47608d123954c11cd4f1339`
- GitHub Actions run: [34076542106](https://github.com/tugrapaydiner/SPECTRA/actions/runs/34076542106)
- Focused native/contract/acceptance tests: **22 passed in 31.97 s**
- Full fast suite: **169 passed, 16 slow tests deselected in 34.58 s**
- Native extension backend in the accepted run: **AVX2**
- Forced scalar CMake build: `spectra_compiled_with_avx2() == 0`
- Evidence artifact: [m02-native-contract-evidence / 10002280753](https://github.com/tugrapaydiner/SPECTRA/actions/runs/34076542106/artifacts/10002280753)
- Evidence ZIP SHA-256: `f6a433edd7b5d79a343a503a0e07f97fc04a03acc3f44aa2f09ea7c3c0ace0a8`

## Gate 1 — native paths match the independent reference over the supported contract

**PASS.**

Evidence:

- `tests/test_kernel_contract.py::test_all_int8_values_all_ternary_signs_against_wide_reference` checks every signed INT8 activation value (`-128..127`) against ternary signs `-1/0/+1` in both scalar and AVX2 builds.
- The same test checks both packed sparse and decoded weight-stationary paths.
- Representative widths `1, 3, 4, 31, 32, 33, 63, 64, 65` cover sub-byte row padding, the 32-lane AVX2 boundary, and vector tails.
- `tests/test_m02_acceptance_gate.py::test_acceptance_native_paths_match_independent_reference` adds interacting multi-lane vectors, deliberately places multiple `INT8_MIN * -1` lanes, exercises shifts `0,1,7,15,31,62`, zero/ordinary/saturating multipliers, and compares both native paths to an independent Python wide-integer dot/requantization implementation.
- Saturation, frozen rows, zero-active behavior, duplicate indices, row padding, and malformed input cases remain covered by `tests/test_kernel_contract.py`.

The accepted contract remains the one documented in `docs/RESEARCH_STATE.md`: full signed INT8 activations, ternary weights, non-negative int32 multipliers, shifts `0..62`, independently zero-padded `ceil(width/4)` rows, and dot widths bounded so the mathematical accumulator fits signed INT32.

## Gate 2 — invalid calls are rejected before raw-pointer compute/scatter execution

**PASS.**

There are three layers of evidence:

1. `deploy/torch_kernel.py` performs CPU, dtype, rank, contiguity, dimension, active-index, multiplier, shift, and structural buffer-length validation before `load_extension()` is invoked.
2. `deploy/cpp_sparse_kernel/extension.cpp` independently rechecks CPU/dtype/rank/contiguity/dimensions/lengths/indices before invoking the checked C ABI.
3. `deploy/cpp_sparse_kernel/spectra_kernel.cpp` validates dimensions, shift, exact lengths, null pointers, packed codes/padding, multipliers, and active indices before entering dot-product, requantization, scatter, decoded-row reuse, or FFN compute loops.

Executable acceptance proofs:

- `tests/test_m02_acceptance_gate.py::test_acceptance_public_invalid_calls_never_reach_native_backend` replaces `load_extension()` with a trap and verifies structurally invalid public calls fail before the backend is reached.
- `tests/test_m02_acceptance_gate.py::test_acceptance_invalid_native_calls_do_no_compute_or_scatter` initializes output buffers to a sentinel and verifies invalid raw-C calls return an error while leaving output bytes unchanged in both scalar and AVX2 builds.

Packed-content validation necessarily reads the supplied packed buffer at the checked native boundary; it occurs before any numerical kernel/scatter execution. A raw C caller can still lie about the true allocation behind a pointer/length pair, which remains an explicit unsupported property rather than something C can prove.

## Gate 3 — the reported edge case is a regression test

**PASS.**

`tests/test_kernel_contract.py::test_reported_int8_min_negation_case` permanently freezes the reported case:

- hidden dimension `32`
- activation `x[0] = -128`, all other activation entries zero
- weight `w[0] = -1`, all other weights zero
- multiplier `1`
- shift `7`
- expected output `+1`

It runs in both scalar and AVX2 builds and checks both packed sparse and decoded weight-stationary paths.

The historical audit also reconstructs the M01 kernel and verifies the original failure remains reproducible as evidence: M01 scalar produces `+1`, M01 AVX2 produces `-1`, while both M02 scalar and AVX2 produce `+1`.

## Scope boundary

This acceptance decision applies only to **Milestone 02 native numerical correctness and input contracts**. It does not certify the README's target-hardware throughput/cache/energy headlines, trained scaling, reasoning quality, end-to-end recursive reuse, compute-saving halting, VQ real-task guarantees, or verifier grounding. Those remain separate claims/milestones.
