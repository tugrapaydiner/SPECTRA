# SPECTRA Research State

## Milestone 01 — trustworthy baseline

**Stage status:** COMPLETE, with explicit environment/provenance limitations below.

Milestone 01 establishes a reproducible, bounded baseline only. No substantive model, search, training, or evaluation algorithm was changed.

### Repository state

- Repository: `tugrapaydiner/SPECTRA`
- Source baseline (`main` when Milestone 01 began): `ece509593198dd780d71969bc7749556d40b2e7e`
- Working branch: `research/m01-baseline`
- Baseline code/infrastructure commit exercised by the final clean audit: `7a1b8ba6a6796ab2581315784764d4426022b42d`
- Final clean workflow run: GitHub Actions run `34073485472`
- Evidence artifact: `m01-baseline-evidence`, artifact id `10001279309`
- Evidence ZIP SHA-256 reported by Actions: `005d197a2d0c6fffb538282acc3461dccdf1c61143cc511f8928091c955b832b`
- Artifact retention configured by the audit workflow: 14 days. This document is the persistent summary after the ephemeral artifact expires.

`main` was not modified during M01 until the milestone was accepted and merged.

### Working-tree limitation

The model execution sandbox could not directly clone GitHub (`Could not resolve host: github.com`), so the user's own local checkout and any unpushed/uncommitted working-tree edits were **unavailable for inspection**. They are therefore neither described as clean nor modified by this milestone.

The GitHub Actions checkout used for the reproducible audit **was clean before audit outputs were created**.

### Applicable repository instructions

No `AGENTS.md` or equivalent agent-specific instruction file, and no additional `CONTRIBUTING.md` instruction layer, was found in the inspected repository tree. The applicable baseline instructions were therefore the repository README/build guidance, `requirements.txt`, `pyproject.toml`, `setup.py`, and pytest configuration.

### Minimal setup changes

Only setup/audit compatibility changes were made:

1. Added `.github/workflows/m01-baseline.yml` to make the baseline commands, budgets, host probes, and evidence collection repeatable.
2. Changed the optional PyTorch C++ extension compile standard from C++17 to C++20 in `setup.py`, `deploy/cpp_sparse_kernel/setup.py`, and `deploy/torch_kernel.py` because the then-current unconstrained `torch>=2.4` resolver selected PyTorch 2.14.0+cpu, whose headers reject C++17.

No test was weakened, skipped by modification, or rewritten to accommodate a scientific failure.

### Reproducible environment

Final clean M01 audit host:

- GitHub-hosted runner: Ubuntu 24.04.4 LTS
- Python 3.11.16
- torch 2.14.0+cpu
- numpy 2.4.6
- PyYAML 6.0.3
- einops 0.8.2
- tqdm 4.70.0
- pandas 3.0.5
- psutil 7.2.2
- pytest 9.1.1
- AVX2: supported
- GPU/CUDA: unavailable
- readable RAPL `energy_uj`: unavailable

### M01 exact baseline commands

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.4"
python -m pip install -r requirements.txt
python -m pytest -m "not slow" -ra
python -m pytest tests/test_kernel.py -ra
python setup.py build_ext --inplace
```

A bounded two-optimizer-step CPU training smoke was also run from `.github/workflows/m01-baseline.yml`.

### M01 results

#### Passed

- dependency installation
- repository energy API graceful-unavailable probe
- existing fast test gate: **151 passed**, 16 slow tests deselected
- native kernel correctness: **4 passed**
- optional PyTorch extension build
- tiny CPU training smoke with finite gradients and a parameter update

#### Failed

Final clean baseline: none.

Failures encountered while establishing the baseline and retained in the record:

1. PyTorch 2.14 required C++20 while the repository requested C++17.
2. The first evidence upload excluded hidden `.m01/` files.
3. The first status formatter emitted a spurious timeout line due to shell operator precedence.

#### Skipped

- 16 `slow` pytest items explicitly deselected by `-m "not slow"`.

#### Timed out

None.

#### Unavailable

- user's unpushed local working tree
- GPU/CUDA
- readable RAPL counters / physical Joule measurements
- target-device physical-performance recertification
- complete raw provenance sufficient to independently regenerate all README performance figures

### M01 claim boundaries

M01 established that Python plumbing, tested ternary/INT8 mechanics, AVX2 correctness on the then-tested aligned shapes, extension buildability, synthetic VQ boundedness, MCTS/verifier mechanics, postprocessed halting mechanics, and tiny trainability worked. It did **not** establish the README's broad physical performance/energy numbers, a trained scaling law, end-to-end sequential `K` reuse, arbitrary-depth real-task VQ guarantees, actual compute-saving halting, or grounded verifier self-improvement.

---

## Milestone 02 — native-kernel correctness and input contracts

**Stage status:** COMPLETE; accepted and merged to `main` before M03.

M02 changes only native-kernel correctness, packing/backend/input contracts, native-facing benchmark callers, tests, and audit infrastructure. It does not change the reasoning/search/training algorithms.

### Verified implementation / evidence

- M02 base: M01 merge on `main`, `40745dbe185c069aeee9eff3cf63dd411d9e17da`
- Verified implementation commit: `ba8fa0acf3fe9015235d281be807f40124027cec`
- Green GitHub Actions run: `34075718044`
- Evidence artifact: `m02-native-contract-evidence`, artifact id `10001994671`
- Evidence ZIP SHA-256: `d2367d39eae331ff86a47d2fdbe521963ea6a4c3da1d7e05b9d5dffb6ce82aa9`
- Focused tests: **16 passed in 34.26 s**
- Full fast suite: **163 passed, 16 deselected in 33.05 s**
- PyTorch extension backend in the accepted run: **AVX2**
- Forced CMake scalar build probe: `spectra_compiled_with_avx2() == 0`

### Root cause: `INT8_MIN * -1`

The requested reproduction is real in the M01 kernel:

- hidden dimension: 32
- one output
- activation: `x[0] = -128`, all other activations `0`
- ternary weight: `w[0] = -1`, all other weights `0`
- multiplier: `1`
- requantization shift: `7`

The independent audit compiled the M01 source twice and observed:

| Path | accumulator | requantized output |
|---|---:|---:|
| M01 scalar packed | +128 | +1 |
| M01 scalar decoded | +128 | +1 |
| M01 AVX2 packed | -128 | -1 |
| M01 AVX2 decoded | -128 | -1 |
| M02 scalar packed/decoded | +128 | +1 |
| M02 AVX2 packed/decoded | +128 | +1 |

The M01 AVX2 code used `_mm256_sign_epi8(x, w)`. `VPSIGNB` performs signed **byte** negation. For `x=-128` and `w=-1`, the true product is `+128`, which cannot be represented in signed INT8. The byte-level negate therefore remains `0x80`, interpreted as `-128`. The scalar path first promotes the operands and therefore produces the mathematical `+128`.

### Numerical fix

M02 deliberately keeps the **full INT8 activation contract**; it does not ban `-128`.

Both packed and already-decoded AVX2 dot-product paths retain the fast `VPSIGNB` product for ordinary lanes and detect the unique exceptional lane condition:

```text
activation == -128 AND ternary_weight == -1
```

For each exceptional lane, the byte result `-128` is exactly `256` below the mathematical `+128`, so M02 adds **+256 per exceptional lane** to the widened vector sum. Scalar tails continue to multiply after integer promotion.

An earlier correctness-first M02 prototype widened every activation/weight before multiplication. It was correct but measurably slower, so it was replaced with this exact exceptional-lane correction before M02 completion.

### Native numerical contract

#### Activations

- type: signed INT8
- valid values: **all `[-128, 127]`**
- public PyTorch/native extension tensors: CPU only, rank 2, contiguous

#### Ternary weights and packing

- semantic values: `{-1, 0, +1}`
- codes: `00 -> 0`, `01 -> +1`, `10 -> -1`
- code `11`: reserved/invalid and rejected by checked native entry points
- each output row uses exactly `ceil(hidden_dim / 4)` packed bytes
- rows are padded **independently**
- unused 2-bit codes in the final byte of a row must be zero
- `deploy.pack_ternary.pack_ternary_rows` is the native row-padded producer
- non-multiple-of-four widths are supported through a scalar tail; representative tails are tested

#### Dimensions and accumulator bound

For an INT8 activation and ternary weight, the maximum absolute per-element mathematical product is 128. Native dot widths are therefore restricted to:

```text
1 <= hidden/inter width <= floor(INT32_MAX / 128) = 16,777,215
```

This keeps the mathematical dot-product accumulator inside signed INT32. Token/output dimensions passed through the public extension must be positive and fit the native `int` interface.

The contract limit is enforced; M02 does not allocate/test a 16.7-million-wide tensor in CI.

#### Requantization

- accumulator: checked signed INT32 domain above
- fixed-point multiplier: non-negative signed INT32
- shift: **integer `0..62` inclusive**
- `shift=0` is handled explicitly without a negative/undefined rounding shift
- intermediate `acc * multiplier` and rounding are performed in signed INT64
- final result saturates to `[-128,127]`

The public scale helper implements:

```text
mult[o] = round(weight_scale[o] * act_scale / out_scale * 2**shift)
```

with:

- finite non-negative weight scales
- finite non-negative activation scale
- finite strictly-positive output scale
- rejection if the resulting multiplier exceeds signed INT32

Negative fixed-point multipliers are unsupported by design and rejected.

#### Active indices / sparse semantics

- every active index must satisfy `0 <= index < num_tokens`
- out-of-range indices are rejected before native work begins
- zero active indices is valid and performs no writes
- frozen/unselected rows in the raw C API are untouched
- the public PyTorch extension allocates a zero output, so unselected rows are zero there
- **duplicate indices are valid**: each duplicate deterministically recomputes and overwrites the same output row; there is no scatter-add/accumulation behavior

#### Buffer lengths

The checked C ABI receives explicit element counts for every raw buffer and requires exact expected lengths. Examples:

```text
X elements       = num_tokens * hidden
W packed bytes   = out_dim * ceil(hidden/4)
requant entries  = out_dim
Y elements       = num_tokens * out_dim
```

The fused FFN applies the corresponding exact checks to both layers.

Raw C pointers cannot prove the actual allocation size behind a dishonest pointer/length pair and do not encode a device concept. CPU/device/dtype/rank/contiguity are therefore enforced by both the public Python wrapper and the PyTorch C++ extension boundary; the raw C ABI independently rechecks dimensions, lengths, packed contents, multipliers, shifts, indices, null pointers, and allocation failure.

### Backend contract

- `deploy/torch_kernel.py` detects host AVX2 conservatively.
- On a supported x86 host the JIT/setup extension builds the AVX2 implementation.
- On a host without AVX2 it builds the same source without `-mavx2`, activating the real scalar implementation.
- CMake exposes `SPECTRA_AVX2=AUTO|ON|OFF`; M02 CI explicitly builds `OFF` and confirms the library reports scalar.
- Native/JIT build failures are raised with the backend/flags and original exception rather than silently treated as success.
- The ctypes adapter refuses to load a library compiled for AVX2 on a CPU that does not advertise AVX2.

### M02 test coverage

`tests/test_kernel.py` plus `tests/test_kernel_contract.py` exercise:

- exact requested `-128 * -1` reproduction/fix in scalar and AVX2
- both packed and decoded dot-product paths
- **every one of the 256 INT8 activation values** against all ternary signs `-1/0/+1`
- independent wide-integer Python reference
- representative widths `1, 3, 4, 31, 32, 33, 63, 64, 65`
- vector and scalar-tail boundaries
- positive and negative saturation
- zero active indices
- frozen tokens
- duplicate active indices
- invalid shifts
- negative multipliers
- active-index bounds
- off-by-one raw buffer lengths
- reserved ternary code `11`
- nonzero row-padding bits
- row-packed producer round-trip
- public Python CPU/dtype/rank/contiguity/length checks
- scale conversion checks
- direct PyTorch extension invalid-input checks
- backend detection and visible loader failure
- fused FFN correctness with non-aligned widths
- weight-stationary decoded correctness with non-aligned widths

### Before/after microbenchmark

This is a small CI-host microbenchmark of the internal dot kernels only; it is **not** a publication or target-device performance claim. Configuration: hidden width 512, 32 input rows, 20,000 dot calls, five rounds, median ns/dot.

| Path | M01 buggy AVX2 | M02 corrected AVX2 | ratio |
|---|---:|---:|---:|
| packed dot | 54.01875 ns | 57.33455 ns | **1.0614x** (+6.1%) |
| decoded dot | 23.8339 ns | 28.19465 ns | **1.1830x** (+18.3%) |

The checksums differ because the M01 AVX2 implementation computes incorrect values whenever the random benchmark contains `-128 * -1`; the after checksum reflects corrected arithmetic. The timing comparison is retained only to expose the cost of correctness on that workload.

### M02 failures/iterations retained explicitly

No final M02 check failed, but several issues were found while establishing the gate and remain part of the record:

1. A test that constructed reserved code `11` used Python `~0x3` against NumPy `uint8`; NumPy 2.4 rejects the negative mask. The test was corrected to unsigned `0xFC`. Production code was unchanged.
2. An extension smoke probe attempted to import the built PyTorch extension before importing PyTorch, so `libc10.so` was not loaded. The probe now imports `torch` first; the extension build itself had succeeded.
3. The first optimized exceptional-lane rewrite accidentally supplied 30 arguments to a 32-byte AVX2 LUT initializer. GCC rejected the AVX2 build. The LUT is now defined once as an exact 16-byte table and broadcast to both AVX2 lanes.
4. The audit initially captured compiler stderr and surfaced only `CalledProcessError`. It now prints the complete failed compiler command/stdout/stderr before raising.
5. A full-widen arithmetic prototype was correct but caused a larger measured regression; it was replaced by the exact `+256` exceptional-lane correction. This optimization did not relax the tests or the full INT8 contract.

### Remaining unsupported / unproven cases

- Windows/MSVC execution was not tested by the M02 GitHub Actions gate.
- A forced scalar build was tested on x86; non-x86/ARM compilation and execution were not exercised.
- The maximum allowed width (`16,777,215`) is contract-checked but not allocated/executed in CI.
- Raw C callers can lie about pointer allocation size; native code can validate only the explicit lengths it is given.
- Negative requantization multipliers/scales are intentionally unsupported.
- M02 does not re-certify the README's historical throughput/cache/energy headline measurements on target hardware. The small before/after microbenchmark is only a regression check.

### M02 decision

The M01 known AVX2 shape/numerical boundary is closed for the documented M02 contract: full INT8 values are supported, representative vector tails are tested, row padding is explicit, invalid inputs fail loudly, and a scalar backend is real and testable.

**Next action:** stop here for M02. Do not treat this milestone as evidence for broader system-level performance, energy, reasoning-quality, or trained-scaling claims; those remain separate later milestones.

---

## Milestone 03 — trustworthy task/data/evaluation contracts

**Stage status:** COMPLETE ON `research/m03-task-data-eval`; not merged at the time of this entry.

M03 changes task/data construction, symbolic validation, task metrics, split/manifests, retained task configuration, tests, and audit infrastructure only. It does **not** change SPECTRA's reasoning/search/training algorithms.

### Verified implementation / evidence

- M03 base: accepted M02 merge on `main`, `e0781ec8b4e649ab4ccd48d4cd5f432a9b88d249`
- First complete green implementation commit: `03c4a8070bd250f4c7ff17a5eb98073293ce8c67`
- Green implementation workflow run: `34077748173`
- Evidence artifact: `m03-task-data-evidence`, artifact id `10002690239`
- Evidence ZIP SHA-256: `60ec3325d84c529c779f9ac32eafec053c61fbca7b2a053ded81b15b2cf74c20`
- Evidence artifact size: 17,500 bytes; retention: 14 days
- Focused M03/data/verifier gate at the recorded run: **42 passed, 3 deselected in 1.31 s**
- Full fast suite at the recorded run: **182 passed, 16 deselected in 64.00 s**
- Manifest reference seed: `314159`
- Manifest reference sizes for each retained task: train 8 / validation 4 / test 4
- Generator version: `spectra-m03-data-v1`
- A subsequent parity refinement also made the tensor Sudoku validator reject non-integral tensor dtypes to match the NumPy contract and added `tests/test_m03_validator_parity.py`; final branch CI must remain green before merge.

### Task-construction root cause and explicit retained contracts

M03 removed the old `scripts/_common.py` shortcut that effectively behaved as **Sudoku vs. everything-else-is-maze**. The shortcut passed maze-shaped kwargs to every non-Sudoku task. This silently left ARC-style construction at generator defaults even though `config/arc.yaml` declared a different shape.

Each retained task now resolves through `data/task_contracts.py` before construction. The executable task contract validates shape, vocabulary, spatial dimensions, sequence length, padding/mask policy, task-specific generator parameters, and benchmark scope.

Retained local configurations are:

| Task | Retained M03 contract | Benchmark scope |
|---|---|---|
| Sudoku | 9×9, seq 81, vocabulary 0..9, box 3, 30–50 requested clues, uniqueness on, randomized completion | generated local Sudoku |
| Maze | 15×15, seq 225, tokens 0=wall/1=open/2=start/3=goal/4=path, shortest-path optimality required | synthetic perfect-maze path overlay |
| ARC-style | 30×30 padded canvas, seq 900, pad 10, local `flip_h`, source max 6×6 | **synthetic local variant; NOT official ARC/ARC-AGI** |
| BabyAI-style | 8×8, seq 64, exact local 8-token vocabulary, wall probability 0.2 | **synthetic local one-step variant; NOT official BabyAI** |
| SmartHome | 1×16, seq 16, exact local 17-token vocabulary, pad 0 | synthetic deterministic policy snapshot |

`GridDataset` now rejects generator/config disagreements in flattened length or declared vocabulary and carries input/target content masks. ARC-style masks exclude pad token 10. SmartHome input masks exclude PAD while the repeated action target remains fully evaluated.

### Sudoku validation and generation contract

The NumPy validator in `data/sudoku.py` now requires:

- exact `(N,N)` shape implied by `box`
- integral NumPy representation
- partial-grid domain `0..N`
- completed-grid domain `1..N`
- no repeated nonzero digit in any row, column, or box

Malformed or contradictory puzzles return invalid; `count_solutions` returns 0 and `solve` returns `None` rather than entering solver work with contradictory masks.

`model/verifier.py` implements the corresponding tensor partial-grid semantics and now requires integer tensor dtypes as well. M03 adversarial/parity tests cross-check NumPy and tensor validity on valid puzzles and contradictory clues and include an explicit dtype-parity regression.

Sudoku completed-board generation is no longer restricted to row/column/digit/transposition transformations of one canonical completed solution. The retained M03 config uses randomized MRV/backtracking **from an empty board**. The legacy canonical-symmetry construction remains available only through an explicit `solution_method: canonical_symmetry` experiment.

This broadens the construction process but does **not** establish uniform sampling over all Sudoku solutions or equivalence classes, and does not make a real-world Sudoku distribution/generalization claim.

### Maze semantic-success contract

`data/maze.py::candidate_success` is independent of the stored reference target. A successful candidate must:

- have the declared shape/token domain
- preserve exactly one declared start and one goal
- preserve walls exactly
- place path cells only on originally open cells
- form one connected simple 4-neighbour start-to-goal route
- have endpoint degree 1 and interior route degree 2 (no branches/disconnected path components)
- satisfy BFS shortest-path length when `require_optimal` is true

The retained maze config requires optimality. Copying a non-trivial unsolved input does **not** count as success. Adversarial tests cover unsolved copies, wall crossings, branches, longer-but-valid routes when optimality is disabled, and rejection of those longer routes when optimality is required.

### Primary task metrics

`eval/metrics.py` separates:

1. `exact_reference_match`: candidate equals the retained target cell-for-cell.
2. `semantic_validity`: strict task success independent of target identity where a complete symbolic checker exists (currently Sudoku and maze).
3. `blank_cell_accuracy`: Sudoku accuracy only on cells blank in the input.
4. `content_cell_accuracy`: padding-aware content accuracy for the local ARC-style padded task.

These metrics are intentionally not interchangeable. M03 includes a 4×4 Sudoku regression where a candidate can be semantically valid while differing from a separate valid reference solution, so semantic validity is 1 while exact reference match is 0.

No official benchmark semantic metric is claimed for the synthetic local ARC-style or BabyAI-style generators.

### Reproducible grouped split/manifests contract

`data/splits.py` and `scripts/build_data_manifests.py` now provide reproducible train/validation/test manifests with:

- one root integer seed
- NumPy `SeedSequence.spawn(6)` with **separate generation and augmentation child streams for train, validation, and test**
- stable example IDs
- a base `group_id` computed from the **unaugmented** input/target pair before augmentation
- augmentation applied only after split ownership is fixed
- cross-split regenerated base-group rejection
- cross-split exact-fingerprint audit
- generator version, task scope, generator kwargs, RNG spawn keys, split counts, stable IDs, and per-example/task-specific difficulty metadata

SmartHome is a special finite-universe case: all 64 exact binary states are deterministically partitioned into disjoint 52/6/6 train/validation/test state pools before sampling. Repeats may occur **within** a split when sampling from its assigned finite pool; that is reported rather than hidden. Cross-split state/group overlap remains forbidden.

### M03 reference manifests and duplicate audit

The retained evidence artifact contains `arc.json`, `babyai.json`, `maze.json`, `smarthome.json`, and `sudoku.json`, each generated at seed `314159`, sizes 8/4/4.

For **all five** reference manifests:

- train/validation group overlap: 0
- train/test group overlap: 0
- validation/test group overlap: 0
- train/validation exact overlap: 0
- train/test exact overlap: 0
- validation/test exact overlap: 0

Within-split exact/group duplicates in this small reference run:

- Sudoku: 0 / 0 / 0 for train/validation/test
- Maze: 0 / 0 / 0
- ARC-style: 0 / 0 / 0
- BabyAI-style: 0 / 0 / 0
- SmartHome: 2 / 1 / 2 repeated examples/groups in train/validation/test, reflecting finite split-specific state sampling; this does not cross split boundaries

### Recorded reference difficulty distributions

The manifests persist full counts/min/max/mean. Selected ranges from the retained 8/4/4 reference run:

- Sudoku clues: train 30–50, validation 31–45, test 30–48; corresponding blanks are recorded.
- Maze shortest-path length: train 33–57, validation 37–49, test 49–77. Perfect-maze wall fraction was 0.568888… for all reference samples at 15×15.
- ARC-style source grids: source height/width are recorded per split within the declared 2..6 generated range.
- BabyAI-style: wall counts and whether the one-step agent moved are recorded; reference wall counts ranged 8–18 across splits.
- SmartHome: action-token and non-pad sensor-token distributions are recorded; every encoded state contains six non-pad feature tokens.

These are descriptive distributions of the small reference manifests, not claims that the splits are population-matched or IID.

### M03 failures/iterations retained explicitly

No final accepted M03 check may be described as passing until the final branch gate is green. During construction the following failures were exposed and fixed without weakening tests:

1. Initial split augmentation passed `height/width` both positionally and through task kwargs, causing Python argument collisions for retained configs. The split/build layers now remove duplicate spatial kwargs before augmentation.
2. The direct legacy ARC builder inherited the generator's default pad token 10, but the first M03 mask implementation required an explicit pad token. The builder now inherits the same historical default while config-driven construction remains explicit.
3. The direct manifest CLI initially lacked the repository root on `sys.path`; it now mirrors the repository's other directly runnable script entry points.
4. A final validator parity review found NumPy Sudoku rejected floating representations while the tensor validator could accept integral-valued floats. The tensor validator now requires an integer dtype and has a dedicated regression test.

### Remaining limitations / unsupported claims

- The local ARC-style generator is **not** ARC/ARC-AGI and has no official ARC dataset/evaluation integration.
- The local BabyAI-style generator is **not** the official BabyAI environment/benchmark and evaluates only a local one-step next-state construction.
- The split mechanism prevents observed base/augmentation leakage and audits exact/group duplicates; it does not prove IID sampling, matched population difficulty, absence of semantic near-duplicates, or external benchmark generalization.
- Sudoku randomized backtracking is not proven uniform over completed solutions/equivalence classes.
- Uniqueness is enforced for the retained Sudoku puzzle maker but the manifest difficulty field is clue/blanks, not a calibrated human difficulty rating.
- Maze difficulty records path length/wall fraction; no human/navigation difficulty calibration is claimed.
- SmartHome has only 64 exact binary states; large datasets necessarily reuse states within their split-specific pool unless the experiment explicitly caps examples to unique states.
- M03 does not establish trained task accuracy, search advantage, scaling behavior, energy savings, or published hardware performance claims.

### M03 decision

M03 closes the task/data/evaluation **contract** layer for the retained local experiments: construction is explicit, malformed Sudoku and false maze successes are rejected, primary metrics have distinct meanings, and split/manifests are reproducible and leakage-audited.

**Next action:** stop here for M03. Do not expand this milestone into training/scaling/performance claims. Merge only after the final branch CI including the dtype-parity regression is green and the user accepts the milestone.
