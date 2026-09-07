# Milestone 06 Acceptance Gate — Controlled Trained Baseline

**Decision: PASS.**

M06 acceptance is defined by evidence, not by winning a comparison. The gate is:

1. at least one real trained checkpoint from the declared experiment,
2. a held-out result table from the frozen test split,
3. retained learning curves,
4. exact reproduction commands,
5. a clear diagnosis of failures or limitations.

**Progress is not conditioned on SPECTRA beating a baseline.** A negative, tied, or weaker result passes this milestone if the experiment is real, reproducible, held out, and honestly diagnosed.

The accepted M06 run satisfies all five requirements. It is **not** a pass for a reasoning-capability, recursion-advantage, ternary-parity, scaling-law, search, energy, or target-hardware claim.

## Research question

On one fixed validated 9×9 Sudoku distribution, does a small floating-point recursive TRM learn measurable held-out task structure, how much of that learning survives a matched W1.58A8 recursive model at full quantization, and how do both compare with a larger simple single-pass baseline under the same data, optimizer, batch, step, and seed protocol?

The full preregistration and the pre-experimental 1.5% parameter-match amendment are in [`M06_PROTOCOL.md`](M06_PROTOCOL.md).

## Frozen experiment

- Task: generated 9×9 Sudoku, unique solutions, 30–35 clues.
- Data seed: `20260907`.
- Split: train `384`, validation `96`, test `128`.
- Primary metric: strict Sudoku `semantic_validity` on the frozen test split.
- Secondary metrics: exact reference match, blank-cell accuracy, all-cell accuracy.
- Optimizer: AdamW, `lr=1e-3`, `weight_decay=0.01`.
- Batch size: `32`.
- Gradient clipping: global norm `1.0`.
- Main target: `200` optimizer steps per model.
- No architecture-specific LR search, no early stopping, no best-checkpoint selection, no test-time search.
- Recursive models use the existing deep-supervision objective; the single-pass baseline uses token cross-entropy because it has no recursion or halting head. This objective difference is an architectural limitation of the comparison.

### Models

| Model | Trainable parameters | Main structure |
|---|---:|---|
| FP recursive | `30,829` | TRM, `dim=48`, 1 block, `n=1`, `T=1`, `N_sup=2`, FP32 |
| W1.58A8 recursive | `30,397` | same recursive dimensions/schedule, ternary projections + A8 recurrent state |
| larger single-pass | `228,394` | `System1Student`, `dim=96`, 2 blocks, FP32; confidence head frozen |

The existing FP and ternary projection implementations differ in bias conventions, giving a `1.401278%` trainable-parameter gap. An architecture-only preflight discovered this before any M06 timing or accuracy result; the preregistered tolerance was transparently amended from `1.0%` to `1.5%` rather than changing either core implementation.

## Timed resource probe and budget decision

Five optimizer steps were timed per model before the main fits. Probe models were discarded and probe accuracy was not inspected.

| Model | 5-step elapsed | seconds / step |
|---|---:|---:|
| FP recursive | `0.256232 s` | `0.0512465` |
| W1.58A8 recursive | `0.231664 s` | `0.0463327` |
| larger single-pass | `0.238884 s` | `0.0477768` |

Slowest observed probe rate: `0.0512465 s/step`.

The preregistered timing rule therefore selected the full repeated-seed branch:

- seeds: `1101`, `2202`
- `200` steps per model per seed
- six main fits total
- timing-derived estimated main-training time: `61.4958 s`
- recorded main-training wall budget: `1080 s`
- actual summed model-training time: `52.9538 s`

No result-dependent budget adjustment occurred.

## Held-out result table

**Every model produced zero valid solved boards on the 128-example frozen test split at both seeds.**

| Model | Seeds | Semantic solve rate | Exact reference match | Blank-cell accuracy | All-cell accuracy |
|---|---:|---:|---:|---:|---:|
| FP recursive | 2 | `0.0000` | `0.0000` | `0.15925 ± 0.00065` | `0.49894 ± 0.00039` |
| W1.58A8 recursive | 2 | `0.0000` | `0.0000` | `0.11458 ± 0.00453` | `0.47232 ± 0.00270` |
| larger single-pass | 2 | `0.0000` | `0.0000` | `0.32821 ± 0.01570` | `0.59963 ± 0.00936` |

The `±` values above are population standard deviations across the two preregistered seeds, not confidence intervals.

### Per-seed secondary results

| Model / seed | Blank-cell accuracy | All-cell accuracy | Semantic solve | Exact match |
|---|---:|---:|---:|---:|
| FP recursive / 1101 | `0.159896` | `0.499325` | `0` | `0` |
| FP recursive / 2202 | `0.158602` | `0.498553` | `0` | `0` |
| W1.58A8 recursive / 1101 | `0.110050` | `0.469618` | `0` | `0` |
| W1.58A8 recursive / 2202 | `0.119113` | `0.475019` | `0` | `0` |
| single-pass / 1101 | `0.312510` | `0.590278` | `0` | `0` |
| single-pass / 2202 | `0.343907` | `0.608989` | `0` | `0` |

## Real trained checkpoints

The accepted artifact contains all six final checkpoints:

```text
.m06/checkpoints/fp_recursive_seed1101.pt
.m06/checkpoints/fp_recursive_seed2202.pt
.m06/checkpoints/ternary_recursive_seed1101.pt
.m06/checkpoints/ternary_recursive_seed2202.pt
.m06/checkpoints/single_pass_seed1101.pt
.m06/checkpoints/single_pass_seed2202.pt
```

These are trained final-step checkpoints, not random-init or smoke fixtures. The test predictions were generated only after final checkpoints were written and reloaded.

## Learning curves and optimization diagnosis

The artifact retains the complete per-step JSONL learning curves for every fit:

```text
.m06/curves/fp_recursive_seed1101.jsonl
.m06/curves/fp_recursive_seed2202.jsonl
.m06/curves/ternary_recursive_seed1101.jsonl
.m06/curves/ternary_recursive_seed2202.jsonl
.m06/curves/single_pass_seed1101.jsonl
.m06/curves/single_pass_seed2202.jsonl
```

Zero exact solves did **not** automatically mean optimization failure. The preregistered failure rule required less than 5% training-loss improvement or zero/non-finite gradients.

All six fits cleared that rule:

| Model / seed | Initial-window loss | Final-window loss | Relative improvement | Max pre-clip grad norm | Train time |
|---|---:|---:|---:|---:|---:|
| FP recursive / 1101 | `2.39634` | `1.29827` | `45.82%` | `1.2651` | `8.8775 s` |
| FP recursive / 2202 | `2.59600` | `1.29393` | `50.16%` | `1.4984` | `8.8840 s` |
| W1.58A8 recursive / 1101 | `2.48694` | `1.33376` | `46.37%` | `1.3595` | `8.8997 s` |
| W1.58A8 recursive / 2202 | `2.38910` | `1.32407` | `44.58%` | `1.1203` | `8.9933 s` |
| single-pass / 1101 | `2.08138` | `0.76017` | `63.48%` | `1.1844` | `8.6971 s` |
| single-pass / 2202 | `1.98616` | `0.74136` | `62.67%` | `0.9740` | `8.6022 s` |

Therefore the M06 gradient/overfit/minimal-4×4 failure branch was **not triggered**. 4×4 Sudoku was not used in the main experiment or as a post-hoc retuning surface.

### Failure diagnosis

The observed failure is **task-level capability failure at this pilot budget**, not a localized optimization failure:

- every model had zero strict held-out solves;
- gradients remained finite and nonzero;
- training loss decreased strongly for all six fits;
- secondary cell accuracy shows partial structure was learned;
- the larger single-pass baseline learned substantially more partial structure than either small recursive model;
- the W1.58A8 recursive model was weaker than the FP recursive model on both secondary metrics in both seeds.

This diagnoses what M06 actually establishes: the training pipelines learn, but this small data/step regime is insufficient for complete 9×9-board capability. It does not justify claiming that recursion is intrinsically worse, that ternary reasoning cannot work, or that the baseline architecture is universally superior.

## Exact reproduction commands

From the repository root, the accepted experiment entry point is deterministic with the M06 protocol constants embedded in the runner.

Install the same dependency families used by the accepted CPU CI run:

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.4"
python -m pip install -r requirements.txt
```

Run the exact M06 experiment command used by the accepted workflow:

```bash
mkdir -p .m06
timeout 1800s python scripts/m06_protocol_runner.py --out .m06 2>&1 | tee .m06/run.log
```

Run the post-experiment regression gate:

```bash
timeout 600s python -m pytest -m "not slow" -ra 2>&1 | tee .m06/pytest_fast.txt
```

The GitHub Actions definition that executed these commands is `.github/workflows/m06-controlled-baseline.yml`. The accepted run is `34124798931` at experimental head `9fd96e72b9542c33c4ae0639d44847f104a83df8`.

## Declared ternary inference state

The W1.58A8 quantization-strength warmup was `50` steps. Each final ternary checkpoint was reloaded before test evaluation. The evidence gate required every `FakeBitLinear.quant_strength` buffer to equal exactly `1.0`.

Both seeds passed for all seven ternary modules:

```text
blocks.0.attn.q    = 1.0
blocks.0.attn.k    = 1.0
blocks.0.attn.v    = 1.0
blocks.0.attn.proj = 1.0
blocks.0.ff.0      = 1.0
blocks.0.ff.2      = 1.0
out_head           = 1.0
```

The workflow evidence verifier reported `quantization_strength_check=pass`.

Final ternary distributions were non-degenerate:

- seed 1101: negative `0.36156`, positive `0.35904`, zero `0.27940`
- seed 2202: negative `0.36032`, positive `0.36121`, zero `0.27848`

This establishes the declared evaluation representation. It does not establish accuracy parity with the FP model.

## Training-compute accounting

Each run sampled `6,400` training examples (`200 × 32`). The recursive models execute four shared-block applications per example under `N_sup=2, T=1, n=1`; the single-pass model executes two blocks once.

Per run:

- FP recursive: `25,600` recorded block applications sampled
- W1.58A8 recursive: `25,600`
- larger single-pass: `12,800`

These are structural compute counters, not FLOP-equated or joule-equated costs. Physical energy was not measured on the GitHub runner.

## Environment

Accepted run environment:

- Ubuntu `24.04.4` hosted runner
- AMD EPYC 7763
- x86_64, 4 logical CPUs exposed
- Python `3.11.16`
- PyTorch `2.14.0+cpu`
- NumPy `2.4.6`
- CUDA unavailable
- experiment device: CPU
- PyTorch threads: `2`

This is controlled CI evidence, not target-edge-hardware benchmarking.

## Interpretation

M06 provides **no evidence that recursion substitutes for parameter count on this task/budget**. The deliberately larger single-pass baseline was materially better on the partial-cell metrics while all three models remained at zero strict solves.

M06 also provides **no evidence that W1.58A8 preserves the floating-point recursive pilot's quality at this budget**. The full-strength ternary recursive model had lower blank-cell and all-cell accuracy than the matched FP recursive model in both seeds.

These observations do not establish the reverse universal claims either. The comparison is a small generated-data pilot with only two seeds, a much larger single-pass model, different architectural objectives (deep supervision versus single-pass CE), and existing FP/ternary projection-bias differences. It therefore cannot prove that recursion is intrinsically worse or that ternary reasoning cannot work.

The important positive result is narrower: all three optimization paths learned nontrivial token/cell structure without collapse, the controlled experiment/evidence machinery works, and the project now has an honest trained baseline against which later hypotheses can be tested.

## Accepted evidence

Authoritative accepted execution:

- branch head: `9fd96e72b9542c33c4ae0639d44847f104a83df8`
- Actions run: `34124798931`
- job: `101750849574`
- artifact: `m06-controlled-baseline-evidence`
- artifact id: `10019726497`
- artifact ZIP SHA-256: `63c533e96fba488baac7bb8a44052dd1408a342b21a6313cd8c3c7d1e6b5acb8`
- artifact size: `6,697,316` bytes
- retention: 14 days
- experiment exit: `0`
- evidence verifier exit: `0`
- fast regression exit: `0`
- fast regression: **204 passed, 16 deselected in 59.98 s**

Key retained file hashes:

```text
summary.json          1a3041d541aee77d50e65f9d4314ff70ba1fe81ccd418a15959adcc55db802aa
data_manifest.json    bbe21422059fb901d9bb53503110c33250c910e780fade7a9511fc26634237f1
timing_probe.json     f17a7c51e44fb506ab698fd51a08b740ebb01e74bfd13bb5987e55fb9f179194
budget_decision.json  4508b46fc2397a69f02b9b974a2c8032eda2474a275fbb1afd6dfb844bb8ab39
parameter_audit.json  edea84e4ad13f1b9b2d873bdb88fd46ee578d9af03250378d93da103e8d49e9a
environment.json      6548bcd5c2abc219a941217ae8d45cb4a4433aee206f72ca476d3c36089370cf
```

The artifact also retains all six checkpoints, all six JSONL learning curves, all six 128-example prediction files, failures, diagnostics, raw run logs, environment/install metadata, and per-file SHA-256 records.

### Pre-acceptance iterations

Two pre-experimental failures are retained as process evidence, not result evidence:

1. Run `34124091722` stopped at architecture-only preflight because the original 1.0% recursive parameter tolerance was slightly too strict for the repository's existing bias conventions. No timing probe or accuracy ran. The protocol was amended to 1.5% before experimental execution.
2. Run `34124576768` passed the amended architecture check but failed on a direct-script import-path bug before the timing probe or accuracy ran. The entry point was fixed without changing the experiment protocol.

Neither failed run exposed an M06 performance result.

## M06 decision

M06 satisfies the user-defined acceptance gate: **real trained checkpoints, a frozen held-out result table, retained learning curves, exact commands, and a clear failure diagnosis are all present.**

The fact that SPECTRA did not beat the larger single-pass baseline is an experimental result, not a milestone failure. Likewise, zero strict solves are an observed limitation, not a reason to retroactively fail this gate.

Future milestones may choose to improve strict task capability, add a size/compute-controlled comparator, or test new mechanisms under separately preregistered contracts. Those are scientific next steps, **not conditions for M06 acceptance or permission to progress**.

**Stop here for M06.**
