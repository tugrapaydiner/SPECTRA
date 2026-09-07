# SPECTRA Research State

This is the live milestone register. Detailed historical state is preserved without rewriting prior evidence:

- M01–M04: [`RESEARCH_STATE_M01_M04.md`](RESEARCH_STATE_M01_M04.md)
- complete pre-M06 live register, including M05 in full: [`RESEARCH_STATE_M05.md`](RESEARCH_STATE_M05.md), archived from blob `08ed850a4e9b58b9c4b58367c3c15d79a4c0decc`

Git history retains the original cumulative registers. The live register below records the accepted milestone index and M06 in full.

## Accepted milestone index

| Milestone | Scope | Accepted / merged state |
|---|---|---|
| M01 | trustworthy baseline | accepted and merged; `main` merge `40745dbe185c069aeee9eff3cf63dd411d9e17da` |
| M02 | native-kernel correctness and input contracts | accepted and merged; `main` merge `e0781ec8b4e649ab4ccd48d4cd5f432a9b88d249` |
| M03 | trustworthy task/data/evaluation contracts | accepted and merged; `main` merge `01638b10777029fb28bb35229e374f0865c6e5d4` |
| M04 | reproducible training and checkpoint state | accepted and merged through PR #4; `main` merge `370caf708755e1c68c59d5696778597f0290ea68` |
| M05 | checkpoint-backed evaluation | accepted and merged through PR #5; `main` merge `1003c59e17dc17e652438317b7480c9e898379af` |
| M06 | controlled trained baseline experiment | **COMPLETE ON `research/m06-capability-gates`; accepted evidence run `34124798931`; not merged at the time of this entry** |

---

## Milestone 06 — controlled trained baseline experiment

**Stage status:** COMPLETE ON `research/m06-capability-gates`; accepted experiment/evidence gate is green; not merged at the time of this entry.

M06 intentionally does one thing only: it establishes a controlled trained baseline experiment on a non-debug task. It does **not** add or evaluate MCTS, routing, halting-policy training, a process reward model, distillation, a self-improvement flywheel, energy optimization, scaling-law machinery, or target-hardware performance.

The preregistered protocol is [`M06_PROTOCOL.md`](M06_PROTOCOL.md). The concise accepted evidence surface is [`M06_ACCEPTANCE_GATE.md`](M06_ACCEPTANCE_GATE.md).

### Research question and primary endpoint

On one fixed validated 9×9 Sudoku distribution, does a small floating-point recursive TRM learn measurable held-out task structure, how much of that learning survives a matched W1.58A8 recursive model evaluated at full quantization strength, and how do both compare with a deliberately larger simple single-pass baseline under a fixed data/training protocol?

The primary endpoint was fixed before training:

```text
strict Sudoku semantic solve rate (`semantic_validity`)
```

Exact-reference match, blank-cell accuracy, all-cell accuracy, loss curves, gradient norms, and training time were retained as secondary/diagnostic quantities. A zero exact-solve rate was explicitly allowed by the protocol.

### Flagship task and frozen data

The main pilot used the existing validated Sudoku pipeline at a harder-than-debug setting:

- 9×9 Sudoku (`box=3`)
- unique solutions required
- 30–35 clues
- existing validated generator/augmentation path
- data seed `20260907`
- train `384`
- validation `96`
- test `128`
- zero cross-split group overlap required
- zero cross-split exact overlap required

The frozen split passed the duplicate audit. The test split was not used to select architecture, optimizer, step count, seed count, or checkpoint.

4×4 Sudoku was reserved only for a preregistered learning-failure diagnostic. That branch was not triggered, so **4×4 was not used in the M06 main experiment or for post-hoc retuning**.

### Models and declared differences

#### A. Floating-point recursive reference

```text
TRM
FP32
dim=48
n_layers=1
heads=4
n=1
T=1
N_sup=2
trainable parameters=30,829
```

#### B. Matched W1.58A8 recursive model

Same recursive dimensions/schedule as A, with ternary projections and A8 recurrent-state fake quantization.

```text
trainable parameters=30,397
quantization warmup=50 of 200 steps
```

The repository's FP and ternary attention/projection implementations use different existing bias conventions. An architecture-only preflight therefore found a `1.401278%` trainable-parameter difference. The original 1.0% preregistered match tolerance was amended to 1.5% **before any M06 timing probe, validation metric, test prediction, or accuracy result was produced**. The implementations themselves were not changed to force equality.

#### C. Larger single-pass baseline

```text
System1Student
FP32
dim=96
n_layers=2
heads=4
trainable parameters=228,394
single pass
```

The unused confidence head was frozen and excluded from the trainable parameter count because M06 did not train or test confidence routing.

The larger single-pass baseline is deliberately not parameter matched. It asks whether a simple larger model is a stronger baseline than the small recursive systems. A future causal comparison still needs a size/compute-controlled single-pass comparator.

### Optimization protocol

Common settings were fixed before results:

```text
optimizer        AdamW
learning rate    1e-3
weight decay     0.01
batch size       32
gradient clip    1.0
main steps       target 200 / model / seed
```

There was no architecture-specific LR search, accuracy-based early stopping, best-checkpoint selection, or test-time search.

The recursive models used the repository's existing deep-supervision objective, including the existing halting/improvement terms. The single-pass model used token cross-entropy because it has no recursive supervision or halting head. This objective difference is an architectural limitation and prevents interpreting M06 as a clean causal test of recursion alone.

### Timed probe and recorded compute budget

Before the main fits, exactly five optimizer steps per architecture were timed. Probe models were discarded; no probe accuracy was inspected.

| Model | 5-step elapsed | seconds / step |
|---|---:|---:|
| FP recursive | `0.256232 s` | `0.0512465` |
| W1.58A8 recursive | `0.231664 s` | `0.0463327` |
| larger single-pass | `0.238884 s` | `0.0477768` |

The slowest observed rate was `0.0512465 s/step`. Under the preregistered timing-only rule, the `1080 s` main-training budget allowed the full repeated-seed branch:

```text
seeds             1101, 2202
steps / model      200
models / seed      3
main fits          6
estimated time     61.4958 s
actual summed fit  52.9538 s
budget             1080 s
```

No accuracy-dependent budget adjustment occurred.

Training compute is retained separately from evaluation metrics. Each run sampled `6,400` training examples. Under the declared architectures:

- FP recursive: `25,600` shared-block applications sampled per run
- W1.58A8 recursive: `25,600` per run
- larger single-pass: `12,800` block applications per run

These are structural counters, not FLOP-equated or joule-equated costs. Physical joules were not available on the runner.

### Observed primary result

**The primary endpoint was zero for every model at both seeds.**

Across the frozen 128-example test set:

| Model | Seeds | Semantic solve rate | Exact match | Blank-cell accuracy | All-cell accuracy |
|---|---:|---:|---:|---:|---:|
| FP recursive | 2 | `0.0000` | `0.0000` | `0.15925 ± 0.00065` | `0.49894 ± 0.00039` |
| W1.58A8 recursive | 2 | `0.0000` | `0.0000` | `0.11458 ± 0.00453` | `0.47232 ± 0.00270` |
| larger single-pass | 2 | `0.0000` | `0.0000` | `0.32821 ± 0.01570` | `0.59963 ± 0.00936` |

The `±` values are population standard deviations over the two fixed seeds, not confidence intervals.

Per-seed strict solve and exact-reference rates were all exactly zero. M06 therefore does **not** establish useful exact 9×9 Sudoku reasoning capability for any of these small-budget fits.

### Did training fail?

Not under the preregistered optimization-failure definition.

All six fits had finite/nonzero gradients and reduced final-window training loss by far more than the required 5%:

| Model / seed | Initial-window loss | Final-window loss | Relative improvement |
|---|---:|---:|---:|
| FP recursive / 1101 | `2.39634` | `1.29827` | `45.82%` |
| FP recursive / 2202 | `2.59600` | `1.29393` | `50.16%` |
| W1.58A8 recursive / 1101 | `2.48694` | `1.33376` | `46.37%` |
| W1.58A8 recursive / 2202 | `2.38910` | `1.32407` | `44.58%` |
| larger single-pass / 1101 | `2.08138` | `0.76017` | `63.48%` |
| larger single-pass / 2202 | `1.98616` | `0.74136` | `62.67%` |

No fit met the preregistered learning-failure trigger. Consequently the gradient-inspection / 8-example overfit / minimal-4×4 diagnostic branch was not executed. `diagnostics.json` and `failures.json` are retained and empty for the accepted run.

This distinction matters: the pilot learned partial token/cell structure but did not reach the much stricter complete-board capability endpoint.

### Ternary inference verification

The W1.58A8 result was evaluated only after each final checkpoint was reloaded. Evaluation was rejected unless every `FakeBitLinear.quant_strength` buffer was exactly `1.0`.

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

The accepted workflow printed `quantization_strength_check=pass`.

Final ternary state was non-degenerate:

- seed 1101: negative `0.36156`, positive `0.35904`, zero `0.27940`
- seed 2202: negative `0.36032`, positive `0.36121`, zero `0.27848`

This validates the declared M06 evaluation representation. It does not establish parity with FP accuracy.

### Interpretation / claims boundary

The M06 evidence supports only narrow conclusions:

1. A repeated-seed trained baseline experiment now exists on a non-debug 9×9 task under a frozen data/resource/evaluation protocol.
2. All three training paths learned partial token/cell structure under the preregistered optimization criterion.
3. None solved a complete held-out Sudoku at this data/training budget.
4. The deliberately larger single-pass baseline was materially stronger on the secondary cell metrics than either small recursive model.
5. The full-strength W1.58A8 recursive model was weaker than the matched FP recursive model on the same secondary metrics in both seeds.

Therefore M06 provides **no evidence that recursion substitutes for parameter count**, and **no evidence that W1.58A8 preserves FP recursive quality** at this pilot budget.

It also does not prove the reverse universal claims. The single-pass comparator is much larger; recursive and single-pass training objectives differ; the FP and ternary implementations have the documented bias difference; only two model seeds were run; and the task distribution is generated rather than an external benchmark. M06 cannot establish that recursion is intrinsically worse or that ternary reasoning cannot work.

No search/routing/PRM mechanism should be added to reinterpret this result inside M06.

### Accepted environment and regression state

Accepted experiment environment:

```text
Ubuntu 24.04.4 hosted runner
AMD EPYC 7763
x86_64
4 logical CPUs exposed
Python 3.11.16
PyTorch 2.14.0+cpu
NumPy 2.4.6
CUDA false
PyTorch threads 2
```

This is controlled CI evidence, not target-edge-hardware benchmarking.

After the experiment/evidence verifier succeeded, the full fast regression passed:

```text
204 passed, 16 deselected in 59.98 s
```

The 16 slow legacy/capability tests were outside the user-defined M06 experiment scope; M06 does not convert them into acceptance evidence.

### Repository / evidence state

Authoritative accepted experiment:

- M06 base: M05-accepted `main`, `1003c59e17dc17e652438317b7480c9e898379af`
- accepted experimental head: `9fd96e72b9542c33c4ae0639d44847f104a83df8`
- Actions run: `34124798931`
- job: `101750849574`
- artifact: `m06-controlled-baseline-evidence`
- artifact id: `10019726497`
- artifact ZIP SHA-256: `63c533e96fba488baac7bb8a44052dd1408a342b21a6313cd8c3c7d1e6b5acb8`
- artifact size: `6,697,316` bytes
- retention: 14 days
- experiment exit: `0`
- evidence-check exit: `0`
- fast-regression exit: `0`

Key retained evidence hashes:

```text
summary.json          1a3041d541aee77d50e65f9d4314ff70ba1fe81ccd418a15959adcc55db802aa
data_manifest.json    bbe21422059fb901d9bb53503110c33250c910e780fade7a9511fc26634237f1
timing_probe.json     f17a7c51e44fb506ab698fd51a08b740ebb01e74bfd13bb5987e55fb9f179194
budget_decision.json  4508b46fc2397a69f02b9b974a2c8032eda2474a275fbb1afd6dfb844bb8ab39
parameter_audit.json  edea84e4ad13f1b9b2d873bdb88fd46ee578d9af03250378d93da103e8d49e9a
environment.json      6548bcd5c2abc219a941217ae8d45cb4a4433aee206f72ca476d3c36089370cf
```

The artifact retains all six final checkpoints, six per-step learning curves, six complete 128-example prediction JSONL files, raw timing/budget/parameter records, environment/install metadata, failures/diagnostics files, test output, raw run logs, and per-file SHA-256 records.

### Pre-acceptance iterations

Two pre-experimental failures are retained because they explain the final protocol provenance:

1. Run `34124091722` stopped at architecture-only preflight when the original 1.0% parameter-match tolerance encountered the existing 1.401278% FP/ternary bias-related difference. No M06 timing probe, validation accuracy, test prediction, or performance result ran. The tolerance was amended to 1.5% before experiment execution.
2. Run `34124576768` passed the amended architecture preflight but failed on a direct-script import-path bug before the timing probe or accuracy ran. The entry point was repaired without changing the experiment protocol.

Neither failed run exposed a model-performance result.

### M06 decision

M06 closes the **controlled trained baseline** layer for its declared pilot scope. The result is deliberately not upgraded into a central-hypothesis claim: every model's strict solve rate was zero.

**Evidence required before a later milestone can test additional reasoning mechanisms:** first establish an adequate trained baseline with nonzero strict solves under a separately preregistered, larger-but-bounded data/training budget, and add a size/compute-controlled single-pass comparator. Only after complete-task capability exists should search, routing, verifier, energy, or scaling mechanisms be evaluated as scientific improvements.

**Next action: stop here for M06. Do not stack new mechanisms onto this pilot.**
