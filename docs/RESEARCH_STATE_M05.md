# SPECTRA Research State

This is the live milestone register. The complete pre-M05 M01–M04 state log is preserved **verbatim** in [`RESEARCH_STATE_M01_M04.md`](RESEARCH_STATE_M01_M04.md), copied from the exact M04-era blob `0a3a815e040d12aa9686dfe164111a686c64d08b` before this register was compacted. Git history therefore retains both the original cumulative record and the live register below.

## Accepted milestone index

| Milestone | Scope | Accepted / merged state |
|---|---|---|
| M01 | trustworthy baseline | accepted and merged; `main` merge `40745dbe185c069aeee9eff3cf63dd411d9e17da` |
| M02 | native-kernel correctness and input contracts | accepted and merged; `main` merge `e0781ec8b4e649ab4ccd48d4cd5f432a9b88d249` |
| M03 | trustworthy task/data/evaluation contracts | accepted and merged; `main` merge `01638b10777029fb28bb35229e374f0865c6e5d4` |
| M04 | reproducible training and checkpoint state | accepted and merged through PR #4; `main` merge `370caf708755e1c68c59d5696778597f0290ea68` |
| M05 | checkpoint-backed evaluation | **COMPLETE ON `research/m05-checkpoint-eval`; not merged at the time of this entry** |

Detailed M01–M04 evidence, numerical contracts, test counts, failures, and limitations are unchanged in the archived register linked above. The sections below record M05 in full.

---

## Milestone 05 — checkpoint-backed evaluation

**Stage status:** COMPLETE ON `research/m05-checkpoint-eval`; not merged at the time of this entry.

M05 changes evaluation loading, immutable evaluation snapshots, research-vs-smoke boundaries, inference-setting contracts, realized-compute accounting, per-example provenance, tests, and audit infrastructure. It does **not** change TRM training mathematics, native-kernel mathematics, or the underlying LatentNativeMCTS search algorithm; the MCTS subclass added in M05 only counts calls around the existing implementation.

### Repository / evidence state

- M05 base: accepted M04 merge on `main`, `370caf708755e1c68c59d5696778597f0290ea68`
- Final green implementation head: `5ccd87d907e635b0e47a2e296f65de4c4de78071`
- Final green implementation workflow: GitHub Actions run `34082746876`
- Job: `101621021178`
- Evidence artifact: `m05-checkpoint-eval-evidence`
- Artifact id: `10004234258`
- Evidence ZIP SHA-256: `2dc88643726db54eec5c15a66d6833bc178a113ac1d0bad311094bdf67856ed5`
- Artifact size: 91,098 bytes; workflow retention: 14 days
- Focused M05 gate: **14 passed in 4.43 s**
- Full fast suite: **202 passed, 16 deselected in 61.09 s**
- CI host: Ubuntu 24.04.4, Python 3.11.16, torch 2.14.0+cpu, NumPy 2.4.6, pytest 9.1.1
- CUDA availability: false
- RAPL energy on the CI host: unavailable

### Root causes closed by M05

Before M05, the research-facing evaluation paths were not trustworthy enough for scientific result labels:

1. `scripts/eval_scaling_laws.py` constructed fresh random TRMs for requested parameter budgets instead of loading trained checkpoints.
2. The same scaling path constructed fresh random `LatentEnergyVerifier` and `LatentActionCodebook` objects while presenting the configuration as learned search.
3. `scripts/eval_edge.py` allowed a missing checkpoint to fall through to random initialization.
4. Scaling rows treated `N_sup` as a generic recursion/search-depth knob even though `LatentNativeMCTS._step()` does not consume `N_sup`; changing it could therefore label two search rows differently while executing identical MCTS transition computation.
5. M03 split manifests recorded stable IDs/split provenance but did not contain the exact input/target arrays, so evaluation replay still depended on generator code and seed reconstruction.
6. M04 EMA state tracks trainable parameters only, while ternary `quant_strength` is deliberately a non-persistent buffer stored separately in checkpoint quantization provenance. A research evaluator must restore both pieces explicitly rather than instantiate a fresh rho or substitute raw trainable weights.

### Immutable evaluation snapshot contract

`eval/evaluation_manifest.py` introduces:

```text
format  = spectra.eval_manifest
version = 1
```

A research evaluation snapshot embeds the exact selected split:

- input arrays
- target arrays
- input masks
- target masks
- stable data IDs
- pre-augmentation group IDs
- per-example metadata
- task/scope/official-benchmark label
- exact task configuration
- height, width, sequence length, vocabulary and pad token
- SHA-256 of the source M03 data manifest when available
- a canonical `manifest_sha256`

The canonical digest excludes only its own digest field. Loading recomputes it and rejects tampering. Evaluation constructs the dataset directly from the embedded arrays; it does **not** regenerate examples at evaluation time.

The M05 reference snapshot was built from `config/m04_cpu_reference.yaml`, seed `20260907`, test size 4. Its canonical SHA-256 was:

```text
d8d4bb166a632fd735f7e2c1380e6cb815c8f5f0f86aefd42a39106e47908747
```

The four frozen test IDs were:

```text
9d3eda1cb2139845b28a16b2
9dfaae0192c1ac074bba42eb
fb66cbcdc44afb5d0818ddff
1c7f1319ecce8ca1e48f2c37
```

Both the `N_sup=1` and `N_sup=2` reference evaluations consumed these exact four IDs in the same order.

### Strict trained-core checkpoint restoration

`eval/checkpoint_eval.py::load_research_trm_checkpoint` requires the current versioned M04 training-state format and rejects:

- missing files
- legacy weights-only checkpoints
- non-resume-capable/incomplete checkpoint structures
- `global_step <= 0` checkpoints
- unsupported model family metadata
- missing architecture fields
- architecture/task shape or vocabulary disagreement
- incompatible state dicts
- invalid raw/EMA identity
- missing EMA trainable parameters
- missing or inconsistent ternary quantization provenance

The evaluator reconstructs the actual checkpoint model rather than accepting YAML guesses:

- model family
- dimension and exact parameter count
- vocabulary / sequence length
- layer count
- `n`, `T`, `N_sup`
- attention heads
- residual scales
- max grid size
- ternary setting
- A8 setting
- raw versus recorded EMA evaluation identity

It then forces `model.eval()`; research evaluation itself runs under `torch.inference_mode()`.

#### EMA and non-persistent quantization state

For an EMA-labelled result, every trainable parameter must come from the checkpoint EMA mapping. Raw trainable substitution is forbidden. Persistent non-parameter model state can come from the matching raw state where required by PyTorch state reconstruction.

`FakeBitLinear.quant_strength` is intentionally non-persistent and therefore is not carried by either raw `state_dict()` or EMA. M05 separately restores `training.quantization.strengths` through `load_quant_strength_state` and verifies the restored per-layer values. The ternary+A8 adversarial fixture checkpointed at training step 1 with rho `0.25`; the M05 loader test verifies the research model is restored at rho `0.25` rather than constructor default rho `1.0`.

### Checkpoint / evaluation-manifest compatibility

Research evaluation requires the checkpoint and immutable snapshot to agree on:

- task name
- height and width
- sequence length
- vocabulary
- exact resolved task configuration

A modified manifest whose digest is deliberately recomputed still fails when its task configuration no longer matches the trained checkpoint. Distribution-shift evaluation therefore requires an explicitly different declared contract; it cannot occur accidentally through regeneration or a config override.

### Learned-search auxiliary checkpoint contract

Research-labelled learned Latent MCTS never creates fresh random auxiliaries.

M05 defines versioned learned auxiliary metadata:

```text
format  = spectra.learned_auxiliary
version = 1
```

The supported M05 auxiliary kinds are:

- `latent_energy_verifier`
- `latent_action_codebook`

Each artifact must record `trained_steps > 0`, architecture metadata, state dict, and compatibility tied to:

- the **exact core checkpoint SHA-256**
- task
- latent dimension
- vocabulary
- sequence length

Learned search requires both a compatible verifier checkpoint and a compatible action-policy checkpoint. Missing either one, wrong auxiliary kind, incomplete architecture, malformed state, zero-step/untrained metadata, or a core-checkpoint hash mismatch fails before a research result is emitted.

The M05 adversarial tests use one-step-optimized auxiliary fixtures only to verify this loader/search contract. They are **not** evidence that a scientifically trained verifier or action policy is good, calibrated, or beneficial.

### Compute-knob contract

M05 separates requested labels from computation actually consumed.

#### Ordinary greedy forward

`N_sup` is a real knob for the ordinary TRM forward pass. For a checkpoint with `T` outer cycles and inner recurrence parameter `n`:

```text
recursive_cycle calls / example = N_sup * T
shared-operator applications / example = N_sup * T * (n + 1)
```

The reference checkpoint had `T=1`, `n=1`.

Reference realized totals over the same four examples:

| setting | forward calls | recursive-cycle calls | shared-operator applications | stopping reason |
|---|---:|---:|---:|---|
| `N_sup=1` | 4 | 4 | 8 | `greedy_forward_complete` ×4 |
| `N_sup=2` | 4 | 8 | 16 | `greedy_forward_complete` ×4 |

The tiny reference checkpoint happened to produce the same task metrics at those two settings. This is **not** evidence that the knob was ignored: realized cycle/operator counters doubled exactly as specified.

#### Current native learned MCTS

`LatentNativeMCTS._step()` consumes the checkpoint's `T` and `n` through `recursive_cycle`; it does **not** consume `N_sup`. M05 therefore:

- rejects `ordinary_n_sup` / `N_sup` as an MCTS transition setting
- keeps the checkpoint's original `N_sup` in provenance only
- excludes it from the effective search-compute signature
- records `N_sup_consumed_by_search_transition = false`

Current `LatentNativeMCTS` is also deterministic: there is no random rollout sampling, random tie break, root noise, or stochastic policy sampling. A `search_seed` is therefore currently a decorative ignored setting and is rejected. M05 records:

```text
search_stochastic = false
search_seed_consumed = null
```

If a future stochastic search implementation is introduced, it must define and consume an independent search RNG stream before exposing a research seed knob.

`uncertainty_beta` is consumed only when the loaded verifier implements `value_with_uncertainty`. The strict M05 learned-verifier loader currently supports a single `LatentEnergyVerifier`, which does not implement that interface, so nonzero `uncertainty_beta` is rejected rather than silently ignored. A future ensemble/uncertainty verifier checkpoint must explicitly satisfy that interface before such a sweep is valid.

The effective current MCTS settings therefore include only knobs/metadata actually used by the loaded implementation, such as rollout budget, loaded action count/codebook, checkpoint `T/n`, and PUCT coefficient; uncertainty is valid only with a compatible uncertainty-aware verifier.

### Realized MCTS compute accounting

`CountingLatentNativeMCTS` wraps the existing MCTS methods without changing search math and records per example:

- child transition calls (`_step`)
- node expansions (`_expand`)
- verifier calls (`_value`)
- rollouts requested
- rollouts completed
- stopping reason
- corresponding recursive-cycle and shared-operator applications

The adversarial contract fixture uses three learned actions and two rollouts. It verifies exactly:

```text
expansions             = 3
transition calls       = 9
verifier calls         = 2
rollouts completed     = 2
stopping reason        = rollout_budget_exhausted
```

These values establish accounting for that tested search topology; they are not a claim about learned-search quality.

### Per-example research output

Research evaluation emits a record for every frozen example containing at least:

- checkpoint path and SHA-256
- checkpoint format/version/global step
- actual model family and parameter count
- ternary/A8 settings
- checkpoint `n/T/N_sup`
- requested and realized raw/EMA identity
- actual evaluation backend/device/dtype
- training runtime metadata
- evaluation-manifest canonical/file SHA-256
- split/task/scope
- data ID and group ID
- inference setting
- realized setting
- verifier/action-policy checkpoint hashes when search is used
- prediction tokens
- exact-reference correctness boolean
- task metrics
- realized compute counters and stopping reason

The result kind is `research_checkpoint`. Random initialization can never produce that label.

### Random-initialization smoke boundary

Historical parameter-budget random models remain only behind explicit smoke mode:

```text
--smoke-random-init
```

Smoke rows are permanently marked:

```text
result_kind     = smoke_random_init
research_result = false
```

The old random-initialized dense-vs-SPECTRA null-hypothesis result path was disabled entirely rather than retaining an easy-to-misread pseudo-experiment. A scientific comparative/null-hypothesis result now requires separately trained compatible checkpoints and a fixed evaluation snapshot.

### M05 reference checkpoint-backed run

The accepted implementation CI trained the tiny eight-step M04 CPU teacher only to exercise the full checkpoint→snapshot→evaluation path.

Reference teacher checkpoint file SHA-256:

```text
2bb51758b5d4bfcbf2f73cbbb7b17a77b24779e5bdf73eb1c2a66501259e59e4
```

Loaded evaluation provenance:

```text
model family       = TRM
parameter count    = 4,488
ternary            = false
A8                 = false
weight identity    = ema
backend            = pytorch_eager
device             = cpu
manifest examples  = 4
```

Reference metrics for both `N_sup=1` and `N_sup=2` on the same frozen examples were:

```text
exact reference match = 0.0
semantic validity      = 0.0
cell accuracy          = 0.203125
blank-cell accuracy    = 0.03125
```

These low values are expected from a tiny mechanics checkpoint and are not a task-capability or scaling-law claim. The single-run CI latency samples (~0.745 ms and ~1.290 ms) are retained only as execution smoke evidence and are **not target-hardware performance claims**. No physical energy reading was available.

### Exact reproducible M05 commands

Create a versioned trained checkpoint:

```bash
python scripts/train_teacher.py \
  --config config/m04_cpu_reference.yaml \
  --train-size 24 --val-size 12 \
  --out outputs/m05_teacher
```

Freeze the exact test examples:

```bash
python scripts/build_eval_manifest.py \
  --config config/m04_cpu_reference.yaml \
  --train-size 24 --val-size 12 --test-size 4 \
  --split test --seed 20260907 \
  --out outputs/m05_test_manifest.json
```

Checkpoint-backed scaling/inference sweep:

```bash
python scripts/eval_scaling_laws.py \
  --checkpoint outputs/m05_teacher/teacher.pt \
  --manifest outputs/m05_test_manifest.json \
  --weights recorded --device cpu \
  --greedy-n-sup 1 2 \
  --latency-runs 1 \
  --out outputs/m05_scaling.csv \
  --predictions-out outputs/m05_scaling_predictions.jsonl
```

Checkpoint-backed edge report:

```bash
python scripts/eval_edge.py \
  --ckpt outputs/m05_teacher/teacher.pt \
  --manifest outputs/m05_test_manifest.json \
  --weights recorded --device cpu \
  --n-sup 1 2 --latency-runs 1 \
  --out outputs/m05_edge_report.json
```

A learned-search research invocation additionally requires one compatible `--verifier-checkpoint` and `--action-checkpoint` per core checkpoint. Current native MCTS is deterministic, so `--search-seed` is intentionally rejected; nonzero `--uncertainty-beta` is also rejected by the single-verifier path because it would not be consumed.

### M05 failures / iterations retained explicitly

No final M05 test was weakened. The following failures or hidden-contract problems were found and closed:

1. **Legacy null-hypothesis expectation.** The first M05 CI run passed all new focused tests and both checkpoint-backed CLIs, but the full suite had one old test expecting random-init dense-vs-SPECTRA rows. The research path was not restored; the legacy test was changed to assert the new rejection boundary.
2. **Non-persistent ternary rho.** A focused run exposed that `FakeBitLinear.quant_strength` is not in `state_dict`, so a reconstructed ternary evaluator started at rho `1.0` instead of checkpoint rho `0.25`. The loader now restores the explicit M04 quantization-strength snapshot and verifies it.
3. **MCTS `N_sup` provenance trap.** An early effective-setting record included checkpoint `N_sup`, which could make otherwise identical MCTS settings appear different. It was removed from the effective signature; direct search `N_sup` input is rejected.
4. **Decorative search seed.** A final search audit confirmed current `LatentNativeMCTS` is deterministic. `search_seed` is now rejected instead of pretending to alter the computation.
5. **Ignored uncertainty beta.** A nonzero beta would be ignored by a single `LatentEnergyVerifier`. M05 now rejects it unless the loaded verifier implements the uncertainty interface.

Earlier failed-run artifacts remain useful debugging provenance but are not acceptance evidence. The authoritative green implementation evidence is run `34082746876` above.

### Remaining limitations / unsupported claims

- The M05 CI research CLI exercised the **greedy checkpoint-backed path** using a tiny eight-step teacher; it did not establish useful reasoning quality.
- Learned-search loading/accounting is adversarially unit-tested using one-step auxiliary fixtures. M05 did **not** evaluate a scientifically trained verifier/action-policy pair and therefore does not establish learned-search advantage.
- The strict M05 auxiliary loader supports a single `LatentEnergyVerifier`; no strict ensemble-verifier checkpoint loader was added in this milestone. Nonzero uncertainty beta is therefore intentionally unavailable in research mode.
- Current native MCTS is deterministic. The requested independent stochastic-search RNG rule is enforced prospectively: a future stochastic implementation must define/use a separate stream before a seed can be exposed. No stochastic search RNG exists to test today.
- M05 does not prove a trained scaling law. The harness is now capable of comparing multiple real checkpoints on one frozen snapshot, but the reference evidence contains only one tiny checkpoint and two ordinary-forward compute settings.
- M05 does not establish a scientific System-1 null-hypothesis comparison; the misleading random-init comparison was disabled and no trained baseline checkpoint experiment was supplied.
- CUDA/GPU evaluation was unavailable in CI.
- Physical RAPL energy measurement was unavailable in CI.
- The small single-run latency values are not target-device benchmarks.
- Research core loading is intentionally strict to the current `spectra.training` v1 TRM metadata contract. Legacy weights-only files are not research checkpoints.
- An evaluation manifest is immutable by content hash, but intentional creation of a different rehashed manifest is a different experiment identity; compatibility rules still prevent silent task-config drift.
- M05 does not establish official ARC/BabyAI results, external benchmark generalization, search-quality gains, scaling laws, energy savings, or production hardware performance.

### M05 decision

M05 closes the **evaluation provenance and execution-contract** layer for the tested scope: research rows require a trained versioned core checkpoint and exact frozen examples; architecture/precision-family settings are restored from metadata; EMA and ternary quantization state are explicit; learned search cannot fabricate random auxiliaries; ignored knobs are rejected; realized compute is counted; and each prediction carries enough provenance to trace it back to checkpoint and data identity.

**Next action:** stop here for M05. Do not convert the tiny reference checkpoint into a capability/scaling claim, and do not claim learned-search benefit until genuinely trained compatible auxiliary checkpoints are evaluated. Merge only after the final documentation-inclusive branch-head CI is green and the user accepts the milestone.
