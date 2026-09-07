# Milestone 01 Acceptance Gate

This page is the concise acceptance surface for the SPECTRA Milestone 01 baseline. Detailed provenance remains in [RESEARCH_STATE.md](RESEARCH_STATE.md), and the exhaustive claim ledger remains in [CLAIMS.md](CLAIMS.md).

## 1. Clean-environment setup command

Linux / GitHub Actions baseline (Python 3.11):

```bash
python3.11 -m venv --clear .venv-m01 && . .venv-m01/bin/activate && python -m pip install --upgrade pip && python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.4" && python -m pip install -r requirements.txt
```

The authoritative automated form, including budgets and hardware probes, is [.github/workflows/m01-baseline.yml](../.github/workflows/m01-baseline.yml).

The resolver used by the accepted audit selected Python 3.11.16, PyTorch 2.14.0+cpu, NumPy 2.4.6, PyYAML 6.0.3, einops 0.8.2, tqdm 4.70.0, pandas 3.0.5, psutil 7.2.2, and pytest 9.1.1. The full resolved environment was captured with `pip freeze` in the run artifact.

## 2. Honest baseline report

Accepted clean audit: [GitHub Actions run 34073485472](https://github.com/tugrapaydiner/SPECTRA/actions/runs/34073485472). Raw audit logs/environment snapshot: [m01-baseline-evidence artifact 10001279309](https://github.com/tugrapaydiner/SPECTRA/actions/runs/34073485472/artifacts/10001279309) (14-day retention). Persistent summary: [RESEARCH_STATE.md](RESEARCH_STATE.md).

| Category | Check | Result |
|---|---|---|
| **passed** | dependency setup | passed; 30 s |
| **passed** | `pytest -m "not slow" -ra` | **151 passed**, 16 slow tests deselected; pytest 48.04 s |
| **passed** | native `tests/test_kernel.py` | **4 passed** on AVX2 host; pytest 1.85 s |
| **passed** | `python setup.py build_ext --inplace` | passed after minimal C++20 compatibility fix; 27 s |
| **passed** | tiny CPU training smoke | finite loss/gradients and parameter update; ~0.955 s internal loop |
| **passed** | energy API graceful-unavailable behavior | `rapl_available=False`, measurement `None` |
| **failed** | final clean audit | **none** |
| **timed out** | final clean audit | **none** |
| **skipped** | slow pytest items | **16 deselected by design**; not counted as passing |
| **unavailable** | GPU/CUDA | unavailable on audit host |
| **unavailable** | readable RAPL `energy_uj` | unavailable on audit host |
| **unavailable** | user's unpushed/local working tree | unavailable to this execution environment; not touched |
| **unavailable** | target-device physical performance/energy recertification | not established by this CI baseline |

### Known setup/audit failures retained in the record

These occurred while establishing the baseline and are deliberately not erased from the history:

1. PyTorch 2.14 rejected the repository's old C++17 extension flags because current headers require C++20. The minimal setup fix changed only extension/JIT build flags to C++20 in [setup.py](../setup.py), [deploy/cpp_sparse_kernel/setup.py](../deploy/cpp_sparse_kernel/setup.py), and [deploy/torch_kernel.py](../deploy/torch_kernel.py).
2. The first evidence upload missed the hidden `.m01/` directory because the artifact action excludes hidden files by default. The workflow now sets `include-hidden-files: true`.
3. The first status formatter emitted a spurious `timed_out` line due to shell operator precedence. It was replaced by explicit `if/elif/else` bookkeeping.
4. The native packed-weight path still has a known **non-multiple-of-4 hidden-dimension boundary** that is not guarded or proven correct. The accepted AVX2 result applies only to tested aligned shapes.

## 3. Claim register with evidence links

Status meanings are defined in [CLAIMS.md](CLAIMS.md). A test passing only upgrades the exact behavior exercised by that test; it does not automatically validate a broader scientific claim.

| Claim | Direct evidence | Boundary / known failure | Status |
|---|---|---|---|
| Recursive TRM/deep-supervision plumbing runs and trains at tiny scale. | [model/trm.py](../model/trm.py), [train/losses.py](../train/losses.py), [tests/test_recursion.py](../tests/test_recursion.py), [tests/test_smoke.py](../tests/test_smoke.py), [accepted audit](https://github.com/tugrapaydiner/SPECTRA/actions/runs/34073485472) | Trainability only; not task-level reasoning quality. | **unit-tested** |
| W1.58 ternary and INT8 activation machinery exists and passes functional tests. | [model/bitlinear.py](../model/bitlinear.py), [model/fake_quant.py](../model/fake_quant.py), [tests/test_fake_bitlinear.py](../tests/test_fake_bitlinear.py), [tests/test_int8_activation.py](../tests/test_int8_activation.py), [tests/test_ternary_trm.py](../tests/test_ternary_trm.py) | Does not establish end-to-end task accuracy at target precision. | **unit-tested** |
| AVX2 kernel equals scalar/reference on supported tested shapes. | [native kernel](../deploy/cpp_sparse_kernel/spectra_kernel.cpp), [tests/test_kernel.py](../tests/test_kernel.py), [accepted audit](https://github.com/tugrapaydiner/SPECTRA/actions/runs/34073485472) | 4/4 passed on AVX2, but tested widths are aligned; arbitrary hidden dimensions are not established. | **unit-tested** |
| PyTorch C++ extension builds in the accepted current environment. | [setup.py](../setup.py), [extension bridge](../deploy/torch_kernel.py), [accepted audit](https://github.com/tugrapaydiner/SPECTRA/actions/runs/34073485472) | Required C++20 compatibility fix after the first clean attempt exposed C++17 incompatibility with PyTorch 2.14. | **unit-tested** |
| Weight-stationary kernel can reuse one decoded weight row across precomputed `K` inputs. | [scripts/bench_kernel.py](../scripts/bench_kernel.py), [native kernel](../deploy/cpp_sparse_kernel/spectra_kernel.cpp), [tests/test_kernel.py](../tests/test_kernel.py) | Benchmark precomputes all `K` activation vectors; does not prove causal end-to-end recursive reuse. | **implemented** |
| Published `K`-reuse throughput, SIMD speedups, cache-residency and GOP/s headlines are currently re-certified. | [README.md](../README.md), [scripts/bench_kernel.py](../scripts/bench_kernel.py), [scripts/bench_cache.py](../scripts/bench_cache.py), [scripts/bench_sparse.py](../scripts/bench_sparse.py) | Complete raw publication measurement provenance was not established and CI VM is not the claimed target machine. | **hypothesis** |
| VQ projection is idempotent and remains bounded in the repository's constructed 30-step expansive recurrence. | [model/latent_vq.py](../model/latent_vq.py), [tests/test_death_traps.py](../tests/test_death_traps.py) | Synthetic constructed system only. | **unit-tested** |
| VQ guarantees covering-radius-bounded task-relevant error/topology for arbitrarily deep real SPECTRA trajectories. | [model/latent_vq.py](../model/latent_vq.py), [README.md](../README.md) | No trained real-trajectory proof/test at README strength. | **hypothesis** |
| Latent MCTS searches without decoding every latent node; verifier/LCB plumbing executes. | [eval/latent_mcts.py](../eval/latent_mcts.py), [model/energy.py](../model/energy.py), [tests/test_latent_mcts_native.py](../tests/test_latent_mcts_native.py) | Mechanics only; verifier calibration/task quality not established. | **unit-tested** |
| Adaptive halting can select an earlier generated answer. | [model/halting.py](../model/halting.py), [tests/test_halting.py](../tests/test_halting.py) | Current helper computes the full recursion first, then postprocesses the chosen halt step. | **unit-tested** |
| Current halting saves wall time/compute/energy through execution-time early exit. | [model/halting.py](../model/halting.py) | False as an established result for the current helper: recursion has already run before selection. | **hypothesis** |
| Scaling harness demonstrates a trained parameter-scaling law. | [scripts/eval_scaling_laws.py](../scripts/eval_scaling_laws.py), [eval/scaling.py](../eval/scaling.py), [tests/test_scaling_laws.py](../tests/test_scaling_laws.py) | Inspected harness creates fresh models and does not load/train checkpoints before accuracy evaluation. | **hypothesis** |
| Verifier backups can be propagated and distilled mechanically. | [eval/latent_mcts.py](../eval/latent_mcts.py), [train/distill.py](../train/distill.py), [tests/test_prm.py](../tests/test_prm.py) | Generated targets originate from verifier/search values; correctness against external task truth is not established. | **unit-tested** |
| Verifier backups are grounded estimates of true task quality and drive real self-improvement. | [model/energy.py](../model/energy.py), [tests/test_prm.py](../tests/test_prm.py) | Current tests validate self-distillation plumbing, not externally grounded reward accuracy. | **hypothesis** |
| RAPL code handles missing counters safely. | [eval/edge_energy.py](../eval/edge_energy.py), [tests/test_energy_measurement.py](../tests/test_energy_measurement.py), [accepted audit](https://github.com/tugrapaydiner/SPECTRA/actions/runs/34073485472) | Accepted host had no readable RAPL counters, so no Joule value was produced. | **unit-tested** |
| README Joule/energy frontier is measured by Milestone 01. | [README.md](../README.md), [eval/edge_energy.py](../eval/edge_energy.py) | RAPL unavailable on accepted host; target-device energy experiment was not rerun. | **hypothesis** |

For the exhaustive per-claim mapping, including measurement-artifact assumptions and additional README claims, use [docs/CLAIMS.md](CLAIMS.md).

## Gate decision

**PASS for Milestone 01 baseline acceptance.**

The gate passes because there is now one copy-paste clean-environment setup command, an executed baseline report that separates passed/failed/skipped/timed-out/unavailable states, and a clickable evidence-backed claim register. Known failures and unavailable measurements remain explicit rather than being reclassified as passing.