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

`main` was not modified. The work was isolated on `research/m01-baseline` so existing repository state was not overwritten.

### Working-tree limitation

The model execution sandbox could not directly clone GitHub (`Could not resolve host: github.com`), so the user's own local checkout and any unpushed/uncommitted working-tree edits were **unavailable for inspection**. They are therefore neither described as clean nor modified by this milestone.

The GitHub Actions checkout used for the reproducible audit **was clean before audit outputs were created**. The workflow records `git status --short` before creating `.m01/`; the final clean run recorded no entries.

### Applicable repository instructions

No `AGENTS.md` or equivalent agent-specific instruction file, and no additional `CONTRIBUTING.md` instruction layer, was found in the inspected repository tree. The applicable baseline instructions were therefore the repository README/build guidance, `requirements.txt`, `pyproject.toml`, `setup.py`, and pytest configuration.

## Minimal setup changes

Only setup/audit compatibility changes were made:

1. Added `.github/workflows/m01-baseline.yml` to make the baseline commands, budgets, host probes, and evidence collection repeatable.
2. Changed the optional PyTorch C++ extension compile standard from C++17 to C++20 in:
   - `setup.py`
   - `deploy/cpp_sparse_kernel/setup.py`
   - `deploy/torch_kernel.py`

Reason: the current unconstrained requirement `torch>=2.4` resolved on the audit host to `torch==2.14.0+cpu`, whose headers reject the repository's previous `-std=c++17` build. The first audit reproduced that failure; C++20 was the minimum compatibility fix. Standalone C++ kernel logic was not changed.

No test was weakened, skipped by modification, or rewritten to accommodate a failure.

## Reproducible environment

Final clean audit host:

- GitHub-hosted runner: Ubuntu 24.04.4 LTS (`ubuntu-24.04`, image `20260831.293.1`)
- Linux: `6.17.0-1022-azure`, x86_64
- CPU allocation: 4 logical CPUs
- Reported CPU: AMD EPYC 9V74 80-Core Processor, virtualized under Microsoft/Azure
- L3 visible to runner: 32 MiB
- AVX2: **supported** (`avx2` present in `/proc/cpuinfo`)
- `g++`: 13.3.0
- `clang++`: 18.1.3
- GPU: **unavailable** (`nvidia-smi` unavailable; no `/dev/nvidia*` devices)
- CUDA in PyTorch: **unavailable** (`torch.version.cuda=None`, `torch.cuda.is_available() == False`, device count 0)
- Intel RAPL root: **unavailable** (`/sys/class/powercap/intel-rapl` absent)
- Readable `energy_uj` counters: **none available**
- Repository energy API probe: `rapl_available=False`; `measure_energy_joules(...)` returned `None` without error

Resolved primary Python packages:

- Python 3.11.16
- pip 26.2.1
- torch 2.14.0+cpu
- numpy 2.4.6
- PyYAML 6.0.3
- einops 0.8.2
- tqdm 4.70.0
- pandas 3.0.5
- psutil 7.2.2
- pytest 9.1.1

The Actions artifact contains the sorted full `pip freeze` from the run.

## Exact baseline commands

The authoritative executable form is `.github/workflows/m01-baseline.yml`. The final clean run executed the following relevant commands.

### Host/repository probes

```bash
git rev-parse HEAD
git branch --show-current
git status --short
python --version
python -m pip --version
uname -a
getconf _NPROCESSORS_ONLN
lscpu
g++ --version
clang++ --version
grep -qm1 -w avx2 /proc/cpuinfo
nvidia-smi
ls /dev/nvidia*
find /sys/class/powercap -type f -name energy_uj
# Each discovered energy_uj path is checked for readability and read with cat.
```

### Dependency setup

```bash
timeout 360s python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.4"
timeout 180s python -m pip install -r requirements.txt
python -m pip freeze | sort
```

### Energy API probe

```bash
python - <<'PY'
from eval.edge_energy import rapl_available, measure_energy_joules
print("rapl_available=", rapl_available())
result = measure_energy_joules(lambda: sum(range(1000)), n_runs=1)
print("measurement=", result)
PY
```

### Existing fast test gate — 360 second budget

```bash
timeout 360s python -m pytest -m "not slow" -ra
```

### Native correctness — 180 second budget

The workflow first requires both AVX2 and a C++ compiler, then runs:

```bash
timeout 180s python -m pytest tests/test_kernel.py -ra
```

### Optional PyTorch C++ extension build — 180 second budget

```bash
timeout 180s python setup.py build_ext --inplace
```

### Tiny training smoke — 30 second budget

```bash
timeout 30s python - <<'PY'
import time
import torch
from model.trm import TRM
from train.losses import deep_supervision_loss

torch.manual_seed(0)
t0 = time.perf_counter()
model = TRM(
    dim=16, num_tokens=5, seq_len=4, n_layers=1, n=1, T=1,
    N_sup=1, heads=1, max_grid_size=4,
)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
x = torch.randint(0, 5, (2, 4))
y = torch.randint(0, 5, (2, 4))
before = next(model.parameters()).detach().clone()
losses = []
for _ in range(2):
    _, steps = model(x, height=2, width=2)
    loss = deep_supervision_loss(steps, y)
    assert torch.isfinite(loss), loss
    opt.zero_grad()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    opt.step()
    losses.append(float(loss.detach()))
after = next(model.parameters()).detach()
assert not torch.equal(before, after), "optimizer did not update parameters"
print("device=cpu")
print("steps=2")
print("losses=", losses)
print("parameter_update=yes")
print("elapsed_s=", round(time.perf_counter() - t0, 6))
PY
```

## Check results

### Passed

| Check | Result | Recorded budget / elapsed |
|---|---|---|
| Dependency installation | passed | 30 s elapsed; install sub-budgets 360 s + 180 s |
| Repository energy API graceful-unavailable probe | passed | returned `rapl_available=False`, measurement `None` |
| Existing fast test gate | **151 passed** | 360 s budget; 50 s workflow elapsed; pytest 48.04 s |
| Native kernel correctness | **4 passed** | 180 s budget; 3 s workflow elapsed; pytest 1.85 s |
| Optional PyTorch C++ extension build | passed | 180 s budget; 27 s elapsed |
| Tiny training smoke | passed | 30 s budget; 3 s workflow elapsed; internal loop ~0.955 s |

Tiny smoke losses were `1.9163410663604736` then `1.8603845834732056`; gradients were finite and at least one model parameter changed after the optimizer steps. The smoke is a trainability/plumbing check, **not** a learning-quality claim.

### Failed

**Final clean baseline:** none.

Failures encountered while establishing the baseline, before the final clean run:

1. The optional PyTorch extension failed because PyTorch 2.14 headers require C++20 while the repository requested C++17. Fixed minimally by switching extension/JIT build flags to C++20.
2. The first audit workflow did not upload `.m01/` because hidden files/directories were excluded by the artifact action default. Fixed with `include-hidden-files: true`.
3. The first audit status formatter used a shell expression with incorrect operator precedence and emitted a spurious `timed_out` status line for otherwise successful dependency installation. Replaced with explicit `if/elif/else` status assignment.

These were setup/audit failures; none was hidden or described as a passing scientific check.

### Skipped / not run by design

- **16 pytest items** carrying the excluded `slow` marker were deselected by the explicitly requested fast baseline command `-m "not slow"`. They are **not** counted as passing.
- Dependency-gated “unavailable” fallback bookkeeping was skipped in the final run because dependency installation succeeded.
- Long training/evaluation campaigns and publication benchmark reruns were intentionally outside Milestone 01.

### Timed out

**None** in the final clean baseline.

### Unavailable

- User's actual local working tree / unpushed edits: unavailable to this execution environment; not touched.
- GPU/CUDA: unavailable on the GitHub Actions host.
- Readable RAPL/`energy_uj` counters: unavailable on the GitHub Actions host.
- Physical-target energy measurements: unavailable because no readable counters exist on this host.
- Claimed target-device physical performance validation: unavailable from this AMD EPYC virtual CI host.
- A complete raw-data/provenance set sufficient to regenerate every published README performance figure was not established in the inspected commit; therefore Milestone 01 did not re-certify the numeric performance headlines.

## Baseline findings that constrain later claims

The corresponding evidence map is maintained in `docs/CLAIMS.md`. The most important baseline boundaries are:

1. **AVX2 boundary:** native bit-exact tests pass on tested hidden dimensions divisible by 4. The packed kernel's `hidden_dim / 4` stride and bridge contract do not establish correctness for unaligned hidden dimensions.
2. **Scaling is checkpoint-free in the inspected path:** the scaling harness creates fresh models and does not establish a trained scaling law.
3. **`K`-reuse benchmark precomputes the input vectors:** it demonstrates kernel-level weight reuse under that benchmark setup, not end-to-end reuse across causally dependent recursive states.
4. **VQ evidence is narrower than the strongest prose claim:** VQ snapping/idempotence and boundedness in a constructed 30-step expansive recurrence are tested; arbitrary-depth real-task trajectory error/topology preservation is not.
5. **Halting is postprocessed:** `run_with_halting` first executes the full recursion and selects/freezes an earlier result afterward, so it does not currently establish wall-time/energy savings from early termination.
6. **Verifier backups are mechanically valid but not grounded:** PRM/MCTS tests validate backup/distillation plumbing using verifier-produced values; they do not establish that those values correspond to true task quality.

## What this baseline can support

At Milestone 01, it is defensible to state that:

- the selected non-slow Python test suite passes in the recorded CPU environment;
- the standalone native kernel passes its four correctness tests on an AVX2 host and the tested aligned shapes;
- the optional PyTorch extension builds under the currently resolved PyTorch after the minimal C++20 setup update;
- a tiny CPU forward/backward/optimizer loop produces finite gradients and updates parameters;
- the repository's synthetic VQ boundedness test, latent-MCTS/verifier mechanics, quantization/export mechanics, and postprocessed halting mechanics pass their selected fast tests.

This baseline **cannot** support, without later evidence:

- the README's exact physical throughput/speedup/cache/energy headline numbers;
- a trained parameter-scaling law from the current checkpoint-free scaling path;
- end-to-end sequential recursive `K`-reuse from a benchmark that precomputes the `K` inputs;
- arbitrary-depth real-task VQ trajectory/topology guarantees;
- real compute/wall-time/energy savings from the current postprocessed halting helper;
- task-grounded verifier backup quality or a real self-improvement/flywheel claim;
- energy/Joule claims on the Milestone 01 CI host.

## Current blockers

1. **Evidence provenance:** raw measurement artifacts/checkpoints needed to independently substantiate the broad README performance/scaling claims are incomplete or not established by this baseline.
2. **Energy hardware:** no readable RAPL counters on the reproducible CI host.
3. **AVX2 shape contract:** non-multiple-of-4 hidden dimensions are not guarded/tested at the native bridge boundary.
4. **Scientific grounding:** scaling, verifier-backup, halting-savings, and broad VQ claims need trained/task-grounded experiments rather than plumbing-only tests.
5. **Local state visibility:** this environment cannot certify or preserve by inspection any unpushed user-local changes; it avoids the risk by keeping all work on a separate remote branch.

## Next action

**Do not change the algorithm yet.** The next milestone should begin by closing the highest-value evidence/provenance gaps: identify or regenerate the raw benchmark/checkpoint artifacts under explicit hardware/configuration records, add a precise test/guard for the AVX2 hidden-dimension boundary, and define trained/task-grounded evaluation protocols before interpreting scaling, verifier, halting, VQ, or energy outcomes.
