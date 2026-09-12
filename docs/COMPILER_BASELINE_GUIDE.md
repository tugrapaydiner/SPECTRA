# Compiler baseline: a faster nonbitwise option

This experiment compares the trained FP path from PR27 with functioning PyTorch
Inductor controls. It does not change the default backend or the exact runtime.
The compiler adapter is experimental and explicitly outside the bitwise contract.

## Use and boundary

Use the pinned CPU research environment and the supported FP32 4x4 Sudoku model
from [the trained runtime guide](TRAINED_FP_GUIDE.md). The model must be frozen,
eval-mode, CPU FP32 with the audited n=T=1 graph. No checkpoint is converted or
retrained by this interface. Base CNF imports still do not require Torch.

Set freezing before importing Torch, including before any indirect Torch import:

```python
import os
os.environ["TORCHINDUCTOR_FREEZING"] = "1"
os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"

from spectra.compiler_runtime import CompilerFPSudoku

# frozen_model and input_tokens come from the existing trained-runtime guide.
runtime = CompilerFPSudoku(frozen_model, mode="max-autotune")
# Compilation is lazy. A full trace warms the complete recurrent region.
runtime.trace(input_tokens, max_steps=4)
answer, work = runtime.solve(input_tokens, max_steps=4)
assert isinstance(work["final_semantic"], bool)
# False means budget exhausted, not a verified solution or an UNSAT proof.
```

The other modes are `default` (ordinary Inductor regional compilation) and `eager`
(an uncompiled adapter useful for contracts). The benchmark's eager competitor is
the stronger pre-existing `PreparedEagerControl`, not an intentionally weak path.
Compilation errors propagate. Suppressed compiler errors are rejected. There is
no automatic fallback, hidden default switch, or exact-output promise.

Only a recurrent cycle plus its output head is compiled, with `fullgraph=True`
and `dynamic=False`. Every request still embeds its own input, creates a fresh
native checker and zero recurrent state, and checks the answer after every cycle.
The checker and data-dependent exit are deliberately outside the compiled region.
Owned model snapshots isolate later caller edits; callers must not mutate the
source concurrently with construction or change global dispatch configuration
while execution is in progress. The graph remains restricted to the supported
PyTorch 2.10.x stack; newer compiler behavior has not been tested by these results.

## Why regional compilation

A separate strict full-solve capture probe fails at the Python binding for the
native checker. That error is retained in `whole_solve_probe.json`. It is not a
reason to dismiss the compiler: both regional controls compile and execute the
neural region successfully. The benchmark rejects new captured graphs during
supposedly warm timing and retains compiler counters and generated source hashes.

Max-autotune alone is not evidence of frozen-weight packing. The first exploratory
probe requested freezing only as a compiler option and did not establish packed
frozen parameters. The final launcher sets `TORCHINDUCTOR_FREEZING=1` before
import, records its effective configuration, and retains generated packed-kernel
and frozen-parameter source evidence. A subsequent over-restrictive configuration
guard aborted an initial formal attempt before timing; it was corrected rather
than silently counting that failed arm as a competitor.

## Reproduce

```bash
python -m pytest tests/test_compiler_runtime.py tests/test_compiler_evidence.py
python scripts/bench_compiler_baseline.py run --out outputs/compiler-new
python scripts/bench_compiler_baseline.py verify --out outputs/compiler-new --replay
```

Use a fresh output directory. The executable and fixture identities are frozen
before measurement. The [prospective protocol](COMPILER_BASELINE_PROTOCOL.md) was
committed at `8b35c0879864fa8fd958d457d2442d173a32d784` before implementation and
measurement. All 578 files from the preceding PR27 source are unchanged.

The matrix retains both accepted checkpoints, 256 distinct ordinary/shifted
DEVELOPMENT puzzles, and 512 model/puzzle cases. Four arms and seven randomized
rounds yield 14,336 timing observations, not that many independent problems.
No training, filtering, new confirmation/reserve experiment or reference-answer
lookup occurs. A separate full-budget trace covers 2,048 arm executions and
8,192 recurrent steps. Each returned answer is checked independently against
its original clues and complete Sudoku constraints.

## Final local observation

Single pinned AMD EPYC 9V74 core, Python 3.13.5, Torch 2.10.0+cpu, GCC 14.2.
Ratio of sums of per-case medians, relative to PR27 prepared execution:

| Arm | Ratio | Conditional problem-cluster 95% interval | Valid outcomes |
|---|---:|---:|---:|
| Matched-preparation eager | 1.544225 | [1.532491, 1.557499] | 478/512 |
| Prepared C++/ATen reference | 1.000000 | [1.000000, 1.000000] | 478/512 |
| Inductor default | 0.947538 | [0.942831, 0.952244] | 478/512 |
| Inductor max-autotune + freezing | 0.732840 | [0.728216, 0.737636] | 478/512 |

Frozen Inductor has about 26.7% lower warm complete-request latency, approximately
1.36x faster, than the prepared custom path in this local matrix. This is a result
in favor of the compiler, not a speed claim for the custom runtime. Both compiler
arms preserve every answer, validity flag, stopping depth and work record in the
fixed cases. Neither is bitwise equivalent: internal trajectories differ in all
512 cases. Maximum observed absolute logit differences are 3.09944e-5 (default)
and 3.50475e-5 (frozen). There are no nonfinite states in the audited traces.

Local charged total time per verified output, including failed attempts, is
0.800 ms eager / 0.517 ms prepared / 0.488 ms default / 0.380 ms frozen. This is
not mean latency computed only on successful calls. The frozen arm has zero
case-median regressions but 47 empirical case-P95 regressions against prepared;
the default arm has 88 and 156 respectively. Seven samples per arm/case do not
support a production-tail guarantee. All observations and strata are retained.

The earlier complete local run is retained separately, along with the aborted
setup attempt and exploratory probes; no observations were overwritten. Final
CI outcomes and exact source identity belong to the PR receipt, not a claim that
this configured workflow has already completed.

## What this does not establish

The exact track fails for both compilers, while the bounded task-output track
passes locally. Unchanged answers on these previously observed cases are not a
population-level quality guarantee or proof of equivalent reasoning. The exact
PreparedFPSudoku API remains available and unchanged. No new learned capability,
classical-solver superiority, universal compiler win, memory or energy advantage
is established. The earlier symbolic comparator remains faster and more accurate
on its separately retained experiment; it was not retimed in this compiler matrix.

Warm timing includes embedding, fresh state/checker, recurrence, decoding/checking
and returned work. Input tensor creation, the extra external audit, serialization,
model loading and lazy compilation are excluded and separately identified.
Compilation starts with an empty per-run Inductor cache, but controls are prepared
in a fixed sequence and can reuse compiler work within that cache. Native
extensions are already cached in the local matrix. Per-arm setup receipts are
not independent cold-install times. The clean installed-wheel probe uses a fresh
native build directory and fresh Inductor cache, with those costs reported
separately. Generated-code hashes are not a cross-hardware reproducibility promise.

The strongest next execution baseline is the best functioning compiler control,
not merely historical eager or our custom runtime. Any later batching/scheduling
experiment must compare complete cost and task outcomes against that control.

## Primary implementation references

- PyTorch CPU max-autotune: https://docs.pytorch.org/tutorials/unstable/max_autotune_on_CPU_tutorial.html
- PyTorch regional compilation: https://docs.pytorch.org/tutorials/recipes/regional_compilation.html
- torch.compile API: https://docs.pytorch.org/docs/stable/generated/torch.compile.html

Online documentation may describe a newer version. Configuration, generated code
and observed behavior above are verified against the installed 2.10.0 CPU stack.
