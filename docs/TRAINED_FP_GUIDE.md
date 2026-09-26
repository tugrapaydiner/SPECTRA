# Prepared execution of trained FP checkpoints

This optional runtime targets the **FP32, n=T=1, 4x4 Sudoku TRM**, not the packed
ternary/A8 runtime from the earlier PR27 experiment. These are separate model
formats and comparisons: their speedup numbers must not be multiplied together.

## Use

Install the CPU research environment from the repository README (PyTorch 2.10.0,
Ninja and a compatible C++ compiler), then install this checkout. The ordinary
compiler-free CNF package remains unchanged. This branch is not a newly published
package-index release.

```python
from pathlib import Path
from scripts.m16_evidence import Evidence
from spectra.fp_runtime import PreparedFPSudoku

# Repository research loader: verifies the existing accepted checkpoint hashes.
evidence = Evidence(Path("results/m16/sources"))
model, checkpoint_sha = evidence.load_model(
    "fp_recursive_dim64", 1401, Path("outputs/source-checkpoints")
)
runtime = PreparedFPSudoku(model)
answer, work = runtime.solve(input_tokens, max_steps=4)
# input_tokens: CPU torch.int64 [1,16], symbols 0..4; zero denotes a blank.
# Check work["final_semantic"]. A budget-exhausted answer is NOT a valid solution.
```

The source must be eval-mode, frozen CPU FP32 with the audited module structure.
Unsupported module classes, per-instance graph overrides, hooks, nonfinite
weights, autocast, incompatible normalization/GELU settings, disabled attention
fast-path dispatch and inappropriate shapes/budgets are rejected. The runtime
uses the existing internal ATen native-MHA operation and is deliberately bounded
to the tested PyTorch 2.10.x series. Compiler or future-PyTorch compatibility is
not assumed; upgrades require the full contracts and trained execution replay.

The source is neither retained nor mutated: owned copies form a snapshot. Do not
mutate a source concurrently with construction. Later source edits do not update
the snapshot; prepare another instance explicitly. Returned state tensors do not
alias weights or other requests. Global dispatch configuration must remain stable
while calls execute. There is no optimizer, target answer, puzzle-answer cache,
request-specific prepared state or recurrent-state reuse across solves.

## What changed

Preparation validates and owns the FP weights and computes only the fixed spatial
position embeddings. Each request still creates its own exact Sudoku checker,
embeds its actual input, initializes zero y/z state and performs one complete
recurrent cycle at a time. The C++ context calls the same PyTorch ATen RMSNorm,
native MHA, linear, exact GELU, addition and multiplication operations in the same
order. It removes repeated Python module dispatch and per-request eval/position
setup; it does not introduce a new matrix multiplication or attention algorithm.

The historical exact native checker decodes and checks each completed step. A
valid answer stops computation immediately; otherwise the original trained
four-step budget is used. No intermediate recurrent step or validity check is
skipped. The new `trace` method is a separate diagnostic which executes the full
requested budget and returns states/logits; it is not used inside timed solves.

All old training, deployment, search, input-generation and checker implementations
remain unchanged. There is no quantization of the accepted FP checkpoints.

## Evidence and reproduction

The [protocol](TRAINED_FP_PROTOCOL.md) was committed before implementation and
measurement. The four-arm primary uses both accepted seeds, 1401 and 2402, and
all 128 inputs from each of two already observed development surfaces: ordinary
M16 Sudoku and shifted M17 Sudoku. There are **256 puzzles, 512 model/puzzle
cases**, four arms and seven rounds: **14,336 observations**. These are NOT
14,336 independent tasks. The confirmation/reserve surfaces are not evaluated.
Known symmetry/generalization limitations of the historical tiny-Sudoku data
remain in force.

```bash
python -m pytest tests/test_fp_runtime.py tests/test_fp_evidence.py
python scripts/bench_trained_fp.py run --out outputs/trained-fp
python scripts/bench_trained_fp.py verify --out outputs/trained-fp --replay
python scripts/bench_fp_controls.py run --out outputs/fp-controls
python scripts/bench_fp_controls.py verify --out outputs/fp-controls --replay
```

Use fresh output directories. The four primary arms are historical Python
semantic exit, the existing native-checker semantic exit, prepared FP execution,
and the existing exact symbolic solver plus checking. Every row retains its
answer, validity, depth, work, randomized position and integer-nanosecond timing.
Independent NumPy and standard-library checks validate the original clues and
complete Sudoku constraints. Execution replay checks all four arms; full-budget
recurrent-state and logit bit comparisons check the graph separately.

A separately declared **matched-preparation eager ablation** gives the historical
Python graph its own immutable copied model and fixed positions without repeated
eval setup. It tests whether native dispatch elimination adds value beyond just
hoisting preparation. Its two-arm, seven-round matrix retains **7,168 additional
observations** and replays **1,024 distinct executions**. It does not redefine the
original four-arm primary or make a new capability claim.

The stdlib analysis reports all case medians, conditional paired problem-cluster
bootstrap intervals with both fixed models retained, family/model strata,
failures, empirical tails and total charged time per verified answer. The latter
includes unsuccessful attempts, not just successful-call latency. Seven samples
per arm/case are insufficient for a production P95/P99 guarantee. Performance
gates are recorded as PASS or FAIL without suppressing negative outcomes.

The source/fixture digests are frozen before timing. A deterministic replay does
not reproduce historical wall times. The verifier rejects source/fixture changes,
missing or duplicate observations, altered ordering, answer/work corruption,
invalid summaries and incomplete trajectory coverage. Hashes do not authenticate
the truth of wall-clock observations.

## Setup, memory, and measurement limits

JIT loading/compilation, source checkpoint loading, owning-context construction
and first-call measurements are separate from warm complete-solve time. The
cached-library break-even calculation must not be presented as a cold compiler
break-even. The outside-checkout installation probe uses a fresh build directory
and records compilation-plus-preparation time separately.

Owned tensor payload is not peak memory or RSS. Keeping the original model in
memory alongside the runtime duplicates much of its storage. No energy reduction
is established. Python cProfile can omit native/released-GIL execution time;
its attributed time is not used for the performance claim. Separate instrumented
ATen profiles verify recurrent operator counts; ordinary uninstrumented rows
remain the latency evidence.

Bitwise fidelity is within a matched environment, not cross-hardware portability.
A runtime speedup is not new learned reasoning, stronger generalization, or
superiority to the symbolic solver. The symbolic baseline remains visible even
when faster and more accurate. See the PR's source-pinned receipts and the
`trained-fp-execution` workflow for completed runs rather than interpreting this
reproduction guide as evidence that a run has passed.
