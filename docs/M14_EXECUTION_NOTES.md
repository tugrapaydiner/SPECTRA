# M14 implementation and environment notes

## Before accuracy inspection

The declaration and initial experiment implementation were committed locally
as `b115fc520df2f0663dc3357b0a9a9341f88bd0fc`. The GitHub connector created the
equivalent remote commit `52c1dee301d0d313e42d941cda33d46b81fdcb65`; both have the
identical verified Git tree `8eb9940ac38170adb12114ebe09ab7e9e8e0c821`.
The evidence includes `source_commit_mapping.json`, original source hashes and
the immutable experiment configuration hash. Commit metadata differs, source
bytes do not.

The initial local environment lacked PyTorch, pytest, Ninja and psutil.
After installing the declared dependencies, the existing native backend passed
its focused correctness checks. Ninja must be on PATH; installing it into a
venv without activating/exposing its bin directory is insufficient.

An early full fast test attempt recorded 276 passes, five failures, one skipped
test and 16 deselections. Four failures concerned the unavailable Ninja/native
extension; the fifth was the existing benchmark's unsupported `None` memory
value when psutil was absent. These setup failures are preserved in execution
logs; they are not new scientific outcomes. A separately invoked benchmark
test was interrupted and scheduled for rerun with explicit thread limits.
Initial fitting overlapped a portion of that test attempt, so elapsed training
times are measured bookkeeping, not clean training-throughput comparisons.
Controlled inference comparisons run without concurrent tests or model fitting.

The runtime exposes no readable physical CPU-package energy counter and no GPU.
This does not change the predeclared latency endpoint. No physical joule,
GPU-performance or target-edge-hardware claim is made.

Before the first tuning/development accuracy inspection, code review tightened
the confirmation guard to bind authorization to the exact phase, selection and
native-backend freeze, added stage-specific source hashes/wall times, and added
an independent prediction checker and decoder-only control runner. No model,
optimizer, training-step budget, seeds, data, primary metric, statistical rule
or pass threshold was changed. Initial fits use the original committed training
code; later stage records identify the executed revision separately.

The numerical dependencies were pinned to the versions actually installed:
PyTorch 2.8.0+cpu, NumPy 2.3.5, SciPy 1.17.0 and Matplotlib 3.10.8. The complete
tested dependency list is retained with the experiment evidence.
