# Milestone 13 Acceptance Gate — Defensible Measurement Protocol

**Decision: PASS as a measurement-integrity milestone.**

M13 establishes a shared, auditable measurement path for physical CPU-package energy, representative complete-solve timing, memory scope accounting, instrumentation overhead, and arithmetic/traffic conventions. It does **not** establish energy superiority, cache residency, a bandwidth bottleneck, GPU energy, or whole-system energy.

Authoritative preregistration: [`M13_PROTOCOL.md`](M13_PROTOCOL.md).
Durable accepted raw evidence: [`../results/m13/`](../results/m13/).

## Accepted implementation measurement run

```text
branch        research/m13-measurement-protocol
head          4a700938cece1b041205d8ad16a48d280cfc9a99
Actions run   34155993914
job           101847823445
artifact      m13-defensible-measurement-evidence
artifact id   10031016693
ZIP SHA256    a64cf2e073eef2c636d68fef9c8ce7c4d517e79714bd27e051e0c4fbfc9c48a1
size          402,570 bytes
```

Accepted gate:

```text
compile       0
focused       0
precomputed   0
measurement   0
render        0
evidence      0
fast          0
focused       29 passed
full fast     282 passed, 16 deselected, 1 pre-existing M10 warning
```

The warning is the already-known M10 test-only tensor-to-scalar warning. No M13 regression failed.

---

## 1. Shared physical-energy counter implementation

`eval/edge_energy.py` and `common/telemetry_logger.py` no longer maintain separate RAPL readers. Both use `common/energy_counters.py`.

The shared reader records for every usable powercap counter:

- stable domain id and path;
- kernel-provided domain name;
- parent domain id;
- classified domain type;
- per-counter `max_energy_range_uj`;
- whether the counter participates in package-total aggregation.

The package total sums only independent top-level package domains. Nested DRAM/core/uncore/graphics-style child counters can be reported individually but are never added on top of their package parent.

A physical energy window is invalid if the selected counter set changes, discovery becomes partial, an endpoint is malformed/out of range, a counter range changes, or any required domain delta cannot be validated.

Unavailable or invalid energy is represented as:

```text
available      false
energy_joules  null
```

not `0.0`.

A genuine valid zero delta remains distinguishable as `available=true, energy_joules=0.0`.

## 2. Wraparound / reset semantics

Each counter uses its own declared range. A decreasing endpoint is accepted as one wrap only when the start is near the upper declared boundary and the end is near the lower boundary. The delta is then:

```text
(max_energy_range_uj - start_uj) + end_uj
```

A non-boundary decrease is classified `reset_or_corrupt_decrease` and the measurement becomes unavailable. M13 does not infer multi-wrap energy from two snapshots.

## 3. Controlled fixtures are not hardware measurements

The focused M13 suite uses synthetic filesystem counter fixtures to verify:

- nested package/DRAM domains do not double count;
- a valid zero remains zero;
- one valid per-range wrap is reconstructed;
- reset-like decreases are rejected;
- malformed/out-of-range endpoints remain unavailable;
- disappearing/partial counter sets invalidate the window;
- invalid discovery cannot fabricate zero energy.

Accepted focused result:

```text
29 passed in 1.17 s
```

These fixtures are contract tests only and make no physical-energy claim.

---

## 4. Physical-energy result on the accepted runner

The accepted hosted runner exposed no usable package powercap domain:

```text
root                        /sys/class/powercap
package domain ids          []
package discovery complete  false
physical energy available   false
failure reason              no_package_domain
energy joules               null
joules / complete solve     null
idle subtraction            false
```

This is an accepted **unavailable measurement**, not a failed milestone and not a zero-joule solve.

The scope string is explicitly:

```text
cpu_package_rapl_not_gpu_not_whole_system
```

RAPL package energy, when available, is energy attributed by the CPU package counter and can include CPU cores plus other on-package components. M13 never presents it as discrete-GPU energy or whole-system/wall energy.

---

## 5. Host provenance

Accepted host observation:

```text
CPU model          INTEL(R) XEON(R) PLATINUM 8573C
OS/kernel          Linux 6.17.0-1022-azure
process affinity   CPUs 0,1,2,3
logical CPUs       4
PyTorch threads    2
interop threads    2
frequency driver   intel_cpufreq
observed governor  performance
CUDA               unavailable
backend            spectra_m10_native
native operator    packed_ternary_fp32_scalar_linear_v1
compiler           g++ 13.3.0
runtime flags      -O3 -std=c++20
```

The recorded frequency values are observations only; M13 does not claim a locked frequency. `/sys/class/power_supply` exposed no usable supply information on the runner, so power-source context is retained as unavailable rather than guessed.

---

## 6. Real workload timing

The primary M13 performance workload is not `x[:1]` repeated and is not the K-input microkernel. It is the actual sequential `CPURecursiveRuntime.forward` path on distinct B=1 held-out examples.

Each timed solve includes:

- recurrent native/mixed-precision execution;
- complete declared supervision/recurrent schedule;
- final logits / argmax decode;
- symbolic Sudoku semantic validation.

Training and export are excluded from warm latency. The first cold solve is separate and includes artifact load/validation, runtime construction, native extension build/load, and one complete solve.

Accepted timing:

```text
held-out instances         24 distinct
warm-up instances           4 distinct
warm timing rounds          3
raw warm rows              72
inference_mode            true
reference targets used    false
warm mean              7.602571 ms
warm median            7.585985 ms
warm p95               7.819384 ms
warm min               7.351030 ms
warm max               8.089556 ms
cold first solve   16417.175752 ms
```

Raw rows are retained in [`../results/m13/full_solve_timings.csv`](../results/m13/full_solve_timings.csv).

---

## 7. Memory labels

M13 replaces the historical post-run-RSS-as-peak label with a window-local background RSS sampler and records Linux lifetime `VmHWM` separately.

Accepted memory observations:

```text
packed ternary weights                3,112 B
full model state                     53,184 B
controlled search-tree tensor store  73,472 B
process RSS current                 374.719 MiB
process RSS peak sampled            374.719 MiB
Linux lifetime VmHWM                375.426 MiB
RSS sampling interval                 0.5 ms
RSS samples                            324
```

The sampled peak can miss allocations shorter than the sampling interval. `VmHWM` is process-lifetime high-water and is not called a window-local peak. Search-tree storage counts unique tensor storages and explicitly excludes Python object/list allocator overhead. The retained reference tree uses the trained reasoner but untrained search auxiliaries and carries no search-quality claim.

---

## 8. Instrumentation overhead

Accepted overhead observation:

```text
direct complete solve mean       7.759915 ms
RSS-sampled solve mean           11.245103 ms
observed sampler overhead        +3.485188 ms
energy snapshot read mean        22.266 us
energy snapshot read p95         30.759 us
```

These are environment-dependent observations, not constants. They are retained so instrumentation cost is visible instead of being silently included in workload timing.

---

## 9. MAC / operation convention

M13 adopts one convention everywhere:

```text
1 MAC = 2 arithmetic operations
```

Both GOP/s and arithmetic intensity use arithmetic operations. Raw rows retain both `macs` and `operations` and the gate asserts `operations == 2 * macs`.

The old K-input benchmark is now explicitly:

```text
workload = precomputed_input_reuse
```

because the complete `[K,H]` activation matrix already exists when the native call begins.

Accepted AVX2 precomputed-input rows include:

```text
K=1     0.358 GOP/s      7.642 operations/B
K=64   17.526 GOP/s    252.062 operations/B
K=256  31.134 GOP/s    407.056 operations/B
```

The traffic denominator includes packed weights, all K input activations, all K outputs, and requant metadata. It is labelled:

```text
kernel_external_first_touch_logical_bytes_not_measured_dram
```

These rows do not establish actual recurrence reuse, cache residency, or a bandwidth bottleneck.

---

## 10. Actual sequential recurrence traffic model

For one complete M13 recurrent solve the native work counters report:

```text
native linear MACs           1,583,104
arithmetic operations        3,166,208
```

The declared logical traffic accounting records:

```text
persistent packed weights                 3,112 B
persistent scales/biases                   1,192 B
persistent FP tensors                      3,340 B
native FP32 input/output boundaries      304,384 B
recurrent y/z read/write lower bound      32,768 B
input embedding output                     2,048 B
input tokens                                  128 B
logits/decode                               1,408 B
accounted logical bytes                   348,380 B
operations/accounted logical byte          9.0884
```

Algorithmic attention-score storage (`32,768 B` if materialized) is disclosed separately and excluded from the denominator because the PyTorch attention implementation may fuse or stream it.

The traffic scope is explicitly:

```text
logical_tensor_and_interface_bytes_not_measured_cache_or_dram_transactions
```

and the accepted records set:

```text
cache_residency_established       false
bandwidth_bottleneck_established  false
```

---

## 11. Raw evidence and regenerated figure

Durable repository evidence includes the complete 72 timing rows, host/environment record, physical-energy record, memory scopes, instrumentation overhead, sequential recurrence accounting, precomputed-input rows, controlled fixture result, and a regenerated figure.

Figure:

[`../results/m13/m13_full_solve_latency.svg`](../results/m13/m13_full_solve_latency.svg)

The durable figure provenance records:

```text
raw timing CSV SHA256  e9fa3ca8aa09970aec79459c5ef1bac4c987b32bb7eca1e1b6f66d7e53a11ba2
accepted CI PNG SHA256 de2f4a748c0cf1d554778c26cc305b49dae02600179a922077feed49722ad8d4
operation convention   1_MAC_equals_2_arithmetic_operations
cache claim             false
bandwidth claim         false
```

Pixels are presentation only; the retained raw rows are the source of truth.

---

## M13 decision

**PASS — measurement integrity only.** Counter failures are not fabricated as zero energy; nested domains and per-counter ranges are handled explicitly; real complete-solve performance covers distinct held-out instances; cold/warm timing is separated; memory scopes are accurately labelled; instrumentation overhead is visible; throughput/intensity use consistent units; precomputed K-input reuse and real sequential recurrence are separated; activation/state traffic is represented; and the figure is regenerated from auditable raw data.

M13 does **not** establish physical energy per solve on the accepted hosted runner, cache residency, a bandwidth bottleneck, whole-system energy, GPU energy, universal latency, search quality, controller quality, or improved task capability.

**Stop here for M13. Do not begin M14 automatically.**
