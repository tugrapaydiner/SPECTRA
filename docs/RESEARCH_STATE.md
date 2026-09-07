# SPECTRA Research State

This is the live milestone register. Detailed completed history is preserved rather than rewritten:

- M01–M04: [`RESEARCH_STATE_M01_M04.md`](RESEARCH_STATE_M01_M04.md)
- M05: [`RESEARCH_STATE_M05.md`](RESEARCH_STATE_M05.md)
- M06: [`RESEARCH_STATE_M06.md`](RESEARCH_STATE_M06.md)
- M07: [`RESEARCH_STATE_M07.md`](RESEARCH_STATE_M07.md)
- M08: [`RESEARCH_STATE_M08.md`](RESEARCH_STATE_M08.md)
- complete M09-era live register: [`RESEARCH_STATE_M09.md`](RESEARCH_STATE_M09.md)
- M10–M12 accepted live register snapshot: [`RESEARCH_STATE_M10_M12.md`](RESEARCH_STATE_M10_M12.md)

## Milestone index

| Milestone | Scope | State |
|---|---|---|
| M01 | trustworthy baseline | accepted / merged; `40745dbe185c069aeee9eff3cf63dd411d9e17da` |
| M02 | native-kernel correctness/input contracts | accepted / merged; `e0781ec8b4e649ab4ccd48d4cd5f432a9b88d249` |
| M03 | task/data/evaluation contracts | accepted / merged; `01638b10777029fb28bb35229e374f0865c6e5d4` |
| M04 | reproducible training/checkpoint state | accepted / merged through PR #4; `370caf708755e1c68c59d5696778597f0290ea68` |
| M05 | checkpoint-backed evaluation | accepted / merged through PR #5; `1003c59e17dc17e652438317b7480c9e898379af` |
| M06 | controlled trained baseline | accepted / merged through PR #6; gate clarification PR #7 |
| M07 | grounded verifier training/evaluation | accepted / merged through PR #8 |
| M08 | correct inspectable MCTS reference | accepted / merged through PR #9 |
| M09 | trained search action mechanism | **INCOMPLETE**; negative result preserved/merged through PR #10 |
| M10 | faithful CPU deployment for one trained configuration | accepted / merged through PR #11 |
| M11 | real adaptive execution | accepted / merged through PR #12 |
| M12 | grounded router/halter RL training path | accepted / merged through PR #13; **learned-control quality NEGATIVE / COLLAPSED** |
| M13 | defensible measurement protocol | **COMPLETE on `research/m13-measurement-protocol`; accepted measurement run `34155993914` at `4a700938cece1b041205d8ad16a48d280cfc9a99`** |

M09 remains scientifically incomplete, and M12 did not establish useful learned adaptive control. M13 changes the measurement contract and public evidence boundary; it does not retroactively turn earlier kernel curves into cache-residency/bandwidth evidence or produce physical joules where the host exposes no valid package counter.

---

# Milestone 13 — defensible measurement protocol

**Stage status: COMPLETE on `research/m13-measurement-protocol`.**

Authoritative preregistration: [`M13_PROTOCOL.md`](M13_PROTOCOL.md).
Accepted evidence/claims boundary: [`M13_ACCEPTANCE_GATE.md`](M13_ACCEPTANCE_GATE.md).
Durable raw measurements: [`../results/m13/`](../results/m13/).

## Counter integrity

M13 replaces the two historical RAPL implementations in `eval/edge_energy.py` and `common/telemetry_logger.py` with one shared reader in `common/energy_counters.py`.

The shared reader explicitly discovers domain identity, hierarchy and `max_energy_range_uj`; validates endpoints; handles a single conservative per-counter wrap; rejects reset/corrupt decreases and partial counter sets; and aggregates only independent top-level CPU-package domains. Nested child domains are not double counted.

Unavailable/invalid energy remains:

```text
available      false
energy_joules  null
```

A genuine valid zero remains distinguishable as a valid `0.0` delta.

Controlled synthetic counter fixtures cover zero, wrap, reset/corruption, malformed/out-of-range values, partial/disappearing counters and nested package/subdomain behavior. They are contract tests, not physical-energy results.

Accepted focused result:

```text
29 passed
```

## Accepted physical-energy observation

The accepted GitHub runner exposed no readable CPU-package powercap domain:

```text
powercap root               /sys/class/powercap
package domains             []
physical energy available   false
failure reason              no_package_domain
energy joules               null
joules / complete solve     null
```

This is the intended failure-safe behavior. M13 does not fabricate `0 J`.

Scope is explicitly:

```text
cpu_package_rapl_not_gpu_not_whole_system
```

CPU-package RAPL can include cores and other on-package components. It is not discrete-GPU energy and not whole-system/wall energy.

## Host provenance

Accepted run:

```text
CPU model          INTEL(R) XEON(R) PLATINUM 8573C
OS/kernel          Linux 6.17.0-1022-azure
process affinity   0,1,2,3
logical CPUs       4
PyTorch threads    2
interop threads    2
frequency driver   intel_cpufreq
observed governor  performance
backend            spectra_m10_native
native operator    packed_ternary_fp32_scalar_linear_v1
compiler           g++ 13.3.0
runtime flags      -O3 -std=c++20
CUDA               false
power-supply sysfs unavailable
```

Frequency metadata is observational; no frequency-lock claim is made.

## Complete-solve performance

The primary M13 workload is actual sequential `CPURecursiveRuntime.forward`, not the historical `[K,H]` microbenchmark and not repeated `x[:1]`.

Each timed B=1 solve includes the complete recurrent runtime, decode and symbolic Sudoku semantic validation. Reference solution targets are not used by timing/validation.

```text
held-out instances       24 distinct
warm-up instances         4 distinct
warm timing rounds        3
raw warm rows            72
inference_mode          true
warm mean            7.602571 ms
warm median          7.585985 ms
warm p95             7.819384 ms
warm min             7.351030 ms
warm max             8.089556 ms
cold first solve 16417.175752 ms
```

Cold start includes artifact load/validation, runtime construction, native extension build/load and the first complete solve. It is separate from steady-state latency.

## Memory accounting

Accepted memory scopes:

```text
packed ternary weights                3,112 B
full model state                     53,184 B
controlled search-tree tensor store  73,472 B
current process RSS                 374.719 MiB
sampled window RSS peak             374.719 MiB
Linux lifetime VmHWM                375.426 MiB
RSS sampling interval                 0.5 ms
samples                                324
```

The sampled RSS peak can miss allocations shorter than the sampling interval. `VmHWM` is lifetime high-water, not a window-local peak. Search-tree bytes count unique tensor storages only and exclude Python allocator overhead.

## Instrumentation overhead

```text
direct complete solve mean       7.759915 ms
RSS-sampled solve mean           11.245103 ms
observed sampler overhead        +3.485188 ms
energy snapshot read mean        22.266 us
energy snapshot read p95         30.759 us
```

These are retained environment-specific observations.

## Arithmetic and traffic convention

All M13 throughput/intensity rows use:

```text
1 MAC = 2 arithmetic operations
```

The historical K-input native benchmark is now explicitly `precomputed_input_reuse` because the complete `[K,H]` input matrix exists before the call. Its traffic model includes packed weights, K input activations, K outputs and requant metadata and is labelled logical first-touch bytes, not measured DRAM traffic.

Accepted AVX2 microbenchmark endpoints:

```text
K=1      0.358 GOP/s      7.642 operations/B
K=256   31.134 GOP/s    407.056 operations/B
```

Those rows do not establish sequential recurrence reuse, cache residency or a bandwidth bottleneck.

Actual sequential recurrence is measured separately. One complete retained solve reports:

```text
native linear MACs          1,583,104
arithmetic operations       3,166,208
accounted logical bytes       348,380
operations/logical byte        9.0884
```

The logical-byte model includes persistent packed/scales/FP tensors, native FP32 input/output boundaries, recurrent y/z traffic lower bounds, input and decode traffic, and separately discloses potential attention-score materialization. It is not measured cache or DRAM traffic.

Accepted claim flags:

```text
cache_residency_established       false
bandwidth_bottleneck_established  false
```

## Raw evidence / figure

The accepted 72 raw timing rows, environment, physical-energy record, memory record, instrumentation overhead, sequential accounting and precomputed-input rows are retained under `results/m13/`.

The regenerated latency figure is:

[`../results/m13/m13_full_solve_latency.svg`](../results/m13/m13_full_solve_latency.svg)

Its provenance binds it to the raw timing CSV SHA256:

```text
e9fa3ca8aa09970aec79459c5ef1bac4c987b32bb7eca1e1b6f66d7e53a11ba2
```

and records the accepted CI PNG hash plus the same operation convention and no-cache/no-bandwidth claim flags.

## Accepted execution

```text
head          4a700938cece1b041205d8ad16a48d280cfc9a99
run           34155993914
job           101847823445
artifact      m13-defensible-measurement-evidence
artifact id   10031016693
ZIP SHA256    a64cf2e073eef2c636d68fef9c8ce7c4d517e79714bd27e051e0c4fbfc9c48a1
size          402,570 bytes
```

```text
compile       0
focused       0
precomputed   0
measurement   0
render        0
evidence      0
fast          0
full fast     282 passed, 16 deselected, 1 pre-existing M10 warning
```

## M13 decision

**PASS — defensible measurement protocol.** Counter failure is no longer fabricated as zero; memory labels distinguish model/search/process scopes; complete-solve timing covers representative distinct held-out instances; cold/warm and instrumentation overhead are separated; precomputed K-input reuse and actual recurrence are separate; MAC/op units are consistent; traffic models disclose activations and memory-level scope; and the regenerated figure is backed by raw, hashed data.

M13 does **not** establish CPU-package joules per solve on the accepted hosted runner, whole-system/GPU energy, cache residency, a bandwidth bottleneck, universal latency, task-capability improvement, learned-controller superiority or learned-search benefit.

**Stop here for M13. Do not begin M14 automatically.**
