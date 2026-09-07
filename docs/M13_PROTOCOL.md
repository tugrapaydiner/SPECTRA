# Milestone 13 Protocol — Defensible Measurement

**Status: preregistered before M13 implementation measurements.**

M13 is a measurement-correctness milestone. It does not attempt to rescue prior performance, cache-residency, energy, or memory claims. If the retained hardware lacks a readable physical counter, the accepted result is `unavailable`, not zero. If a workload is slower or uses more memory after including omitted work, that result is retained.

Base commit: `802b1a8eb5df7091383a127e3d853a1fbbb62690` (M12 merged `main`).

## 1. Audit findings that M13 must correct

### 1.1 Duplicated RAPL logic

`eval/edge_energy.py` and `common/telemetry_logger.py` independently discover `/sys/class/powercap/intel-rapl` and sum every readable `energy_uj` file. This has four correctness failures:

1. a failed counter read is silently skipped, so a partial snapshot can look valid;
2. invalid/non-monotonic deltas can be clamped to zero or reported negative;
3. counter-specific `max_energy_range_uj` and wraparound are ignored;
4. summing nested package/sub-domain counters can double-count the same physical energy.

M13 replaces these readers with one shared implementation.

### 1.2 Memory labels

`eval/memory.py::measure_peak_ram` currently samples RSS only before and after the call and labels the post-run value `peak_ram_mb`. That is not a peak/high-water measurement.

### 1.3 Throughput / arithmetic-intensity convention

`scripts/bench_kernel.py` reports throughput as `2 * MACs / second`, while `eval/roofline.py` computes arithmetic intensity with `1 * MAC / byte`. M13 uses one convention throughout:

- `1 MAC = 2 arithmetic operations`;
- throughput is GOP/s where `operations = 2 * MACs`;
- arithmetic intensity is **operations / byte** using the same 2-op/MAC numerator;
- raw rows also record MACs explicitly so conversion is auditable.

### 1.4 Precomputed K-input benchmark

`spectra_weight_stationary_gemv` receives a pre-created `[K,H]` activation matrix. That benchmark measures **precomputed-input reuse within one kernel call**. It does not by itself establish that real sequential recurrence keeps weights in any cache.

M13 renames/labels that benchmark accordingly and adds a separate measured sequential-recurrence workload.

---

# 2. Shared physical-energy counter contract

## 2.1 Discovery

The shared reader scans a configurable powercap root (default `/sys/class/powercap`) for directories containing:

- `energy_uj`;
- `max_energy_range_uj`;
- `name`.

A counter is usable only when all required files exist and are readable, the energy value parses as an integer, and `max_energy_range_uj` is a positive integer.

Each discovered domain records:

- stable domain id / directory name;
- human-readable kernel domain name;
- source path;
- parent domain id when nested;
- domain class (`package`, `dram`, `core`, `uncore`, `psys`, `gpu_or_uncore`, `other`);
- per-counter range in microjoules;
- whether it participates in the package-total aggregate.

## 2.2 Aggregate scope / no nested double counting

The primary reported energy is **CPU package-domain energy only**. It sums independent top-level package domains and never adds their nested child domains to that total.

Sub-domain deltas may be reported separately for diagnosis but are not added to package energy. `psys`, DRAM, core, uncore and graphics-related subdomains are not silently equated with package energy.

The report states explicitly:

> RAPL package energy is energy attributed by the host package counter. Depending on the processor it can include CPU cores and other on-package components. It is not a discrete-GPU measurement and it is not whole-system/wall energy.

No M13 result may label package joules as `GPU energy`, `system energy`, or `whole-system energy`.

## 2.3 Snapshot validity

A physical-energy measurement selects the package counters at the start of the window. The final snapshot must contain exactly the same selected package counter identities and every selected reading must be valid. Otherwise aggregate energy is unavailable with an explicit failure reason.

No missing or corrupt counter may contribute `0`.

Raw per-domain readings and validation status are retained.

## 2.4 Range and wraparound

Every reading must satisfy `0 <= energy_uj <= max_energy_range_uj`.

For a normal interval (`end >= start`):

`delta = end - start`.

For a decreasing interval, M13 accepts one wrap only under a conservative boundary rule: start must lie in the upper wrap guard and end in the lower wrap guard of that counter's declared range. Then:

`delta = (range - start) + end`.

A decrease away from the declared wrap boundary is classified `reset_or_corrupt` and returns unavailable. Multi-wrap windows cannot be inferred from two samples and are outside the accepted contract.

Controlled fixtures test normal deltas, exact zero delta, one valid wrap, non-boundary reset/decrease, out-of-range values, unreadable files, changed counter sets, malformed ranges and nested domains. These fixtures are explicitly **synthetic counter-contract tests**, not hardware measurements.

## 2.5 Measurement return type

The public measurement API always returns an explicit record with:

- `available: bool`;
- `energy_joules: float | None`;
- `joules_per_run: float | None`;
- selected domain metadata;
- per-domain deltas or failure status;
- failure reason when unavailable;
- wall time.

A valid physical zero-energy delta may be `0.0`; an unavailable/invalid reading is `None`. Those states are never conflated.

---

# 3. Telemetry logger contract

`common/telemetry_logger.py` imports the shared energy-counter implementation. It does not maintain a second RAPL reader.

Background hardware samples retain counter snapshot validity rather than flattening an invalid sample to zero. Summary energy is computed from validated start/end counter snapshots using the same shared delta function.

Hardware telemetry records physical scope and domain ids. A partial/failing energy window is logged as unavailable with a reason.

---

# 4. Host and power provenance

Every retained M13 hardware measurement records at least:

- CPU model string;
- architecture / OS / kernel;
- process CPU affinity;
- logical CPU count;
- `torch.get_num_threads()` and inter-op thread count where available;
- observed CPU frequency governor/policy per accessible CPU policy;
- accessible min/max/current frequency metadata, labelled as observations rather than a frequency lock;
- backend identity;
- compiler and compile flags for any native component measured;
- Python, PyTorch and NumPy versions;
- CUDA availability but no GPU energy inference from RAPL;
- cgroup/affinity constraints actually applied;
- AC/battery/power-source state when readable from `/sys/class/power_supply`, otherwise unavailable;
- RAPL package-domain scope statement.

Unknown values remain unknown.

---

# 5. Timing protocol

## 5.1 Inference mode

PyTorch inference measurements execute inside `torch.inference_mode()` with models in eval mode.

## 5.2 Cold vs warm

Cold-start and warm-run timings are separate fields.

Cold start includes the declared one-time setup for the measured backend (for example artifact load and native extension build/load when applicable) and the first complete solve. It is never averaged into steady-state warm latency.

Warm measurements begin only after explicit untimed warm-up.

## 5.3 Representative held-out instances

M13 does not repeatedly time only `x[:1]`. A fixed held-out set of at least 16 distinct examples is measured B=1. The raw output contains one latency row per `(example, repetition)` or per example for bounded full-solve measurements, plus aggregate mean/median/p95/min/max.

The held-out identities/data hash are retained.

## 5.4 Complete solve path

The primary workload is the complete declared inference path for one example: input preparation already present in the runtime, recurrent execution through all declared steps/search, final logits/decode, and task-level semantic validation when that validation is part of the benchmark command.

Component/microkernel timings remain secondary and cannot substitute for the primary full-path latency.

---

# 6. Energy protocol

Physical energy, when available, is measured around the same representative warm full-solve workload. The retained record reports:

- package joules for the complete measurement window;
- number of distinct solves in the window;
- joules/solve;
- wall seconds and average package power;
- selected package domain ids/names;
- sub-domain readings separately when available;
- whether instrumentation was active.

Idle subtraction is not performed by default in M13 because it changes the estimand and can produce unstable negative net energy for short windows. If an idle observation is recorded it is reported separately, never silently subtracted.

If physical counters are absent or invalid, physical energy fields are null and the reason is retained.

---

# 7. Memory protocol

M13 reports distinct memory quantities:

1. `packed_weights_bytes` — serialized/native packed weight tensors only;
2. `full_model_state_bytes` — in-memory model parameters + persistent buffers under their actual dtypes;
3. `search_tree_bytes` — explicit tensor/storage estimate attributable to a completed search tree when a search workload is measured, otherwise `0`/not-applicable with scope;
4. `process_rss_current_bytes` — current process RSS;
5. `process_rss_peak_sampled_bytes` — maximum RSS observed by a background sampler over the measured window;
6. Linux process high-water mark (`VmHWM`) when readable, recorded separately because it is lifetime high-water, not window-local;
7. optional CUDA allocator peak only if a CUDA workload is actually measured, labelled separately.

The window-local RSS sampler records interval, number of samples, before/after RSS, max sampled RSS and acknowledges that short allocations between samples can be missed. A post-run RSS value is never called peak RAM.

Instrumentation overhead is measured separately for the sampler and energy reads.

---

# 8. Arithmetic / traffic convention

## 8.1 Operation convention

For dense dot products:

- MACs = multiply-accumulate pairs;
- arithmetic ops = `2 * MACs`.

All GOP/s and operations/byte values use arithmetic ops. Raw rows retain `macs`, `operations`, and time.

## 8.2 Precomputed-input reuse model

The existing K-input kernel benchmark is renamed/labeled `precomputed_input_reuse`. Its traffic model includes at minimum:

- packed weights;
- all K input activation bytes;
- all K output bytes;
- requantization/scales/multipliers consumed by the kernel.

It reports a **modelled traffic level** (kernel-call visible bytes / first-touch memory model), not measured DRAM bytes. It does not assert any cache level.

## 8.3 Sequential recurrence model

A separate benchmark executes actual sequential recurrence where output/state from step `k` is produced before step `k+1`. The record includes complete recurrent model traffic components relevant to the runtime: packed/native weights, activation/state reads/writes, attention/intermediate traffic where modelled, and output/decode traffic.

Any traffic estimate states which memory level it models (`logical bytes touched`, not DRAM transactions) unless hardware PMU counters directly establish another level.

## 8.4 Claim boundary

A flat throughput curve, a small packed model, or precomputed K-input reuse does **not** establish cache residency or a bandwidth bottleneck. M13 may report measured throughput and modelled bytes, but cache residency/bandwidth-bound classification requires independent hardware evidence not provided by the shape of the curve alone.

---

# 9. Raw data and regenerated figure

The accepted run writes machine-readable raw data before rendering:

- environment/provenance JSON;
- energy discovery and physical measurement JSON;
- controlled counter-fixture test evidence;
- per-instance full-solve timing CSV/JSON;
- memory measurement JSON;
- instrumentation-overhead JSON;
- kernel precomputed-reuse CSV;
- sequential-recurrence CSV/JSON;
- figure-input CSV/JSON.

Exactly one M13 figure is regenerated from retained raw data by a deterministic rendering script. The figure carries the raw-data SHA256, operation convention, workload labels, and generated-at commit SHA in adjacent provenance metadata. Figure pixels are not the source of truth.

The M13 figure will compare **precomputed-input reuse** and **actual sequential recurrence** using consistent GOP/s / operations-per-byte units or, if those workloads are not directly comparable on one axis, will show the representative full-solve latency distribution. It will not label modeled traffic as measured DRAM bandwidth.

---

# 10. Acceptance gate

M13 passes only if all of the following hold:

1. both old RAPL readers use one shared implementation;
2. controlled fixtures prove unreadable/partial/corrupt/reset counters remain unavailable and prove valid single-wrap handling using per-counter ranges;
3. nested child domains are not added to package total;
4. physical measurement records package-domain scope and never labels it GPU/whole-system energy;
5. host provenance contains CPU model, affinity, thread count, frequency-policy observation, backend, compiler flags where relevant, OS and power context;
6. timing uses inference mode, separates cold/warm, and covers at least 16 distinct held-out examples through the complete declared solve path;
7. memory output separates packed weights, full model state, search-tree memory, sampled process RSS peak and lifetime high-water where available; sampling limits are disclosed;
8. instrumentation overhead is measured and retained;
9. MAC/op convention is consistent in throughput and arithmetic intensity, with raw `macs` and `operations` fields;
10. the old K-input benchmark is explicitly labeled precomputed-input reuse and an actual sequential recurrence benchmark is retained separately;
11. activation/state traffic is represented in the stated traffic model and no cache-residency/bandwidth-bound conclusion is inferred solely from a flat curve;
12. raw data plus one deterministic regenerated figure and provenance are retained;
13. focused M13 tests and the full fast regression pass;
14. `docs/RESEARCH_STATE.md` records the measured availability/limitations exactly.

A host with no valid physical RAPL access can still pass M13 if the protocol correctly reports physical energy as unavailable and all controlled counter contracts pass. M13 acceptance is about measurement integrity, not producing a non-null joule number.

**Stop after M13. Do not begin M14 automatically.**
