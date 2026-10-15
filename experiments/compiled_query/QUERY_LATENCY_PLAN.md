# Faster short requests without losing checked answers

This is a development continuation of the published prepared service at
`39e9ee0b89f1261106231ee94546758c510291ba`. The 48 synthetic structural-panel
graphs are already exposed. This comparison is neither a new confirmation nor
an operator workload. The original frozen protocol and failed speed gate remain
unchanged.

The changes remove a duplicate routed-graph sort and reuse a fixed-graph
contradiction proof inside a single bank. Every current request is still parsed,
empty domains keep their own proof, and the independent observer checks every
emitted certificate. Nothing is cached between calls.

The optional `auto` packet mode uses the raw engine for fewer than 64 requests or
a base-infeasible graph. For other banks it uses routed proofs if the SCC count
is at most the vertex count, and plain SCC batching otherwise. This fixed policy
uses graph structure and bank length, without consulting request answers or
timings. The half-size threshold is a heuristic, not a proven optimal rule.
The response reports the selected engine. No default changes.

## Comparison

`query_latency.py` builds the published and candidate C++ sources with the same
compiler, flags and Python ABI. Both complete sources, build receipts and binary
hashes are retained. Six arms are fixed: published raw/route and candidate
raw/SCC/route/auto. The five bank lengths remain 1, 63, 64, 65 and 1,024, with nine
paired repetitions and two warmups. Forward, reverse and fixed shuffled orders
cycle across repetitions; arm order is randomized deterministically. Every
graph gets a fresh process, one available CPU, a 120-second deadline, a 2 GiB
address-space ceiling and a 512 MiB native array admission cap.

The timer includes input hashing, the complete checked SBC1-to-SCQ2 native call,
and output hashing. Packing, compilation, process startup, evidence writes and
Python replay are outside it. Worker wall time and peak RSS are separate. The
native payload limit does not bound total process RSS.

Every measurement is a separate immutable, fsynced file. A worker receipt binds
every record and both binaries; missing, changed, duplicate or failed cells fail
the functional check. All unique packet/output pairs receive original-input
Python replay. Same-mode candidate/published outputs must match byte for byte;
auto must match its reported candidate engine. Native verification occurs on
every call, including repeated identical outputs.

Report all 48 graphs, all eight families and all five bank lengths. Aggregate
ratios use graph medians and geometric means, not a pooled query count. Fixed
family bootstrap intervals use 4,000 graph resamples. Retain every case over 25%
slower. The development target for auto against the equally improved raw engine
is a long-bank ratio <= 0.90, upper 95% bound < 1, every family <= 1.10, and a
single-request ratio <= 1.10. This is a new development target, not a revision of
the failed v1 routed gate. An incomplete run cannot pass or report a complete
panel speedup. Do not retune the policy based on this run and call it confirmation.

## Reproduce

From a clean commit with a C++17 compiler and Python development headers:

```bash
python -m experiments.compiled_query.query_latency run --out query-latency-run
python -m experiments.compiled_query.query_latency audit --out query-latency-run
```

Use a new output directory each time. Failed runs are retained. Project-owned
x86/ARM CI establishes portability; it cannot establish outside reproduction,
adoption, novelty or production latency. Windows and ordinary-laptop measurements
remain open.
