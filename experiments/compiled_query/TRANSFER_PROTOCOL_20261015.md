# Prospective structural transfer protocol — 15 October 2026

This protocol is frozen before this panel is generated or measured. It challenges
the unified SBC1 endpoint; it is not a new WAP confirmation, operator workload,
external replication, or proof of priority. Never describe synthetic instances as
real operational evidence. No benchmark result changes a default automatically.

## Fixed inputs, before outcomes

`transfer_panel.py` specifies eight randomized graph families: random recursive
trees, percolated grids with sparse diagonals, radius geometric graphs,
preferential-attachment graphs, rewired rings, four-community graphs,
percolated hypercubes and random bipartite graphs. Six realizations per family
cross 128/512 vertices with three domain regimes: shared binary colours,
unplanted random two-of-eight colours, and the latter with deterministic singleton
positions. Seeds are SHA256-derived from the published version/family/replica.
All 48 instances are generated once after publication. No solving, repair,
SAT filtering, topology rejection or seed search is allowed during generation.
Base-infeasible instances remain in the panel and receive separate interpretation.

Each case has 1,024 original-address requests, with fixed distributions of widths
1/2/4/8, empty requests, full allowed domains, duplicate conflicting pins,
empty allowed masks, and irrelevant bit 63. Every request is retained.
The five fixed bank sizes are 1, 63, 64, 65, and 1,024. Prefixes, ordering variants
and timed repetitions are not independent graph samples.

## Fixed implementations and costs

Four explicit same-certificate arms: the prior prepared routed C++ source from
commit `3d085050ff48a907d95571900c1f6481af8c2972`, and the new prepared raw, SCC,
and routed implementations. The old source is retained and compiled on the same
host with the same compiler options; no old measurements are imported.
All arms return complete original SAT labels or portable SBP1 contradictions
serialized as SCQ2. Full independent native output checking is inside every call.

Primary endpoint: already available immutable SBC1 bytes -> checked SCQ2 bytes,
including input SHA256, native construction, solving, witnesses/proofs,
independent checking, serialization, output SHA256, and native disposal.
Packet packing is measured separately, not silently free in a Python-input claim.
File I/O, native compilation, process startup, warmups and Python replay are outside
this warm ready-byte endpoint. Cold worker wall time and whole-worker peak RSS are
reported separately. Native payload admission is NOT a hard total RSS limit.

Every graph uses a fresh process with a 120-second wall deadline, 2 GiB virtual
address-space bound and 512 MiB native payload cap. Worker affinity is pinned to
one available CPU. Each bank size has nine paired repetitions, with forward,
reverse and deterministically shuffled request orders cycled three times.
Configuration order is randomized within each paired block. Two warmups per arm
and bank size occur at the first block. Every timed record is retained; no retries
until success, dropping outliers, or selecting a winning comparator afterwards.

## Fixed gates

Functional gate: all 8,640 scheduled timed calls complete, agree on decisions,
and supply outputs accepted by the independent original-input Python verifier.
Every output and packet is stored by SHA256; byte-identical repeats can share
storage and post-run semantic replay, while EVERY native call still fully verifies.
Count unique instances, repeated calls, unique outputs and checked decisions separately.

Speed gate for the routed endpoint versus the equally optimized raw endpoint:
- long-bank graph-geometric mean of within-graph median time ratios <= 0.90;
- upper endpoint of the fixed-family, graph-resampled 95% bootstrap interval < 1;
- no long-bank family geometric ratio > 1.10;
- overall single-request geometric ratio <= 1.10.

Report prior-route and SCC comparisons regardless of outcome. Flag every
per-graph >25% regression. Missing/error/timeout cells cause gate failure and
cannot produce a finite complete-panel speedup. No threshold changes after data.
Nine repetitions do not justify production p99 claims. The bootstrap has 4,000
fixed-seed resamples within the eight fixed families; it does not estimate all
possible graph families. Nonrandom domain/size strata limit generalization.

## Reproducibility and scope

Bind the published protocol commit, every executing Python/C++ source, every
case and packet, both native libraries, compiler receipts, host and interpreter
identities. Preserve full stdout/stderr and interrupted workers. Run current
semantic tests and sanitizers before inference; test fixtures may use tiny custom
examples but MUST NOT consume these 48 instances before the freeze.

Run the same frozen inputs on local Linux x86-64 and project-controlled CI with
Python 3.11/3.13 and Linux ARM64 where the infrastructure permits. These executions
are portability evidence, not independent-team scientific replication. An exact
code or harness change invalidates this v1 freeze: retain v1 and use a separately
named version for subsequent changes.
