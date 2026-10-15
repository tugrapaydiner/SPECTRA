# Faster queries: results and remaining gaps

The published routed service spent too much time preparing short requests and
rebuilding a contradiction proof when the fixed graph was already impossible.
This continuation fixes both costs and adds an optional `auto` engine choice.
The original frozen result and failed transfer gate remain unchanged.

Native implementation: `ece18d869674e1d9eb988eb07dc8b2d7cba5014d`.
Final audit: `9d48b7dccf61283937a5eceb7401b4bfc7d43c1d`.
Published comparison source: `39e9ee0b89f1261106231ee94546758c510291ba`.
Branch: `faster-queries`. [Plan](QUERY_LATENCY_PLAN.md) and
[machine-readable results](QUERY_LATENCY_RESULTS_20261015.json).

## What changed

- Routed construction now builds its DAG and proof bridges from one sorted edge
  pass. It no longer sorts a second equivalent list.
- Within a batch, an impossible fixed graph reuses its original contradiction
  proof. Every current request is still parsed, an empty domain keeps its own
  proof, and the separate observer checks every emitted proof. No request result
  or mutable state is shared between calls.
- Optional `mode='auto'` uses raw traversal for fewer than 64 requests or a
  base-infeasible graph. Otherwise it chooses routed proofs when SCCs at least
  halve the literal graph, or plain SCC batching when they do not. The result
  reports `info['selected_mode']`. This is a fixed heuristic, not an optimality
  claim. Existing modes and defaults keep their meanings.
- The exact extension builds on Windows x64 through the existing MSVC build
  support. Its portable bit operations do not require POPCNT. Linux build flags
  and the SBC1, SCQ2, SBP1 and ABI-3 contracts are retained.
- The new comparison writes each timing record once, binds it to a worker
  receipt, rejects partial graph rosters, and replays original-input witnesses
  and proofs. It does not edit or reuse the old frozen harness as a new study.

## Measured costs

These are **already exposed synthetic graphs**: eight families, six realizations
each, five bank sizes and nine paired repetitions. All six arms return complete
checked answers. The primary timer includes input hashing, native construction,
solving, independent native checking, output serialization, disposal and output
hashing. Packing, file I/O, compilation and process startup are reported outside
the ready-packet endpoint.

The Linux comparison of the final native implementation completed **12,960
timed calls per environment** and passed original-input replay on all three
environments. Ratios below one favour
auto. Each value is the geometric mean of 48 within-graph median ratios.

| Environment | One request: auto / published route | 1,024 requests: auto / published route | 1,024 requests: auto / improved raw | New 10% target |
|---|---:|---:|---:|---|
| Linux x86-64, Python 3.11 | 0.6570 | 0.8292 | 0.9700 | **FAIL** |
| Linux x86-64, Python 3.13 | 0.7242 | 0.7647 | 0.9583 | **FAIL** |
| Linux ARM64, Python 3.13 | 0.7412 | 0.7697 | 0.9716 | **FAIL** |

The new service has about **26–34% lower latency for one request** and
**17–24% lower latency for long banks** than the published routed endpoint on
this panel. There are no
per-graph regressions above 25% against that published endpoint at either length
in these final CI runs.

The stronger control changes the interpretation: raw traversal also benefits
from fixed-proof reuse. Auto's long-bank advantage over that improved raw engine
is only **2.8–4.2%**, short of the fixed 10% target. Its graph-bootstrap 95% ratio
intervals are `[0.9193, 1.0160]`, `[0.9068, 1.0056]` and `[0.9285, 1.0099]` in
the table's order. All three cross one. The same graph is over 25% slower than
improved raw on both x86 runs; its name and exact ratios remain in the JSON.
Explicit routed mode still has
short-request overhead. Neither the target nor the default is changed to hide it.

Across the panel, **33 of 48 fixed graphs are already infeasible**. Reusing their
proofs explains part of the gain; this mixture is not evidence about an operator's
production workload. Repetitions, prefixes and machines are not additional
independent graph samples. Nine repetitions do not establish production p99.

## Verification and evidence

- [Final Linux and ARM check](https://github.com/tugrapaydiner/SPECTRA/actions/runs/38689473310):
  **124 focused tests per environment**, full paired comparison and replay; both
  address/undefined and undefined-only sanitizer jobs passed. Leak detection was
  disabled, so this is not a LeakSanitizer result.
- [Windows run](https://github.com/tugrapaydiner/SPECTRA/actions/runs/38688566636):
  **133 passed** on Python 3.11 and **133 passed** on Python 3.13, including actual
  MSVC builds, exhaustive small problems, native MiniCard comparisons, proof
  mutation checks, malformed packets, memory admission and concurrent sessions.
  These are correctness results; no Windows speed or laptop claim is made.
- Local stable-package checks: **957 passed**, all **11 installed-wheel checks**
  passed, navigation passed, and **449 protected historical files** remained
  byte-identical. Historical model-training tests were not rerun in this change.
- The first constructor change matched **960 complete outputs byte for byte**
  against the published source, across every panel graph, bank length and explicit
  engine. The full comparison also requires same-mode byte equality and requires
  auto's output to equal its reported engine's output.
- The earlier local development run at `d1422a4` completed 12,960 calls and
  independently replayed **1,636 distinct outputs / 484,546 decisions**. Its
  target failed too. Peak fresh-worker RSS was **21,708,800 bytes** on these cases,
  not a hard runtime memory bound. Its separate
  [CI reproduction](https://github.com/tugrapaydiner/SPECTRA/actions/runs/38687643074)
  and all its outcomes are retained in the
  [earlier results JSON](QUERY_LATENCY_EARLIER_RUNS_20261015.json).
- The preceding [portable-native comparison](https://github.com/tugrapaydiner/SPECTRA/actions/runs/38688363961)
  also passed all correctness checks and failed the speed target. Its short-bank
  gains were 26–37%, long-bank gains 17–23%, and gains over improved raw 2.8–6.3%.
  Those results are retained in the earlier JSON. The final run above followed a
  fixed-roster audit correction with no native code change; the differences
  illustrate timing variation, not a reason to select the most favourable run.
- [Prior frozen transfer CI](https://github.com/tugrapaydiner/SPECTRA/actions/runs/38682131221)
  already passed correctness on x86 and ARM but failed its routed speed gate.
  [Its summaries](TRANSFER_CI_RESULTS_20261015.json) are retained separately.

The complete local evidence archive is named
`SPECTRA_Query_Evidence_20261015.zip`, with SHA-256
`96462b7f7d2c912e9f774843912712da23099b3d2bb9b4d6b0c47231096b6684`.
It contains raw inputs, outputs, timing records, receipts, logs and audits.
Source commits are on GitHub; the archive does not duplicate the repository.
CI artifacts have 90-day retention and are not a permanent archive.

Two failed attempts remain visible. The initial local reproduction of the old
harness failed because four append-only timing logs did not match their worker
receipts; it was retained without repairing hashes or claiming a valid result.
The cause of that mismatch was not established. The first
[Windows run](https://github.com/tugrapaydiner/SPECTRA/actions/runs/38688364007)
failed before compilation because the workflow's temporary Python script could
not import the checkout. A separate commit fixed the import path before the
successful Windows run.

## What still prevents a flagship claim

The stronger complete-cost speed target still fails. The optional policy has
only exposed-development evidence, and the service remains an experimental
binary-domain API. There is no outside-team reproduction, operator-defined
adoption, independent novelty review or production-tail evidence. The old 75/100
review is not automatically replaced by a new whole-project score, and this work
does not support a 100/100 claim.

Next, investigate the remaining cost on feasible mixed-outcome banks while
keeping the improved raw control. Any new algorithm needs a separately recorded
development comparison, followed by a committed protocol and genuinely unseen
workload. Stable API promotion also needs a compatibility policy and deployment
evidence. External replication and an application owner's workload require
outside participation; project-controlled CI cannot supply those results.
