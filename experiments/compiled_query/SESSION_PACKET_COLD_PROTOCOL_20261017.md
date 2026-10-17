# First-use exact-session protocol — 17 October 2026

## Question

Can the installed SPECTRA API turn an already admitted fixed problem and a
1,024-request bank into a complete, independently checked answer bank with a
large advantage over its own optimized raw engine?

This is the cold path that matters to an application using `spectra.exact`:

```python
packet = pack_session_packet(problem, requests)
answer = runtime.solve_session_packet(packet, mode="scc")
```

Both calls are timed. Native construction is not hidden. The packet may also be
persisted and reused, but this study does not grant SPECTRA that advantage.

## Classification

Development evidence on the already exposed WAP05 and WAP06 graph/query banks.
It is not a new holdout, outside replication, operator traffic, or a universal
SAT/colouring claim. The prior JSON-harness result remains recorded as a failed
composite gate; this protocol does not edit or reinterpret it.

The old harness began from a multi-megabyte research JSON envelope and rebuilt
several canonical representations common to both engines. That is useful audit
work, but it is not the installed `spectra.exact` input contract. This protocol
starts from an `ExactProblem` plus request rows and charges every public API step
needed to create and execute immutable SBC1.

## Frozen source and inputs

- Base implementation: merged `main` commit
  `4a647e54e7e7e0756e74b52575f86f03d2764455`.
- Upstream graph repository: `marijnheule/clicolcom` at commit
  `4932048642da2144f387961b595112277afff82f`.
- Cases: `wap05a.col` and `wap06a.col` reconstructed through the retained
  answer-free identity checks and historical disclosed query generator.
- Each case contains exactly 1,024 historically SAT-accepted restrictions.
- No input is selected or removed after timing.

## Candidate change

The candidate may improve only work performed before or during the public
session call while preserving exact API 1, native ABI 3, SBC1/SBQ1/SCQ2/SBP1
bytes, decisions, complete labels, proofs, and independent checking.

The intended changes are:

1. recognize an already canonical immutable edge tuple in one linear validation
   pass instead of rebuilding a set and sorting it again;
2. serialize admitted integer banks with bulk native arrays instead of one
   `struct.pack_into` call per value;
3. avoid a second graph/request validation pass inside `pack_session_packet`;
4. preprocess the immutable problem once during standard-library SCQ2 replay,
   rather than rebuilding its edge set and colour table for every answer.

The fallback path must continue to normalize unordered, reversed, duplicated,
or non-tuple edge inputs exactly as before. Prepared bytes remain strict and are
never repaired silently.

## Timed endpoint

For each arm and repetition, the timer includes:

- `ExactProblem` re-admission performed by `pack_session_packet`;
- validation and freezing of all 1,024 request rows;
- complete SBC1 allocation and serialization;
- two independent native SBC1 parses;
- fresh solver and observer construction;
- all 1,024 exact decisions;
- complete original-vertex labels or portable contradiction proofs;
- independent native checking of every answer;
- SCQ2 serialization and immutable Python-byte ownership; and
- native disposal before the call returns.

Outside the timer, but retained:

- process startup and native compilation;
- WAP graph acquisition and case reconstruction;
- one full standard-library SBC1/SCQ2 replay per graph;
- evidence writing and analysis.

## Arms and schedule

- Explicit `raw` and explicit `scc` only.
- CPython 3.13 on GitHub Ubuntu 24.04 x86-64 and native ARM64.
- One available CPU per process where affinity control is supported.
- Five untimed warmups per mode and graph.
- Thirty-one paired repetitions per graph.
- Mode order shuffled deterministically for every pair.
- Garbage collection disabled only across the paired timing region.
- Every packet and output must remain byte-identical across repetitions.

## Correctness gate

Before any performance statement:

- the scalar-reference and bulk SBC1 serializers must be byte-identical;
- canonical and noncanonical `ExactProblem.create` paths must agree;
- raw and SCC SCQ2 outputs must be byte-identical on both graphs;
- standard-library `verify_session_output` must accept both complete outputs;
- the existing 65-row/two-tile contracts must pass;
- malformed prepared inputs must remain rejected;
- all exact API, installed-package, sanitizer, Windows and CPU regression gates
  triggered by the final pull request must pass.

Any wrong answer, invalid proof, malformed packet, crash, timeout, unavailable
baseline, or missing case blocks the result.

## Meaningful-win gate

Ratios are complete `pack_session_packet + solve_session_packet` time for SCC
divided by the same complete time for raw. Lower is better.

The gate passes independently on x86-64 and ARM64 only if:

- both graph cells complete and replay;
- the geometric mean ratio is below `0.75`; and
- each graph ratio is below `0.80`.

The thresholds are fixed before remote timing. They are not relaxed after
seeing results. A pass closes the remaining first-use cold-path issue for this
exposed WAP scope; it does not authorize an automatic-selector or universal
solver claim.
