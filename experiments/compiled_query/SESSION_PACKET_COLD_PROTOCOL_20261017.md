# First-use exact-session protocol — final pass, 17 October 2026

## Question

Can the installed SPECTRA API turn an already admitted fixed problem and an
ordinary 1,024-request bank into a complete, independently checked answer bank
while using at most 60% of the time required by its own optimized raw engine?

The primary measured path is:

```python
packet = pack_session_packet(problem, requests)
answer = runtime.solve_session_packet(packet, mode="scc")
```

Both public calls are timed. A secondary guard also reconstructs the immutable
`ExactProblem` inside each timed call, so moving fixed-graph work to admission
cannot hide a regression for a one-bank application.

## Classification

This is development evidence on the already exposed WAP05 and WAP06 graph/query
banks. It is not a new holdout, outside replication, operator traffic, or a
universal SAT or colouring claim. The earlier JSON-bound failed gate and the
first 0.75 session-packet pass remain in Git history and are not edited.

## Frozen implementation scope

The candidate may:

1. enforce `ExactProblem` invariants at construction so a valid immutable
   instance is not re-admitted for every bank;
2. encode masks and canonical edges once during problem admission;
3. validate and serialize each request pin in one pass; and
4. preprocess immutable replay data once per output bank.

It may not change exact API 1, native ABI 3, SBC1/SBQ1/SCQ2/SBP1 bytes, solver
logic, raw mode, decisions, complete labels, proofs, default modes, public
limits, or independent checking.

## Inputs and schedule

- Upstream graphs: `wap05a.col` and `wap06a.col` at clicolcom commit
  `4932048642da2144f387961b595112277afff82f`.
- Exactly 1,024 historically SAT-accepted restrictions per graph.
- Explicit `raw` and `scc` modes only.
- CPython 3.13 on GitHub Ubuntu 24.04 x86-64 and native ARM64.
- Five untimed warmups; 31 paired repetitions per graph.
- Deterministically shuffled mode order in every pair.
- Garbage collection disabled only during paired timing.
- Stable packet and output bytes required across repetitions.

The final evidence binds the exact implementation source hashes, native binary,
input packets, output banks, environment and CPU receipt. Any later executable
code change requires a new identified timing run; a documentation-only follow-up
cannot silently inherit an earlier code result.

## Timed boundaries

### Admitted-problem bank boundary

The primary timer includes request validation, SBC1 request-array serialization,
two independent native parses, fresh solver and observer construction, all
decisions, full labels or proofs, independent native checking, SCQ2
serialization, immutable Python-byte ownership, and disposal.

Problem admission, process startup, compilation, graph acquisition, case
reconstruction, one standard-library replay, and evidence writing are outside
this paired timer. Problem admission time and cached fixed-graph bytes are still
recorded.

### One-bank application guard

A second paired timer begins before `ExactProblem.create`, then performs packet
packing and the same complete native call. This guard charges canonical graph
validation and fixed-byte creation once per compared bank. It exists to prevent
the primary result from merely relocating cost outside the measured path.

## Correctness gate

Before any performance statement:

- bulk and scalar SBC1 bytes must match;
- direct and factory problem construction must enforce the same invariants;
- raw and SCC SCQ2 bytes must be identical;
- standard-library replay must accept every complete bank;
- SAT, empty-domain UNSAT, implication-path UNSAT, 65-row/two-tile,
  malformed-input, resource-limit, proof-mutation, Windows, sanitizer, package,
  CPU-regression and WAP-regression contracts must pass; and
- any wrong answer, invalid proof, malformed packet, crash, timeout or missing
  case blocks the result.

## Predeclared meaningful-win gates

Ratios are SCC time divided by raw time. Lower is better. The result passes only
if both architectures satisfy all gates.

### Admitted-problem bank

- geometric-mean ratio below `0.60`; and
- every graph ratio below `0.65`.

### One-bank application guard

- geometric-mean ratio below `0.75`; and
- every graph ratio below `0.80`.

These thresholds are frozen before final remote timing and are not relaxed
afterward. A pass closes the first-use long-bank issue only for this exposed WAP
scope; it does not authorize automatic-selection or universal solver claims.
