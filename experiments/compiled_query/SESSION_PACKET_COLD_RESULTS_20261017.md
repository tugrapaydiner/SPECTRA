# First-use exact sessions — 17 October 2026

## Result

The installed exact API now keeps the SCC advantage when a caller starts from an
already admitted `ExactProblem` and an ordinary 1,024-request bank.

The measured path is the public first-use sequence:

```python
packet = pack_session_packet(problem, requests)
answer = runtime.solve_session_packet(packet, mode="scc")
```

Both calls are timed. The result therefore includes Python validation and SBC1
serialization, two independent native SBC1 parses, fresh solver and observer
construction, all decisions, complete labels or contradiction proofs,
independent native checking, SCQ2 serialization, immutable Python-byte ownership,
and native disposal.

On the two exposed WAP development banks, SCC used **68.7% of raw time on
x86-64** and **67.9% on native ARM64**. That is a **31.3–32.1% reduction in
complete first-use latency**, with both individual graphs below the fixed 0.80
limit.

| Environment | WAP05 SCC/raw | WAP06 SCC/raw | Geometric mean | Reduction | Gate |
|---|---:|---:|---:|---:|---|
| Linux x86-64, CPython 3.13 | 0.689349 | 0.684990 | **0.687166** | **31.3%** | PASS |
| Native Linux ARM64, CPython 3.13 | 0.685691 | 0.672915 | **0.679273** | **32.1%** | PASS |

Every environment ran 31 paired repetitions per graph after five untimed
warmups. Mode order was shuffled deterministically for each pair. All packet and
output bytes were stable across repetitions, raw and SCC outputs were
byte-identical, and the complete SCQ2 banks passed standard-library replay.

## What changed

The previous cold diagnostic started from a multi-megabyte research JSON envelope
and repeatedly paid work shared by both engines. It was useful as an audit, but
it did not represent the installed `spectra.exact` input boundary.

This change removes avoidable work from the real public path without changing the
native solver or weakening raw:

- an already canonical immutable edge tuple is admitted in one linear pass rather
  than copied into a set and sorted again;
- validated SBC1 integer banks are written with bulk native arrays instead of one
  `struct.pack_into` call per value;
- `pack_session_packet` no longer validates the same graph and request bank twice;
- standard-library SCQ2 replay preprocesses the immutable graph once per bank
  instead of rebuilding edge and colour tables for every answer.

Noncanonical edge inputs still take the normalizing fallback path. Strict SBC1,
SBQ1, SCQ2 and SBP1 bytes, API version 1, native ABI 3, decisions, labels, proofs,
default modes and resource limits are unchanged.

## Correctness

The candidate passed 17 focused public contracts on both x86-64 and ARM64 before
timing. These include:

- byte equality with the previous scalar SBC1 serializer;
- canonical and noncanonical `ExactProblem` admission;
- a 65-row bank spanning two tile64 blocks;
- SAT, empty-domain UNSAT and implication-path UNSAT;
- malformed prepared-input rejection;
- strict SCQ2 decoding and proof mutation checks; and
- public standard-library replay without loading native code.

The benchmark then replayed all 4,096 distinct graph/query decisions represented
by the two raw and two SCC output banks in each environment. No decision,
complete label, proof, packet byte or output byte differed.

## Relationship to the earlier audit

The earlier JSON-bound composite audit remains a recorded failure: SCC/raw was
0.929 on x86-64 and 0.948 on ARM64 because shared JSON ingestion and case
reconstruction dominated the endpoint. Its thresholds are not edited or
retroactively reinterpreted.

That same audit independently established large current-main advantages over
external competitors on these exposed banks:

- 2.81× over persistent MiniCard on x86-64 and 2.83× on ARM64;
- 2.86× over persistent CaDiCaL on x86-64 and 2.93× on ARM64; and
- 3.94× over reduced native CUDD on x86-64.

The new result closes the remaining product-boundary problem: the public
`ExactProblem -> SBC1 -> SCQ2` first-use path now preserves a large SCC/raw
advantage instead of allowing unrelated JSON work to erase it.

## Boundaries

This is exposed development evidence, not a new holdout or outside replication.
It covers two all-SAT WAP banks of 1,024 restrictions on CPython 3.13 CPU hosts.
It does not establish one-query superiority, high-pin ITC superiority, CPython
3.11 performance, universal SAT or graph-colouring dominance, production-tail
latency, or an automatic-selector result.

The next issue must remain separate. Automatic selection should begin only after
this candidate passes the repository's full cross-platform, sanitizer, package,
CPU-regression and WAP-regression matrices.

## Identities

- Base `main`: `4a647e54e7e7e0756e74b52575f86f03d2764455`
- Candidate implementation: `415036763dc9ea918cbcf69e4a87da23d05c43d4`
- Candidate tree: `79b3ed08cf84d89d16f938753ab02771c5137eb4`
- Benchmark workflow run: `39072623159`
- x86-64 artifact: `12115876545`, ZIP SHA-256
  `492d21e9e47309b9a6eb0c7c1986c24ced14c1cf685605e68ab0b004cfced6c8`
- ARM64 artifact: `12116240371`, ZIP SHA-256
  `85f175214ce6991d506804a7b95b5e1c0af73d95abcb2c6d488c6e19d9b2fb67`
