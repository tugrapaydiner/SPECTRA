# Six-family fresh transfer study

**Frozen before graph acquisition:** 2026-10-20

## Question

Does the public SPECTRA `auto` exact service beat both:

1. official CaDiCaL 3.0.1; and
2. the fastest median of the separately implemented direct 2-SAT closure, DAG,
   and tile-64 engines,

on previously unopened graph topologies when every graph receives the same
proof-carrying first-use contract and a deterministic bank containing exactly
512 SAT and 512 UNSAT requests?

This is a narrow repeated exact-query comparison. It is not a general SAT,
ordinary graph-colouring, production, adoption, or asymptotic claim.

## Source freeze

The generator, benchmark, inventory, competitors, schedule, and gates were first
committed on the former `fresh-transfer-study` line before any candidate graph
blob was acquired:

- generator commit: `f4473163e5928f101237365c6b05ea9314b5cfb9`;
- benchmark commit: `7c2556adbaa7f4c5de79caee4027602fffb61533`;
- graph-free contract commit: `b3899b286904a2b28ba26f44bf4503ec21c1ed7f`.

The branches were subsequently consolidated, before graph opening, into clean
commit `c0d56123bcc97f60d777de0d52d5b85a70808a25`, tree
`102bf99ce3ad5447a12c98c29049be61e8914969`.

The benchmark-relevant frozen blobs are:

| File | Git blob SHA-1 |
|---|---|
| `fresh_transfer.py` | `64cecf816806db431c69b3529b4b2d13cb970d25` |
| `fresh_transfer_benchmark.py` | `4d5adb1101d3a94d2bdb95bdf7911efe37c595fe` |
| `direct_twosat.py` | `f73619902d0c9d2a3400c192d7c997ccc9ca6396` |
| `cadical_exact.py` | `313c9538451746085622fd7eced82355e6fa689d` |
| `spectra/_native/exact_binary.cpp` | `953388e463dbec3ee43b22032bfae61af2e574d6` |

### Consolidation amendment

The consolidation retained the newer structural-v11 and direct-conflict runtime
and added the completed edgeless session helper from the release-verification
line. That helper is reachable only when the original graph has zero edges.
Every frozen transfer candidate is required to contain at least 512 edges, so the
helper is outside this study's admitted input domain. No graph identity, query,
expected answer, measured arm, timing schedule, threshold, or competitor changed
during consolidation.

The exact source commit used by the opening workflow must descend from
`c0d56123bcc97f60d777de0d52d5b85a70808a25` and may add this protocol and the
opening workflow only. Any benchmark-relevant source change requires a new
unopened inventory and a new freeze.

## Unopened graph inventory

Upstream repository: `marijnheule/clicolcom`

Pinned upstream commit:
`4932048642da2144f387961b595112277afff82f`

The names, families, sizes, and Git blob identities below were selected from the
upstream tree metadata before reading the file contents.

| Family | Graph | Bytes | Git blob SHA-1 |
|---|---|---:|---|
| random DIMACS | `DSJC500.5.col` | 599,069 | `5732c76326e835fa9dd61fb5c4251d93d9a34648` |
| flat/planted | `flat300_20_0.col` | 198,295 | `3e37b08bfe90ed4dd57654ac959ef8c758729b1c` |
| Leighton | `le450_25c.col` | 165,335 | `615254414a3ef5784d0fc130573f2c14cbf7bba0` |
| geometric | `miles1500.col` | 43,473 | `937ba541922a0e2ef4d1e9bf49cef49271a8e909` |
| queen | `queen16_16.col` | 58,010 | `d1e1057d1e70fbea9a1318a0bf53c7321ec62b4c` |
| school scheduling | `school1.col` | 180,775 | `10c1165916a0edfb5941af2c5c914b1b6199ff28` |

A missing graph, byte or blob mismatch, strict parser failure, fewer than nine
vertices, fewer than 512 edges, more than 63 deterministic DSATUR colours,
duplicate request, resource failure, timeout, malformed answer, or checker
failure is retained as a failed graph. No replacement is allowed after opening.

## Answer-free workload construction

For each graph:

1. Parse the original DIMACS topology strictly.
2. Compute the existing deterministic DSATUR colouring using saturation, degree,
   then vertex ID as tie-breakers.
3. Add one new trap colour. Each original vertex receives exactly two colours:
   its DSATUR witness colour and the shared trap colour.
4. Shuffle the original edge and vertex lists from the fixed master seed and
   graph name.
5. Generate exactly 1,024 unique requests in fixed order:
   - row zero is the unrestricted SAT base case;
   - every other even row encodes its index as a unique nonempty subset of nine
     witness-colour pins and is SAT by the retained DSATUR witness;
   - every odd row chooses a distinct original edge, pins both endpoints to the
     shared trap colour, and is UNSAT by that edge alone.
6. Verify every expected result directly from the disclosed construction without
   calling a solver.

The outcome vector is fixed before opening:

- 512 SAT rows;
- 512 UNSAT rows;
- alternating SAT/UNSAT from row zero;
- no solver filtering, rejection sampling, timing feedback, or post-open deletion.

Master seed: `0x6D4B1F29A7C3E850`.

## Measured arms

Every arm receives byte-identical immutable `SBC1` packets and must return complete
proof-carrying `SCQ2` output:

- SPECTRA public `auto`;
- official CaDiCaL 3.0.1 adapter pinned to commit
  `c60730422e758ef1cebe7aeddf2dda31c996bf04`;
- direct 2-SAT `auto`;
- direct 2-SAT scalar DAG; and
- direct 2-SAT tile-64.

For each cell, SPECTRA is compared against the fastest median of the three direct
engines, giving the challenger hindsight rather than forcing a weak policy.

## Measurement boundary

The immutable packet is prepared before timing. Every timed call includes:

- packet parsing and validation;
- implication graph and component construction;
- all requested SAT/UNSAT decisions;
- complete original-colour labels or fresh portable contradiction proofs;
- independent original-input checking inside each service;
- `SCQ2` serialization and immutable return bytes; and
- native state disposal.

Compilation, workflow startup, graph acquisition, case generation, and later
standard-library replay are retained separately and excluded symmetrically.

Counts are `1`, `8`, `64`, and `1,024`. Each architecture uses 30 measured
repetitions after three untimed warmups. The five arms rotate through complete
position-balanced blocks from a fixed per-cell seed. Short cells use calibrated
batched calls with per-call costs; every repetition is retained.

Architectures are judged separately:

- GitHub-hosted Ubuntu 24.04 x86-64, CPython 3.13;
- GitHub-hosted Ubuntu 24.04 native ARM64, CPython 3.13.

## Correctness gates

Every graph, count, architecture, and arm must satisfy all of the following:

1. graph bytes match the frozen identities;
2. generated cases and expected-status receipts replay from scratch;
3. the complete frozen SAT/UNSAT sequence is returned;
4. every SAT row contains a complete original-vertex colouring;
5. every UNSAT row contains an independently replayable original-literal proof;
6. every complete output passes standard-library replay;
7. CaDiCaL identifies the pinned official commit and returns portable proofs; and
8. no failure is converted into SAT, UNSAT, a timeout win, or a missing cell.

Any correctness failure makes the study fail regardless of speed.

## Frozen performance gates

Ratios are `SPECTRA auto / competitor`; lower is better for SPECTRA. Both the
geometric-mean and every-graph limits must pass independently on each architecture.

### Official CaDiCaL 3.0.1

| Requests | Geometric mean | Every graph |
|---:|---:|---:|
| 1 | `<= 0.30` | `<= 0.50` |
| 8 | `<= 0.25` | `<= 0.40` |
| 64 | `<= 0.15` | `<= 0.25` |
| 1,024 | `<= 0.10` | `<= 0.20` |

### Fastest direct 2-SAT engine

| Requests | Geometric mean | Every graph |
|---:|---:|---:|
| 1 | `<= 0.85` | `<= 1.10` |
| 8 | `<= 0.75` | `<= 1.00` |
| 64 | `<= 0.50` | `<= 0.75` |
| 1,024 | `<= 0.35` | `<= 0.60` |

The overall verdict is PASS only if every correctness gate and all 16 numerical
conditions pass on both architectures. Results may not be pooled across machines,
thresholds relaxed, graphs replaced, explicit SPECTRA modes substituted for
`auto`, or failed cells removed after opening.

## Claim boundary

A pass would show a large, prospectively frozen advantage for SPECTRA's public
automatic exact service over official CaDiCaL 3.0.1 and this strong project-authored
direct 2-SAT challenger on sustained mixed-result banks across six unopened graph
families under the stated proof-carrying first-use contract.

It would still not establish general SAT leadership, ordinary graph-colouring
leadership, superiority to every possible 2-SAT implementation, production value,
external adoption, or independent replication.
