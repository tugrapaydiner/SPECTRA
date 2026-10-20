# Mixed direct 2-SAT holdout protocol

**Frozen before graph acquisition:** candidate names, Git-blob identities and byte
lengths were read from upstream tree metadata only. The graph blobs themselves
must not be fetched until this protocol, generator and gates are committed.

## Question

Does the public SPECTRA `auto` policy beat a separately implemented direct 2-SAT
service on previously unopened graph topologies when every request bank contains
both SAT and UNSAT decisions and both systems pay the same complete ready-packet
cost?

This is a narrow exact-service question. It is not a general SAT or ordinary
graph-colouring competition.

## Frozen source family

Upstream repository and commit:

```text
marijnheule/clicolcom
4932048642da2144f387961b595112277afff82f
```

The four graphs were selected by filename/family and metadata before reading their
contents. None of the four names was present in SPECTRA's active indexed source at
freeze time.

| Family | Graph | Bytes | Git blob SHA-1 |
|---|---|---:|---|
| random DIMACS | `DSJC250.5.col` | 143,142 | `b8a80c7ea4ae5e2df00ac901dae80d637e6d419c` |
| flat/planted | `flat300_28_0.col` | 201,265 | `e05f69de8d2dece0ad1327ccc3c2ea22a4dc24a1` |
| Leighton | `le450_15c.col` | 158,884 | `f3dc12ae8d0ea898b0816df4680bf661620c4bb7` |
| geometric | `miles1000.col` | 26,899 | `0aef7ac921ea7b7acac9ded3f446e17af02ed1fa` |

A missing graph, byte-length mismatch, Git-blob mismatch, parser failure, more than
63 deterministic DSATUR colours, fewer than 512 edges, duplicate generated query,
or any other admission failure is retained as a failed graph. No replacement is
allowed after opening.

## Frozen workload construction

For each graph:

1. Strictly parse the original DIMACS graph.
2. Compute the existing deterministic DSATUR colouring using saturation, degree,
   then vertex ID as tie-breakers.
3. Add one new `trap` colour above the DSATUR palette. Every original vertex gets
   exactly two allowed colours: its DSATUR witness colour and the trap colour.
   The DSATUR colouring is therefore a complete known witness for the base case.
4. Shuffle the original edge list and vertex list with the fixed generator seed
   and graph filename. No solver answer or timing is consulted.
5. Generate exactly 1,024 unique queries in fixed order:
   - even rows are SAT by construction and contain only pins to the known DSATUR
     witness; row zero is the empty base query;
   - odd rows are UNSAT by construction and pin both endpoints of a distinct
     original edge to the shared trap colour, optionally with additional witness
     pins.
6. Retain the complete case, expected SAT/UNSAT vector, graph hash, DSATUR witness
   hash, masks, queries and generation receipt before timing.

The bank contains exactly 512 SAT and 512 UNSAT rows. There is no SAT filtering,
solver-guided rejection, answer-dependent regeneration or post-open case deletion.

Generator master seed:

```text
0xD12EC72A75A7C0DE
```

## Frozen arms

All six arms receive byte-identical `SBC1` packets and must return complete `SCQ2`:

- SPECTRA `raw`;
- SPECTRA `scc`;
- SPECTRA public `auto`;
- direct 2-SAT closure policy;
- direct 2-SAT scalar DAG; and
- direct 2-SAT tile-64.

The direct challenger has its own parser, SCC construction, solver, proof builder,
serializer and original-input observer. It does not consume SPECTRA SCC IDs,
answers or timing history. For each cell the primary competitor is the fastest
median among its three engines, giving the challenger hindsight rather than
forcing a weak policy.

The primary SPECTRA result is `auto`. The faster explicit SPECTRA arm is secondary
and cannot rescue a failed public-auto gate.

## Measurement boundary

The packet is prepared before timing. Each timed call includes:

- packet parsing and validation;
- implication graph and component construction;
- all requested decisions;
- complete original-colour SAT labels or fresh `SBP1` UNSAT proofs;
- an independently constructed native original-input observer;
- `SCQ2` serialization and immutable return bytes; and
- native state disposal.

Packet generation, native compilation, workflow startup and later Python replay
are retained separately and excluded symmetrically.

Each architecture uses 30 measured repetitions after three untimed warmups.
Six-arm execution order rotates through complete position-balanced blocks from a
fixed seed. Sub-millisecond cells use calibrated repeated calls with per-call
costs; calibration is bounded and recorded. No measured repetition may be removed.

Environments:

- GitHub-hosted Ubuntu 24.04 x86-64, CPython 3.13;
- GitHub-hosted Ubuntu 24.04 native ARM64, CPython 3.13;
- UBSan and ASan+UBSan correctness runs before graph opening.

Request prefixes are `1`, `8`, `64` and `1,024`. The 1- and 8-query cells are
reported diagnostics. The primary sustained endpoints are 64 and 1,024.

## Correctness gates

Every graph and architecture must satisfy all of the following:

1. graph bytes match the frozen metadata;
2. generated case and expected-status receipts replay from scratch;
3. every arm returns the exact frozen 512/512 SAT/UNSAT sequence at 1,024 rows;
4. every SAT row has a complete original-vertex colouring;
5. every UNSAT row has an independently replayable original-literal proof;
6. every full `SCQ2` bank passes standard-library replay; and
7. no timeout, allocation failure, sanitizer finding, malformed output or checker
   failure is converted into a mathematical result.

Any correctness failure makes the holdout a failure regardless of speed.

## Performance gates

Ratios are `SPECTRA auto / fastest direct engine`; lower is better for SPECTRA.
Architectures are judged separately and are never pooled.

### 64-query endpoint

- geometric mean across the four unopened graphs `<= 0.90`; and
- every graph ratio `<= 1.05`.

### 1,024-query endpoint

- geometric mean across the four unopened graphs `<= 0.75`; and
- every graph ratio `<= 0.90`.

The one-query endpoint has no pass threshold because development evidence already
contains a real ARM64 loss. It must remain visible. The eight-query endpoint is
also diagnostic and cannot substitute for either primary endpoint.

The overall verdict is PASS only if every correctness gate and all four primary
performance conditions pass on both architectures. Thresholds may not be relaxed,
results pooled, graphs replaced, or explicit modes substituted for `auto` after
opening.

## Claim boundary after a pass

A pass would establish that SPECTRA's public automatic exact service beats this
strong project-authored direct 2-SAT challenger for sustained mixed-result banks on
four prospectively unopened DIMACS graph topologies under the frozen contract.

It would still not prove general SAT leadership, ordinary graph-colouring
leadership, asymptotic novelty, external implementation superiority, production
readiness, operator value or independent replication.