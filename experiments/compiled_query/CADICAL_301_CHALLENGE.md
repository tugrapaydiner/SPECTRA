# Official CaDiCaL 3.0.1 challenge

**Status:** protocol frozen before adapter compilation or timing  
**SPECTRA source:** `63b724b8cb897faadeba57d2ae5e48c4362dca5d`  
**External solver:** `arminbiere/cadical` tag `rel-3.0.1`  
**Pinned external commit:** `c60730422e758ef1cebe7aeddf2dda31c996bf04`

## Question

Does SPECTRA's public `auto` exact service beat the official CaDiCaL solver core by
a large margin when both systems receive the same ready `SBC1` request packet and
must return the same complete, independently checked `SCQ2` witness/proof bank?

This challenge addresses a stronger source boundary than the project-authored
purpose-built 2-SAT challenger. The SAT decision engine and incremental-assumption
implementation come unmodified from the official CaDiCaL repository. The packet
adapter, original-colour reconstruction, portable `SBP1` proof translation,
observer, and benchmark harness remain project-authored and are charged in full.
The result therefore tests an external solver core; it is not outside-team
replication.

## Frozen external source boundary

The workflow must check out exactly:

```text
repository: arminbiere/cadical
commit:     c60730422e758ef1cebe7aeddf2dda31c996bf04
release:    rel-3.0.1
```

The source commit, build commands, compiler identity, resulting static/shared
library hashes, adapter source hash, and final combined binary hash are retained.
No vendored or modified CaDiCaL source is permitted. A source, tag, signature,
header, build, or runtime-identity mismatch is a failed run, not a substitution.

## Matched formula and query contract

Each admitted vertex has one or two original colours. The adapter uses one Boolean
variable per original vertex:

- false selects the lower original colour;
- true selects the higher original colour;
- a singleton domain adds the corresponding unit clause; and
- every shared-colour conflict on an original edge adds the exact binary clause
  forbidding both endpoints from selecting that colour.

For each request row, repeated restrictions are intersected before solving. An
empty original domain produces a portable kind-1 proof. Otherwise each forced
choice is supplied to one persistent CaDiCaL instance as an assumption. Learned
clauses remain available across the complete bank. SAT answers are reconstructed
from `val` into all original colours. UNSAT answers are translated into an
original-literal kind-2 path proof; a fixed inconsistent formula uses kind 3.
Every answer is checked against separately parsed original inputs before return.

CaDiCaL may use `failed` assumptions to reduce proof roots, but a failed-assumption
set is not accepted as the public certificate. The emitted `SBP1` paths must replay
without CaDiCaL, compiler state, clause identifiers, or internal solver data.

## Frozen workload

The first matrix reuses the four already opened mixed-result banks from the direct
2-SAT holdout:

- `DSJC250.5.col` — random DIMACS;
- `flat300_28_0.col` — flat/planted;
- `le450_15c.col` — Leighton;
- `miles1000.col` — geometric.

Each complete bank has exactly 1,024 rows: 512 SAT and 512 UNSAT by construction,
with no solver filtering. Prefixes are `1`, `8`, `64`, and `1,024` rows. Because
these graph bytes and outcomes were already opened before this protocol, this
matrix is explicitly **development evidence**, not prospective confirmation.
No graph may be removed or replaced after the adapter is run.

## Frozen measured arms

Exactly two primary arms are timed:

1. SPECTRA public `auto` using `structural-v11`;
2. the official CaDiCaL 3.0.1 core through the frozen proof-carrying adapter.

Diagnostic explicit SPECTRA arms may be retained, but they cannot rescue a failed
public-`auto` gate. No retrospective portfolio of CaDiCaL configurations is used.
The adapter uses the official default solver configuration except for options that
are strictly required for supported incremental assumptions or deterministic quiet
embedding; every changed option must be listed in the build/run receipt.

## Complete ready-packet timing boundary

Packet creation happens before timing for both systems. Every timed call includes:

- complete `SBC1` parsing and validation;
- formula/implication construction;
- CaDiCaL object construction and clause insertion, or SPECTRA native construction;
- every request decision;
- complete original-colour SAT reconstruction;
- complete portable UNSAT proof construction;
- independent original-input checking of every row;
- `SCQ2` serialization and immutable return allocation; and
- solver, observer, and temporary-state disposal.

Native compilation, workflow startup, case generation, packet preparation, and the
later Python replay are excluded symmetrically and retained separately. Build time
must never be presented as query-service time.

## Frozen schedule

- Environments: GitHub-hosted Ubuntu 24.04 x86-64 and native ARM64, CPython 3.13.
- Correctness: exhaustive three-vertex formulas and restrictions, random mixed
  differential banks, all three `SBP1` kinds, malformed-input rejection, output
  mutation rejection, UBSan, and ASan+UBSan for project-owned adapter code.
- Measurement: three untimed warmups, then 30 measured repetitions per cell.
- Order: deterministic position-balanced two-arm blocks from a frozen seed.
- Short cells: calibrated repeated calls with the per-call value retained; no
  measured sample is removed.
- Statistics: median per cell; geometric mean across the four graph families.
- Architectures are judged separately and are never pooled.

## Correctness gates

Every architecture and workload cell must satisfy all of the following:

1. the official source and all build identities match the frozen commit;
2. the two arms return the same 512/512 full-bank status sequence;
3. every SAT answer contains a complete valid original-vertex colouring;
4. every UNSAT answer contains a replayable `SBP1` proof;
5. the complete `SCQ2` bank passes the standard-library verifier;
6. a mutated output is rejected;
7. no `UNKNOWN`, crash, timeout, sanitizer finding, resource error, malformed
   packet, or checker failure is converted into SAT or UNSAT; and
8. every measured repetition is retained.

Any correctness failure makes the challenge a failure regardless of speed.

## Large-gap performance gates

Ratios are `SPECTRA auto / official CaDiCaL adapter`; lower is better for SPECTRA.
The gates were chosen before timing the external adapter.

### 8-query endpoint

- geometric mean across four graphs `<= 0.50` (at least 2.0×); and
- every graph ratio `<= 0.75`.

### 64-query endpoint

- geometric mean `<= 0.30` (at least 3.33×); and
- every graph ratio `<= 0.45`.

### 1,024-query endpoint

- geometric mean `<= 0.20` (at least 5.0×); and
- every graph ratio `<= 0.30`.

The one-query result is mandatory and visible but diagnostic. It cannot substitute
for a failed sustained endpoint. The overall development verdict is PASS only if
all correctness gates and every 8/64/1,024 performance gate pass independently on
both architectures. Gates may not be relaxed, graphs pooled, samples removed, or
explicit SPECTRA modes substituted after timing.

## Claim boundary after a pass

A pass would show that SPECTRA's public exact service substantially outperforms the
official CaDiCaL 3.0.1 decision core under this narrow, proof-carrying repeated
binary-list query contract on four disclosed mixed-result graph families.

It would not show general SAT leadership, ordinary graph-colouring leadership,
algorithmic priority, superiority over every specialized implementation,
production readiness, operator value, independent replication, or broad field
leadership. A fresh transfer suite and outside-team execution remain separate gates.
