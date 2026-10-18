# Structural automatic selector v7 — frozen protocol

## Purpose

Policy v6 passed development and real-workload guards but failed its prospective
confirmation because it sent every dense-pin bank to raw. This protocol freezes
v7 before timing. The failed v6 confirmation is retained as development evidence
and is not reused as v7 confirmation.

The selector remains deterministic and timing-blind. It may inspect only immutable
graph and request-bank geometry available during strict SBC1 admission. It may not
inspect answers, previous measured times, query outcomes, solver caches, or a
hidden trial execution.

## Frozen policy

Rules are evaluated in order:

1. A bank of at most eight requests on a graph above 512 vertices uses raw without
   constructing SCCs: `large_tiny_bank`.
2. A base-infeasible graph uses raw: `base_infeasible`.
3. A dense bank averages at least eight pins per request and covers at least one
   original vertex in 24.
   - If it contains at least 64 requests and at least 128 vertices, use SCC:
     `dense_long_tile64`.
   - Otherwise use raw: `dense_pin_fraction`.
4. An uncompressed graph with at least four original edges per vertex and at most
   128 requests uses raw: `dense_uncompressed_warmup`.
5. Any remaining feasible bank of at least 64 requests uses SCC: `scc_tile64`.
6. A bank of at least eight requests uses SCC when SCCs compress the literal graph
   by at least fourfold: `scc_strong_compression`.
7. Everything else uses raw: `short_weak_compression`.

Automatic route selection remains disabled. Explicit raw, SCC, sparse, and route
modes retain their existing semantics.

## Development evidence

V7 development is allowed to use:

- the original 104-case, 720-cell structural development panel;
- the 48-case v6 confirmation that rejected v6, now explicitly retrospective;
- exposed WAP05/WAP06 development banks; and
- the already opened ITC set9, set10, and set12 repair banks.

The ITC cases are retrospective development for v7, not untouched holdouts.

### Development gates

For each structural panel cell, the explicit oracle is the minimum median complete
time among raw, SCC, and route. Automatic timing includes packet packing, strict
native parsing, construction, solving, complete witness/proof materialisation,
independent checking, SCQ2 ownership, and disposal.

The original development panel must satisfy:

- global realised regret geometric mean `<= 1.025`;
- global selected-engine regret geometric mean `<= 1.015`;
- every family/count group geometric mean `<= 1.05`;
- at least 97% of cells within 5% of oracle;
- worst cell `<= 1.10`;
- zero policy-rule, decision, witness, proof, or replay failure.

The former v6 confirmation, now retrospective, must satisfy the same gates on both
x86-64 and ARM64.

The WAP guard must select SCC for 64, 128, and 1,024 requests on both graphs and
both architectures. At 1,024 requests, automatic/raw geometric mean must be
`<= 0.75` and every graph `<= 0.80`.

The retrospective ITC guard must:

- select SCC for set9 and set10;
- select raw for set12;
- remain within 5% of the faster explicit raw/SCC engine on every case; and
- preserve complete independently replayed answer banks.

## Fresh v7 confirmation

The confirmation namespace is fixed as
`SPECTRA-auto-selector-v7-confirmation`. It contains eight topology families
crossed with eight regimes, for 64 cases. Each case is evaluated at
1, 8, 32, 64, 128, and 1,024 requests, producing 384 cells per architecture.

The regimes are generated before timing:

1. `dense-small-96`: 96 vertices, eight pins/request — dense but below the v7
   vertex threshold, so long banks stay raw.
2. `dense-boundary-127`: 127 vertices, eight pins/request — raw side of the exact
   boundary.
3. `dense-boundary-128`: 128 vertices, eight pins/request — SCC side for banks of
   at least 64 requests.
4. `dense-medium-192`: 192 vertices, ten pins/request — SCC for long dense banks.
5. `strong-sparse-384`: 384 vertices, two pins/request, strong SCC contraction.
6. `edge-heavy-warmup-768`: 768 vertices, twelve pins/request, at least five
   original edges per vertex and no useful SCC contraction.
7. `low-density-large-1024`: 1,024 vertices, sixteen pins/request — many pins but
   below the density threshold.
8. `base-infeasible-512`: 512 vertices with a fixed contradiction.

The confirmation panel uses CPython 3.13 on GitHub Ubuntu 24.04 x86-64 and native
ARM64. It uses five untimed warmups and 31 paired repetitions per cell, with a
deterministically shuffled raw/SCC/route/auto order. Every output must be stable
and independently replayable.

### Confirmation gates

On each architecture independently:

- global realised regret geometric mean `<= 1.04`;
- global selected-engine regret geometric mean `<= 1.025`;
- every family/count group geometric mean `<= 1.06`;
- at least 95% of cells within 5% of oracle;
- worst cell `<= 1.10`;
- every topology family's auto/raw geometric mean `<= 1.03`;
- zero structural-rule, correctness, output-stability, or replay failure.

The policy, cases, thresholds, schedules, and namespaces are not changed after a
confirmation result is opened. A failed confirmation keeps Issue 2 open and any
next selector version requires another fresh namespace and fresh cases.

## Repository acceptance

A confirmation pass is necessary but not sufficient. The final one-commit PR must
also pass:

- Linux x86-64, native ARM64, and Windows/MSVC installed exact APIs;
- Python 3.11 and 3.13 package checks;
- UBSan and ASan+UBSan;
- compiler-free wheel execution;
- full CPU regression;
- WAP flagship regression;
- SVM portability; and
- repository audit.

After merge, the signed `main` commit must pass its post-merge workflows before
Issue 2 can be closed.

## Claim boundary

A pass would establish a high-quality structural dispatcher for the repository's
binary-domain exact-query workload families. It would not establish universal SAT
or graph-colouring dominance, production traffic performance, learning-based
adaptation, or outside-team replication.
