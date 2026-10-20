# Native 2-SAT challenge

**Status:** first exposed development comparison completed
**Branch:** `part3/release-replication-readiness`
**Base:** merged Part 2 commit `527590406c2c624a6aa050e17c6f45b0f000ac5b`

## Why this challenge exists

SPECTRA's exact service admits fixed conflict graphs whose original domains have
one or two colours. That contract has a direct 2-CNF interpretation. Persistent
MiniCard, CaDiCaL and reduced CUDD are serious controls, but they are not the
cheapest purpose-built algorithm for this exact problem class.

This track therefore builds the opponent most likely to erase SPECTRA's advantage:
an independently written native 2-SAT service using the same prepared request and
certified-answer formats.

Adding the challenger does not itself establish a performance win. The first
complete comparison is recorded in `TWOSAT_CHALLENGE_RESULTS_20261019.md`; its
pure-service result is negative for SPECTRA and remains part of the active record.

## Independence boundary

The challenger may use established implication-graph, SCC, condensation-DAG,
unit-closure and bit-parallel techniques. Those techniques are prior art.

It must not receive or import:

- SPECTRA's native `Index` object;
- SPECTRA SCC identifiers, quotient graph, route tables or base assignment;
- cached answers, measured timings or development-case labels;
- an output previously produced by SPECTRA; or
- private data derived from unopened evaluation cases.

Its only semantic input is immutable public `SBC1` bytes. Its only semantic output
is public `SCQ2` bytes. The standard-library `verify_session_output` function judges
both systems from original graph, domain and query data.

The implementation lives in `native_twosat.cpp`, with a separate build receipt and
Python loader. It independently parses the packet, builds the implication graph,
computes SCCs, propagates query units, reconstructs complete original-colour labels,
and emits `SBP1` contradiction paths.

## Implemented challenger modes

- `raw`: scalar multi-source implication closure on the original graph;
- `scc`: scalar SCC-DAG closure, with 64-query bit-parallel propagation for long
  banks; and
- `auto`: a frozen structural rule owned by the challenger, not SPECTRA's selector.

All modes return full witnesses for SAT and portable proof kinds 1, 2 or 3 for
UNSAT. Resource failure remains an exception; it is never relabelled UNSAT.

## Phase A acceptance gates

Before timing, all of the following must pass:

1. exhaustive decision agreement with a brute-force oracle for every one-, two-
   and three-variable palette-2 graph/domain combination and every per-vertex
   restriction vector, including empty domains;
2. a fixed randomized four-variable panel with duplicate and mixed restrictions;
3. arbitrary palette positions through bit 63, including SAT and base-UNSAT cases;
4. standard-library replay of every emitted `SCQ2` bank;
5. explicit exercise of empty-domain, query-implication and base-contradiction
   proofs;
6. explicit exercise of scalar raw, scalar SCC and multi-tile tile64 SCC kernels;
7. malformed packets, tampered build receipts and an actual native index rejection
   at the 1 MiB memory cap remain visible;
8. Linux x86-64 Python 3.11/3.13, native ARM64, Windows/MSVC, UBSan and
   ASan+UBSan; and
9. no source sharing with `spectra/_native/exact_binary.cpp` beyond public format
   constants and the mathematical contract.

## Phase B development comparison

Only after Phase A passes, compare complete calls on exposed development cases.
Both systems receive identical immutable `SBC1` bytes. The six mandatory arms are
SPECTRA `raw`, `scc`, and `auto`, plus challenger `raw`, `scc`, and `auto`.

Two endpoints are reported separately:

1. **service:** native parsing, construction, every query, complete labels/proofs,
   serialization, Python ownership of the returned bytes, and disposal; and
2. **verified service:** the same complete call followed by one standard-library
   replay of that arm's own output bank.

Compilation, packet creation and artifact transport are reported separately. A
shared replay must never be added once to a whole comparison because that can hide
arm-specific output/proof costs.

Required banks include 1, 8, 64 and 1,024 queries; SAT and UNSAT mixtures; weak and
strong SCC compression; sparse and dense implication graphs; arbitrary palette
positions; and low/high pin density. Every timeout, resource failure and malformed
result remains in the matrix.

Timing uses complete position-balanced blocks across all six arms, identical batch
sizes within a cell, deterministic schedules, raw per-repetition samples, and
semantic replay rather than requiring byte-identical SAT assignments. No
performance threshold is attached to this exposed development phase, and the
fastest explicit challenger may be reported only as a descriptive post-hoc oracle.
Development timing cannot be converted into a holdout.

## Phase C prospective competition result

A field-leading claim requires a new freeze before unseen data are opened:

- one fixed source identity for both systems and the harness;
- fixed hardware/software environments, repetitions, schedules and memory limits;
- multiple topology/workload families rather than only WAP;
- both satisfiable and unsatisfiable requests;
- single-query and sustained-bank endpoints;
- complete output and proof checking;
- effect-size gates and failure rules fixed in advance; and
- a full raw evidence archive, including negative cells.

The final comparison must report SPECTRA against challenger `raw`, challenger
`scc`, challenger `auto`, persistent MiniCard, persistent CaDiCaL, and the strongest
working reduced-CUDD control that applies to the same cases.

## Claim boundary

Passing a future prospective challenge would support superiority only within its
frozen repeated binary-domain query contract. It would still not establish general
SAT, chromatic-number, arbitrary-domain CSP, production deployment, algorithmic
priority, or asymptotic superiority.
