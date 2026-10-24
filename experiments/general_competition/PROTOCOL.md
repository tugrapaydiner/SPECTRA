# Phase 4 — general-competition protocol

**Status:** development protocol, not frozen confirmation evidence
**Starting source:** `832302a30d8d4d04a48afb418610ad49b94c3ad1`

**Current correctness status (2026-10-22):** the frozen campaign at source
[1f6a913](https://github.com/tugrapaydiner/SPECTRA/commit/1f6a9133a08b9d8e5a6a6690979f55fbe633e0c5)
completed all 768 cells and produced ten invalid SAT models from standalone
SPECTRA-G. That source remains disqualified and fails G0. The
[retained failure record](https://github.com/tugrapaydiner/SPECTRA/blob/8f7cee1a9c82e7da34cc2704ff7012290a554a91/experiments/general_competition/EXPOSED_CAMPAIGN_20261021.md#confirmed-native-model-failures)
identifies the cases and evidence. The occurrence-tracking repair at
`cb14031d4187c4cba5cccb37984ecf59d59bfdae` completed its separate 96-case
validation with seven checked SAT answers, two checked UNSAT proofs and 87
timeouts. A separate longer-budget diagnostic and retained-proof check now
provide corrected, independently checked answers for two original failure cases;
eight remain unresolved. Both original runs and all timeouts are unchanged. Any performance
experiment uses its own frozen plan and acceptance criteria; neither bounded
repair validation nor an optimization replaces the failed baseline or
establishes a leadership label.

The objective of this phase is to earn the broadest technically defensible solver
leadership claim. It is not permission to relabel the existing binary-list exact
service as a general SAT solver.

## 1. Claims are separated before implementation

Three claims must remain distinct.

1. **General CNF leadership.** A solver accepts arbitrary valid DIMACS CNF,
   returns a complete model or a checked UNSAT proof, and ranks first against the
   predeclared field on a prospectively unseen competition-style benchmark set.
2. **Repeated-CNF service leadership.** A fixed arbitrary CNF is prepared once
   and answered under a bank of assumptions. Preparation, every answer, complete
   evidence, checking and disposal are charged under a separately declared
   service metric.
3. **Binary-list exact-service leadership.** The current release-facing API over
   fixed conflict graphs and one-or-two-colour domains. Existing Part 3 evidence
   applies only here.

MaxSAT, pseudo-Boolean optimisation, SMT, CSP optimisation and exact chromatic
number are separate problem classes. They are not included in a SAT-feasibility
claim merely because they can sometimes be encoded into CNF. A future MaxSAT
claim requires a public objective contract and the official MaxSAT scoring rules.

The phrase **“beats all competition generally”** is not an admissible result
label. The target publication wording is:

> SPECTRA-G ranked first among the complete predeclared solver field on the
> prospectively frozen [track, benchmark inventory, hardware and metric].

A broader statement is allowed only when the corresponding additional tracks
have independently passed.

## 2. Why a new architecture is mandatory

The current `spectra.exact` contract has a fixed simple conflict graph and exactly
one or two allowed colours per original vertex. Its native reasoning reduces to a
binary implication structure. It does not accept arbitrary CNF clauses and cannot
inherit ordinary SAT Competition results.

Phase 4 therefore introduces a separate experimental solver, **SPECTRA-G**. It
must not change the semantics or claims of `spectra.exact`.

The intended architecture is:

1. strict, solver-free DIMACS admission and source receipts;
2. proof-preserving normalization with the original formula retained for checking;
3. a unit/binary implication core;
4. a residual engine for clauses of length three or greater;
5. interaction between binary closure and residual propagation without trusting a
   previous answer;
6. complete SAT model reconstruction and independent checking;
7. certified UNSAT output accepted by the declared checker;
8. an outcome-blind dispatcher based only on pre-solve structural features; and
9. explicit failure when a required backend, proof path or resource is unavailable.

A hidden fallback to a competitor is prohibited. A portfolio using an upstream
solver is allowed only when the upstream component, source pin, license, measured
cost and contribution are explicit. Such a result supports a **SPECTRA-assisted
portfolio** claim, not a from-scratch standalone-solver claim.

## 3. Competitor floor

The initial field is chosen from public competition leaders, not from convenient
legacy baselines.

### Sequential general SAT

At minimum retain exact source snapshots for:

- SAT Competition 2026 Main overall winner `satsuma-iter+kissat`;
- 2026 Main author-award runner-up `Kissat-MAB-HyPre`;
- 2026 Main third-place author-award entry `EDA-Kissat-v2`;
- the 2026 SAT-subtrack AI winner `Lymphosat`, reported separately because of its
  family-specialized generated portfolio;
- 2025 Main winner `AE_kissat2025_MAB`;
- 2024 Main winner `kissat-sc2024`;
- current upstream Kissat;
- current upstream CaDiCaL; and
- the strongest independently maintained proof-producing baseline that remains
  buildable in the official execution environment.

Author-award positions are distinct from raw solver ranks. The retained
[2026 score accounting](https://github.com/tugrapaydiner/SPECTRA/blob/8f7cee1a9c82e7da34cc2704ff7012290a554a91/experiments/general_competition/EXPOSED_CAMPAIGN_20261021.md#declared-solver-field) places
HyPre third and EDA-Kissat-v2 seventh in the raw Main solver ordering. This
clarification does not change the required competitor entries.

The official [2025 slides](https://satcompetition.github.io/2025/satcomp25slides.pdf#page=19)
name AE-Kissat-MAB as Main winner. The 2025 HTML results page repeats the
[2024 table](https://satcompetition.github.io/2024/downloads/satcomp24slides.pdf#page=18),
whose winner is kissat-sc2024. Both entries remain required; this attribution
correction leaves earlier frozen fields and results unchanged.

The virtual best solver is retained as a diagnostic upper bound. It is never
reported as an executable competitor and cannot be “beaten” by changing its
construction after results are visible.

### Specialized service controls

The repeated-service track additionally retains:

- the strongest direct engine selected prospectively from its disclosed engine
  family;
- persistent CaDiCaL where its public incremental interface applies;
- MiniCard or another native-cardinality solver when the input contains a natural
  cardinality structure;
- noncontracting SPECTRA; and
- any independently submitted challenger received before the confirmation set is
  opened.

Every competitor gets its intended native representation when one exists. No
pairwise-CNF handicap may be imposed on a solver that supports the relevant
constraint natively.

## 4. Benchmark boundaries

### Development inventory

SAT Competition Main benchmarks from 2023–2025 may be used for implementation,
profiling and selector development. All development runs are permanently marked
open and cannot later become confirmation evidence.

### Exposed diagnostic inventory

The 2026 Main inventory is already public and its rankings are known. It may be
used as a harsh diagnostic and to reproduce official scoring, but it is not a new
prospective confirmation set for this project.

### Primary confirmation inventory

The primary general-CNF claim requires one of:

1. the next official SAT Competition hidden selection, with SPECTRA-G submitted
   before the solver deadline; or
2. an outside-controlled sealed inventory with public family-selection code,
   source hashes and an opening ceremony after the solver/container hash is frozen.

The confirmation inventory must contain at least 300 formulas, both SAT and UNSAT,
and multiple independently sourced families. Family counts, deduplication,
near-duplicate detection, sanitization and exclusions are fixed before any solver
is run. A failed opening remains the result; instances, limits and competitors are
not replaced.

The repeated-service confirmation is separately generated from unseen base
formulas and unseen assumption banks. Prefixes of one bank are not counted as
independent formulas.

## 5. Official-style one-shot track

The primary one-shot contract follows the current SAT Competition Main shape:

- arbitrary valid DIMACS CNF, including unsanitized edge cases;
- one sequential process;
- 5,000-second timeout;
- 32 GiB memory limit;
- SAT model required;
- UNSAT proof required;
- checker timeout or rejection means unsolved;
- incorrect model or proof disqualifies the complete run;
- cold build is excluded only when every competitor receives the same official
  pre-build boundary; and
- ranking uses official PAR-2, with SAT, UNSAT and overall results retained.

No warm preparation, learned state from another benchmark, answer cache or
family-specific generated solver is silently carried between instances. An AI- or
family-tuned track is reported separately under its own rules.

## 6. Repeated-service track

The service track measures a different user problem and never substitutes for the
one-shot Main result.

For each fixed formula, the charged endpoint includes:

- bounded input read, parsing, validation and hashing;
- all index/preprocessor/solver construction;
- every assumption bank entry;
- complete SAT models or portable UNSAT evidence;
- independent checking from the original CNF and assumptions;
- output-bank serialization and hashing; and
- disposal.

Report bank sizes `1, 4, 16, 64, 256, 1024` unless a predeclared resource bound
rejects the case before timing. Report complete time, p50, p95, peak RSS, output
bytes, proof-check time and solved count. Preparation-only and query-only numbers
are diagnostics, not the primary result.

## 7. Selector and tuning firewall

The final dispatcher may inspect only predeclared structural features available
before solving, such as variable/clause counts, clause-length histogram, binary
fraction, implication geometry and assumption-bank size. It may not inspect:

- solver answers;
- measured trial times from the confirmation inventory;
- benchmark filenames or family labels;
- cached models or proofs;
- a hidden lookup table keyed by hashes; or
- post-opening manual exceptions.

Training and selector fitting use development families only. The exact model,
code, feature schema and thresholds are hashed before the confirmation inventory
is opened.

## 8. Gates for a general-leadership claim

All gates are conjunctive.

### G0 — correctness

- zero wrong SAT models;
- zero rejected UNSAT proofs;
- zero malformed outputs accepted by the project checker;
- complete raw logs and checker outputs retained; and
- differential fuzzing against at least two independent solvers before any timed
  confirmation run.

### G1 — general input contract

- strict DIMACS admission passes official unsanitized robustness cases;
- formulas with empty, unit, binary and long clauses are handled;
- compressed and uncompressed source identity is retained;
- no input is rejected because its structural features predict poor performance;
- resource rejection is deterministic, pre-solve and reported as unsolved.

### G2 — official ranking

On the unopened primary inventory, SPECTRA-G must have lower overall PAR-2 than
every predeclared executable competitor. It must also avoid a catastrophic
subtrack trade: SAT and UNSAT PAR-2 must each be within 5% of the best competitor,
unless SPECTRA-G itself wins that subtrack.

A first-place overall result with a severe hidden SAT or UNSAT collapse is reported
as an overall-track win, not “general superiority.”

### G3 — robustness and portability

The exact frozen source must pass x86-64 Linux, native ARM64 Linux, and Windows
model/checking contracts where the implementation supports those platforms.
The official ranking remains tied to the declared competition host. Sanitizer,
malformed-input and isolated-install checks must be green.

### G4 — generalization

- no confirmation family appears in selector training;
- no source or benchmark change after opening;
- all negative cells and timeouts retained;
- results summarized first by family and then across families; and
- duplicate or near-duplicate instances do not inflate the effective sample.

### G5 — external challenge

At least one outside team must reproduce the frozen executable result or submit an
independently written challenger under the same contract. Project-owned CI is not
external replication.

## 9. Result labels

Permitted labels are mechanically tied to passed gates:

- `GENERAL_FRONTEND_READY` — G0 and G1 only;
- `DEVELOPMENT_LEADER` — first on an exposed development inventory;
- `PROSPECTIVE_TRACK_WINNER` — G0–G4 and first overall on the unopened inventory;
- `EXTERNALLY_REPLICATED_TRACK_WINNER` — G0–G5;
- `GENERAL_SAT_LEADERSHIP` — prospective first place plus the SAT/UNSAT balance
  rule and external replication.

None of these imply MaxSAT, SMT, CSP, graph-optimisation, asymptotic or universal
algorithmic leadership.

## 10. Immediate implementation order

1. Add strict arbitrary-CNF admission and deterministic receipts.
2. Reproduce official 2025 and 2026 scoring from pinned solver and benchmark
   snapshots without SPECTRA in the field.
3. Implement a proof-preserving binary-core extractor and independent replay.
4. Connect the binary core to a residual long-clause engine with full model/proof
   reconstruction.
5. Run differential fuzzing and small exhaustive truth-table checks.
6. Measure the exposed 2023–2026 development inventories and retain failures.
7. Freeze the SPECTRA-G container, competitors, selector and confirmation protocol.
8. Enter the next official competition or transfer custody to an outside holder
   before opening the sealed inventory.

The program stops and records a negative result at the first failed correctness or
admission gate. Performance tuning never outranks semantic correctness.

## 11. Historical engineering checkpoint

The repair at `1310c9f192a4a1e2f25303934f4f3e0fd4fd3a86` made the native
arbitrary-CNF matrix a permanent repository contract and kept the workflow
inventory guards synchronized with it. It also closed two validation gaps found
while exercising the first exposed panel:

- an UNSAT result whose PySAT/Lingeling proof stream omits the terminal empty
  clause is completed with the standard `0` line only after the solver reports
  UNSAT, and the resulting proof is still required to pass the pinned external
  `drat-trim` checker; and
- the exposed 2026 SAT-only diagnostic now uses the receipt-bound native binary
  frontend instead of the Python reference reducer, while both arms still use the
  same CaDiCaL backend and every accepted SAT model must satisfy the exact original
  admitted CNF.

This checkpoint recorded an engineering repair, not a performance result or a
passed G0 gate. The exposed panel remains development-only. The later native model
failures above remain part of the record, and none of the general-leadership
labels may be used until their complete gates pass.
