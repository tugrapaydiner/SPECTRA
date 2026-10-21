# SPECTRA — Roadmap to a Real 100/100

> **Continuation checkpoint — 2026-10-21:** This document preserves the supplied
> audit and its strict scientific gates. Development continues on `finish-part3`
> in draft PR #76. The historic scores below have not been promoted. The latest
> engineering and measured evidence is tracked in
> [the exposed campaign report](../experiments/general_competition/EXPOSED_CAMPAIGN_20261021.md).
> The frozen native baseline has produced independently confirmed invalid SAT
> models. Its correctness gate is failed; the repair and validation are tracked
> separately from the unchanged baseline evidence.
> Neither a hidden win nor independent replication has been established.

**Document type:** living research and engineering roadmap
**Last reviewed:** 2026-10-21
**Canonical working branch:** `finish-part3`
**Original audit head:** `904629a1acfbdca964f8355a7e74b182007eb6e5`
**Continuation checkpoint:** the measured source and later review head are recorded in the linked campaign report.
**Current pull request:** draft PR #76
**Current rule:** do not open another working branch and do not merge merely because CI is green.

---

## 1. What “100/100” means

For this project, **100/100 must be a result that a serious outside SAT researcher could inspect without finding a hidden shortcut, unfair comparison, missing failure, or inflated claim**.

The target claim is:

> **SPECTRA-G ranked first against the complete predeclared modern solver field on a prospectively unseen arbitrary-CNF benchmark inventory, under official-style scoring, with checked SAT models, checked UNSAT proofs, fixed hardware and limits, and outside replication.**

This is deliberately narrower than “beats every solver at every problem.” MaxSAT, SMT, CSP, theorem proving, exact graph colouring, pseudo-Boolean optimization, and other problem classes require separate campaigns.

A green repository is not 100/100.
A fast result on four exposed SAT formulas is not 100/100.
A CaDiCaL-assisted portfolio is not a standalone SPECTRA-G win.
A hidden-set victory without outside replication is not 100/100.

The gates in this file are **conjunctive**. One major red gate blocks the final claim even when the other areas are excellent.

---

## 2. Original strict audit snapshot

### Overall position

| Area | Strict score | Current verdict |
|---|---:|---|
| Repository and CI health | **92/100** | Strong |
| Proof-carrying binary-list service | **88/100** | Genuine specialized leadership |
| General-CNF correctness infrastructure | **72/100** | Credible development foundation |
| General-CNF competitive performance | **22/100** | Not established |
| Independent scientific validation | **3/100** | Essentially absent |
| Ready to merge under the “beat everyone” objective | **42/100** | No |
| Broad “we beat the competition generally” claim | **22/100** | Not defensible yet |

These are the supplied audit's working scores at the original audit head, not
scientific statistics or a reassessment after the newly observed native failures.
The current gate ledger and linked campaign report take precedence for new work.

### Current repository health

- [x] One canonical working branch: `finish-part3`
- [x] Only `main`, `finish-part3`, and the archive branch are active
- [x] Repository audit passes
- [x] General-CNF workflow passes
- [x] CPU regression passes on Python 3.11 and 3.13
- [x] Exact-service matrix passes
- [x] Public package and clean-install checks pass
- [x] SVM portability passes
- [x] WAP regression passes
- [x] Linux x86-64 passes
- [x] Native Linux ARM64 passes
- [x] Windows/MSVC passes
- [x] GCC and Clang warning-clean builds pass
- [x] UBSan and ASan+UBSan pass
- [x] External `drat-trim` replay passes
- [ ] Branch protection is configured
- [ ] Current freeze commit is signed
- [ ] Final review surface is small enough for an outsider to audit
- [ ] Permanent, non-expiring research evidence is deposited

### Current exposed general-CNF panel

The latest four-instance exposed SAT panel contains 12 repetitions:

- **3 meaningful reducer samples** on one unit-heavy formula
- **9 fallback-parity samples** where both arms effectively run the same CaDiCaL backend

Latest measurements:

| Quantity | Result |
|---|---:|
| Median candidate/control ratio | `0.9947` |
| Overall geometric mean | `0.6846` |
| Reported samples below 1.0 | `7 / 12` |
| Actual reducer samples | `3 / 12` |
| Plain/fallback samples | `9 / 12` |
| Best meaningful median ratio | `0.1900` |
| Meaningful best-case advantage | approximately **5.26×** |

The `7 / 12` number must **not** be described as seven SPECTRA algorithm wins. Four of those “wins” are ordinary timing variation between two executions of the same backend.

The accurate description is:

> One strong structural regime produced three approximately 5× wins. The remaining nine samples demonstrated fallback parity rather than a distinct SPECTRA speed advantage.

---

## 3. What is already genuinely good

These parts should be protected rather than constantly rewritten.

### A. Specialized exact-service result

The binary-list repeated-query service has strong evidence:

- prospective holdout structure;
- complete original-address witnesses;
- independently checked results;
- direct 2-SAT comparison;
- official CaDiCaL service comparison;
- x86-64 and ARM64 evidence;
- honest single-query losses retained;
- strong sustained-bank performance;
- explicit claim boundaries.

This is the strongest part of SPECTRA today.

### B. General-CNF correctness foundation

SPECTRA-G now has real arbitrary-CNF machinery:

- strict DIMACS admission;
- source and semantic receipts;
- unit/binary preprocessing;
- watched-literal propagation;
- first-UIP learned clauses;
- activity-based branching;
- phase saving;
- restarts;
- LBD-aware learned-clause handling;
- clause-database reduction;
- complete SAT models;
- independently checked models;
- retained UNSAT proof output;
- external `drat-trim` replay;
- deterministic `UNKNOWN` on resource exhaustion;
- differential testing against CaDiCaL and Glucose;
- Linux, ARM64, Windows, sanitizer, GCC, and Clang coverage.

This is not yet a winning general solver, but it is a serious platform on which one can be built.

### C. Failure discipline

The project normally retains:

- failed admission;
- failed timing thresholds;
- timeouts;
- platform limitations;
- Python-version regressions;
- exposed development status;
- distinction between internal and independent validation.

That discipline is one of the project’s strongest credibility advantages. Do not weaken it to make the final score look better.

---

## 4. What is currently really bad

These are the largest blockers, in priority order.

### P0 — We do not have a real general-SAT score for SPECTRA-G

The official 2026 score field is reproduced, but SPECTRA is not in it.

Missing:

- SPECTRA solved count;
- SPECTRA SAT solved count;
- SPECTRA UNSAT solved count;
- SPECTRA overall PAR-2;
- SPECTRA SAT PAR-2;
- SPECTRA UNSAT PAR-2;
- SPECTRA family-level rank;
- SPECTRA memory profile;
- SPECTRA proof-checking cost;
- SPECTRA position against the full modern field.

Until these exist, broad competitive discussion is mostly speculation.

### P0 — The main exposed performance panel does not measure the standalone native solver

The current panel primarily measures:

```text
SPECTRA routing/preprocessing + CaDiCaL
versus
plain CaDiCaL
```

That can support a **SPECTRA-assisted portfolio** result. It does not prove that the native SPECTRA-G CDCL engine beats CaDiCaL, Kissat, Satsuma, or any competition leader.

The native solver has excellent correctness checks, but its retained performance receipts are still tiny smoke cases.

### P0 — Same-backend timing noise is counted as “wins”

When the selector chooses `plain`, candidate and control are the same backend on the same formula. Those cells must not contribute to a SPECTRA win count.

This is an evidence-integrity defect and should be repaired before further publicity.

### P0 — There is no hidden confirmation result

All current general-CNF formulas are exposed development data. We have no unopened inventory, no official hidden submission, and no sealed outside-controlled benchmark opening.

### P0 — There is no outside replication

Project-owned GitHub Actions prove engineering portability, not independent scientific replication.

Issue #74 has no outside submission and targets an older exact-service checkpoint, not the current general-CNF campaign.

### P1 — No balanced SAT/UNSAT performance evidence

The latest exposed performance panel is SAT-only. General leadership requires both SAT and UNSAT strength.

### P1 — The PR is too large to review properly

Current review surface:

- 153 commits;
- 82 changed files;
- more than 17,000 additions.

The full raw history should remain preserved, but the final scientific review surface needs a small, coherent commit series.

### P1 — The active tree is nearly full

The audit has recently reported approximately:

```text
994 tracked files
1,000-file limit
6 remaining slots
```

Do not solve this by casually raising the limit. Consolidate or archive obsolete development scaffolding before adding the next campaign.

### P1 — Public identity is inconsistent

The repository description still advertises an old “Edge-native o1 / 1.58-bit / Latent MCTS” identity that the current README explicitly does not claim.

Other stale surfaces include:

- README workflow count;
- PR text that may lag behind completed checks;
- replication tracking pinned to old source;
- missing current general-CNF challenge issue.

This makes careful research look less trustworthy than it actually is.

### P2 — The branch is unprotected and the head is unsigned

This is not the reason we are losing benchmarks, but it matters when freezing a scientific result.

---

## 5. Separate the competition tracks

Never combine results from different contracts into one headline.

| Track | Candidate | Required claim |
|---|---|---|
| Binary-list repeated-query service | `spectra.exact` | Specialized exact-service leadership |
| General CNF, one formula per process | Standalone native SPECTRA-G | General SAT track leadership |
| General CNF assisted portfolio | SPECTRA selector plus disclosed upstream engines | SPECTRA-assisted portfolio leadership |
| Repeated arbitrary-CNF service | Prepared general CNF plus assumption banks | Repeated-CNF service leadership |
| MaxSAT / SMT / CSP / graph optimization | Separate future systems | Separate claims and scoring |

A win in one row does not transfer to another row.

---

# 6. The 100/100 workstreams

## Workstream 0 — Repair evidence and repository integrity

### Required work

- [x] Replace the ambiguous `hybrid_wins` field with explicit classifications:
  - `SPECTRA_NATIVE_WIN`
  - `SPECTRA_FRONTEND_WIN`
  - `UPSTREAM_FALLBACK_PARITY`
  - `LOSS`
  - `TIMEOUT`
  - `UNKNOWN`
  - `INVALID_MODEL`
  - `INVALID_PROOF`
  - `RESOURCE_FAILURE`
- [x] Do not count fallback parity as a SPECTRA win.
- [ ] Update PR #76 to the final current workflow status after every head change.
- [x] Correct the README workflow count.
- [ ] Replace the obsolete GitHub repository description.
- [ ] Update replication issues to the current frozen source when a new freeze exists.
- [x] Recover active-tree file headroom (27 archived raw/scaffolding files; hashes and immutable retrieval retained).
- [ ] Consolidate the final review history while preserving the raw branch in the archive.
- [x] Pin all permanent Actions dependencies to immutable SHAs; enforce in the active-tree audit.
- [ ] Configure required checks before final merge.
- [ ] Sign the final freeze commit and tag.

### Done when

An outsider can identify one canonical branch, one current source SHA, one protocol, one result table, and one evidence bundle without encountering contradictory status language.

---

## Workstream 1 — Establish the standalone native SPECTRA-G baseline

### Required work

Run the native SPECTRA-G engine itself—not the CaDiCaL-assisted router—over a substantial exposed arbitrary-CNF inventory.

Retain:

- complete admitted formula list;
- exact source hashes;
- SAT and UNSAT verdicts;
- complete models;
- UNSAT proofs;
- checker outputs;
- elapsed time;
- peak memory;
- proof generation time;
- proof checking time;
- model checking time;
- conflict/decision/propagation statistics;
- every timeout, crash, `UNKNOWN`, rejection, and resource failure.

### Minimum exposed inventory

Use multiple public competition years and families, including:

- structured verification;
- planning;
- cryptographic;
- combinatorial;
- random;
- crafted;
- graph-derived;
- hardware/software verification;
- SAT and UNSAT;
- easy, medium, and timeout-level cases.

Four selected SAT formulas are not enough.

### Done when

SPECTRA-G has a complete official-style score table and can be ranked like every other solver.

---

## Workstream 2 — Run the real competitor field

### Minimum field

Pin exact source snapshots and configurations for at least:

- 2026 Main winner `satsuma-iter+kissat`;
- 2026 runner-up `Kissat-MAB-HyPre`;
- 2026 third-place `EDA-Kissat-v2`;
- `Lymphosat`, reported separately where appropriate;
- 2025 Main winner `kissat-sc2024`;
- current upstream Kissat;
- current upstream CaDiCaL;
- a strong independent proof-producing baseline;
- standalone native SPECTRA-G;
- the disclosed SPECTRA-assisted portfolio as a separate entry.

### Fairness rules

- identical hardware class;
- identical timeout;
- identical memory limit;
- identical process isolation;
- intended native representation;
- official-style SAT model checking;
- declared UNSAT proof checking;
- no hidden competitor handicap;
- no pairwise encoding when a competitor is entitled to its native constraint format;
- no removed failures;
- no warm state unless the track explicitly permits it.

### Done when

A full table ranks every entry overall, on SAT, on UNSAT, and by family.

---

## Workstream 3 — Reach exposed development leadership

Correctness comes first. Performance work starts only after the full exposed baseline exists.

### Performance areas to investigate

The exact implementation choices will come from profiling, but likely high-value areas include:

- stronger preprocessing without proof breakage;
- more efficient binary implication handling;
- better branching;
- better restart scheduling;
- better phase selection;
- clause minimization;
- learned-clause retention and deletion;
- memory layout;
- watch-list efficiency;
- proof-generation overhead;
- family-aware but outcome-blind routing;
- carefully disclosed portfolios;
- avoiding expensive preprocessing when it cannot recover its cost.

### Every improvement must answer

1. Which exposed families improved?
2. Which families regressed?
3. Did SAT and UNSAT both remain healthy?
4. Did proof size or checking cost become worse?
5. Did memory increase?
6. Did any result become invalid?
7. Was the rule chosen without touching hidden data?
8. Is the benefit present across more than one machine/run?

### Development leadership gate

- [ ] Overall exposed PAR-2 rank is #1.
- [ ] SAT PAR-2 is #1 or within 5% of the best.
- [ ] UNSAT PAR-2 is #1 or within 5% of the best.
- [ ] Wrong models: 0.
- [ ] Rejected proofs: 0.
- [ ] Hidden failures: 0.
- [ ] Family-level collapses are visible and addressed.
- [ ] Standalone native and assisted-portfolio results are separate.

### Done when

SPECTRA-G is actually first on the complete exposed development field, not merely fast on a selected panel.

---

## Workstream 4 — Achieve a convincing large gap

Official first place is not enough for the project’s private “beat them with a big gap” standard.

### Internal big-gap target

Use this stronger target before freezing the hidden campaign:

```text
Overall PAR-2:
    no more than 85% of the runner-up's PAR-2

Solved count:
    at least the runner-up's solved count

SAT and UNSAT:
    each first or within 5% of the best specialist

Family results:
    first on a majority of retained families

Catastrophic regression:
    no major family above 2× the runner-up without a clearly disclosed limitation

Correctness:
    zero wrong models and zero rejected proofs
```

The 15% margin is an internal SPECTRA target, not an official competition rule.

### Done when

The exposed result looks convincing even to a skeptical reviewer who ignores the marketing language.

---

## Workstream 5 — Freeze an outcome-blind dispatcher

The final selector may use only pre-solve structural features, for example:

- variable count;
- clause count;
- clause-width histogram;
- unit fraction;
- binary fraction;
- implication-graph statistics;
- occurrence distribution;
- estimated propagation pressure;
- memory estimate;
- assumption-bank size for a service track.

It may not use:

- filename;
- family label;
- source URL;
- formula hash lookup;
- known solver winner;
- trial timing on hidden formulas;
- answer information;
- cached models or proofs;
- manual exceptions added after opening.

### Required freeze package

- [ ] Feature schema
- [ ] Selector code
- [ ] Model/threshold values
- [ ] Training inventory
- [ ] Training exclusions
- [ ] Exact source hash
- [ ] Exact selector hash
- [ ] Statement of which upstream engines are included
- [ ] Separate label for standalone and assisted entries

### Done when

The selector can be hashed and frozen before any hidden benchmark is opened.

---

## Workstream 6 — Freeze the full competition package

Freeze:

- source commit;
- container image digest;
- compiler and flags;
- build commands;
- standalone solver entry;
- assisted portfolio entry;
- solver configurations;
- selector;
- proof format;
- proof checker;
- SAT model checker;
- competitor sources;
- competitor configurations;
- hardware;
- timeout;
- memory limit;
- scoring script;
- failure classification;
- benchmark selection procedure;
- deduplication and near-duplicate rules.

### Non-negotiable rule

Any post-opening change invalidates the prospective claim unless the entire sealed evaluation is discarded and publicly reported as failed.

---

## Workstream 7 — Win a genuinely hidden inventory

Use either:

1. the next official SAT Competition hidden selection; or
2. an inventory controlled by an outside custodian.

### Minimum hidden-set standard

- [ ] At least 300 formulas
- [ ] Multiple independent families
- [ ] Both SAT and UNSAT
- [ ] Exact duplicate control
- [ ] Near-duplicate control
- [ ] No selector-training family leakage
- [ ] No graph/formula replacement after opening
- [ ] No threshold changes after opening
- [ ] No competitor removal after opening
- [ ] All failures retained
- [ ] Official-style overall, SAT, and UNSAT scoring

### Hidden-set pass gate

- [ ] SPECTRA-G ranks first overall.
- [ ] SAT performance is first or within 5% of best.
- [ ] UNSAT performance is first or within 5% of best.
- [ ] Zero invalid results.
- [ ] The result survives family-level analysis.
- [ ] The large-gap goal is reported honestly, whether passed or failed.

### Done when

The project earns `PROSPECTIVE_TRACK_WINNER`.

---

## Workstream 8 — Obtain outside replication and challenge

Project-owned CI does not count.

### Required external evidence

At least one outside person or team must:

- run the exact frozen source/container;
- disclose their relationship to the project;
- retain complete raw logs;
- retain models and proofs;
- retain checker output;
- disclose hardware and limits;
- report every deviation;
- report negative or conflicting results;
- publish archive hashes.

An independently written challenger under the same contract is also valuable and should be accepted before confirmation opens.

### Done when

The hidden victory is reproduced or challenged by an attributable outside party.

This is required for `EXTERNALLY_REPLICATED_TRACK_WINNER` and the final 100/100 label.

---

## Workstream 9 — Expert review

Seek review from someone familiar with:

- SAT competition methodology;
- CDCL solver engineering;
- proof production/checking;
- benchmark leakage;
- solver portfolios;
- knowledge compilation or repeated-query solving.

They should review:

- competitor fairness;
- source pins;
- hidden-set custody;
- selector leakage;
- proof validity;
- family balance;
- claim wording;
- closest prior art.

### Done when

Review comments and responses are public, and material concerns are resolved or explicitly retained as limitations.

---

## Workstream 10 — Merge, release, and make it usable

Only after the scientific gates:

- [ ] Merge the exact frozen source.
- [ ] Create a signed tag.
- [ ] Publish source, wheel, native binaries, and container digest.
- [ ] Publish SBOM and provenance.
- [ ] Publish benchmark manifests.
- [ ] Publish score receipts.
- [ ] Publish all failures and timeouts.
- [ ] Verify downloaded release assets.
- [ ] Update package version.
- [ ] Update the repository description and README.
- [ ] Create an installation path that does not require research-branch archaeology.
- [ ] Preserve the exact winning configuration permanently.

### Done when

An outside user can install, execute, verify, and reproduce the winning system.

---

## Workstream 11 — Demonstrate real-world value

A competition victory can establish track leadership. A full project-level 100/100 also needs impact.

- [ ] Operator-authored workload
- [ ] Independently chosen acceptance metric
- [ ] Maintained external integration
- [ ] External user who reproduces the benefit
- [ ] Security review for hostile CNF input
- [ ] Reliability and resource-limit documentation
- [ ] Clear support boundaries

---

# 7. Gate ledger

| Gate | Requirement | Current state |
|---|---|---|
| G0 | Correct SAT models and accepted UNSAT proofs | **Failed for the frozen native baseline: invalid models retained; repair under validation** |
| G1 | General arbitrary-CNF input contract | **Mostly passed** |
| G2 | First place under official-style scoring | **Not passed** |
| G3 | Portability and robustness | **Passed internally** |
| G4 | Prospective generalization | **Not passed** |
| G5 | Outside challenge or replication | **Not passed** |
| G6 | Reviewable frozen research surface | **Not passed** |
| G7 | Release and permanent evidence | **Not passed** |
| G8 | External adoption/operator value | **Not passed** |

### Allowed labels

| Passed state | Allowed label |
|---|---|
| G0 + G1 | `GENERAL_FRONTEND_READY` |
| First on complete exposed development field | `DEVELOPMENT_LEADER` |
| G0–G4 and first on hidden inventory | `PROSPECTIVE_TRACK_WINNER` |
| G0–G5 | `EXTERNALLY_REPLICATED_TRACK_WINNER` |
| Balanced prospective first place plus outside replication | `GENERAL_SAT_LEADERSHIP` |
| All scientific, release, and adoption gates | **100/100 project state** |

The supplied audit's `GENERAL_FRONTEND_READY` label describes its previous
frontend evidence. It does not certify standalone native SPECTRA-G. The current
standalone engine is undergoing correctness repair after the exposed campaign
found invalid SAT models. The project is not a `DEVELOPMENT_LEADER` on general
CNF, and no competitive or 100/100 claim is accepted.

---

# 8. Things we must never do

- Do not remove formulas after seeing a bad result.
- Do not replace a failed hidden-set formula.
- Do not relax a gate after opening.
- Do not hide timeouts, crashes, invalid models, or rejected proofs.
- Do not count same-backend timing noise as a SPECTRA win.
- Do not describe CaDiCaL or Kissat fallback as standalone SPECTRA-G performance.
- Do not use benchmark filenames, family labels, or hashes in the final selector.
- Do not tune on the hidden inventory.
- Do not call project-owned CI independent replication.
- Do not combine the exact-service result with general-SAT results.
- Do not multiply speedups from different workloads.
- Do not call a smoke test a competitive benchmark.
- Do not merge merely because the checkmarks are green.
- Do not open several experimental branches and leave them unfinished.
- Do not use “final,” “complete,” or “100/100” in commit names before the relevant gate actually passes.

---

# 9. Commit and branch style — make it sound human

## Branch rule

Stay on:

```text
finish-part3
```

until this work is deliberately merged or stopped.

Do not create:

```text
part3-final
part3-final-v2
part3-clean
part3-real-final
fix-part3-again
```

After Part 3 is genuinely merged, future branches should be named after a concrete goal, such as:

```text
general-sat-field-run
hidden-set-freeze
release-spectra-g
```

## When to commit

Commit when one coherent research thought is complete:

- one evidence defect fixed;
- one benchmark contract added;
- one solver improvement completed;
- one freeze package created;
- one result report finalized.

Do not commit every tiny edit separately, but do not combine unrelated solver, documentation, workflow, and benchmark changes into one giant commit.

## Natural commit titles

Good:

```text
Stop counting fallback runs as SPECTRA wins
Put the native solver on the full 2026 field
Keep rejected proofs in the final score
Skip preprocessing when it cannot pay
Freeze the hidden-set build
Make the README match the actual project
Archive the old panel scaffolding
Report SAT and UNSAT scores separately
Keep failed benchmark cells in the evidence
Pin the competition field
```

Too robotic or vague:

```text
Implement generalized optimization framework v2
Finalize final performance improvements
Update files
Fix stuff
Improve benchmark system
Enhance solver architecture
Complete 100/100 changes
Misc cleanup
```

## Commit-message formula

Use:

```text
<Plain-English action>

<Why the change matters, what evidence it affects, and what it does not claim.>
```

Example:

```text
Stop counting fallback runs as SPECTRA wins

Classify same-backend cells as fallback parity and reserve win counts for
distinct SPECTRA native or frontend work. Preserve the old raw timings and
do not change any benchmark result.
```

---

# 10. End-of-session checklist

Before ending a work session:

- [ ] Did I stay on the one canonical branch?
- [ ] Did I update this roadmap’s status?
- [ ] Did I retain every failure?
- [ ] Did I avoid changing a frozen rule?
- [ ] Did I separate standalone SPECTRA from assisted-portfolio results?
- [ ] Did I run the relevant focused tests?
- [ ] Did I run or trigger the full matrix when the source identity changed?
- [ ] Did I commit one coherent chunk?
- [ ] Does the commit title sound like a human explaining what changed?
- [ ] Did I push the canonical branch?
- [ ] Did I record the exact commit SHA?
- [ ] Did I record the workflow run and artifact hashes?
- [ ] Did I update the PR body only after the checks actually completed?
- [ ] Did I avoid saying “done” when only engineering—not science—was done?

### Friendly reminder

> **Before we stop: make one clean commit with a normal human title. No “final-final-v3,” brother.**

---

# 11. Reusable progress note

Copy this section after each major checkpoint.

```text
Date:
Canonical branch:
Head SHA:

What changed:
-

What passed:
-

What failed:
-

New exposed result:
-

Biggest remaining loss:
-

Claim currently allowed:
-

Claim still forbidden:
-

Evidence artifact and digest:
-

Next highest-leverage task:
-

Natural commit title:
-
```

---

# 12. Immediate priority order

The highest-leverage next sequence is:

1. **Fix the win-accounting defect.**
2. **Clean the repository surface and recover file headroom.**
3. **Update stale PR, README, repository description, and replication tracking.**
4. **Run standalone native SPECTRA-G across a large exposed SAT+UNSAT inventory.**
5. **Run the complete pinned modern solver field under identical rules.**
6. **Calculate the first real SPECTRA overall/SAT/UNSAT PAR-2 ranking.**
7. **Profile failures by family and improve the native solver on exposed data only.**
8. **Reach exposed rank #1 and the internal large-gap target.**
9. **Freeze source, selector, container, field, limits, and checkers.**
10. **Run one sealed or official hidden evaluation.**
11. **Obtain outside replication and expert review.**
12. **Merge, sign, release, and pursue real operator adoption.**

---

# 13. Final definition of done

Part 3 and the general-competition campaign are not complete until all statements below are true:

- [ ] Native SPECTRA-G has a complete official-style exposed score.
- [ ] It competes against the complete predeclared modern field.
- [ ] It ranks first on exposed development data.
- [ ] It meets the internal large-gap standard.
- [ ] SAT and UNSAT are both strong.
- [ ] Models and proofs are independently accepted.
- [ ] The selector is frozen and outcome-blind.
- [ ] A new hidden inventory is opened only after the freeze.
- [ ] SPECTRA-G ranks first on the hidden inventory.
- [ ] No hidden formula or rule is replaced.
- [ ] An outside team reproduces or independently challenges the result.
- [ ] An expert reviews the comparison and claim boundary.
- [ ] The final source and evidence are signed, permanent, and reviewable.
- [ ] The project is merged and released without losing the exact winning identity.
- [ ] At least one external user or operator validates practical value.

Only then is the broad project honestly at **100/100**.

---

## Reference surfaces

Keep these synchronized with this roadmap:

- `experiments/general_competition/PROTOCOL.md`
- `experiments/general_competition/FRONTEND_POLICY.md`
- `docs/PART3_RELEASE_REPLICATION_READINESS.md`
- `README.md`
- draft PR #76
- issue #74 and any future general-CNF replication issue
- latest official-score and exposed-panel artifacts
