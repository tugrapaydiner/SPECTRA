# Exposed general-CNF development campaign — 2026-10-21

**Status:** full restart in progress from `1f6a9133a08b9d8e5a6a6690979f55fbe633e0c5`; the first attempt is retained as incomplete.
**Correctness finding:** two native SAT models have already been confirmed
invalid. SPECTRA-G is disqualified under the frozen campaign rules; the remaining
cells continue to be collected. A subsequent repair cannot erase this result.
**Claim boundary:** exposed development evidence only. This is not an official
competition run, a complete modern field, a hidden confirmation, or independent
replication. The supplied [100/100 roadmap](../../docs/SPECTRA_100_100_ROADMAP.md)
remains the acceptance standard; its historical audit scores have not been raised.

## Why this campaign exists

The previous four-formula panel measured a structural frontend followed by
CaDiCaL. It did not measure standalone SPECTRA-G against modern native solvers.
This campaign establishes an independently checked, retained baseline for the
native solver before changing its algorithm. Every declared instance/solver cell
is retained, including admission failures, timeouts, unknowns, resource failures,
checker failures and invalid answers.

## Frozen inventory and limits

The [inventory receipt](exposed_inventory_20261021.json) bundles the original
selection, acquisition record and source inventory. The selection used GBD's
public SAT Competition 2023–2025 metadata, before running any solver:

| Property | Predeclared value |
|---|---|
| Formulas | 96, all successfully acquired without replacement |
| Per year | 13 metadata-SAT, 13 metadata-UNSAT, 6 metadata-unknown |
| Total strata | 39 SAT, 39 UNSAT, 18 unknown |
| Metadata families | 47 |
| Size filter | At most 10,000,000 bytes in GBD metadata |
| Selection | Deterministic SHA-256 family round robin within year/stratum |
| Seed | `spectra-exposed-baseline-20261021-v1` |
| Exclusions | 1,104 year memberships, all retained with reasons |
| Deduplication | Global GBD ID, isohash and isohash2; exact admitted semantics also reported |
| Solver limit | 5 seconds, fresh process group, sequential execution |
| Checking/admission limit | 30 seconds per process |
| Memory | 4,096 MiB virtual address space **per process** |
| Output limit | 64 MiB per individual output file |
| Admitted DIMACS limit | 64 MiB |
| Ranking | PAR-2; unsolved costs 10 seconds; correctness failure disqualifies the solver |

The selection file SHA-256 is
`dad14ec29395eab1f67ceddfc8bbc023ee9341e6c33ec01ea3daf44be96b9f5e`.
The metadata database hashes and every source URL/hash are in the receipt. This
does not establish near-duplicate or family leakage control. All formulas are
public and exposed. The declared protocol keeps unknown metadata in a separate
stratum. Review found that the frozen runner instead infers some unknown strata
from checked answers; the original calculation is retained as `legacy-v1` and
any corrected table must be reported as a separate derivation.

These deliberately short, size-bounded development limits are not the roadmap's
5,000-second/32-GiB official-style confirmation. The host is shared, not exclusive;
there is no aggregate cgroup memory isolation or security sandbox. Timing covers
process startup, parsing, solving and proof output; independent model/proof checks
are timed separately. Common source acquisition, decompression and admission are
excluded equally from every solver. Rotation changes the first solver on each
formula. One repetition cannot establish a stable small timing difference.

## Declared solver field

The exact executable commands, source identities, build notes and expected binary
hashes are recorded in [the field receipt](exposed_field_20261021.json).
The field SHA-256 is
`bcbc07712517a0f953bab0312c6522f9991085efe4f0d4386bb396c1ade482ce`.
There are eight entries:

| Entry | Role | Evidence |
|---|---|---|
| SPECTRA-G | Standalone native baseline, no upstream engine | Complete models; DRUP checked by pinned `drat-trim` |
| CaDiCaL | Current upstream baseline | Models and DRAT |
| Kissat | Current upstream baseline | Models and DRAT |
| Glucose 4.2.1 | Independently maintained baseline | Models and DRAT |
| AE_kissat2025_MAB | Pinned official 2025 submission | Models and DRAT |
| EDA-Kissat-v2 | Pinned official 2026 submission | Models and DRAT |
| Satsuma-iter+Kissat | Pinned official 2026 Main winner | Models; BSR/DSR checked by its bundled `dsr-trim` |
| Kissat-MAB-HyPre | Pinned official 2026 submission | Models; PB 3.0 checked by VeriPB 3.0.2 |

The official 2026 score CSV reproduces 19 Main entries over 400 formulas. Its raw
solver ranks are Satsuma-iter+Kissat first, the same author's
Satsuma-iter+AE-Kissat-MAB second, HyPre third, and EDA-Kissat-v2 seventh. References
to second/third-place **author awards** in the earlier protocol must not be read
as the raw solver ordering. This eight-entry diagnostic is not that complete
field. It also does not satisfy the full roadmap floor: Lymphosat and the historic
`kissat-sc2024` target remain absent, and the 2025 AE entry is not silently
substituted for that target.

Sources are the [official 2026 submission archives](https://satcompetition.github.io/2026/downloads.html),
the [official 2025 AE archive](https://satcompetition.github.io/2025/downloads/solvers/main/AE_kissat2025_MAB.tar.xz),
and the pinned upstream repositories listed in the field receipt. Original build
scripts and flags are retained. Archive binaries that `make` considered current
were rebuilt from source before measurement. Build logs and skipped unsafe
generated archive symlinks are retained. VeriPB is the exact prebuilt 3.0.2
checker bundled in the official 2026 Lymphosat submission; it was not locally
rebuilt. Hashes bind observed artifacts, not an independently certified supply
chain. The [component manifest](exposed_components_20261021.json) preserves
nested component/license notices, build evidence, and unresolved provenance gaps.

### Retained preflight failure

HyPre's original wrapper returns zero after its cleanup even for SAT and UNSAT;
only this entry permits exit zero with a separately checked verdict. Its
nontrivial UNSAT smoke proof passes VeriPB, but its contradictory-unit smoke
proof is rejected with “There is no contradicting constraint in the database.”
The original input, proof and checker logs are retained. The competitor is not
patched and the negative case is not removed. A table of benchmark outcomes must
not be described as a proof-clean validation of this submission.

## Storage repair and retained incomplete attempt

The initial run used source `9eee856fbac3e9e1bf334f3dca4c8f40e271f82e`. It was
stopped on 2026-10-21 at 17:25:56 UTC after 49 of 768 cells because shared disk
capacity could not hold the projected raw proof output. All 49 completed cells,
the interrupted cell's raw files and a complete list of the 719 uncompleted cells
are retained. This attempt has **no accepted ranking**. Nothing is dropped or
replaced based on a result.

The restart keeps exactly the same 96 formulas, eight entries, binary hashes,
solver/checker limits and order. The only harness change is optional post-check
lossless XZ compression, outside the solve/check timers. Every compressed file
records both original and stored sizes/hashes; decompression must reproduce the
original bytes before the uncompressed copy is removed. Compression failures
abort with retained evidence and no ranking. The published restart source and
full evidence archive identities will be recorded with the results. Evidence
output for the restart uses a memory-backed filesystem because the shared disk
cannot hold the retained campaign. This is applied equally to every solver and
is another reason the result must not be presented as an official-hardware run.
The incomplete-attempt archive is 231,406,636 bytes with SHA-256
`6f8235fc6a133d5fa3b770a8bd52c23de52a8eda6a8b3b44e50d05077892c6ff`.

## Checking and evidence discipline

The runner uses a separate private input copy for each solver and checks answers
against the retained original CNF. Original, private and compressed-source hashes
are checked around execution. A private-input mutation is invalid evidence; any
shared-original mutation invalidates the campaign. Expected executable hashes are
checked before launch and again at the end. Source/checker/manifest drift prevents
an accepted ranking. The runner executes from a clean, frozen source checkout.

Every SAT assignment is reconstructed to the original variable range and checked
against every original clause. Unmentioned variables are assigned false without
search; the resulting total assignment must satisfy the formula. Every UNSAT
answer requires a retained proof accepted by its declared external checker. A
checker timeout earns no solve. Invalid models/proofs disqualify that solver
globally, including all family and SAT/UNSAT tables.

Evidence includes complete cells, raw streams, exit codes, elapsed/CPU time,
per-process peak RSS, native statistics, complete models, proofs, checker logs,
admission receipts, source/build identities and an evidence SHA-256 manifest.
The admitted original is retained once rather than duplicated for every solver.

### Confirmed native model failures

Two independently parsed model/CNF pairs establish actual clause violations,
not missing assignments or a checker-format mismatch:

| Frozen cell | Family | Original clauses falsified | First falsified clause (1-based) |
|---|---|---:|---|
| `00003/solver-000` | modcircuits | 2 | 72: `1 3 2 4` (all four variables false) |
| `00017/solver-000` | sgen | 27 | 13889: `64 65` (both variables false) |

Both models contain exactly one ordered assignment for every declared variable.
A separately written parser and `python-sat` agree with the retained checker.
The source archive hashes match, and decompression reproduces the original
admitted CNFs. The forensic receipt SHA-256 is
`a63e422c00a20a2f124d6deb5525e281d437a344a0d7d6a5e9cec9bc6baf5701`.
The native decision-heap relevance invariant is under investigation; any fix is
validated as a different source identity. Performance tuning is paused.

### Scoring and verification repairs

The current runner records `declared-inventory-v2` for new campaigns. It keeps the
39/39/18 inventory strata fixed, preserves valid answers that conflict with
metadata while requiring adjudication before ranking, and distinguishes the
pinned checkers' exact allocation failures from invalid proofs. These corrections
do not modify the frozen baseline source or its raw records. The historical
official-score replay retains its original default behavior.

A separate offline verifier checks file and decoded-content hashes, private input
integrity, complete cells, process/checker receipts, and recomputed rankings. It
replays archives without a policy field as `legacy-v1`. Optional fresh answer
checks require trusted local code and temporary input copies; stored commands are
never executed by default. See [the reproduction guide](../../docs/DEVELOPMENT.md#scoring-policy-and-offline-evidence-verification).

## Corrected earlier frontend accounting

[The accounting receipt](panel_accounting_20261021.json) preserves both the
original CI panel and a derived classification; no old timing was overwritten.
Original artifact: run `39612064795`, artifact `12351182490`, ZIP SHA-256
`5b2e30bb6448cf7304719d3c27f7497ca69535a158fc11c7b5586c144eec7035`.

| Classification | Samples |
|---|---:|
| SPECTRA frontend win | 3 |
| Upstream fallback parity | 9 |
| Standalone SPECTRA native win | 0 |

The reducer-only median candidate/control ratio is `0.19003438997146574`
(approximately 5.26×); its geometric mean is `0.22300867275580685`. The fallback
median is `1.0005831088474717`. The mixed median/geometric mean remain available
as diagnostics and are not counted as algorithm wins. The frozen
`general-frontend-v1` routing rule has not been retuned.

## Reproduction

Use the frozen commit recorded by the campaign, build each declared source with
the retained commands, and rebase the field's absolute local paths to the new
build locations. Rebuilt artifacts have their own declared hashes; record them
as a reproduction rather than overwriting the original field. Acquire sources
from the inventory's original URLs and require their hashes. Then run:

```bash
python -m experiments.general_competition.campaign run \
  --inventory /absolute/path/acquired/inventory.json \
  --solvers /absolute/path/solvers.json \
  --out /absolute/path/new-evidence-directory \
  --timeout 5 --checker-timeout 30 --memory-mib 4096 \
  --output-mib 64 --dimacs-mib 64 --compress-evidence \
  --checker /absolute/path/drat-trim
```

No result from this inventory may later be called hidden confirmation. Source
identities, results and publication status are recorded separately so a later
documentation commit cannot silently become the measured source.
