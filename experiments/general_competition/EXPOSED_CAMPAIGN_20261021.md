# Exposed general-CNF development campaign — 2026-10-21

**Status:** the original 768-cell restart and separate 96-case repaired validation
are complete. Both archives passed post-run verification and fresh checks of
every reported verified answer. The first attempt remains retained as incomplete.
The separately frozen heap experiment also finished all 256 cells and returned
`NOT_ESTABLISHED`; its candidate is not adopted.
**Correctness finding:** all ten invalid native SAT models have been independently
confirmed. The original SPECTRA-G entry is disqualified under the frozen rules.
The repaired source returned nine checked answers and 87 timeouts in its first
validation. The separate longer-search diagnostic and retained-proof check below
now give verified corrected answers for two of the ten original failure cases;
eight remain unresolved. Later evidence cannot erase the original result.
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
the corrected table is reported as a separate derivation below.

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

## Completed original baseline

The run at `1f6a9133a08b9d8e5a6a6690979f55fbe633e0c5` finished at approximately
18:45 UTC on 2026-10-21 with all **768 unique cells** and exit code zero. Its
[unchanged physical-file manifest](exposed_evidence_manifest_20261021.json)
lists 6,341 files and has SHA-256
`a6bef0af6bd2b7bf603e4eee7f7a421ad20e693d8bb52e14d368fdb68dd59551`.
There were no recorded artifact changes or original-input mutations.

| Position | Solver | Verified SAT | Verified UNSAT | Timeouts | Other outcomes | Overall PAR-2 (s) |
|---|---|---:|---:|---:|---|---:|
| 1 | satsuma-iter+kissat | 9 | 15 | 71 | 1 checker timeout | 7.7091 |
| 2 | Kissat-MAB-HyPre | 6 | 12 | 51 | 27 unknown | 8.2304 |
| 3 | Kissat | 8 | 5 | 82 | 1 resource failure | 8.7491 |
| 4 | CaDiCaL | 9 | 3 | 83 | 1 resource failure | 8.8150 |
| 5 | EDA-Kissat-v2 | 7 | 3 | 85 | 1 resource failure | 9.0071 |
| 6 | AE_kissat2025_MAB | 6 | 3 | 87 | — | 9.0855 |
| 7 | Glucose-4.2.1 | 5 | 2 | 77 | 12 resource failures | 9.3205 |
| **DQ** | **SPECTRA-G** | **6** | **2** | **78** | **10 invalid models** | 9.2272, diagnostic only |

PAR-2 is the mean runtime with ten seconds charged for every unsolved case.
Invalid answers disqualify the whole solver before ranking, regardless of its
numerical mean. SAT/UNSAT columns count verified returned answers, including
answers on metadata-unknown cases. They are distinct from the declared-stratum
score denominators. The table retains the original `legacy-v1` calculation;
the separate declared-strata derivation leaves every overall score, rank,
solved count and global disqualification unchanged.

The [768 original score rows](exposed_scores_20261021.csv),
[unchanged legacy summary](exposed_summary_legacy_20261021.json), and
[separate declared-strata summary](exposed_summary_declared_20261021.json) are
available directly in this tree. Both summaries retain every family table.
These are byte-identical copies of their respective measured or derived files;
the new filenames do not change their policy or source identity.

The 15 solver resource failures retain return code `-25`; they are not silently
described as memory exhaustion. Satsuma's one checker timeout retains a
0.644784352-second solver run and a 30.041780312-second checker run and earns no
solve. The earlier HyPre smoke-test rejection remains outside this 96-case table.

Fresh post-run verification checked all 6,341 physical files, decoded and checked
all 811 compressed files, replayed the complete grid and scores, and accepted
all **56 SAT models and 45 UNSAT proofs** using explicitly pinned local checkers.
No verified answer was left unchecked. The fresh-check log manifest is
`6dc20f9e1c763412d0580393318ce2a28228f8ef0e1451d542409ed140da8562`;
the successful archive-verification report is
`d15b89c4eb0300ce361b3d4793d8580cff9d999c3966d4e08b1032107d42d9b6`.
These are local rechecks, not outside replication or renewed solve timings.

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
abort with retained evidence and no ranking. The restart source and final
evidence-manifest identity are recorded above. Evidence
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

The first two independently diagnosed model/CNF pairs establish actual clause
violations:

| Frozen cell | Family | Original clauses falsified | First falsified clause (1-based) |
|---|---|---:|---|
| `00003/solver-000` | modcircuits | 2 | 72: `1 3 2 4` (all four variables false) |
| `00017/solver-000` | sgen | 27 | 13889: `64 65` (both variables false) |

Both models contain exactly one ordered assignment for every declared variable.
A separately written parser and `python-sat` agree with the retained checker.
The source archive hashes match, and decompression reproduces the original
admitted CNFs. The forensic receipt SHA-256 is
`a63e422c00a20a2f124d6deb5525e281d437a344a0d7d6a5e9cec9bc6baf5701`.
The [diagnosis receipt](native_correctness_20261021.json) confirms the cause:
repeated activity rescaling underflowed some relevant variables to zero, and
heap rebuilding then omitted them. Instrumented original search reproduced both
wrong models byte for byte and found 19 and 88 relevant variables, respectively,
left unassigned with zero activity. A four-variable regression reproduces the
same failure with two clauses.

The completed all-failure audit independently confirms **all ten** retained
`INVALID_MODEL` cells. Every witness is complete and ordered; the independent
parser, Python-SAT clause parser and retained checker's first failure agree.
Across those witnesses, 5,615 original clauses are falsified. The full forensic
receipt is retained with SHA-256
`363a815f91f4cfab2e801e11125d88d58c315746d8fb6be793cd803aebb8e546`.
The compact [verification and failure receipt](exposed_verification_20261021.json)
binds every case to its original CNF and model hashes. Mechanistic instrumentation
reproduced the first two failures above; the clause audit does not claim a
separate instrumented root-cause reproduction for each of the remaining eight.

| Instance index | Family | Falsified original clauses | Repaired outcome |
|---|---|---:|---|
| 00003 | modcircuits | 2 | TIMEOUT |
| 00017 | sgen | 27 | TIMEOUT |
| 00027 | social-golfer | 120 | TIMEOUT |
| 00031 | social-golfer | 252 | TIMEOUT |
| 00041 | heule-folkman | 3,144 | TIMEOUT |
| 00050 | st-connectivity-principle | 20 | TIMEOUT |
| 00055 | binary-pigeon-hole | 2 | TIMEOUT |
| 00064 | random-csp | 31 | SAT_VERIFIED |
| 00071 | pythagorean-triples | 1,683 | TIMEOUT |
| 00074 | fixed-shape-random | 334 | TIMEOUT |

### Separate repaired validation

The repair at source `cb14031d4187c4cba5cccb37984ecf59d59bfdae` records variable
occurrence independently of floating-point priority. The regression fails on the
original engine and passes with the repair; all 21 native tests pass locally with
external DRAT checking. The [separate validation plan](native_validation_plan_20261021.json)
froze a native-only rerun of all 96 formulas after the original 768-cell baseline.
It started at 18:45:37 UTC and finished at 18:54:07 UTC with exit code zero.
The result is **7 SAT_VERIFIED, 2 UNSAT_VERIFIED and 87 TIMEOUT**. There are no
recorded invalid answers, checker failures, resource failures or admission
failures. The original six verified SAT and two verified UNSAT answers retain
their verdicts; one formerly invalid case now has a verified SAT answer and the
other nine remain unresolved. All 78 original native timeouts remain timeouts.

The [repaired physical-file manifest](repaired_evidence_manifest_20261021.json)
has SHA-256
`0ac536b565d759164e7462e5e6c614fc9b293475525ff57eee51e0893ec00408`.
Fresh post-run checks accepted all seven models and both proofs. The separate
verification manifest, including all 96 status transitions and every fresh log,
is `2f36665a698a59c49fe54176b0382462c726e053a8a087c8a0ff9a4ff07a4ab4`.
The measured native library is
`89ad45c8af2c4aaa7416eadbdd9b9c1f63e11547d465f0774ead97eaec28572f`.

A timeout supplies no corrected answer or long-search correctness result. This
separate one-entry run does not supply a new eight-entry ranking or paired
speedup. The earlier v1 heap optimization plan remains suspended with zero
ablation cells; its replacement v2 plan and outcome are separate below.

### Longer-search correctness diagnostic and retained-proof check

The [separate diagnostic plan](native_long_search_plan_20261021.json) was locally
frozen and its SHA-256 recorded before solving on 2026-10-21 at 23:54 UTC. It was
not an externally registered hidden experiment. All ten original invalid-model
cases were reacquired from the recorded public URLs, with both compressed and
original decoded hashes checked. The unchanged repaired source `cb14031` was
rebuilt with GCC 14.2.0 and run in a separate checkout on a shared Linux x86-64
host with Python 3.12.14. The exact compiler/build, source, checker, environment,
case order and budgets are retained in the plan.

Each case received one 60-second solve attempt, with 30-second admission and
independent-check budgets, 4,096 MiB per-process address space, and 64 MiB input
and per-file output limits. All ten cells completed: **1 SAT_VERIFIED, 8 TIMEOUT,
and 1 CHECKER_TIMEOUT**. The verified random-CSP case was already verified in the
five-second repaired run. This diagnostic alone therefore left the same nine
original cases unresolved. All 139 evidence files and 11 compressed payloads
passed post-run identity and lossless round-trip verification; a fresh offline
archive check also rechecked all ten compressed sources and the SAT model.

The sgen case `26114429758a203d651f4df87c94de4c` returned UNSAT after approximately
41 seconds and retained a 15,087,704-byte DRUP proof. Its original checker was
killed at the unchanged 30-second limit. A [separately frozen proof-check plan](native_long_proof_plan_20261022.json)
then allowed one 300-second verification of those exact proof/CNF bytes with the
same pinned checker and flags, without rerunning the solver. That check exited
zero with `s VERIFIED` after **37.19 seconds**. The proof, original CNF and checker
hashes were unchanged. The original `CHECKER_TIMEOUT` cell remains untouched;
this is additional correctness evidence, not a rescored benchmark cell.

The [compact evidence record](native_long_search_results_20261021.json) retains
all ten outcomes, both plans' hashes, the physical-file manifest, lossless
storage receipts, the separate checker logs and the archive-verification report.
Two original failure cases now have checked corrected answers, one SAT and one
UNSAT; eight remain unresolved. This does not close full G0 or establish a
performance rank, speedup, hidden-set result or outside replication. Local raw
payload retention and hashes do not establish a permanent public archive.

### Bounded learned-prefix and terminal-state audit

The [combined audit record](native_derivation_audit_20261022.json) retains a
separate locally frozen plan, exact harness/control sources, all ten outcomes,
checker logs and the 102-file raw manifest. The final plan was reviewed before
any of the ten original cases executed. Its SHA-256 is
`9ca0f4a4e8226702d15a62bdc30cc52794f81bb207f83780bab6a25cf65bcff2`.
The two superseded, unexecuted pre-review plans remain in the raw bundle.

The unchanged `cb14031` native kernel was included in a private C++ harness that
retains learned clauses on SAT and UNKNOWN as well as UNSAT. All ten original
failure cases ran once in their original order, with a configured 50,000-conflict
budget, 60-second solver/checker limits and the same 4,096-MiB per-process and
64-MiB per-file ceilings. The existing solver increments its conflict counter
before testing `conflicts > max_conflicts`, so a budget stop records **50,001
conflicts and 50,000 learned additions**. This behavior was declared before the
run; neither the source nor the observations were changed to conceal it.

All **473,640 learned additions / 25,353,801 literals** passed pinned
`drat-trim -f -U` forward RUP checking. Its exact `s DERIVATION` result and exit
one mean an accepted incomplete derivation, **not an UNSAT refutation**. Positive,
unsound, malformed, false-empty-clause, RAT-only and deadline-boundary controls
are retained. The runner rejects elapsed time above the frozen budget even when
a just-late process exit is reaped without the supervisor's timeout flag.

All ten terminal states passed checks of causal live reasons, watched-clause
registrations, unassigned relevant-variable heap coverage, cleared analysis
scratch, and proof/statistics consistency. The nine budget-limited searches each
reached 13 restarts and nine database reductions. Nine terminal states contained
zero-activity relevant variables, up to 6,490 in one case; this is a terminal
observation, not an event trace proving which variables were requeued or chosen.

The outcomes are **nine UNKNOWN_WITH_VERIFIED_DERIVATION and one already-known
SAT_AND_PREFIX_VERIFIED**. The SAT model was also checked against all 14,577
original clauses. No new defect or additional solved case was established:
corrected answer coverage remains two of ten, with eight unresolved. These
prefix/state checks are not exhaustive solver correctness, ABI serialization
coverage, a performance comparison, full G0 closure or outside-team replication.
Raw prefixes are retained losslessly outside the active tree; the user-retained
bundle is not a signed permanent public archive.

### Completed repaired-source heap experiment

The [v2 plan](heap_ablation_plan_20261021.json),
[unchanged 32-formula subset](heap_ablation_inventory_20261021.json), and
[preparation receipt](heap_ablation_freeze_20261021.json) were published before any
candidate build or A/B execution. The subset was selected from structural
metadata before observing outcomes; it is not chosen from successful solves.
The plan SHA-256 is
`ae2915bb84026dfc28fecb5c688e21a400e1efb046dfd21e25e5d58be1aafa32`;
the controller SHA-256 is
`f6d3e8954e00e2f835f828a60721b21b77fd848434862ccecea2929977b5318e`.
The original plan and all preparation revisions remain retained.

Execution required the complete original 768-cell baseline and the separate
96-case repaired validation, each bound to an independently retained final
manifest hash. Every returned repaired answer must pass its checker; timeouts
and unknowns remain unresolved. After those checks, the controller built both arms
from repaired source `cb14031d4187c4cba5cccb37984ecf59d59bfdae`. The sole candidate
change avoids queuing an activity bump for an already assigned variable.

The experiment retains 64 fixed-budget cells (256 conflicts, two seconds) and
192 endpoint cells (three paired repeats, two seconds), with both arms rebuilt
using identical compiler settings. All 32 fixed-budget traces must match exactly.
No endpoint repeat may regress in overall PAR-2. The plan also requires at least
eight common completed cases, a minimum amount of measured native work, and at
least a 5% reduction in each repeat and in aggregate fixed-budget native ABI CPU
time. Every mismatch, timeout, checker failure, regression and inconclusive
result is retained. A missing criterion yields `NOT_ESTABLISHED`; thresholds
will not be relaxed after results. Passing permits review, not automatic merging
or a full-field, hidden-set, independent-replication or 100/100 claim.

The run started at 19:27:16 UTC and finished at 19:38:17 UTC on 2026-10-21.
All **256 cells** and all four complete 64-cell campaign grids are retained.
The frozen conclusion is **`NOT_ESTABLISHED`**, and the heap change is not
incorporated into the production engine. The fixed-budget comparison has
22 exact returned pairs and ten pairs where both arms timed out. Two timeouts
do not establish identical search traces. Each endpoint repeat has one exact
completed SAT pair and 31 pairs where both arms timed out; the common completed
population is one case, below the required eight. Original native work on that
population is also below the required one second in every repeat.

No completed trace conflicts, invalid answers, resource failures, checker
failures, admission failures or artifact drift were recorded. Overall PAR-2
did not regress in any endpoint repeat, but that does not overcome the unmet
coverage and work criteria. Conditional phase totals from the 22 returned
fixed-budget pairs remain diagnostic only. They include approximately 3.18
seconds of original Python packing wall time and 0.56 seconds of original
native ABI wall time; these measurements motivate a separate packing study,
not a relaxed interpretation of this experiment.

The complete experiment manifest is
`e01dc39bc5657520173fa30aca965a0348d9928a6e628634c70fb1def8759239`,
and its frozen report is
`cfc8ebd164a06c2b5658f4073cc5080871e0e1c14ae341129972c41f3e130681`.
The [combined results receipt](exposed_results_20261021.json) binds the baseline,
all fresh checks, the separate scoring derivation, repaired validation and this
experiment to their original manifests. Its SHA-256 is
`0b2b7dcd795a01461005950c16c87c3fd2956d6932cf30e77eddebfaef47b85b`.
Raw preservation is separate from this receipt: these local checks do not
establish a permanent public release or independent replication.

### Packing study v1: retained admission failure

The separate [packing-study freeze](packing_study_freeze_20261021.json) has
SHA-256 `b41de4391dd74046d421305224a0ae34ade30a78c371cb6fd85e2b170354b29f`.
It publishes the complete exact controller, unchanged helper and tested test
source as losslessly encoded payloads, with their individual byte counts and
hashes. Both existing public inventory references and their exact reconstruction
rules are included. The controller requires the immutable published Git blob
and an identical HTTPS readback before creating a candidate. At the original
v1 freeze checkpoint, no candidate had been created or measured.

Both arms start from repaired source
`cb14031d4187c4cba5cccb37984ecf59d59bfdae` and use the same measured native
library. The sole prospective change replaces Python list/splat ABI packing with
native-width array buffers and lifetime-retaining ctypes views. Canonicalization,
native C++, search policy, and the prior unsuccessful heap change remain outside
this transformation. A width mismatch takes the original packing path over the
already canonical data.

The study requires 17 mandatory unit groups, exact canonical/literal/offset
comparisons on all 96 formulas, and three paired full-corpus packing repeats
(576 timed calls). Each clock encloses the complete `_pack_cnf` call, including
canonicalization; imports, admission, hashing and checking are outside it.
The native comparison retains 64 fixed-budget cells on the original structural
32-case subset (256 conflicts, 30-second ceiling) and 576 endpoint cells
(all 96 formulas, both arms, three repeats, five-second ceiling).

Eligibility requires at least a 10% reduction in aggregate full-corpus packing
CPU time in **every** repeat, with at least one second of original CPU work per
repeat. Every byte probe must pass, all 32 fixed-budget pairs must return exact
matching traces, and overall all-cell endpoint PAR-2 must not regress in any
repeat. Invalid answers, admission/artifact/checker/resource failures, incomplete
coverage and conflicting completed traces block eligibility. Timeouts retain
their penalties and cannot establish trace equivalence. No result-driven subset,
threshold change, retry or extra repetition is permitted. Passing allows manual
review; it does not merge a change or establish general solver leadership.

Before publication, 45 synthetic harness checks passed and static preflight
verified 528 prerequisite evidence files plus the pinned inputs and artifacts.
These preparation checks did not execute the candidate or benchmark workloads.
The freeze is a prospective local exposed protocol: its public payloads reconstruct
the harness, but its host-specific runtime paths and prerequisite evidence hashes
are not a self-contained public runtime bundle. Permanent public raw evidence,
hidden confirmation, outside replication and the roadmap's 100/100 claim remain
open. Earlier failed and inconclusive experiments are unchanged.

The v1 execution ran from 21:03:24 to 21:06:58 UTC on 21 October 2026 and
ended **NOT_ESTABLISHED**. All 96 untimed pair workers were attempted: 93
returned exact canonical, literal and offset identities; three exited before
either packing call. All 17 mandatory unit groups passed separately, including
35 untimed packing calls and 10 small direct native ABI calls. The corpus phase
completed 186 untimed calls and retained six unstarted calls. All **576 timed
packing calls and 640 native campaign cells are NOT_RUN**. This attempt provides
no packing-time or endpoint-performance result, and no candidate was adopted.

The three errors came from a harness assertion equating GBD selection metadata
with DIMACS declared variable counts. Both counts remain retained:

| Original index | Instance ID | Catalog variables | DIMACS declared variables | Clauses in both |
|---|---|---:|---:|---:|
| 5 | `587150fb7b12a6b5dd7e4a9446b9713b` | 1,889 | 1,890 | 12,666 |
| 18 | `211938776d92f11870a687abd11d55a4` | 1,000 | 1,001 | 9,897 |
| 33 | `f746dae2050dd4aa48bcd53e8bd897b4` | 74,776 | 74,804 | 393,322 |

The strict parser's declared counts are correct for the admitted inputs. Their
source hashes did not change. The full frozen output contains 2,855 files; all
2,854 payload hashes and the exact file inventory were independently verified.
The final manifest is
`e82ba27e9cf37bf6e462d245a3eab851b2eb4bc591af1e929b77346238621f5c`.
A read-only report replay returned the byte-identical report
`bf014313158ac8c8aafab539834805594d370f00d28648ac53f5dd139eb974b3`.
The [retained v1 result](packing_study_v1_result_20261021.json) publishes exact
lossless report, planned-cell accounting and error payloads. Its complete embedded
decoder reconstructs the original manifest from the pinned public Git source
tree, one candidate-file hash override, and all 896 other manifest rows. The
reconstructed 2,854-row manifest must match the original byte count and hash.
Subsidiary arithmetic comparisons on zero timing data are not performance evidence.

### Prospective packing study v2

The [separately versioned v2 freeze](packing_study_v2_freeze_20261021.json) has
SHA-256 `383caac9c1b3efd77eb6427cf96ddf4dac0475f3861b206d25f6eda8767a909d`.
It corrects only the harness's input identity binding and retains the complete
failed v1 attempt. It pins all 96 previously verified strict admission receipts:
DIMACS declared variables and clauses, exact decompressed bytes/hash, semantic
hash, and original compressed bytes/hash. The catalog metadata stays unchanged.
All 96 compressed identities and raw decompressed identities were independently
checked against those earlier receipts without rerunning either packer or solver.

Every new pair retains its complete admission receipt before checking those
seven frozen fields. Both live execution and report replay require the expected
receipt, configuration and exact integer dimensions. Missing or changed identity
records fail closed. The candidate transformation, source/native binary, original
96-case population and 32-case subset, order, phase counts, resource limits,
deadlines, clocks, thresholds and rejection rules are unchanged. This is a new
full experiment; no selected v1 cells are retried or reused as v2 measurements.

Before this v2 publication, 67 synthetic regression checks passed, an independent
review found no blocking issue, and static preflight verified the same 528
prerequisite files plus all pinned inputs and artifacts. No v2 candidate has been
created or measured at publication. Its published payloads reconstruct the exact
controller, helper and tests; host-specific runtime/prerequisite dependencies
remain explicit. Neither freeze establishes a self-contained public release,
hidden-set validation, external replication or general solver leadership.

### Prospective minimization scratch-buffer study — 2026-10-22

The [separate study record](minimization_scratch_study_20261022.json) freezes one
candidate: reuse the stack and touched-variable buffers in clause minimization,
clearing both at every invocation. Traversal order, marker cleanup, learned-clause
logic and every other native operation stay the same. Existing code allocates
fresh buffers for each attempted literal removal; this motivates testing the
allocation cost, without assuming a measured benefit.

All ten original failure cases remain in their original order. Three paired
repeats alternate baseline/candidate order by case and repeat, with 50,000
conflicts, 100 million cumulative learned literals, 60-second solve/check limits,
4,096 MiB per-process address space, and 64 MiB per-file output. The plan binds
newly built baseline/candidate harnesses, the exact original and canonical inputs,
Python, checker and execution environment before any exposed solve. These new
builds do not replace the absent frozen packing-study artifacts.

Eligibility requires exact ordered learned prefixes, models, native counters and
terminal invariant receipts in all 30 pairs, with identical output across repeats;
every prefix/model must independently check. Each repeat must save at least 5% aggregate native solve CPU, with at least
one second of baseline aggregate work. Cases with baseline median CPU of at least
0.25 seconds must avoid a greater-than-10% median slowdown. Every pair must keep
candidate post-exec peak memory within 110% of baseline plus 1 MiB. All failures,
timeouts and incomplete cells remain disqualifying evidence; thresholds are not
changed after results open.

This is a narrow fixed-work experiment, with no adoption, solved-count, end-to-end
ranking, hidden-set, outside-replication or G0 claim. Production native source is
unchanged. Measurements are not started at the freeze checkpoint.

### Continuation environment recovery — 2026-10-22

The [recovery checkpoint](reproduction_recovery_20261022.json) records work at
source `74c296d5b83ec5764594a98fdc538f1ac69c0c7f`, tree
`6096814dd15f343573ddfb0845ee3d78957d33ac`. Receipt validation at that source
passed 374 general-CNF tests with no skips, including the native sanitizer and
external proof checks. All seven automatic workflows completed successfully;
their run IDs and GitHub-reported artifact ZIP digests are retained in the
checkpoint. These are internal engineering checks, and the CI artifacts expire.

All 96 original public sources were reacquired with zero failures and zero
replacements, totaling 20,309,108 compressed bytes. The recovered inventory is
byte-identical to the frozen inventory, SHA-256
`b2b7c072b538e6e70645cd0b9974cdc873c061b01ffcdb3315bdfae157737501`.
Current strict admission and clause replay matched all seven frozen v2 identity
fields on every input: declared variables/clauses, decoded bytes/hash, semantic
hash, and compressed source bytes/hash. All 96 new admission receipts, the exact
checking script and its complete output are retained in the recovery checkpoint.
This validates input identity, without invoking a solver or packing candidate.

The exact published v2 controller, helper and test payloads were reconstructed;
all 67 synthetic harness checks passed again. Public data also restored both
inventories, all 96 compressed sources, nine pinned source files, both original
archive-verification summaries and the completed result report at their frozen
paths. Each of these 110 files matched its predeclared hash. The pinned Python
executable already matches. This is partial reconstruction of public bytes, not
recovery of complete original evidence trees or an executable runtime bundle.

The first static preflight stopped at the missing original inventory path.
After exact public reconstruction, a separately retained static preflight stopped
at the missing pinned `drat-trim` binary. The native library/build receipt and
the three original prerequisite manifests are also absent, as is the repaired
completion receipt. Those manifests bind further supporting files; recovering
the seven directly identified missing files alone would not establish full
preflight success. Expected paths and hashes are retained in the checkpoint.
Newly built tools and newly generated checks cannot substitute for these pinned
bytes under the existing freeze.

Both failed preflights and their outputs remain retained. No v2 candidate was
created, no packing or native campaign cell executed, and no threshold or frozen
rule changed. The original v1 failure, heap experiment, baseline invalid models
and all timeouts remain unchanged. Corrected answer coverage is still two of ten
original failure cases, with eight unresolved. Hidden evaluation, independent
replication, permanent public raw archives and the 100/100 claim remain open.

### Scoring and verification repairs

The current runner records `declared-inventory-v2` for new campaigns. It keeps the
39/39/18 inventory strata fixed, preserves valid answers that conflict with
metadata while requiring adjudication before ranking, and distinguishes the
pinned checkers' exact allocation failures from invalid proofs. These corrections
do not modify the frozen baseline source or its raw records. The historical
official-score replay retains its original default behavior.

For the completed original campaign, the explicit derivation policy is
`declared-v2-preserve-original-global-dq-v1`. The original legacy summary has
39 SAT, 42 UNSAT and 15 unknown cases because three declared-unknown formulas
received verified UNSAT answers. The separate derivation restores the declared
39/39/18 strata and retains those five verified solver answers in overall scoring.
Satsuma has 12 verified answers in the declared-UNSAT stratum and three UNSAT
answers on unknown cases; HyPre has ten and two respectively. No metadata
conflicts were found. There were no `INVALID_PROOF` cells, so **zero cell
classifications change**. Original cells, score CSV and summaries are retained
byte for byte; all original global disqualifications remain in force.
The derivation manifest is
`32c29f82e589c11f82efccf2aa1d868c073e12cc69699fd714b4e58acf32fd52`.

A separate offline verifier checks file and decoded-content hashes, private input
integrity, complete cells, process/checker receipts, and recomputed rankings. It
replays archives without a policy field as `legacy-v1`. Optional fresh answer
checks require trusted local code and temporary input copies; stored commands are
never executed by default. See [the reproduction guide](../../docs/DEVELOPMENT.md#scoring-policy-and-offline-evidence-verification).

## Corrected earlier frontend accounting

[The accounting receipt](panel_accounting_20261021.json) preserves both the
original CI panel and its derived classification in a lossless XZ/Base85 wrapper.
Its embedded Python decoder accepts both this wrapper and the original plain
JSON, verifies the complete original bytes and summary, and writes only to a new
output file. The restored document is 1,207,539 bytes with SHA-256
`9f7f534b36563531175d2bc99e2fd356db95e2fab986a490df83989bead5253e`.
The [original plain JSON at the immutable prior commit](https://raw.githubusercontent.com/tugrapaydiner/SPECTRA/166cefd0e0d53cc0bf490321a0405a0e06ad8c59/experiments/general_competition/panel_accounting_20261021.json)
remains available for readers expecting its original structure. Every observation,
model, timing, classification and metadata field is retained; this representation
change adds no measurement and overwrites no earlier result.
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
