# SPECTRA closeout and partial source recovery — 2026-10-03

Work is stopped. The Improve SPECTRA continuation was checked and is disabled.
No new experiment, repository write, merge, release or default change was made
during this closeout.

## Durable verified work

Draft PR #50: https://github.com/tugrapaydiner/SPECTRA/pull/50

Exact head: 037c20b11465f0031f66b157757f01badcfb0d64.
Its Python 3.11 and 3.13 public-contract CI was rechecked and both passed.
The optional deductive backend solves forced and binary constraints before bounded
search. Its complete saved synthetic evaluation reports 124/160 solved formula/seed
cases versus 97/160; mean full time 8.588 versus 12.520 ms; mean worker peak RSS
21.051 versus 20.943 MiB with common interpreter/native imports. General 3-SAT
quality is unchanged, overall p95 is 2.3% slower, and native Glucose4 is faster
on structured inputs. There is no learned-model/general-intelligence gain claim.

The committed results, raw evidence and limitations are available at:
https://github.com/tugrapaydiner/SPECTRA/blob/037c20b11465f0031f66b157757f01badcfb0d64/experiments/deductive_search/RESULTS.md

Earlier maintenance work remains in draft PRs #47, #48 and #49, covering bounded
SVM iterators and consistent tree/compiler/FP evidence. They are not intelligence gains.

## Later attempt: useful source, no completed fresh evaluation

The later age-guided focused-search candidate and tests below are reconstructed
from the code edits visible in this conversation after workspace maintenance
removed the temporary checkout. They are a PARTIAL SOURCE RECOVERY, not a copy
of an intact original run directory or a newly verified implementation.

Files contains the new focused solver and its correctness tests. The integration
patch adds the opt-in Python API and CLI backend against the exact PR #50 head.
It also includes the previously proposed CI base-branch entry. Nothing is applied
automatically. Review and revalidate before using the recovered patch.

The candidate uses a dense swap-delete unsatisfied-clause pool and a break/age
variable heuristic. Other attempted policies and restarts remain in the recovered
source. The historical compact default is unchanged by the patch. This is classical
search, not original Novelty equivalence or new learned intelligence.

The old tool output reported 370 public tests passing and these development results
on 32 fresh formulas, each with two search seeds. These numbers are TRANSCRIBED
FROM THE CONVERSATION, not retained raw observations or new measurements:

| Development-02 arm | Solved /64 | Mean complete ms reported |
|---|---:|---:|
| Existing indexed | 30 | 16.891 |
| Dense polynomial | 23 | 12.869 |
| Freebie | 24 | 13.119 |
| Minbreak/noise | 30 | 12.679 |
| Sharp polynomial | 28 | 12.196 |
| Freebie/restarts | 26 | 13.471 |
| Minbreak/restarts | 27 | 13.533 |
| Anti-reversal | 29 | 13.164 |
| Break/age candidate | 37 | 10.619 |
| Native Glucose4 reference | 28 | 34.520 |

The age candidate was selected by highest development SAT count, then mean time.
Reported candidate/indexed mean ratio was 0.6287064802483228. These results are
provisional because the raw files and complete original environment were not
recovered. Do not promote them as confirmed improvement or combine them with
PR #50's independent measurements.

The first separate untimed diagnostic reported 63/64 rows; a second same-input
diagnostic reported all 64 with indexed trajectory agreement. Both raw diagnostic
files are unavailable now. Development timing output reported inventories of
1,024 rows in the first run and 1,280 in the second; those raw rows are also missing.

The GitHub save/approval step was interrupted. A fresh lookup found no
research/focused-search-20261002 branch and no corresponding PR. The 96-formula
fresh evaluation was not started. Installed-wheel completion and new memory
measurements were not confirmed. The recovered files have only a syntax and patch
structure check during closeout; the 370-test claim belongs to the former run.

## Reproduction context for a future, separately authorized continuation

Development: planted/uniform random 3-SAT at density 4.2, sizes 128/512, eight
formulas per family/size cell, generation seeds 202610024000 through 202610024031
in family/size/instance order, search seeds 17/73, 2,048 total flips and two timed
rounds. Variants with restarts used an interval of 4*nvars while retaining the
same total flip cap. Native Glucose4 used python-sat 1.9.dev15 and requested 2,000
conflicts, which are unequal to flips and not a hard time bound.

The unrun evaluation declaration was 96 planted/uniform formulas at 128/512/1024
variables, 16 per cell, generation seeds starting 202610025000, the same search
seeds/flip cap, and three rounds. Controls were indexed, dense polynomial,
minbreak/noise, selected age policy and native Glucose4. It required a published
candidate freeze before input generation, independent original-clause witness
checks, complete timing including setup/checking, separate Python/RSS memory,
retention of failures, and no tuning on evaluation inputs.

The proposed optional-backend gate was at least five additional solved cases out
of 192 over indexed, no greater mean full time, no cell losing more than two cases,
and no invalid SAT witness. That gate has NOT been evaluated. Old sealed inputs,
models, tolerances and all 449 historical files must remain untouched.

This bundle contains no raw benchmarks, native dependency, model, full repository,
or certification. Checksums bind these recovered files only.
