# CPU reliability integration review

## Scope and branch reconciliation

Base `main`: `1e29cb10662cb83ba9e28e4d30164fc373a547a5`.
The integration starts from `research/verifier-aligned-cpu-search` at
`630efebde51781e707028abc1fc8fcba3339a71c`, whose M16/M17 implementation is
already present together with durable source/checkpoint/result archives.

`research/m16-verifier-aligned-cpu` at
`2910d0c8447d03e353b515bc3a57c1cfe5bda839` contributes its useful battery
observation repair and portability tests. Its conflicting protocol drafts and
one-off snapshot workflow are not adopted as current experimental instructions;
their historical commit remains available. The empty branch
`research/cpu-search-reliability-overhaul` has no changes beyond the base.
No fourth research branch is needed.

The previously described local `SPECTRA_cpu_search_reliability.zip` is not an
input to this integration: it was not available to verify. Claims made about it
are not substituted for actual repository contents or test records.

## What is improved

The M16/M17 code adds content-based ancestral exclusions; target- and source-bound
value checkpoints; optional checked search; fixed-pool/closed-loop experiments;
CPU native validated owning weights; native semantic checking; and durable raw
records. It preserves existing model/default deployment entry points and retains
classical comparators. The integration additionally repairs the following:

* Search transitions receive private state snapshots. In-place transitions and
  scratch-buffer reuse can no longer corrupt sibling paths, retained states, or
  caller inputs. Decode/value callbacks receive separate snapshots; returned
  answers and incumbent/checker inputs are also separated.
* Timing aggregators reject missing, duplicate, unmatched, nonfinite, nonpositive,
  or malformed observations, and inconsistent repeated correctness. Crossed
  bootstrap inputs must use matched examples across model seeds.
* Battery observations distinguish unavailable readings from numeric control
  defaults. Missing sysfs and invalid provider values do not crash the controller
  or become fake measurement evidence.
* Native packed geometry uses overflow-safe row sizing and rejects impossible
  products before signed integer overflow or allocation.
* A pinned offline evidence verifier validates archive inventories, consumed-data
  ancestry, manifest/array identities, independent semantic correctness of stored
  answers, and replayed summaries. Corrupt, duplicate, missing, extra, unsafe, and
  changed archive members fail closed.

Copying state has a real cost inside the optional search timing window. Historical
M17 timings belong to its recorded source commit, not automatically to the hardened
search implementation. Callback isolation is not a security sandbox: an exact
checker must still implement the declared task truthfully. Ancestry checks cover
declared content/groups, not every possible symmetry-equivalence relation.
The C++ accumulation algorithm is unchanged by the overflow repair.

## Scientific results retained, not upgraded by wording

M16's paired mean native-checker/reference ratio is 0.66918, CI
[0.66343, 0.67457], with equal semantic outcomes on 512 model–example cases.
This is a narrow execution-path improvement; the exact solver remains faster.

M17 Sudoku-shift confirmation quality-minus-improvement selection is 0.5859375,
CI [0.52734375, 0.64453125], on 512 model–example pools. Maze development coverage
is only 0.1484375; its effect of 0.06640625 does not pass the frozen gate.
Maze confirmation was not opened. Overall:
`TWO_FAMILY_CLAIM_NOT_ESTABLISHED`.

No thresholds, original records, checkpoint identities, training seeds or
confirmation-selection rules are changed here to turn those failures into wins.
This integration is not described as 100/100, L7 equivalence, an iso-energy result,
or a new generally superior search algorithm.

## Verification

Before modification, the recovered candidate source passed 430 fast tests on
Python 3.13.5 / PyTorch 2.10.0+cpu. The recovered source is the exact archive
from M17 source commit `33e548d3e0f52d0664c00929316b2198d5fcffe4`; comparison to
`630efeb...` shows only M17 evidence additions, no executable changes.
New adversarial regression tests cover the repaired cases without weakening
existing assertion thresholds.

Run the complete fast suite with `python -m pytest -m "not slow" -ra` and
`python scripts/verify_retained_results.py --out outputs/retained-audit.json`.
CI performs both on Python 3.11 and 3.13 with CPU-only PyTorch and records the
exact tested commit, complete source archive, dependency versions, JUnit results,
verification output and checksums. It runs on the integration branch, PRs and main.
It does not regenerate confirmation or select a better scientific run.

A separate full-suite command (`python -m pytest -ra`) includes the expensive
legacy training gates. Those outcomes must be reported separately from fast
regression. On limited-memory hosts `SPECTRA_TEST_MICROBATCH=16` changes only the
shared fixture's backward partitioning, not the sampled batch, update count, or
acceptance assertions; floating reduction order is not promised bit-identical.
See the PR/commit-linked validation records for the actually executed scope.

The offline evidence audit independently checks 11,520 stored complete-solve
answer records and 1,728 manifest rows. Fixed-pool labels are reaggregated, not
independently re-inferred. Integrity success does not mean scientific success.
