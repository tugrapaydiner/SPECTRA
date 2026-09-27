# Ownership and replayable receipts — September 27, 2026

Baseline: `e6f3d6b44ec53c158b9fa8021ddbadded482f1bb`, tree
`101314c0cedea8d3ee1a0b3103419a1158e288df`. All eleven baseline cloud workflows
were green. That did not establish absence of native-lifetime defects.

## Reproduced problem and correction

Twelve controlled dispatch regressions failed on the unchanged parent. They found
same-thread close/reentry at legacy scalar, shared scalar/buffer, certificate and
prepared-owner worker-creation boundaries, plus repeated native destruction on
recursive close. Tests stop before dereferencing a freed pointer: these are not
observed exploit/crash counts. The failed log and original probes are retained.

`_NativeOwner` now supplies one request lease to all those paths and fused calls.
A strong reference owns the native allocation and library through execution and
result validation. Close is deferred during active operations, same-thread native
reentry is rejected, and destruction detaches its pointer before calling C.
Fused capsules are invocation-local and hold the owning resource, not only an
integer address. Arbitrary asynchronous exceptions during cleanup and private
state/native-pointer tampering are not made safe by this contract.

## Stronger optional verification

The old vote checker remains deliberately conditional on supplied pair outcomes.
The new independent receipt verifier also checks the intended model bytes,
independently supplied transformed input, precision, label order and each disclosed
pair sign. It parses the wire format separately and uses ordered Python binary64
arithmetic and system exp; it imports no native executor or numerical framework.

All **4,281 retained model/input pairs** pass isolated replay: **5,064 disclosed
pair decisions** and **1,357,027 kernel evaluations**. A forged partial tournament
can pass the old conditional vote predicate but is rejected by numerical replay.
Model/input substitutions, changed signs, labels, precision and work-budget excess
are rejected in the tests. Signed zero is bound explicitly.

This is local computed-binary64/libm replay, NOT an exact-real proof, cryptographic
signature, truthful real-world label guarantee, preprocessing attestation or outside
researcher's replication. Receipts contain feature values and can expose sensitive
inputs. Replay is an optional slower checking path, not part of ordinary inference.

## Measured correctness cost — regressions retained

AMD EPYC 9V74, one pinned core, Python 3.13.5. The before/after Python packages use
THE SAME portable native SVM and preprocessing binaries in the same process. All
18 frozen models and all saved rows are included, with fifteen randomized repeats
per arm and three scopes. There are **1,620 complete-job records** and **385,290
checked repeated predictions**; they are not new independent examples.

| Scope | Equal-task after/before ratio | Added cost | Regressing model medians |
|---|---:|---:|---:|
| Jobs of individual prepared-buffer calls | 1.136414 | 13.64% | 18/18 |
| Prepared-buffer batches of 32 | 1.029895 | 2.99% | 17/18 |
| Mixed raw-input fused traces | 1.065927 | 6.59% | 15/18 |

These scopes include new request leases, per-call capsules and fresh results, but
exclude model loading, compilation and optional receipt replay. The change was
not selected for speed; no performance gate was invented or relaxed afterward.
Timings use already-consumed convenience datasets, overlapping saved splits and
one machine. They are not production-tail or cross-host guarantees. No native
arithmetic, model, scheduler default or preprocessing operation changed.

## Completed checks

- Full local fast suite: **2,065 passed**, the same **16 historical slow exclusions**,
  two historical warnings, exit zero. This includes 17 new ownership and 49 receipt
  tests; they are not added twice to the total.
- The **418-case SVM subset** also passes with an AVX2 runtime and with BOTH native
  components compiled under UBSan. Repeated builds are not distinct tests. No
  sanitizer diagnostic observed; no new ASan, leak or complete-security claim.
- An actual wheel built through its sdist passes **48 installed-only checks** in a
  fresh environment without numerical frameworks. Both native components compile
  from installed sources. All **144 packaged source files** match the checkout.
- The unchanged history auditor checks **449 protected files** and its recovered
  helper byte-for-byte. All existing benchmark model files and native numerical
  sources remain unchanged.
- A separate standard-library auditor checks every timing record/order, output
  digest, both source snapshots, original input anchor, native binary and aggregate.
  **12 corrupted copied receipts are rejected**; these probes are separate from the
  full pytest count. Independent code is not independent researcher replication.

## Reproduction

Use `docs/SVM_RECEIPTS.md` for the public API. The evidence packet supplies frozen
`inputs/models`, original SDK identity manifest, the old source snapshot, literal
logs, original failure probes, raw timing records and per-input receipts.

```bash
python -I -S experiments/lifetime_receipts/replay.py --mode verify \
  --inputs /evidence/inputs/models --receipts /evidence/receipts \
  --out /new/receipt-replay.json
python -I -S experiments/lifetime_receipts/audit.py --run /evidence/cost \
  --before /evidence/before --after /this/checkout --inputs /evidence/inputs/models \
  --manifest /evidence/inputs/PRIOR_SDK_SHA256.json \
  --library /evidence/lib/libspectra_svm.so \
  --library /evidence/lib/_spectra_preprocess.cpython-313-x86_64-linux-gnu.so \
  --out /new/timing-audit.json
```

Use fresh output paths. Binary names are examples for the tested CPython ABI and
host. No benchmark models were retrained; the existing suite contains small
synthetic compatibility/training tests. No GPU or new accuracy claim. Candidate
package version remains 0.7.1 and must not replace existing published assets.
Exact-head cloud acceptance and any main merge are recorded separately in PR30.
