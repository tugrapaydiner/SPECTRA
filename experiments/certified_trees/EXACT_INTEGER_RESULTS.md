# Exact-integer tree verification — completed local result

## Decision and source boundaries

Continue actual remote `20e36d4851ac37d33748c89ffbf496b8cd8358b5`, not the earlier
interrupted handoff. Reuse the four real frozen models/certificates from the separately
preserved `04173ea8ec7b62afd66309fc9c05cb5c3ef488e9` delivery. Its source branch is
retained separately; the stronger manifest/refinement changes are reconciled onto
the current remote implementation rather than blindly overwriting it.

The new exact-dyadic verifier reconstructs the IDENTICAL certificate bytes, without
running the old per-leaf Fraction parser/compiler. It is the default only within
this experimental module; `backend='reference'` retains the original implementation.
Neither backend skips the proof. No production/main merge, version or release follows.

There is no new classifier, accuracy increase, inference-kernel optimization,
world-first algorithm or outside-researcher replication. This removes a measured
verification/setup bottleneck. Binary-rational arithmetic is established; the
contribution is a tested, byte-identical implementation of this certificate policy.

## What changed

Stored binary64 leaves are converted exactly to integers with one power-of-two
denominator. Leaf contrasts, nearest-even rounding, and tree-wise residual extrema
then operate on integer numerators instead of normalizing general fractions for
every scalar. Bias/scale and source-roundoff calculations remain exact rational
where needed. The original conservative bounds, source-validation restrictions,
canonical digest, tie policy and native representation do not change. All eight
frozen 8-/16-bit files match the preserved rational compiler byte-for-byte.

Read EXACT_INTEGER_CONTRACT.md for the derivation. The new implementation shares
the strict JSON decoder, wire serializer and tiny exact power/log helpers, but not
the per-leaf parser or residual algorithm. This is differential evidence, not a
machine-checked proof that two implementations cannot share any bug.

The original `runtime.cpp` and `reference/certificate_oracle.py` are byte-identical
to the base. All frozen model files and original labels are unchanged. Ordinary
CatBoost agreement remains conditional on the ORIGINAL supported arithmetic/domain
contract; this compiler optimization neither strengthens nor weakens that contract.

## Complete new fresh-process comparison

One pinned CPU: AMD EPYC 9V74 80-Core Processor. Python3.13.5; one process at a time; numerical Python
frameworks are not imported. The exact measurement script and input identities were
frozen before the run. Seed2026092953, three repetitions per cell. All108 attempts
completed: four models, both precisions, both exact backends, proof-only and full
session-setup phases, plus twelve official-library-load controls. No selective
rerun, discarded timed failure, cached proof or model refitting.

The session phase includes source/compact file reads, strict parsing, exact
reconstruction, canonical serialization/full-byte comparison, native loading and
loading the actual original CBM/official fallback library. Interpreter startup and
imports are outside this phase timer, but each child's entire wall time is also
recorded. Own-process Linux VmHWM is read immediately after preparation. One actual
prediction is validated after timing. Filesystem caches may be warm. These are not
cold-disk guarantees, service-tail distributions or worst-case batch-memory bounds.

### Verified 16-bit session setup, three-process medians

| Model | Old rational ms | Exact-integer ms | Speedup | Old peak MiB | New peak MiB | Official load-only ms |
|---|---:|---:|---:|---:|---:|---:|
| Letter | 3705.883 | 625.737 | 5.92x | 181.44–181.44 | 88.36–88.36 | 8.282 |
| Pendigits | 1341.263 | 265.404 | 5.05x | 82.52–82.52 | 47.55–48.05 | 4.613 |
| Satellite | 740.606 | 163.616 | 4.53x | 54.91–54.91 | 39.42–39.65 | 3.840 |
| Optdigits | 1133.312 | 235.290 | 4.82x | 73.07–73.09 | 46.81–46.84 | 4.804 |

The verified-startup improvement is 4.53–5.92x on these four recorded workloads.
Peak preparation memory is lower by approximately28–51%. These ratios compare two
implementations of the SAME proof and native-session boundary, not proof against
no proof. Proof-only and 8-bit cells, CPU times and total process wall times remain
in the full record and are not pooled into the headline.

Official CatBoost loading without exact reconstruction still takes only about4–8ms
here. The new verifier still takes roughly0.16–0.63 seconds for the complete verified
session. This is NOT a cold-start victory over CatBoost, a warm-inference speedup,
or an end-to-end 5x file-processing gain. Verified workers still benefit from reuse.
The source-pair SDK assembler can perform expensive checks offline; this experiment
does not claim to accelerate all of its CBM re-export operations.

## Reconciled correctness boundaries

The local04173ea delivery already contained two important fixes which were not all
present on remote20e36d4. They are credited as reconciled work, not new discoveries:

- SDK v2 validates original source class order, required asset roles, exact file
  closure and source/binding identities. The current remote CBM-to-JSON verifier,
  which compares the full interpreted prediction structure, is retained.
- 8-bit accepted inputs need not be a subset of 16-bit accepted inputs. The native
  counterexample and v2 manifest now use MEASURED refinement coverage. The SDK
  builder executes refinement instead of assuming its count equals the 16-bit count.

Hashes and a self-written manifest do not authenticate a malicious publisher who
replaces the entire source/certificate/receipt set. Input directories and native
libraries remain trusted. The generic explicit fallback API retains its caller
source-pair obligation; the SDK assembler checks that relationship before packaging.

## Completed local acceptance

The341-case suite contains131 inherited cases,188 new exact-compiler differential/
adversarial cases, and22 reconciled manifest/refinement cases. All341 pass on fresh
portable, AVX2 and AVX2-UBSan builds with no skips/failures. The original13 rational
oracle tests also pass unchanged. These are repeated tests, not1,036 unique cases.
The prior baseline131 cases were run before changes and passed too.

Differential cases cover8/16-bit, binary/multiclass, pairwise bounds, negative halfway
rounding, cancellation, subnormals, unusual non-dyadic normalized bias, extreme
scales, invalid schemas and CRC-resealed forged certificates. A monkeypatch check
proves the new path does not call the old per-leaf parser/compiler. Trust/skip
backend names are rejected; no hidden proof bypass exists.

Whole-corpus replay executes128 cases across both verifier-produced objects,
scalar/tiled routing, early checkpoints, compact8/16, shared-route refinement and
both official libraries1.2.8/1.2.10. All indices AND work/tree counts agree, including
unresolved cases. This checks361,440 repeated rows from11,295 underlying source
inputs. The source models retain10,628 ground-truth-correct predictions, not11,295;
certification means agreement with the source classifier, not a true-label proof.

Both freshly assembled SDK targets replay all four models without numerical Python
imports. Compact16 accepts11,284 rows; the actual original model resolves the other
11 (Letter9, Pendigits1, Satellite0, OptDigits1). Those counts are unchanged.
Twenty-four isolated public CLI jobs cover all models, both builds and three
policies, with67,770 repeated rows checked by the independent output auditor.
Eight additional actual CLI attempts check existing-file refusal and late-invalid
input without publishing partial results. No timing claim is derived from these
functional replay runs.

The separate standard-library study auditor checks all108 recorded attempts,
source/model/library inventories, executed commands, exact certificate hashes,
child observations and derived medians. Sixteen altered disposable receipt copies
are rejected. It verifies recorded identities/arithmetic, not elapsed-clock truth
or a hostile wholesale evidence replacement.

One initial negative-audit launch failed because isolated Python excluded its
sibling import path; the retained failure precedes the corrected explicit file
import. A container streaming-session launch also failed before the timing study
started. Neither was a discarded numerical/benchmark cell. No benchmark model was
trained; existing tests fit only small synthetic compatibility models.

## Limits and reproduction

Fresh acceptance here is Linux x86-64 only. No new Windows, ARM, ASan, historical
full-project suite, production release or external user acceptance is implied.
The original prediction algorithms and stronger older model-quality results remain
separate. All prior timings retain their original named sources and are not relabelled.

Use the SDK's `python -I -S selftest.py --target portable --out ../replay.json`.
AVX2 is explicit and requires compatible hardware. Outputs must be outside the
immutable SDK. The original model and official library remain required for full
coverage. Operating-system runtime libraries, Python and compilers are not bundled.

For the exact study, run verification_study.py with the included frozen evidence,
source, native library and official library; audit with audit_verification.py.
Use NEW output directories. Source hashes must match the measured source files.
The reference backend is intentionally slower and remains selectable for diagnosis.

Final remote commit/PR acceptance and extracted archive identities are recorded
in the delivery receipt; previous CI is not assigned to this source until its own
workflow actually completes.
