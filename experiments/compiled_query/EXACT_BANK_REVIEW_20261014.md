# SPECTRA exact-bank checkpoint — recovered code review and hardening

**Review date:** 14 October 2026
**Recovered checkpoint:** `0e73a92882a5ce645270b329043c80d37318ff78`
**Recovered tree:** `291abbacb4f9b5d8a13336d8415cc2c571e49406`

## Bottom line

This is a real second advanced implementation line, not a transport stub. It adds
fused exact calls, serial and 64-lane checked answer banks, SCC and closure modes,
tile accounting, and 8,192-request development schedules. It diverges from the
prepared ABI-3/SBC1 line at `ed10903...`; it does not include that line's strict
binary full-session packet or one-call construction/disposal boundary.

The recovered source is internally strong, but its original resource admission
omitted the worst-case portable UNSAT proof payload. Therefore the old `max_bytes`
claim was not enforced for exact banks. The implementation could admit a request
bank under its formula and then retain output and scratch well above the cap until
allocator failure. This was fail-safe at the process level, but it violated the
stated conservative native-payload contract and could distort resource comparisons.

## Recovered implementation

- `solve_fused` removes a second interpreter transition while preserving the
  separately owned original-input observer.
- `solve_bank_checked` executes an ordered tuple bank and emits checked SCQ2.
- SCC/closure modes propagate up to 64 requests with word lanes.
- Raw/SCC/closure answer-bank controls share complete SAT witnesses and portable
  SBP1 contradiction proofs.
- Tile timings, proof bytes, visits, arcs, complete output hashing and disposal are
  represented in the study schema.
- The workflow defines exposed 1,024- and 8,192-request development matrices.

## Review verification before modification

The seven test modules passed in isolated fresh processes:

| Module | Passed |
|---|---:|
| generic BDD contracts | 83 |
| construction contracts | 39 |
| reduced-compiler contracts | 32 |
| exact service/bank contracts | 72 |
| independent proof replay | 19 |
| exact study/audit contracts | 43 |
| general study contracts | 83 |
| **Total** | **371** |

The affected bank modules were also rerun independently after recovery. A single
combined process was deliberately not substituted for these isolated results when
an exhaustive CUDD run exceeded the execution envelope; no partial run is counted.

## Concrete resource defect reproduced

On a 1,001-vertex path with 512 contradictory requests, the original Python/native
admission formula accepted a `1,240,234`-byte cap. The returned SCQ2 bank contained
`2,575,376` bytes, including `2,060,288` proof bytes. A conservative lower bound
for retained owners, scratch and output was `3,246,437` bytes: **2,006,203 bytes
above the admitted cap**.

The problem was not proof unsoundness. Every proof remained independently checked.
The defect was resource-accounting correctness.

## Hardening applied in this review

1. Charge every SCQ2 row for the complete portable-proof envelope: decision,
   witness/padding, proof length, and two simple paths of at most `2n` literals.
2. Charge native query copies, 64-lane propagation, selected-label, parent, proof,
   and vector metadata scratch before execution.
3. Recover the total session cap from the separately owned solver and observer,
   subtract retained owner arrays, and enforce the remaining bound in C++ even
   when the Python wrapper is bypassed.
4. Reserve the admitted output envelope once and fail if serialized SCQ2 exceeds it.
5. Bind `build.json` to Python executable, version, implementation, SOABI and
   platform, and remove a published library if receipt publication fails.
6. Reject combined solver/observer capacity exhaustion as `MemoryError` before
   passing a negative remainder to the unsigned native parser.
7. Add regression tests that reproduce the old proof-output undercharge through
   both the public wrapper and direct native call.

## Post-hardening verification

| Check | Result |
|---|---:|
| Seven isolated compiled-query modules | **374 passed** |
| Exact service/bank module | **75 passed** |
| Exact study/audit module | **43 passed** |
| General study module | **83 passed** |
| Exact module under UBSan | **75 passed** |
| Exact module under ASan + UBSan | **75 passed** |
| Python compilation | **PASS** |
| Git whitespace check | **PASS** |

ASan used `detect_leaks=0`; this is not a LeakSanitizer claim.

## Grade and integration decision

| Object | Grade | Reason |
|---|---:|---|
| Hardened exact-bank engineering checkpoint | **84 / 100** | Strong checked batching and tests; old ABI, monolithic native file and no final result package. |
| Scientific evidence at this checkpoint | **not promoted** | No committed final bank audit/result report was recovered. Code and workflow existence are not a measured win. |
| Value as a kernel for the prepared line | **high** | The 64-lane propagation and matched closure/raw controls are the right next implementation challenge. |

This checkpoint should **not** replace the ABI-3 prepared SBC1 line wholesale. The
next implementation step is to port the word-parallel propagation kernel and
matched controls into the stricter full-packet architecture, retain its independent
solver/observer parsers and output admission, and rerun all measurements. Results
from either parent line must not be inherited by the unified endpoint.
