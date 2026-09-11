# SAT workload-admission pilot: a retained negative result

The pre-measurement source and protocol are committed at
`d47859aedad1a7b73cf48d0fa7a6347f411c17d3`. The initial native execution is GitHub
Actions run **34635415241**, job **103381976975**, artifact **10278195498**.
The tested PR-merge commit is `383410688b348f32b956933d21318e4d8c32c812`, exact tree
`8d8cac4f041ecfab041d0846857764640edc264f`. This tree matched all 543 tracked local
source files byte-for-byte and mode-for-mode. Publication preceded the measured
pilot; neither the ordering nor later CI reruns are external preregistration or
independent confirmation.

## Outcomes

All 48 generated inputs and all 288 solver/round observations are present:
189 independently checked SAT witnesses, 69 budget-limited unknowns, 30 UNSAT
reports, no execution errors and no supervising-process timeouts. **All six
admission gates fail**, with zero qualifying cases. No thresholds were lowered.
UNSAT reports are not independently checked proofs. Unknown does not mean UNSAT.

| Distribution | Variables | Inputs verified SAT in every solver/round | Qualifying cases | Maximum faster-solver median complete time among fully verified cases |
|---|---:|---:|---:|---:|
| Uniform | 64 | 5/8 | 0 | 0.993190 ms |
| Uniform | 128 | 3/8 | 0 | 3.365767 ms |
| Uniform | 256 | 0/8 | 0 | Not applicable |
| Planted | 64 | 8/8 | 0 | 0.670963 ms |
| Planted | 128 | 8/8 | 0 | 1.682771 ms |
| Planted | 256 | 6/8 | 0 | 7.004387 ms |

The highest conservative solve/check ratio among fully verified cases was
14.71912, below the declared 100x requirement. The largest measured completed
in-process duration across **all** statuses was 779.779798 ms. The table must not
be read as a runtime bound for unknown inputs or other generated formulas.
These small development strata support an admission decision, not a power
analysis, impossibility theorem, tuned-solver comparison or general SAT result.

The host reported AMD EPYC 9V74, four available CPUs / affinity 0-3, Python
3.13.15 and python-sat 1.9.dev15. Both solvers are serial configurations; no
GPU experiment was run. Every raw record includes solver counters, requested
budget, setup/solve/first-check/full in-process times and separate supervised
startup/import/IPC/cleanup duration. Physical energy remains null.

### Observed budget overshoot

A 2,000-conflict request was **not a strict observed-work cap**. CaDiCaL
recorded up to **2,004 conflicts** (21 observations above the request); Glucose
recorded up to **37,797 conflicts** (33 observations above the request). The
independent replay exposes these diagnostics from the original counters. Do not
claim equal actual conflicts or at-most-2,000-conflict execution. Full measured
latency, raw counters and the separate process deadline remain the valid scopes.
The original protocol already labels this a solver-specific request; the result
is retained without rewriting counters, censoring overshoot or changing gates.

## Permanent, lossless evidence storage

`manifest.json` binds the original execution and artifact identities. `pilot.zip`
is the byte-identical original 244,291-byte archive with all seven payloads,
including the complete input file. No inputs or observations are omitted. The
original generator separately regenerates the cases to check their exact identity.

```bash
python scripts/verify_sat_admission.py
```

This requires no SAT library and performs no new timing run. It checks the original archive
and payload hashes, source identities, all
input formulas and every SAT witness, then independently reproduces the original
summary. Corrupt/missing archives, altered source or changed records fail. Re-sealing
a falsified witness with new hashes does not bypass semantic verification.
Both existing CPU CI versions run it, and the retained-pilot tests exercise its
failure modes. The native solver CI also reruns the full development protocol;
those extra timing observations are CI reproduction, not new test-set evidence.

The original downloaded 51,423,766-byte CI archive has SHA256
`e6dfb39b147c94424db20c99a49c47be71671c3a29e495ffb4977543b10de8d4`.
Its CRC, every member checksum, full source bytes/modes and exact tree were
independently checked. The original seven-file pilot ZIP SHA256 is
`8446d6ce013f14c8b767cae8ce042412cd5061fbcfd6478df75c793cfa3778bd`.
The remote artifact follows the 90-day CI retention policy; the original archive
here does not depend on that link remaining available.

A one-time publication job in run `34637226337` independently checked the archive
and all member digests, then created immutable Git blob
`fe6986a461064c80a1251f697562f52f18a72f07`. Its temporary job had contents-write
permission only for this publication and did not update a branch or commit. The
publication workflow is removed from the final source; normal CI remains read-only.
The final review/merge still requires the ordinary checks on the exact final head.

`local_validation.json` records the original 716-test baseline, the initial
focused test failure and subsequent 800-test full pass. The initial failure was
a test-double class-scope NameError and is retained in `initial_test_failure.txt`.
Later retention tests extend the suite; CI carries their final exact counts.
Sixteen existing slow retraining tests remain outside the fast suite.

## Research decision

This change establishes a working witness/repair/evidence instrument. It does
**not** establish a learned SAT solver, competitive solve speed, generalization,
novelty or a neural training improvement. Do not train a large controller merely
because these files now exist. Refine the task-distribution hypothesis and strong
classical comparisons while preserving this failed admission result. Any later
structured-SAT experiment needs new declared development/confirmation boundaries.
The frontier-impact assessment remains approximately **35/100**.
