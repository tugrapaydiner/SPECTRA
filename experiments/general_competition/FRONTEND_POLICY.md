# General-CNF frontend policy

**Policy:** `general-frontend-v1`  
**Frozen:** 2026-10-21  
**Development source before freeze:** the already exposed four-instance SAT panel

## Rule

The selector uses only admitted formula geometry:

1. If the canonical formula contains an empty clause, use the checked binary-core reducer.
2. Otherwise, if it contains at least one unit clause, use the checked binary-core reducer.
3. Otherwise, send the original canonical formula directly to the matched solver.

The rule does not inspect a solver answer, measured runtime, benchmark identity, file name, source URL, or historical winner. A direct route still constructs a fresh solver instance and checks the complete SAT model against the exact canonical original CNF.

## Why this rule was frozen

The exposed development panel showed two distinct regimes:

- A unit-heavy, binary-dense formula was reduced to an empty residual and produced a large end-to-end win.
- No-unit formulas paid the SCC and substitution cost without enough residual savings; one exposed cell was more than twice as slow as the direct control.

The frozen rule preserves the demonstrated propagation opportunity and removes that known no-unit regression by construction. Earlier negative measurements remain part of the development record.

## Claim boundary

This is an outcome-blind development dispatcher, not a prospective result. Its performance on the exposed panel may guide engineering, but it cannot establish general SAT leadership. The policy must remain unchanged before any genuinely new hidden or prospective inventory is opened; every timeout, loss, wrong answer, proof failure, and resource failure must remain visible.

## Accounting for the exposed panel

Panel schema `spectra.general_exposed_panel.v4` removes the ambiguous
`hybrid_wins` counter. Every repetition has a `result_classification` and every
case and full report retain counts for all nine outcomes:

- `SPECTRA_NATIVE_WIN`
- `SPECTRA_FRONTEND_WIN`
- `UPSTREAM_FALLBACK_PARITY`
- `LOSS`
- `TIMEOUT`
- `UNKNOWN`
- `INVALID_MODEL`
- `INVALID_PROOF`
- `RESOURCE_FAILURE`

A valid `plain` route is always `UPSTREAM_FALLBACK_PARITY`, whether its measured
ratio is below, equal to, or above one. A checked `reduce` route earns
`SPECTRA_FRONTEND_WIN` only when its complete charged time is lower than the
control. Equal time is a non-winning `LOSS`. This panel uses an upstream solver
in both arms, so it cannot emit `SPECTRA_NATIVE_WIN`; the native binary reducer is
still a frontend, not the standalone SPECTRA-G CDCL entry.

Raw timings, including slower fallback repetitions, remain in the receipt.
Frontend-only and fallback-only ratio summaries are reported separately; an
all-cell ratio is a mixed diagnostic, not an algorithm speedup. Unchecked UNSAT
reports are `UNKNOWN`, not accepted proof results. Explicitly rejected proofs or
contradictory UNSAT reports against a checked SAT model are `INVALID_PROOF`.

Acquisition, admission, setup and comparison exceptions retain the scheduled
cells with phase and error information. Rejected SAT witnesses and completed
arms are retained when model validation fails. A detected model/proof error
stops further solving, records remaining cells as unrun `UNKNOWN`, and makes the
report `FAILED_CORRECTNESS`. Other incomplete panels remain `INCOMPLETE`; the CLI
writes their receipts before exiting unsuccessfully. The in-process diagnostic
does not provide hard per-cell resource isolation; competition scoring belongs
to the separately bounded standalone campaign.

The routing rule and earlier raw receipts are unchanged. Historical v3
`hybrid_wins` values count timing ratios, including fallback variation, and must
not be quoted as SPECTRA algorithm wins.
