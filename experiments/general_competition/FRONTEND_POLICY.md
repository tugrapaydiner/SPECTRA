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
