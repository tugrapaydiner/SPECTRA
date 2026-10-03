# Focused-search candidate recovery — 2026-10-03

This is a clean recovery of an experimental implementation, not a benchmark
promotion or release. It is stacked on deductive-search PR #50 at
`037c20b11465f0031f66b157757f01badcfb0d64`. The compact default, indexed and
deductive solver implementations, models and sealed inputs remain unchanged.

## What survived

The earlier focused-search push did not complete, and its temporary checkout was
subsequently unavailable. Source and tests were reconstructed from the recorded
edits. The recovered `spectra/cnf/focused.py` provides a dense residual-clause pool,
six move policies and bounded optional restarts. Only the break/age policy is the
focused backend's default. See [usage and semantics](../../docs/FOCUSED_GUIDE.md).

The recovery bundle had SHA-256
`a82e00bf7929c739231f16f8f5cc78637e6e48fc22eea1b0af55de4d551a8832`.
Its two new source/test files were copied into a fresh checkout. The test name
was then corrected from "frozen" to "selected": no successful published freeze
existed for this candidate. API/CLI integration was reapplied against the actual
base. New usage documentation and an installed-wheel probe complete this recovery.
The original complete experiment harness, protocol files and raw run directory
have not been recovered. This tree is not claimed to match the lost prepared tree.

## Evidence that must not be promoted

The [historical closeout](HISTORICAL_CLOSEOUT.md) is preserved as written before
the new push authorization. Its stopped/unverified statements describe that
earlier checkpoint. Its development summaries are conversation transcriptions,
not raw observations or current verified measurements. They preserve negative
policy attempts, the incomplete first diagnostic and split exposure information.

Consumed development generation seeds were `202610024000` through `202610024031`:
planted/uniform 3-SAT, density 4.2, sizes 128/512, eight formulas per cell; search
seeds 17/73, 2,048 flips and two rounds. Do not reuse them as fresh confirmation.
The later 96-formula evaluation starting at seed `202610025000` was not run.
No new benchmark or model experiment was run during this recovery. Original
benchmark timing, memory and complete raw observations remain missing. Correctness
tests below cannot establish solve-quality, speed or memory gains.

Any future evaluation needs its own published source/generator freeze before
generating fresh declared inputs, independent original-clause checks, retained
failures and raw rows, classical baselines, complete solve time and separate
Python-allocation/process-memory measurements. No default promotion is justified
by this recovery.

## Recovery failure retained

The first `git apply --check` against the real PR #50 checkout failed at
`spectra/cnf/__init__.py:5`. The recovery bundle's staged base snapshot had an extra
trailing newline, so its patch context did not match the repository bytes. This
was a dry-run check; no partial application occurred. The small integration
changes were reapplied directly and the solver source copied unchanged. This is
a recovery-context failure, not a failed solve or missing benchmark observation.

Current checks, exclusions and source identities are recorded in the
[maintenance checkpoint](../../maintenance/reviews/focused-recovery-20261003.md).
