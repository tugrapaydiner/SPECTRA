# Focused-search recovery checkpoint

Isolated branch `research/focused-search-recovery-20261003`, based on PR #50 at
`037c20b11465f0031f66b157757f01badcfb0d64`. This checkpoint saves the recovered
classical candidate after the user requested a clean push. It is not a release,
default promotion or continuation of the stopped benchmark campaign.

## Changes and limits

Added optional `solve_focused` / `--backend focused`: dense unsatisfied-clause
sampling, a break/age move policy, retained alternative policies and bounded
restarts. Source recovery, missing raw development observations, unrun evaluation,
consumed inputs and the initial patch-context failure are disclosed in the
[recovery record](../../experiments/focused_search/RECOVERY.md). The historical
closeout is preserved separately. No quality/time/memory advantage is claimed.

The implementation is accompanied by API/CLI documentation, 20 new public tests
and an installed-only probe exercised by public-package CI. CI is enabled for
this PR's deductive-search base branch. Earlier solver sources, models, raw
evidence, tolerances and default selection are unchanged.

## Local verification

Python 3.12.14 / pytest 9.0.2, fresh full checkout and isolated environments:

```bash
.venv/bin/python -m pytest tests/public --confcutdir=tests/public --junitxml=.review/recovery/public.xml
.venv/bin/python -m build --no-isolation --outdir .review/recovery/dist
.venv/bin/python scripts/check_current_installation.py --dist .review/recovery/dist --out .review/recovery/install
.review/recovery/install/venv/bin/python -I scripts/check_focused_installation.py
.venv/bin/python scripts/audit_workspace.py --out .review/recovery/workspace.json
.venv/bin/python scripts/check_release_readiness.py --out .review/recovery/navigation.json
.venv/bin/python -I -S experiments/deductive_search/check_evidence.py
```

- 370 public tests passed, including the 20 recovered focused tests (not counted
  twice). The new tests exercise cache consistency, all six policies, original
  clause witnesses, budget/restart bounds and deterministic records.
- The actual sdist-built wheel passed all 11 existing installed-only checks;
  148 packaged source files byte-match. The new installed-only probe passed
  512 exhaustive two-variable formula checks and two CLI solve/check commands
  outside the checkout, with no Torch, NumPy, pytest or PySAT installed.
- The workspace audit preserved all 449 historical files and the recovered
  helper byte-for-byte. Release/navigation validation passed.
- Existing deductive evidence verified: all 1,440 evaluation observations,
  80 formulas and 192 unchanged general-3-SAT trajectory pairs. Its known
  incomplete first development run remains reported, not repaired or concealed.
- The build retains existing setuptools license-metadata deprecation warnings.
  No full neural/slow/native-model suite was run: those implementations and
  parameters were not changed. No new benchmark, memory measurement or fresh
  focused-search evaluation was run.

[Validation manifest](../../experiments/focused_search/VALIDATION.json) binds the
tested sources, wheel/sdist identities and retained logs. The PR records the final
commit/tree and its exact-head CI status; local checks do not imply remote success.

## Handoff

The earlier recurring continuation remains paused. This is a recoverable save
point for review. Keep PRs #47–#49 separate and PR #50 as this branch's dependency.
Do not treat lost-run summary transcriptions as verified performance evidence.
