# Research engineering checkpoint — 2026-10-05

Draft [PR #52](https://github.com/tugrapaydiner/SPECTRA/pull/52), branch
`research/research-engineering-20261005`, based on #51 at
`54b91553e5d3be356db7011ff622fb16e495256f`. This is a bounded research experiment
and technical case study. It does not merge the dependency stack or publish a
release. The earlier recurring automation remains paused.

## Concrete contribution

Close the recovered focused solver's fresh-evidence gap: freeze and publish the
candidate, generator, analysis and protocol; compare with indexed, dense-policy
controls and native Glucose4; retain all data; replay local trajectories; explain
the design and measured limits in a reviewer guide. Runtime solver code is exactly
the existing #51 implementation. Tests and documentation are not intelligence gains.

Freeze commit `fceff2360fc7a2d9c7c2c6a44e0a750d878eae17`, tree
`215b5148d52be0832eaf99b2a8ea8cdc1c3a2591`, was published before first evaluation
generation. All 17 frozen source files remain unchanged. This experiment consumes
generator seeds `202610055000`–`202610055095`; do not relabel reruns as fresh evidence.

On this single-host synthetic panel, the selected policy verifies 102/192 pairs
versus 60/192 indexed, with complete mean wall time 18.798 versus 26.705 ms
(29.6% lower). All declared gate conditions pass. Memory remains approximately
21.4 MiB including the interpreter/import baseline on the 12-formula subset.
Four individual indexed successes are lost. Most gains are on planted formulas;
large uniform inputs remain mostly unresolved. Native budgets overshoot and are
unequal to flips. The [complete report](../../experiments/focused_evaluation/RESULTS.md)
retains every control and limitation, not only the favorable pooled number.

## Verification commands and results

```bash
.venv/bin/python -m pytest tests/public --confcutdir=tests/public --junitxml=.review/research-engineering/public.xml
.venv/bin/python -I -S experiments/focused_evaluation/analyse.py experiments/focused_evaluation/evidence-20261005.zip
.venv/bin/python -I -S experiments/focused_evaluation/replay.py experiments/focused_evaluation/evidence-20261005.zip
.venv/bin/python -m build --no-isolation --outdir .review/research-engineering/dist-final
.venv/bin/python scripts/check_current_installation.py --dist .review/research-engineering/dist-final --out .review/research-engineering/install
.review/research-engineering/install/venv/bin/python -I scripts/check_focused_installation.py
.venv/bin/python scripts/audit_workspace.py --out .review/research-engineering/workspace.json
.venv/bin/python scripts/check_release_readiness.py --out .review/research-engineering/navigation.json
.venv/bin/python -I -S experiments/deductive_search/check_evidence.py
```

The public suite passed **382 tests** (12 new study contracts included). Retained
verification accepted **96 formulas, 2,880 timing rows and 36 memory rows**;
all **768 local-search paths** replayed exactly, excluding timing. The 449-file
historical identity audit passed and previous deductive evidence remains valid,
including its explicitly incomplete first development run. Actual installed-wheel
and final-head CI results are recorded in the PR and validation receipt.

Local distribution verification passed: 148 shipped source files byte-match,
all 11 installed-only commands pass, and the installed focused probe passes 512
small-formula checks plus two CLI commands without optional dependencies. The
[validation manifest](../../experiments/focused_evaluation/VALIDATION.json) binds
the logs and final wheel/sdist hashes. The [figure receipt](../../experiments/focused_evaluation/FIGURE.json)
binds the summary, renderer and inspected SVG. Neither receipt promotes the
earlier lost development measurements.

No old model/native implementation, tolerance, raw observation or sealed input
was modified. Full neural/slow/native-model regression was not rerun. Native
Glucose4 was executed as an experimental comparator, not substituted for those
excluded model checks. Hashes do not establish independent scientific validity.

## Retained operational limitations

The old temporary virtual environment was gone; an initial command failed before
any study ran. A new venv was created. A figure preview initially requested absent
cairosvg; the existing matplotlib renderer produced the inspected PNG instead.
An initial distribution build preceded final guide wording; the final distribution
uses a fresh output directory. Existing setuptools metadata warnings remain visible.
Git's whitespace check initially rejected trailing spaces emitted by matplotlib
in SVG path data; the renderer now strips line-ending whitespace without changing
the figure geometry or numeric inputs. The corrected staged diff passes.
No measured row was dropped, rerun or overwritten. The older focused raw development
records are still missing; the new evaluation does not retroactively repair them.

For a future research pass, choose a new task/source distribution and freeze the
candidate before evaluation. The main remaining portfolio gaps are external task
transfer, demonstrated model-training gains and production-scale evidence.
