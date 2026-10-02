# Capability checkpoint: optional deductive CNF search

Draft [PR #50](https://github.com/tugrapaydiner/SPECTRA/pull/50) on
`research/deductive-search-20261002`, based on PR #46 at
`58ae7cd79603a4ab32c4c08392a6ffa9a87a2976`. This is implementation plus a bounded
capability experiment, not a release, merge or default-promotion request.

Candidate/generator tested identity, frozen and published before evaluation:
commit `9fd2265bac0c2aa18fd711b47dc75dbc75a6248f`, tree
`b01b2791f9605d7c906e7ab5fce2559c87fe19e6`. The later evidence/docs commit does
not modify those sources. See the [results](../../experiments/deductive_search/RESULTS.md)
and [freeze](../../experiments/deductive_search/FROZEN.json).

## Outcome

Add dependency-free unit propagation and binary implication solving before
bounded indexed search. Fresh synthetic evaluation: 124/160 verified SAT cases
versus 97/160 indexed; mean complete wall time 8.588 versus 12.520 ms; p95 25.483
versus 24.908 ms. Corrected worker RSS averages 21.051 versus 20.943 MiB, including
common interpreter/native imports. Structured cells supply every extra success;
general-3-SAT solve outcomes are identical and overhead reaches 5.1%. Glucose4 is
faster on structured cells and remains in the report with unequal-budget caveats.

## Checks and exclusions

Commands run with Python 3.12.14, pytest 9.0.2, in an isolated checkout:

```bash
python -m pytest tests/public/test_deductive_search.py --confcutdir=tests/public
python -m pytest tests/public --confcutdir=tests/public --junitxml=.review/capability/public.xml
python -m build --no-isolation --outdir .review/capability/dist
python scripts/check_current_installation.py --dist .review/capability/dist --out .review/capability/install
python scripts/audit_workspace.py --out .review/capability/workspace.json
python scripts/check_release_readiness.py --out .review/capability/navigation.json
python -I -S experiments/deductive_search/check_evidence.py
```

- 350 public tests passed (16 new tests included, not counted twice).
- Actual sdist -> wheel -> separate environment: 11 installed-only checks passed,
  147 packaged source files byte-match. Additional installed exhaustive binary
  probe: 512 formulas passed, no optional dependencies; deductive CLI solve/check
  passed outside the checkout. Documentation packaging and the installed probe
  also passed in fresh `dist-final` / `install-final` directories. Final wheel
  SHA-256: `7c9e3a606fedfa6ae4a3ed0fca4cdb0e973a1c0f3fe0435d2b365b0b2a4780af`.
  [Validation receipts](../../experiments/deductive_search/VALIDATION.json) bind
  the retained logs, source inventory and distribution identities.
- Historical workspace audit: 449 original files and recovered helper preserved.
- Frozen-head GitHub public-contract checks passed on Python 3.11/3.13:
  [run 37060326843](https://github.com/tugrapaydiner/SPECTRA/actions/runs/37060326843).
  This includes the installed distribution and existing compact replay; it does
  not claim results for an untested later head. Final-head status belongs to PR #50.
- Retained evaluation: all 1,440 observations / 80 formulas, two 240-row memory
  records and 192 identical general-3-SAT pairs checked. All reported SAT
  witnesses independently satisfy original clauses. Native UNSAT reports have no
  proof check. Initial development raw evidence is explicitly incomplete (25
  missing observations); original summaries and remaining bytes are retained.
- Original RSS measurements are preserved but rejected because of inherited
  supervisor high-water marks; corrected memory-only measurements are separate.
- Initial network installation and relative installed-probe path errors were
  corrected and disclosed in RESULTS.md. They are not omitted benchmark failures.
- No full neural/slow/native-model suite or retained model replay was rerun:
  no trained parameters, historical solver, model kernels or tolerances changed.
  No accuracy, novelty or general-intelligence claim is supported.

## Resume here

Keep historical maintenance PRs #47–#49 separate. Prioritize actual capability,
not more auditor-only work. This experiment is consumed: freeze a new candidate
before evaluating a newly declared seed range. A useful next question is why
general 3-SAT local walks stall, and whether a bounded restart/noise/search policy
can improve verified successes per full cost against indexed and native controls.
Use new research modules, preserve this optional backend and all failed attempts,
and do not reopen sealed historical confirmation sets. There is no merge/release
authorization. Stop before changes if the user says look, stop or pause.
