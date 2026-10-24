# Current status

This page separates the confirmed research result, the unreleased source tree,
and the latest published Python package. Repository cleanup does not broaden a
scientific claim or turn a negative result into a pass.

## Delivery state

The latest published package release remains
[v0.7.1](https://github.com/tugrapaydiner/SPECTRA/releases/tag/v0.7.1), published
on 2026-09-12. The `main` branch contains later, unreleased research and runtime
source. A source checkout is therefore not byte-identical to the v0.7.1 wheel.
No new package, PyPI publication, or replacement release asset is implied by the
WAP research merge.

The current unreleased source additionally contains `spectra.exact` API version 1
(native ABI 3). It packages the proof-carrying exact-query source and verifies an
explicit local build from an installed wheel. This source availability is not a
v0.8 release, and the exact-query development measurements remain separate from
the confirmed WAP endpoint.

## Part 3 draft branch

Draft pull request #76 on `finish-part3` contains the unreleased exact-service
competition and release-readiness work. It is not part of `main` or the published
v0.7.1 package.

The draft currently includes structural-v11, the edgeless direct-domain kernel,
release-facing self-test/audit/compare commands, a separately implemented direct
2-SAT challenger, and an adapter around the official CaDiCaL 3.0.1 core. Retained
evidence includes:

- a prospectively frozen mixed direct-2-SAT holdout that passed on x86-64 and
  ARM64, with public `auto` winning all 32 cells and full-bank gains of about
  2.19× and 2.50× over the fastest of three direct engines;
- an opened structural-v11 replay with about 4.45× and 4.33× full-bank gains over
  the same project-authored challenger; and
- a disclosed official-CaDiCaL comparison with all 32 cells won and geometric
  gains from roughly 9.26× to 25.82× depending on bank size and architecture.

These are narrow repeated binary-list exact-service results. They do not establish
general SAT or graph-colouring leadership, asymptotic novelty, outside-team
replication, operator value, or production readiness.

The newest six-family prospective transfer is a retained `FAIL_ADMISSION`, not a
speed result: `DSJC500.5.col` exceeded the frozen deterministic-DSATUR 63-colour
limit before any solver was timed. All six candidates are now exposed and cannot
be reused as unopened confirmation. Part 3 remains in progress while the stronger
generator, a genuinely new transfer inventory, external replication, and release
gates remain open.

## General-CNF development

Part 3 also contains a standalone native CDCL engine, optional proof-producing
preprocessing, strict DIMACS admission, and independent original-CNF answer
checks. The general-CNF engine is a research tool, separate from the installed
binary-list exact service.

The original exposed campaign remains disqualified for ten invalid SAT models.
Later repairs and checked development runs do not rescore that campaign. The
current implementation has not passed the complete-field leadership, large-gap,
hidden-inventory, outside-replication or expert-review gates. See the
[competition report](https://github.com/tugrapaydiner/SPECTRA/blob/8f7cee1a9c82e7da34cc2704ff7012290a554a91/experiments/general_competition/EXPOSED_CAMPAIGN_20261021.md)
for the retained measurements and failures.

The frozen packing-v2 study cannot resume from the available handoff: its original
prerequisite archives and exact pinned checker/native build are unavailable.
Recovered inputs and reconstructed summaries do not replace those receipts. A
fresh campaign must declare its own source, tools, complete field and limits;
it cannot be counted as completion of the blocked frozen study.

## Flagship research result

SPECTRA's principal result is the certified original-address quotient for repeated
list-colouring support queries. The protocol, candidate, controls, schedules,
budgets, graph identities, analysis, and thresholds were frozen before opening five
WAP-A optical-conflict graphs.

Under the exact published Python 3.13 contract:

- 105 complete persistent sessions were retained;
- 107,520 complete answers were checked against the original graph, lists, and
  current original-address restrictions;
- SPECTRA's mean complete-session cost was 569.290 ms versus 1,566.119 ms for
  persistent native MiniCard;
- the mean cost ratio was 0.363503 with exact whole-graph 95% interval
  [0.346789, 0.372944];
- the p95 ratio was 0.364735 with interval [0.338157, 0.383226]; and
- SPECTRA won on all five prospectively unopened graphs.

The exact result and immutable identities are in
[the holdout report](../experiments/wap_support/HOLDOUT_RESULTS_20261009.md).
The [gate ledger](../experiments/wap_support/FLAGSHIP_GATE_LEDGER.md) records a
**result-level flagship PASS within this narrow contract**. It explicitly does not
establish that SPECTRA is a generally superior reasoning, SAT, or colouring system.

## Reproduction and portability

The complete 2x gate reproduced on two additional Python 3.13 server environments.
Both tested Python 3.11 environments preserved the exact relation and every answer,
but missed the unchanged 0.50 complete-call mean-ratio gate because the canonical
research-evidence front end added a large common cost. Those failures remain part of
the result and the threshold was not relaxed.

See the
[cross-environment report](../experiments/wap_support/CROSS_ENVIRONMENT_REPRODUCTION_20261009.md),
[certificate reproduction](../experiments/wap_support/CERTIFICATE_REPRODUCTION_20261009.md),
and [replication guide](../experiments/wap_support/REPLICATION.md).
Independent outside-team replication remains open in issue #55.

## Claim boundary

Established:

- exact restrictions at arbitrary original vertices after quotient compilation;
- complete witness lifting and original-input checking for every successful query;
- independently checkable whole-relation certificates;
- a large prospectively frozen complete-cost advantage against persistent native
  MiniCard on the stated WAP support-query workload; and
- cross-environment semantic reproduction plus Python 3.13 performance reproduction.

Not established:

- general SAT or graph-colouring superiority;
- ordinary chromatic-number improvement;
- deployed wavelength-availability or traffic traces;
- invention of SCC substitution, incremental SAT, or knowledge compilation;
- a new asymptotic algorithm;
- Python-version-independent 2x complete-call performance;
- independent-team replication, external adoption, or production impact; or
- validation of historical neural, MCTS, low-bit, or general-reasoner claims.

The precise wording boundary and closest-work review are retained in
[NOVELTY_AND_CLAIM_BOUNDARY.md](../experiments/wap_support/NOVELTY_AND_CLAIM_BOUNDARY.md)
and [PRIOR_ART_AUDIT_20261009.md](../experiments/wap_support/PRIOR_ART_AUDIT_20261009.md).

## Other retained tracks

Backward-compatible runtime modules remain in the package. Bulky historical
evidence and alternate research branch trees are consolidated under
`archive/research-history-20261016` rather than duplicated on `main`. Their failed
gates and narrow claims remain unchanged; none inherits the WAP result.

Use [Development](DEVELOPMENT.md) for current commands, [Releasing](RELEASING.md)
for package publication, and the [history index](history/README.md) for superseded
research records.

## Active-tree boundary

The normal checkout is capped by `scripts/check_active_tree.py`. It contains eight
permanent workflows (seven automatic and one manual ITC suite). It excludes nested
evidence archives, generated historical PNGs, transport directories, and stale
branch-trigger automation. Historical run dumps, internal plans and one-off
diagnostics are outside the active tree. Public result summaries link to their immutable evidence trees;
regression tests exercise maintained behavior using small, focused fixtures.
