# Changelog

## Unreleased (0.8.0.dev0)

### Breaking: focused package and repository cleanup

- The wheel now installs only the `spectra` package. Earlier wheels also
  installed generic top-level packages named `common`, `data`, `deploy`, `eval`,
  `model` and `train`, which could shadow or be shadowed by unrelated packages.
- The CNF core types now live inside the package: `data.cnf` is
  `spectra.cnf.formula`, `eval.cnf_cached` is `spectra.cnf.cached`, and
  `eval.cnf_repair` is `spectra.cnf.repair`. `from spectra.cnf import CNF, ...`
  is unchanged.
- Remove the retired neural, low-bit, packed-runtime, prepared-FP32 and
  compiler-control tracks: `spectra.inference`, `spectra.blocked_runtime`,
  `spectra.fp_runtime`, `spectra.compiler_runtime`, `spectra.fp_evidence`,
  `spectra.compiler_evidence`, their native sources, the PyTorch training,
  evaluation and deployment packages, their scripts, configs, requirements files,
  guides and tests, and the torch-only `setup.py`. None carried a confirmed
  result. The last commit containing them is
  [df63249](https://github.com/tugrapaydiner/SPECTRA/tree/df63249617add008b8ead1538cf52dac40b81615).
- Drop the `research` and `native` extras; add `svm` (NumPy/scikit-learn
  exporters) and give `dev` the linter.
- Plain `pytest` now collects `tests/` and `experiments/` and runs from a clean
  `pip install ".[test]"`. Tests that need optional external tools (`dd.cudd`, a
  pinned CaDiCaL checkout) skip with a reason instead of failing.
- Remove the unreferenced README SVG charts and their provenance files, and
  `.gitattributes`, which only protected those bytes. Tighten the active-tree
  file cap from 1000 to 400.
- Merge the chained installed-wheel checkers into
  `scripts/check_current_installation.py` and one installed CNF probe,
  `scripts/check_cnf_installation.py`; remove two unreferenced SVM benchmarks.
- Add a ruff lint gate (pyflakes rules) and fix every finding: unused imports and
  variables, duplicate imports and one closure over a deleted name in a test.
- Carry a `.devN` version on unreleased source so a source build is never reported
  as the published 0.7.1.

### Release and replication readiness (Part 3, in progress)

- Add `spectra exact selftest`, a deterministic release-facing command that
  explicitly builds the bundled exact runtime, exercises both automatic selector
  paths, SAT, and all three `SBP1`
  contradiction kinds, independently replays every `SBC1 -> SCQ2` answer, rejects
  a mutated output, retains the exact C++ source used for compilation, and writes
  a portable hash/size manifest.
- Add `spectra exact audit`, which requires a fresh process, refuses an already
  loaded native module, verifies the exact ten-artifact inventory and build
  receipt, recomputes retained source/binary identities, reconstructs the frozen
  request banks, replays every proof/witness, and repeats tamper rejection without
  native execution.
- Add `spectra exact compare`, which audits two or more complete bundles and allows
  platform, Python, compiler and native-library differences only when the source,
  packets, certified outputs, proofs and selector receipts remain identical.
- Retain complete installed-wheel bundles from Linux 3.11, Linux 3.13, ARM64 and
  Windows, then compare them in one fresh no-native CI job and publish the semantic
  comparison receipt as a separate artifact.
- Advance the one-call dispatcher to `structural-v11`, retain dense long
  uncompressed banks on raw, and add independently replayed edgeless and direct
  forced-conflict paths without changing the proof-carrying answer contract.
- Restore the prospective mixed direct-2-SAT holdout, the official CaDiCaL 3.0.1
  matched comparison, and their complete claim boundaries after a consolidation
  cleanup accidentally removed those active source files.
- Retain the six-family fresh-transfer opening as `FAIL_ADMISSION`: no solver was
  timed after the first graph exceeded the frozen 63-colour witness limit, no graph
  was replaced, and the exposed candidates cannot be reused prospectively.
- Keep performance, scientific replication, operator adoption and broad
  solver-superiority claims explicitly outside these deterministic release checks.
- Keep the published package at v0.7.1 until a separate reviewed release commit
  updates both version fields and final assets are downloaded and verified.

### Versioned proof-carrying exact-query API

- Add dependency-light `spectra.exact` API version 1 with native ABI 3.
- Package the exact native source in compiler-free wheels; compilation remains an
  explicit local action and publishes a source/Python/compiler/binary receipt.
- Return complete checked SAT colourings or portable `SBP1` UNSAT proofs through
  persistent sessions and one-call `SBC1` to `SCQ2` execution.
- Add strict independent Python replay, `spectra exact verify`, installed-wheel
  native execution, receipt-tamper tests and cross-platform CI contracts.
- Do not publish a new package version or transfer exposed synthetic performance
  evidence into the frozen WAP claim.

### Certified repeated-query compilation

- Integrate SPECTRA's prospectively frozen WAP support-query result into `main`.
- Preserve the exact Python 3.13 headline: mean complete-session ratio `0.363503`
  against persistent native MiniCard, exact whole-graph 95% interval
  `[0.346789, 0.372944]`, p95 ratio `0.364735`, five of five graph wins, and
  `107,520` independently checked full answers.
- Publish the freeze/open identities, full claim boundary, gate ledger, paper draft,
  prior-art audit, cross-environment reproduction, exact certificate reproduction,
  and external replication procedure.
- Retain both Python 3.11 complete-call gate failures. The threshold is not relaxed
  and warm-session timing is not substituted for the frozen endpoint.
- Keep primitive-algorithm novelty, general SAT/colouring superiority, deployed
  network traces, independent-team replication, and external adoption explicitly
  unclaimed.

### Active-tree slimdown

- Consolidate routine CI to seven permanent workflows and remove branch-specific,
  one-shot, acquisition, publication, and superseded benchmark automation.
- Remove nested historical source/evidence archives and six generated PNGs from the
  active checkout while preserving their exact bytes on
  `archive/research-history-20261016` and original commits.
- Add a fail-closed active-tree budget (14 MiB, 1,000 files, 2 MiB per file) and
  archive/workflow guards to prevent repository bloat from returning.
- Make the structural `auto` selector the default for one-call proof-carrying exact
  sessions; explicit raw/SCC/sparse/route modes remain available.
- Replace full-history PR bundling and routine historical evidence replay with
  current source, package, native, and semantic contracts.

### Merge cleanup

- Remove transport-only Base64 patch chunks and their one-shot apply workflows from
  the working tree. Their immutable history remains available in the research
  commits.
- Retire preflight, holdout-opening, development-acquisition, completed diagnostic,
  and failed draft-archive workflows that must not rerun as ordinary `main` CI.
- Add a permanent lightweight WAP regression workflow for source, certificate,
  original-checker, audit, sanitizer, documentation, and packaging contracts.
- Refresh the documentation index and current-status page around the exact result
  that earned promotion.

This merge does not publish a new Python package or replace v0.7.1 release assets.

## 0.7.1 — 2026-09-12

Published as [SPECTRA 0.7.1](https://github.com/tugrapaydiner/SPECTRA/releases/tag/v0.7.1).
Added opt-in indexed/prepared CNF search with preserved seeded paths and an
output-only CPU inference interface with checked final-output equivalence.
Measurements retain small-case regressions, increased cold allocation and the
`UNKNOWN` large-case outcomes. No external SAT superiority or learned-capability
improvement was established. See [the efficiency guide](docs/EFFICIENCY_GUIDE.md).

Earlier packaging and compact-state work is described in
[current status](docs/STATUS.md) and the [history index](docs/history/README.md).
