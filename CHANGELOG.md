# Changelog

## Unreleased

### Release and replication readiness (Part 3, in progress)

- Add `spectra exact selftest`, a deterministic release-facing command that
  explicitly builds the bundled exact runtime, exercises SAT plus all three `SBP1`
  contradiction kinds, independently replays every `SBC1 -> SCQ2` answer, rejects
  a mutated output, retains the exact C++ source used for compilation, and writes
  a portable hash/size manifest.
- Add `spectra exact audit`, which requires a fresh process, refuses an already
  loaded native module, verifies the exact eight-artifact inventory and build
  receipt, recomputes retained source/binary identities, reconstructs the frozen
  request banks, replays every proof/witness, and repeats tamper rejection without
  native execution.
- Add `spectra exact compare`, which audits two or more complete bundles and allows
  platform, Python, compiler and native-library differences only when the source,
  packets, certified outputs, proofs and selector receipts remain identical.
- Retain complete installed-wheel bundles from Linux 3.11, Linux 3.13, ARM64 and
  Windows, then compare them in one fresh no-native CI job and publish the semantic
  comparison receipt as a separate artifact.
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
- Add a fail-closed active-tree budget (14 MiB, 950 files, 2 MiB per file) and
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
