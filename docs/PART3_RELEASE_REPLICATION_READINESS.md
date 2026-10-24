# Part 3 — exact-service competition, release and replication readiness

**Started:** 2026-10-19
**Working branch:** `finish-part3`
**Base:** merged and tagged Part 2 commit `527590406c2c624a6aa050e17c6f45b0f000ac5b`
**Review:** draft pull request #76

## Status

Part 3 is **in progress**. It has not been merged, tagged as complete, or
published as a package release. The branch combines release-facing evidence
tooling with the current structural-v11 exact runtime and the retained
competition studies. Passing project-owned CI is necessary engineering
evidence, not outside replication or a broad state-of-the-art claim.

Earlier Part 3 branches were archived and consolidated into this one working
branch. A later cleanup accidentally removed the newest competition files;
that source was restored before the permanent matrix was pointed at it. The
draft PR records the recovery commits and exact current head.

## Release-facing commands

```bash
spectra exact selftest exact-selftest
spectra exact audit exact-selftest
spectra exact compare first-bundle second-bundle
```

`selftest` explicitly builds the bundled native runtime, exercises raw and SCC
automatic paths, SAT, and all three portable `SBP1` contradiction kinds,
independently replays every `SBC1 -> SCQ2` bank, rejects tampering, retains the
exact C++ source, and writes a portable manifest.

`audit` runs in a fresh process without executing retained native code. It
verifies the frozen inventory, build receipt, source and binary hashes,
packets, complete labels, contradiction proofs, selector receipts, and
mutation checks. `compare` audits multiple bundles and permits only declared
environment differences while requiring one shared semantic result.

A successful self-test directory contains ten manifest-bound files plus
`SHA256.json`: the semantic receipt, retained native source, build receipt,
platform library, three request packets, and three certified output banks.
Failed runs leave partial evidence and must be rerun into a new directory.

## Current runtime identity

The one-call public dispatcher is `structural-v11`. Its decision uses only
immutable graph/SCC geometry, bank length, and pin counts. It keeps long dense
uncompressed banks on raw when a quotient would remove no literals, otherwise
uses the frozen raw/SCC rules inherited from Part 2. Edgeless problems may use
a separately replayed direct-domain kernel. Direct forced conflicts can emit a
minimal original-literal proof without replaying a complete closure.

Explicit `raw`, `scc`, `sparse`, and `route` modes remain available as
controls. Public and research native C++ sources must remain byte-identical.

## Retained competition evidence

### Prospective direct 2-SAT holdout

Four graph identities, a deterministic answer-free bank with exactly 512 SAT
and 512 UNSAT rows, six measured arms, schedules, and numerical gates were
frozen before the graph bytes were acquired. SPECTRA `auto` passed on x86-64
and native ARM64 and won all 32 measured architecture/graph/bank cells. At
1,024 requests it was about 2.19× faster on x86-64 and 2.50× faster on ARM64
than the fastest median of the independently implemented closure, scalar-DAG,
and tile-64 challenger engines.

The original structural-v10 holdout also exposed a negative result: explicit
raw was faster than public auto on every 1,024-row graph. That opened evidence
motivated structural-v11, so the later replay is development evidence rather
than a second prospective confirmation.

### Structural-v11 replay

On the already opened mixed banks, structural-v11 retained complete
proof-carrying output and won all 32 cells. Public auto was about 4.45× faster
on x86-64 and 4.33× faster on ARM64 than the fastest project-authored direct
engine at 1,024 requests. The replay validates the implementation opportunity;
it does not reset the holdout boundary.

### Official CaDiCaL 3.0.1 comparison

The official unmodified CaDiCaL 3.0.1 core was pinned by commit and source
manifest. Under the disclosed matched first-use proof-carrying contract,
SPECTRA won all 32 measured cells. Geometric-mean speedups ranged from about
9.26× at one request to 25.82× at 1,024 requests depending on architecture.

CaDiCaL is a general SAT solver and SPECTRA is specialized for fixed
one/two-colour graph domains. The adapter, proof translation, and benchmark
harness are project-authored, and the workload was already open. This result
therefore does not establish general SAT leadership or outside implementation
superiority.

## Fresh-transfer result: retained failure

The six-family transfer protocol was frozen before opening a new graph set.
Graph-free validation passed, but the first candidate, `DSJC500.5.col`,
exceeded the predeclared 63-colour deterministic-DSATUR admission limit. The
correct verdict is `FAIL_ADMISSION` on both architectures:

- no solver arm was timed;
- no performance cell was produced;
- no graph was replaced;
- no threshold was relaxed; and
- all six acquired candidates are now exposed development inputs.

A stronger deterministic witness-colouring procedure may be developed on that
exposed suite, but a future prospective transfer must use a genuinely new
inventory frozen only after the replacement generator passes its complete
graph-free tests.

## General-CNF evidence boundary

The standalone native CDCL engine and its optional proof-producing preprocessing
remain separate from the installed binary-list service. Their original exposed
campaign is still disqualified for ten invalid models. Repairs and later local
panels do not establish complete-field leadership or close the scientific gates
in [the roadmap](SPECTRA_100_100_ROADMAP.md).

The frozen packing-v2 study is blocked by unavailable original prerequisite
archives and pinned binaries. Its
[recovery record](../experiments/general_competition/reproduction_recovery_20261022.json)
retains expected identities and failed preflights. Reconstructed public summaries
or freshly built tools cannot substitute for those original bytes. Any new
campaign requires a separate declaration and must preserve the failed baseline.

## Merge and release gates

Engineering readiness requires one exact final source identity to pass all of
the following. These checks do not close the separate general-CNF scientific
gates or authorize merging the draft PR:

1. active-tree, release, packaging, and workspace audits;
2. public structural-v11, edgeless, packet, proof, self-test, audit, compare,
   exam-repair, and ITC contracts;
3. isolated installed-wheel execution on Linux Python 3.11/3.13, native ARM64,
   and Windows/MSVC;
4. fresh-process no-native auditing and cross-platform bundle comparison;
5. UBSan and ASan+UBSan for the exact runtime and retained challenger adapters;
6. direct 2-SAT and official CaDiCaL contracts on x86-64 and ARM64;
7. byte equality between public and research native sources; and
8. preservation of every failed admission, negative cell, and claim boundary.

A package release is a separate decision. The published version remains
`0.7.1` until an intentional release commit updates both version fields and
final wheel/sdist assets are downloaded and verified.

## Gates that project-owned CI cannot award

Part 3 remains scientifically incomplete without attributable external work:

- outside-team reproduction with the complete raw bundle;
- an independently written adapter or challenger;
- operator-authored restrictions and acceptance metrics;
- maintained use outside this repository;
- expert closest-work review; and
- production security, reliability, and deployment evidence.

These open gates are not reasons to weaken the engineering contract, and they
cannot be converted into completion by adding more project-owned test runs.
