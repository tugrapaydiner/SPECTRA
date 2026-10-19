# Part 3 — release and replication readiness

**Started:** 2026-10-19  
**Branch:** `part3/release-replication-readiness`  
**Base:** merged and tagged Part 2 commit `527590406c2c624a6aa050e17c6f45b0f000ac5b`

## Goal

Part 2 completed the frozen `structural-v10` automatic-selector milestone. Part 3
does not retune that selector. Its controllable goal is to turn the validated
source tree into a release-facing, independently auditable exact service with one
obvious smoke-test path, portable receipts, clean installed-wheel checks, and an
explicit boundary between engineering verification and scientific replication.

The three public commands are:

```bash
spectra exact selftest exact-selftest
spectra exact audit exact-selftest
spectra exact compare linux311 linux313 arm64 windows
```

`selftest` creates a new directory, explicitly compiles the bundled native runtime,
runs deterministic SAT and UNSAT cases through both the raw and SCC automatic
selector paths, covers all three portable `SBP1` contradiction kinds,
independently replays the serialized `SBC1 -> SCQ2` results, rejects a mutated
output, and writes a hash/size manifest.

`audit` must run in a fresh process. It refuses an already-loaded native module,
executes no native code, verifies the exact frozen inventory and build receipt,
requires the reported Python version, executable and platform to agree with that
receipt, reconstructs the request packets, replays every witness and proof, and
repeats the tamper checks.

`compare` audits two or more complete bundles and permits only environment-level
differences: platform, Python build, compiler and native-library bytes. The exact
source, deterministic packets, certified outputs, proof coverage, API/ABI and
selector receipts must be identical. It emits one semantic fingerprint for the
shared result.

## Evidence layout

A passing self-test contains:

```text
exact-selftest/
├── selftest.json
├── SHA256.json
├── native/
│   ├── exact_binary.cpp
│   ├── build.json
│   └── _spectra_exact_binary<platform extension>
└── cases/
    ├── mixed-current-query.sbc1
    ├── mixed-current-query.scq2
    ├── strong-compression.sbc1
    ├── strong-compression.scq2
    ├── base-static-contradiction.sbc1
    └── base-static-contradiction.scq2
```

`selftest.json` is the semantic receipt. `SHA256.json` uses
`spectra.evidence_freeze.v1` and binds every other retained file. The manifest
establishes byte identity only; it does not establish data independence,
scientific novelty, performance, operator value, or production safety.

The destination must not already exist. A failed run leaves its partial directory
for diagnosis. Do not rerun into that directory or fill in missing artifacts by
hand; use a fresh destination and retain the failed evidence when it matters.

## Reference cross-platform result

The `v2` bundle has been checked on:

- Linux x86-64, Python 3.11;
- Linux x86-64, Python 3.13;
- Linux ARM64, Python 3.13; and
- Windows x86-64, Python 3.13 with MSVC.

The exact final source commit and workflow run are recorded in pull request #73,
outside this source file. Embedding a commit identifier inside the commit itself
would make the identity circular and immediately stale when the document changed.

The four native libraries had four different SHA-256 hashes, as expected for their
different platforms and Python ABIs. All four bundles nevertheless agreed on:

- exact source SHA-256
  `39016dd2256a08e5d2f1c9bbd48bcd2ab61c38aaa5e9773a882154bd273f88e7`;
- three deterministic cases and thirteen decisions;
- ten SAT and three UNSAT results;
- proof kinds `1`, `2`, and `3`;
- `raw` selection for the base-infeasible and short weak-compression cases;
- `scc` selection for the strong-compression case;
- all three packet hashes and all three certified-output hashes; and
- semantic fingerprint
  `99df9e1a42bc396fe451583c42f90dadaf9a47ff838b4508d53b2a59cd304700`.

The comparison process compiled, loaded and executed no retained native library.
It audited the four complete bundles from their manifests, receipts, request
packets and certified outputs. The full bundles and compact comparison receipt are
retained as separate 90-day workflow artifacts.

The earlier two-case `v1` result and its different fingerprint remain in workflow
history. They are not presented as the current result because `v2` adds the SCC
selector path and changes the frozen semantic inventory.

## Frozen acceptance gates for this track

A release candidate containing the exact service is acceptable only when all of
the following pass on one reviewed source identity:

1. public source contracts and active-tree/release/workspace audits;
2. the release-facing exact self-test from a source checkout;
3. the same self-test from an isolated installed wheel with no repository imports;
4. a fresh-process no-native audit of every retained bundle;
5. a cross-platform comparison covering Linux 3.11, Linux 3.13, ARM64 and Windows;
6. Linux x86-64, native Linux ARM64, Windows/MSVC, UBSan, and ASan+UBSan contracts;
7. exact API version, native ABI, source hash, binary hash, packet/output replay,
   proof-kind coverage, evidence-manifest verification and semantic fingerprint;
8. package build and clean-install checks for every supported Python version; and
9. no hidden benchmark threshold, post-hoc case deletion, or conversion of a
   resource/build failure into a mathematical result.

The public package version remains `0.7.1` until an intentional release commit
updates both version fields and the final release assets are downloaded and
verified. Creating this branch, command, tag, comparison receipt or draft release
does not publish a package.

## Non-goals and remaining 100/100 gates

This track improves release quality and makes outside reproduction substantially
easier, but project-owned code and CI cannot manufacture the remaining external
evidence. The following stay open until performed by attributable outside parties
or real application owners:

- independent reproduction with raw receipts and complete evidence;
- an operator-defined workload and acceptance metric;
- maintained use or integration outside this repository;
- closest-work review by relevant experts;
- authentic availability/restriction traces or another genuinely prospective
  transfer family; and
- production deployment, security, reliability, or broad SAT/colouring superiority.

Those are scientific and impact gates, not excuses to weaken the exact engineering
contract. Part 3 must report them honestly rather than award itself completion.
