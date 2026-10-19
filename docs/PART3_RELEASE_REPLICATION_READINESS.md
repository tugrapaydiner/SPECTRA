# Part 3 — release and replication readiness

**Started:** 2026-10-19  
**Canonical branch:** `part3/release-replication-readiness`  
**Base:** merged and tagged Part 2 commit `527590406c2c624a6aa050e17c6f45b0f000ac5b`

## Goal

Part 2 completed the frozen `structural-v10` automatic-selector milestone. Part 3
does not retune that selector. Its controllable goal is to turn the validated
source tree into a release-facing, independently reproducible exact service with
one obvious smoke-test path, portable receipts, clean installed-wheel checks, and
an explicit boundary between engineering verification and scientific replication.

The first command is:

```bash
spectra exact selftest exact-selftest
```

It creates a new directory, explicitly compiles the bundled native runtime, runs
deterministic SAT and UNSAT cases, covers all three portable `SBP1` contradiction
kinds, independently replays the serialized `SBC1 -> SCQ2` results without native
code, rejects a mutated output, and writes a hash/size manifest.

## Evidence layout

A passing run contains:

```text
exact-selftest/
├── selftest.json
├── SHA256.json
├── native/
│   ├── build.json
│   └── _spectra_exact_binary<platform extension>
└── cases/
    ├── mixed-current-query.sbc1
    ├── mixed-current-query.scq2
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

## Frozen acceptance gates for this track

A release candidate containing the exact service is acceptable only when all of
the following pass on one reviewed source identity:

1. public source contracts and active-tree/release/workspace audits;
2. the release-facing exact self-test from a source checkout;
3. the same self-test from an isolated installed wheel with no repository imports;
4. Linux x86-64, native Linux ARM64, Windows/MSVC, UBSan, and ASan+UBSan contracts;
5. exact API version, native ABI, source hash, binary hash, packet/output replay,
   proof-kind coverage, and evidence-manifest verification;
6. package build and clean-install checks for every supported Python version; and
7. no hidden benchmark threshold, post-hoc case deletion, or conversion of a
   resource/build failure into a mathematical result.

The public package version remains `0.7.1` until an intentional release commit
updates both version fields and the final release artifacts are downloaded and
verified. Creating this branch, command, tag, or draft release does not publish a
package.

## Non-goals and remaining 100/100 gates

This track can improve release quality and make outside reproduction much easier,
but project-owned code and CI cannot manufacture the remaining external evidence.
The following stay open until performed by attributable outside parties or real
application owners:

- independent reproduction with raw receipts and complete evidence;
- an operator-defined workload and acceptance metric;
- maintained use or integration outside this repository;
- closest-work review by relevant experts;
- authentic availability/restriction traces or another genuinely prospective
  transfer family; and
- production deployment, security, reliability, or broad SAT/colouring superiority.

Those are scientific and impact gates, not excuses to weaken the exact engineering
contract. Part 3 must report them honestly rather than award itself completion.
