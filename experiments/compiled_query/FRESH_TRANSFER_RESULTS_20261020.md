# Six-family fresh transfer result

**Date:** 2026-10-20  
**Verdict:** `FAIL_ADMISSION`  
**Frozen source:** `9ff8a6130fc76c13171d62345e91532e03a817dc`  
**Workflow run:** `39365661972`

## What happened

The graph-free freeze validation passed before graph acquisition. It checked the
canonical Part 3 ancestry, all five benchmark-relevant source blob identities, the
answer-free generator tests, and the frozen protocol.

The x86-64 and native ARM64 opening jobs then acquired the same pinned upstream
graph tree and stopped on the first frozen candidate:

```text
ValueError: DSJC500.5.col: DSATUR colour count is outside [1, 63]
```

This is an admission failure, not a timing result. No solver arm was timed, no
performance cell was produced, and none of the frozen numerical gates was
reached. The protocol explicitly declared a deterministic DSATUR result above 63
colours to be a failed graph with no replacement after opening, so the study's
correct verdict is failure on both architectures.

## Why the failure is retained

The exact native service currently admits palettes of at most 64 colours. The
frozen construction added one shared trap colour above the deterministic DSATUR
witness palette. `DSJC500.5.col` therefore required a witness using no more than
63 colours. The frozen deterministic heuristic did not satisfy that admission
condition.

The graph may still possess a substantially smaller proper colouring; this run did
not test that question. Changing the colouring heuristic after opening would alter
the frozen workload generator, so it cannot rescue this holdout.

## Exposure boundary

The opening workflow checked out the pinned upstream `graphs/` directory before
case construction. The six named candidates are therefore exposed development
inputs from this point forward:

- `DSJC500.5.col`
- `flat300_20_0.col`
- `le450_25c.col`
- `miles1500.col`
- `queen16_16.col`
- `school1.col`

They may be used to improve and test a new deterministic colouring procedure, but
must not be presented again as unopened or prospective confirmation.

## Evidence

| Evidence | Identity |
|---|---|
| Freeze-validation artifact | `sha256:a4ef372706f2577854253415469b0fff930e74229dd0bd80985348882b5f72a6` |
| x86-64 opening artifact | `sha256:3ac01c47e292e8f57a5ceefc6207ba7462a6015f2a20322ab269c7a22b563b94` |
| ARM64 opening artifact | `sha256:b3f3f43b72c45a17b6a5b205d9690c8d745e40c64237bcd9b31b77b074336d63` |
| Opening workflow source commit | `0736d5f3781a7f2bead63daf3395ad2629d96488` |

Both architecture artifacts retained the source identities, platform receipts,
acquired case directory state, full run log, and nonzero exit code. The one-shot
workflow removed itself after preserving the evidence.

## Next development step

Use the exposed suite to build and challenge a stronger deterministic witness
colourer under the same strict parser and proof-carrying service boundary. The
next prospective study must use a genuinely new, previously unacquired graph
inventory and must be frozen only after the replacement generator passes its
complete graph-free contract suite.

This failed opening does not weaken the earlier prospective direct 2-SAT holdout
or the official CaDiCaL development comparison. It also supplies no new
performance claim.
