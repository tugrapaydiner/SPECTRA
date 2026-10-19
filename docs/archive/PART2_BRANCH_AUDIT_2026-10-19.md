# Part 2 branch audit and recovery map

Date: 2026-10-19

This document records the repository state immediately before consolidating Part 2 (automatic exact-engine selection). The commit containing this file is an octopus archive commit: every branch head listed below is a direct parent, so no experimental or research history is lost when redundant branch refs are removed.

`main` is intentionally unchanged by the cleanup. The only active Part 2 branch after consolidation is intended to be `performance/automatic-engine-selection-final`, which remains the head of draft PR #71 until the full validation matrix passes.

## Preserved heads

| Branch | Archived head | Classification |
|---|---|---|
| `main` | `ac71b3a7777f4869424247df6481ef32f9996a6b` | Released Part 1 baseline; do not modify during cleanup |
| `archive/research-history-20261016` | `9dcccbf24a3536b76c48f9c9c689529bfc56bb4d` | Earlier research archive |
| `audit/tile64-external-competition-20261017` | `409055333f0da476563bc3e4c8ee556a2b5ec326` | Part 1 external competition audit |
| `performance/automatic-engine-selection` | `128b70b696ff05df8e08e8128ee1cfee4dd3bb2f` | Large exploratory Part 2 history |
| `performance/automatic-engine-selection-confirmation-final` | `3d87b4995618614691355795110175709f6c9a7b` | Superseded confirmation attempt |
| `performance/automatic-engine-selection-final` | `0e0fff858b78673b753b5159ee5860bfd25db520` | Existing clean draft-PR head (v7 before consolidation) |
| `performance/automatic-engine-selection-v7` | `9ff01b55af252f515d7a563a349ddf23fe6ad47e` | Superseded v7 experiment |
| `performance/automatic-engine-selection-v8` | `46bde282eecc24c3ad62ad23a26b060af4e7f2dd` | Superseded v8 experiment |
| `performance/automatic-engine-selection-v10` | `ac578fb6c9a3ef180143f7eb5f52714654e1e869` | Latest exploratory v10 history and validation workflows |
| `performance/smarter-auto-selection` | `6bc0852b9fe4601513333200069baac5cce58638` | Early selector experiment |
| `research/beat-raw-tile64-20261017` | `4f930cc1ccead064d3c76e067d287d88ad142dfd` | Part 1 follow-on research |
| `research/champion-gauntlet-20261017` | `1501a3797a775c1b325cd1d76c9a785ec79cd26f` | Part 1 follow-on research |

## Part 2 audit result

The best implementation code is the v10 source snapshot at commit `79520b8340ed9e16f8eda836b03d5ac1ede409e8`, but the full frozen validation run was not a complete pass. It passed tests, the reference gate, WAP, selector-overhead, and x86-64 confirmation. The original ITC timing job and ARM64 confirmation job failed.

Further inspection showed:

- both true ITC holdouts selected the expected engine and passed their proximity checks;
- the original ITC report incorrectly included a development case in its `holdout_failures` field;
- a later paired/balanced ITC rerun passed on both x86-64 and ARM64;
- ARM64 confirmation selected the same raw engine as the oracle in the only failing cell, but a roughly six-microsecond one-query measurement exceeded the 1.100 ceiling by about 0.00041;
- the public Python docstring still named policy `structural-v9` while v10 receipts named `raw_dynamic_safe_v10`.

Therefore v10 is the correct continuation baseline, but it must remain a draft and must be revalidated with the reporting identity fixed and a paired/batched microbenchmark methodology. This archive does not claim that Part 2 is complete or merge-ready.

## Recovery

To recover any removed branch, create a new ref at the exact archived head shown above. Because every head is also a direct parent of this archive commit, ordinary Git reachability is retained even if a branch-name lookup is no longer available.
