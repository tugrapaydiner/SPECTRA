# Documentation

Start with [current status](STATUS.md), [development and installation](DEVELOPMENT.md),
and the [repository README](../README.md). These are the current entry points.

## Current tools

The public API lives in `spectra/`. The CNF CLI accepts strict DIMACS and emits
complete, independently checkable witnesses. The evidence API checks portable
hash/size manifests. Neither API silently loads a neural checkpoint.
The [compact-state protocol](COMPACT_CNF_PROTOCOL.md) is the current bounded
systems experiment; the [older cache protocol](CACHED_RESIDUAL_PROTOCOL.md) stays
unchanged so its evidence still replays.

## Scientific evidence and boundaries

[Research review](RESEARCH_REVIEW_20260911.md) explains the Sudoku, maze and
classical-comparator limitations. [Symmetry audit](SYMMETRY_AUDIT.md),
[fixed-pool replay](FIXED_POOL_REPLAY.md), and
[failure-information study](FAILURE_INFORMATION.md) cover their respective
contracts. The historical [research log](RESEARCH_STATE.md), [claim ledger](CLAIMS.md)
and [architecture description](ARCHITECTURE.md) are retained context, not a claim
that every proposed component has earned promotion.

## Retired work

The [history index](history/README.md) links every retired document and workflow
at a full immutable commit. [File receipts](../maintenance/relocations.json) and
[branch receipts](../maintenance/branches.json) make the cleanup inspectable.
Protocols referenced by executable or retained-evidence checks keep their original
paths and bytes. The project does not rewrite old scientific outcomes to simplify
its presentation.

## Indexed efficiency

[Use, results, invariants and reproduction](EFFICIENCY_GUIDE.md) · [Frozen protocol](INDEXED_SEARCH_PROTOCOL.md).

## Integrated CPU execution

[Runtime API, profiling and reproduction](RUNTIME_GUIDE.md) ·
[Evaluation protocol](INTEGRATED_RUNTIME_PROTOCOL.md). The new kernel is opt-in;
no historical scientific claim is expanded by enabling it.
