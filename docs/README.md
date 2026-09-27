# Documentation

## Use and maintain SPECTRA

| Task | Guide |
|---|---|
| Install and run the public tools | [Quick start](../README.md#quick-start) |
| Understand API, input formats and return values | [API and CLI](API.md) |
| Run optional framework-free certified SVM inference | [SVM runtime](SVM.md) |
| Export fitted raw-input preprocessing and SVM inference | [Pipeline deployment](SVM_PIPELINE.md) |
| Share prepared SVM data and use other feature dimensions | [Shared SVM](SVM_SHARED.md) |
| Independently replay a model/input-bound SVM decision | [Decision receipts](SVM_RECEIPTS.md) |
| Check what is released, measured or experimental | [Current status](STATUS.md) |
| Run tests, build a wheel or reproduce experiments | [Development](DEVELOPMENT.md) |
| Prepare a new release | [Release checklist](RELEASING.md) |
| Submit a change | [Contributing](../CONTRIBUTING.md) |
| Handle external inputs and trusted checkpoints | [Security](../SECURITY.md) |
| Review maintenance changes | [Changelog](../CHANGELOG.md) |

## Measured systems work

[Indexed efficiency](EFFICIENCY_GUIDE.md) is the v0.7.1 implementation guide.
Its [frozen protocol](INDEXED_SEARCH_PROTOCOL.md), the
[compact CNF protocol](COMPACT_CNF_PROTOCOL.md) and the earlier
[cached-residual protocol](CACHED_RESIDUAL_PROTOCOL.md) describe separate
experiments. Do not multiply their speedups or replace their original results
with measurements from a different implementation or host.

## Research evidence

[Research review](RESEARCH_REVIEW_20260911.md) covers Sudoku, maze and stronger
classical comparators. [Symmetry audit](SYMMETRY_AUDIT.md),
[fixed-pool replay](FIXED_POOL_REPLAY.md) and
[failure-information study](FAILURE_INFORMATION.md) define their own contracts.
The [research log](RESEARCH_STATE.md), [claim ledger](CLAIMS.md) and
[architecture description](ARCHITECTURE.md) are historical research context,
not release instructions or proof that every proposed component works.

## Preserved history

The [history index](history/README.md) links retired documents and workflows at
immutable commits. [File receipts](../maintenance/relocations.json) and
[branch receipts](../maintenance/branches.json) record the earlier archival work.
Evidence-bound protocols retain their original locations. Raw evidence remains
under `results/`; research module paths remain compatible with saved records.
The installed wheel is not the complete research archive: use a full Git checkout
for historical replay and the workspace-preservation audit.

## Integrated optional execution

[Prepared FP32](TRAINED_FP_GUIDE.md), [packed runtime](RUNTIME_GUIDE.md) and
[compiler baseline](COMPILER_BASELINE_GUIDE.md) are separate opt-in tracks.

[Optional compiled preprocessing](SVM_PREPROCESS_NATIVE.md) retains the raw-pipeline
contract while reducing Python loop and built-in row-copy overhead.
