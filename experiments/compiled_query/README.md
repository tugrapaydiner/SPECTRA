# Compiled-query validation

This directory contains research adapters and controls for the proof-carrying
binary-list service. The maintained installation and usage contract is the
[exact API](../../docs/EXACT_API.md); build and test instructions are in
[Development](../../docs/DEVELOPMENT.md).

The current one-call dispatcher uses structural-v11. It selects raw or SCC from
immutable graph geometry, bank length and pin counts; edgeless problems may use
the separately checked direct-domain kernel. Explicit modes remain available.
Native builds support Linux and Windows x64 with MSVC and matching x64 Python.

The [exact contract](EXACT_BINARY_CONTRACT.md) and
[reduction contract](REDUCTION_CONTRACT.md) describe original-address witnesses,
proof replay and cost accounting. The direct 2-SAT and CaDiCaL adapters are
project-authored controls. They do not constitute outside replication.

Historical query-cost experiments, their internal plans and diagnostic harness
are retained in the [original source tree](https://github.com/tugrapaydiner/SPECTRA/tree/38c13990a1ba92229cf5cf2a57b1a600b218889e/experiments/compiled_query).
Reproduce an old result from its complete pinned tree. The 48-graph exposed
comparison missed its 10% speed target against improved raw execution. Its
failed cells and the separate [failed transfer gate](TRANSFER_CI_RESULTS_20261015.json)
remain evidence; cleanup does not change those outcomes.

Current specialized results and limitations are summarized in
[Status](../../docs/STATUS.md). Opened development workloads cannot become a new
holdout, and expiring CI artifacts are not permanent research archives.
