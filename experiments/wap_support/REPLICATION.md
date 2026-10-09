# Reproducing the exposed WAP support-query result

The one-shot run at `37883619073` is the confirmation result. Any later execution uses exposed inputs and is **reproduction**, never a second holdout or a basis for post-hoc tuning.

## Exact source and data

Use source commit `0af6fda36c60a110584e7db836ee5724e530fbd4`. Acquire exactly `wap02a.col`, `wap03a.col`, `wap04a.col`, `wap07a.col`, and `wap08a.col` from `marijnheule/clicolcom@4932048642da2144f387961b595112277afff82f`; the repository inventory binds their byte lengths and Git blob hashes.

The canonical environment uses Linux, Python 3.13, a C++17 compiler, `pytest==9.0.2`, and `python-sat==1.9.dev15`. No NumPy, Torch, SciPy, or scikit-learn dependency is required by the SPECTRA runtime.

## Required audit boundary

A valid reproduction must retain:

- all original graph and generated case identities;
- all 105 scheduled sessions, including failed or slow calls;
- all 107,520 complete outputs;
- original graph/list/query verification for each output;
- fresh setup, every query, full output materialisation, verification, diagnostics, and disposal in complete-session timing;
- fixed forward/reverse/shuffle orders;
- whole-graph clustered analysis;
- source, build, environment, evidence, and delivery manifests.

The frozen workflow is intentionally not rerun. `.github/workflows/wap-support-reproduction.yml` executes the exact frozen source in separately configured exposed-data environments and reports relation identity plus descriptive timing. Such automation is cross-environment reproduction, not independent-team replication.

## Independent replication request

An outside replicator should clone the repository, inspect the protocol and certificate verifier, execute from a clean Linux host, publish the complete artifact manifest and machine description, and state any deviations before seeing performance results. A timing mismatch is scientifically useful; correctness or relation-certificate mismatch is a blocking failure.
