# Changelog

## 0.8.0 (unreleased)

### Added

- `spectra.exact`: proof-carrying list-colouring queries for one- and two-colour
  domains, with `SBC1` session packets, `SCQ2` answer banks, `SBP1` proofs, a
  pure-Python verifier, a structural `auto` mode, and `spectra exact
  build|verify|selftest|audit|compare`.
- `spectra.cnf.quotient_query`: the SCC quotient runtime for repeated
  list-colouring support queries, with an independent certificate verifier.
- `spectra.sat`: a native CDCL solver with checked models and DRUP proofs over the
  original variables, a proof-replayable unit/binary simplifier, strict DIMACS
  admission and a DRAT-trim adapter. Builds on Linux and Windows.
- `spectra.migration`: checked two-plan migration with unique minimal move sets
  and original-edge contradiction paths.
- `spectra.exam_repair` and `spectra.itc2007_exam`: bounded exam rescheduling with
  named infeasibility explanations.
- SVM execution: generic feature counts and labels, shared prepared models,
  fitted preprocessing plans, native preprocessing, a binary-classifier fast
  path, replayable decision receipts and `spectra svm run`.
- `benchmarks/`: the WAP workload, independent direct 2-SAT and CaDiCaL baselines,
  a SAT campaign runner and verifier, and CNF and SVM fixture benchmarks.

### Changed

- The wheel now installs only the `spectra` package and has no dependencies.
  Earlier wheels also installed top-level `common`, `data`, `deploy`, `eval`,
  `model` and `train` packages.
- CNF core types moved into the package: `data.cnf` is now `spectra.cnf.formula`,
  `eval.cnf_cached` is `spectra.cnf.cached` and `eval.cnf_repair` is
  `spectra.cnf.repair`. `from spectra.cnf import CNF, ...` is unchanged.
- `pip install ".[test]"` followed by `pytest` runs the whole suite; optional
  external tools are skipped with a reason.

### Fixed

- Exact self-test auditing accepts a null SOABI when CPython does not define one
  (including Windows Python 3.11), while rejecting malformed ABI metadata and
  preserving the native loader's interpreter compatibility checks.
- Installed-wheel checks include native CDCL execution on Linux and Windows;
  cross-platform exact self-test comparison includes Windows Python 3.11 and
  retains its comparison report.
- Runnable guide examples preserve the exact runtime for exam repair, create SVM
  receipts with an open worker, and supply the correct streaming CLI engine and
  native preprocessing library path.

### Removed

- The PyTorch models, training, evaluation and deployment code, and the
  `spectra.inference`, `spectra.fp_runtime`, `spectra.blocked_runtime`,
  `spectra.compiler_runtime`, `spectra.fp_evidence` and
  `spectra.compiler_evidence` modules, together with the `research` and `native`
  extras. They remain available in the 0.7.1 release.

## 0.7.1 (2026-09-12)

- Indexed and prepared CNF local search with the same seeded trajectories as the
  reference solver.
- Output-only CPU inference for the recurrent model.
