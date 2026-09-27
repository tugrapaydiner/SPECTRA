# Reproducible retained-corpus application check

This experiment starts from merged PR30, source tree
`55fd02faf38b547305fa75cb10f9c099a8633178`. It does not change numerical code,
models, the frozen data split, or any previous negative result. The aim is to make
the existing deployment benefit accessible through an actual offline file interface
and check it on native Linux x86-64/ARM64 and Windows x64.

## Byte-identical reconstruction, not new model selection

The previous eighteen model bundles were delivered as attachments rather than
repository binaries. `corpus.py` reproduces them from the complete public dataset
copies distributed with pinned scikit-learn1.8.0. It uses the original numeric and
categorical schema, seeds101/202/303,25% stratified partitions and fixed C10 RBF-SVC.
It does perform these small CPU fitting jobs, but aborts unless every model, fitted
plan, transformed feature array, expected-label file and raw JSONL input matches
`corpus_manifest.json` exactly. The manifest was generated from the previously
accepted89ed670 SDK, not selected after observing platform outcomes.

Original data and OpenML metadata digests are checked against `data_manifest.json`.
No downloading, external pickle loading, test-based tuning, alternative split,
refitting after mismatch or relaxed floating-point tolerance is allowed. The
producer records CPU/wall costs. The targets receive only inert files. An environment
unable to reproduce the original bytes fails instead of issuing a new corpus.

```bash
python -m pip install numpy==2.3.5 scipy==1.17.0 pandas==2.2.3 \
  scikit-learn==1.8.0 joblib==1.5.3 threadpoolctl==3.6.0
python experiments/streaming/corpus.py --out /new/corpus
```

## Installed-only target replay

The portability workflow builds through the sdist, tests the full SVM subset,
installs the wheel without numerical frameworks, and builds its native components.
It then executes `scripts/check_svm_stream_corpus.py` under isolated installed Python.

All4,281 model/input pairs are replayed under two-stage/fused paths and three
schedules, yielding25,686 repeated prediction checks. All207,396 transformed
binary64 values are compared with the old corpus. Seventy-two numerical decision
receipts and eighteen owner-close continuations are separate overlapping checks.
A real `python -m spectra svm run` subprocess is checked using stdin. The independent
standard-library output auditor verifies original labels, order, input/output
identities, completion and the absence of trailing records.

These are six small convenience datasets with three overlapping partitions per
dataset, not eighteen independent tasks. The original repeated transformed vectors
and earlier quality limitations remain. No cross-platform bitwise-libm assertion
follows from matched labels; each platform must pass its own numerical replay.

## Input provenance

Wine and WDBC are UCI copies distributed with scikit-learn. WDBC is used only as a
software compatibility fixture, not clinical evidence. The OpenML copies of Chess3,
Titanic40945 and Zoo62 preserve their public metadata's licence field rather than
being relabelled MIT. Penguins42585 is CC0; UCI also lists Zoo under CC BY4.0.
Original metadata and checksums are in `data_manifest.json`.

- https://archive.ics.uci.edu/dataset/109/wine
- https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
- https://www.openml.org/d/3
- https://www.openml.org/d/40945
- https://www.openml.org/d/42585
- https://www.openml.org/d/62

MIT applies to SPECTRA implementation, not automatically to the original data.
The private working directory is not needed: code, versions, source digests and
expected output identities needed to reconstruct this check are on the branch.
