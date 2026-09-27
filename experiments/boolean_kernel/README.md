# Reproduce the exact Boolean-domain comparison

Read [RESULTS.md](RESULTS.md), the [protocol](PROTOCOL.md), the
[deviations](AMENDMENT.md) and the [API contract](../../docs/SVM_BOOLEAN.md).
No models need retraining for this continuation. Use the hash-bound prior native
comparison corpus and its original outer SHA256.json manifest. The delivered
validation archive includes that retained input subset, both complete source
snapshots, controls, strict compiler commands and raw records.

```bash
python -m pytest tests/svm --confcutdir=tests/svm
python -c 'from spectra.svm import build_runtime; print(build_runtime("new-build",target="avx2"))'
python experiments/boolean_kernel/bench.py --models /evidence/prior/models \
  --controls /evidence/controls --library /new-build/libspectra_svm.so \
  --out /new/run --model chess-101
```

Repeat the command for all seven names declared in bench.py, using the same run
directory; no completed repetition file is overwritten. Build controls from the
included unmodified LIBSVM3.37 and exact previous generated-C sources with the
recorded strict O3/AVX2 flags. HAR generated C remains unavailable; its linear
control is a DIFFERENT model with separately bound predictions. The original
source executable is a control, not the mutable current checkout.

```bash
python experiments/boolean_kernel/audit.py --run /new/run \
  --models /evidence/prior/models --source /path/to/measured/source \
  --controls /evidence/controls --library /new-build/libspectra_svm.so \
  --out /new/audit.json
```

The audit requires all77 complete repetition files and seven protocol records.
It rejects missing, extra, reordered, changed-input, changed-binary or mismatched
source records, then independently recomputes the aggregates. The original corpus
manifest identity is pinned, not learned from a candidate output file. It cannot
prove an honest timer or substitute for outside reproduction.

`observe.cpp` is a test-only shared library compiled twice with
`-DOBSERVER_RUNTIME='"/absolute/path/to/runtime.cpp"'`, once from the original
source and once from the new source with `-DOBSERVER_NEW`. Keep strict arithmetic
flags and the same CPU target. `check_numerics.py` compares actual distance/kernel/
margin bits and output classes; `check_primitives.py` independently checks the
integer primitive, six-bit full domain, ordered synthetic arithmetic and cache
wrap. Neither observer is a deployment ABI.

`bench_raw.py` and `audit_raw.py` reproduce the separately fixed raw-input study.
They require the actual compiled preprocessing extension for the running CPython
ABI. Source hashes bind each timed implementation; a later modified checkout is
not a valid substitute for the original source snapshot. All work counts and
performance claims retain the numerical/hardware/workload boundaries in RESULTS.

Native artifacts are Linux x86-64 test outputs, not universal binaries. Portable
builds have passed their local contracts; target-platform acceptance is separate.
Preserve upstream LIBSVM and data attribution when redistributing evidence.
