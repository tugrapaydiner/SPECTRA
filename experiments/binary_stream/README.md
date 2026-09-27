# Binary streaming: reproduction and evidence

This optional profile follows PR30 source `4faa21280587fa9c4bf13cc2eb465613b5a95d4c`.
Read [the contract](../../docs/SVM_BINARY_STREAM.md), [protocol](PROTOCOL.md) and
[results](RESULTS.md). Keep all inputs and model parameters unchanged; this is
execution of existing classifiers, not new training or accuracy confirmation.

The evidence delivery provides `inputs/models/`, the previous SDK's unchanged
`inputs/PRIOR_SDK_SHA256.json`, both accepted runs, old numerical source, exact
built-library identities and literal commands. It does not redistribute the
numerical-framework environment. Work from a fresh output directory for each run.

```bash
python -m pytest tests/svm --confcutdir=tests/svm
python -c 'from spectra.svm import build_runtime; print(build_runtime("native-build",target="avx2"))'
python -c 'from spectra.svm_preprocess_native import build_preprocessor; print(build_preprocessor("pre-build"))'
python experiments/binary_stream/bench.py --inputs /evidence/inputs/models \
  --library /path/to/libspectra_svm.so --out /new/prepared-run
python experiments/binary_stream/bench_pipeline.py --inputs /evidence/inputs/models \
  --library /path/to/libspectra_svm.so --preprocessor /path/to/your-ABI-preprocessor.so \
  --out /new/raw-run
python experiments/binary_stream/audit.py --kind batch --run /new/prepared-run \
  --inputs /evidence/inputs/models --source . \
  --manifest /evidence/inputs/PRIOR_SDK_SHA256.json \
  --library /path/to/libspectra_svm.so --out /new/batch-audit.json
```

For the raw audit use `--kind raw`, the raw run folder, and TWO `--library`
arguments (SVM and preprocessing libraries). The auditor verifies the complete
expected source/input inventory, source and binary digests, output digests, fixed
randomized order, exact row/field types and every published aggregate. It uses
only Python's standard library. Its fixed prior-manifest hash binds this retained
experiment, not arbitrary future models or an external author's identity.

## Independent ordered-margin check

`margin_probe.cpp` is a test-only observer, not a deployment entry point. Compile
it twice with strict arithmetic: the first includes the unchanged previous
`runtime.cpp`; the second includes this checkout and defines `PROBE_CANDIDATE`.

```bash
g++ -std=c++17 -O3 -mavx2 -fno-fast-math -ffp-contract=off -fPIC -shared \
  '-DPROBE_RUNTIME="/absolute/old/spectra/_native/ovo/runtime.cpp"' \
  experiments/binary_stream/margin_probe.cpp -o /new/base-margins.so
g++ -std=c++17 -O3 -mavx2 -fno-fast-math -ffp-contract=off -fPIC -shared \
  '-DPROBE_RUNTIME="/absolute/current/spectra/_native/ovo/runtime.cpp"' \
  -DPROBE_CANDIDATE experiments/binary_stream/margin_probe.cpp -o /new/stream-margins.so
python experiments/binary_stream/check_margins.py --inputs /evidence/inputs/models \
  --base-library /new/base-margins.so --candidate-library /new/stream-margins.so \
  --out /new/margins.json
```

The observer compares actual binary64 margins, not only labels, to the old source
and a separately decoded Python ordered interpreter. All natural binary inputs
and the fixed synthetic cancellation/subnormal/extreme cases are included. This
is not exact-real RBF arithmetic or independent researcher reproduction. Repeated
backend/path checks are not additional independent examples.

## Preserved negative evidence

A preliminary coefficient-tail-bound screen used approximate NumPy kernels only
as a headroom diagnostic. It needed roughly 92–98% of terms on the expensive
Chess/Titanic binary inputs and did not become an inference implementation. Those
bounds are not an accepted numerical certificate. The successful profile instead
still evaluates every nonzero term, but avoids inappropriate multiclass/cache
machinery and changes SIMD layout without reordering a per-input reduction.
