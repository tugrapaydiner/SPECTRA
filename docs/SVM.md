# Certified SVM execution

The optional `spectra.svm` module executes dense RBF SVC exports without NumPy,
scikit-learn or PyTorch at inference. It includes exact feature dictionaries,
sparse coefficients, request-local kernel reuse and certified pairwise voting.
This is an integration of the previously delivered SVM engines, not another model.
The builders support Linux x86-64/ARM64 and Windows x64 under the explicit
[native-platform contract](SVM_PORTABILITY.md); acceptance depends on completed
platform checks, not just builder support. Import and package installation do
not compile code. Compilation is explicit and requires a suitable C++17 compiler.

## Build and run

```python
from spectra.svm import Session, build_runtime

library = build_runtime("svm-build")  # fresh directory; portable by default
with Session("model.srt", library, tables=False) as session:
    result = session.predict_with_certificate([0.0] * 16)
    print(result.class_index, result.evaluated_pairs)
    classes = session.predict_many([[0.0] * 16, [0.5] * 16])
```

`model.srt` is a real exported classifier, not supplied by this example.
To export a trusted fitted scikit-learn `SVC` in a separate training environment:

```python
from spectra.svm_export import export_svc
export_svc(fitted_svc, "model.srt")
```

Export requires dense RBF SVC weights, exactly 16 features, class labels 0..C-1,
2–128 classes, positive support counts and default vote tie handling. It refuses
overwrite. The format retains binary64 parameters; API inputs are rounded to
binary32 and widened inside the kernel. Compare other implementations on those
same inputs. Arbitrary pickles are not accepted by the inference loader.

## Numerical and execution contract

The result is the earliest class with maximum wins under the engine's computed
pairwise decisions. Partial voting stops only when integer lower/upper bounds
settle that exact winner. A caller-supplied hint changes ordering, not acceptance.
`verify_certificate` independently verifies the discrete vote bounds; it does not
authenticate the model or prove the supplied pair outcomes were computed correctly.
This is not a claim of correct real-world labels, exact real-valued exponentials,
or bitwise compatibility with every library or hardware target.

Supported schedules: `exhaustive`, `lazy`, `knockout_cert`, `beretta_array`,
`beretta_ordered`, and `beretta_cert`. The last is the default. Unsafe knockout-only
selection is not exposed by the public API. Feature tables are explicitly opt-in
because they can regress latency; `tables=True` still falls back losslessly when
exact feature cardinality or counted storage makes dictionaries unsuitable.

A session owns mutable state. Python calls, certificate extraction and destruction
share a lock because ctypes releases the GIL during native execution. Separate
sessions permit parallel work. Every batch row resets its state and kernel cache;
there is no cross-input answer cache or claimed batched-matrix optimization.
Batching amortizes Python/native dispatch and returns a fresh list of class indices.
It does not implicitly expose a certificate for its final row.

Batches are limited to 65,536 rows. Each legacy input row is read through at most
17 values to enforce its 16-feature width; excess iterators are rejected without
exhausting them. The certificate checker similarly consumes at most one more
outcome than the class count permits. These are consumption bounds, not deadlines
for an individual iterator step.

The native bridge checks all finite inputs
before any output write. On an unexpected execution failure, raw C output buffers
are unspecified; the Python API raises and returns no partial result. Native
pointers must be valid, correctly sized and nonoverlapping. The interface is not
a sandbox. Round-to-nearest is required; successful batch calls invalidate the
single-request certificate slot, as do rejected native bridge calls.

## Build and validate the distribution

```bash
python -m pip install build pytest scikit-learn
python -m pytest tests/svm --confcutdir=tests/svm
python -m build --outdir dist/svm-check
python scripts/check_svm_installation.py --wheel dist/svm-check/spectra-0.7.1-py3-none-any.whl --out svm-wheel-check
```

Use a fresh build/output directory. The version above is the current unreleased
source version; do not replace an existing v0.7.1 release asset. The install check
creates a fresh virtual environment, installs the sdist-built wheel, compiles from
its shipped C++/headers, and executes from outside the checkout with no numerical
framework installed. The CI matrix covers Python 3.10 and 3.13 on Linux.

`target="avx2"` is an explicit hardware-specific build option, not a portable binary.
Reported prepared/scratch bytes are counted buffers, not peak process memory.
Previous experiment speedups are not newly measured by this package integration.

## Provenance and prior art

The four retained native files are byte-identical to the corresponding
`research/exact_tables_20260926` and `research/lazy_ovo_20260926` files in the delivered
snapshot `83268b12d9b142af863bb08d8d343cfdb7a9cbd9`. Their dated subdirectories preserve
that provenance; the public bridge/API are new. No historical dataset, negative
result, checkpoint, numerical tolerance or default CNF behavior is altered.

Beretta et al., *An Optimal Algorithm for Finding Champions in Tournament Graphs*,
provides the published-derived selector (https://arxiv.org/abs/2111.13621).
The additional early certificate is separately labelled. The outcome cache here
is quadratic, not the paper's linear-space implementation. Dictionary coding,
sparse storage and kernel caches are established methods, not new inventions.
The SVC format follows the documented LIBSVM/scikit-learn OvO coefficient layout
(https://scikit-learn.org/stable/modules/svm.html#multi-class-classification).
