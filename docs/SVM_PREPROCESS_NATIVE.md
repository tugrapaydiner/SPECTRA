# Optional compiled preprocessing

`PreparedPipeline` keeps its Python preprocessor by default. An explicitly built
CPython extension can execute the same supported plan without per-feature Python
arithmetic or intermediate tuples for ordinary list/tuple rows. SVM numerical code,
models, vote certificates, preprocessing constants and output order are unchanged.

```python
from spectra.svm import build_runtime
from spectra.svm_preprocess_native import build_preprocessor
from spectra.svm_pipeline import PreparedPipeline

svm_library = build_runtime("svm-build")
preprocessor_library = build_preprocessor("preprocessor-build")
with PreparedPipeline("exported-model", svm_library,
                      preprocessor_library=preprocessor_library) as model:
    with model.session() as worker:
        labels = worker.predict_many(raw_rows, columns=model.columns)
```

Use fresh build directories and an existing supported exported model. Build requires
a C++17 compiler and the running CPython's development headers. No numerical Python
framework is needed at inference. Import and pip installation never compile code.
The extension targets the current CPython ABI; it is NOT an abi3 or universal wheel.
Rebuild for each interpreter ABI. Linux and GIL-enabled CPython are the tested scope;
free-threaded Python, Windows, ARM and subinterpreter use are not accepted here.

## Semantics and ownership

The new engine uses the validated inert JSON plan, copies its constants and category
maps, and validates the native operation inventory as well. Numeric subtraction and
division remain separate binary64 operations. Round-to-nearest is required. No
fast-math, reciprocal substitution, quantization, fitting or cross-request cache is
introduced. The existing unsupported-transform and model-binding rules still apply.

Exact built-in list/tuple containers and float/int/None/string scalars use the fast
path with the GIL held, without invoking custom Python callbacks. Custom containers,
iterators or scalar subclasses retain the bounded reference behavior; this can be
slower. No thread-parallel preprocessing throughput is claimed. Native libraries
and caller objects remain trusted; resource caps are not a security sandbox.

Every transform returns new writable binary64 storage, normally a memoryview owning
a bytearray. The reference fallback can return array('d'). Both are iterable and
support `.tobytes()` and the buffer interface. No partial transformed output is
returned on error, and later requests do not modify prior outputs. Existing worker
lifetime, input-schema and per-input SVM-cache isolation rules remain unchanged.

## Measured scope and reproduction

See `experiments/native_preprocessing/PROTOCOL.md` and
`RAW_CONTAINER_AMENDMENT.md` there. The first locally frozen matrix failed its
1.20x-on-every-task point gate narrowly on Titanic; it is retained. The separately
frozen second matrix includes both compiled paths and the specialized NumPy/native
control. This is adaptive engineering on eighteen already-consumed models from six
small datasets, not new accuracy confirmation or external preregistration.

Complete-call timers start at canonical Python raw rows and end at fresh labels.
They include input validation, preprocessing, output allocation and unchanged SVM
execution. Setup, file parsing and domain feature extraction are excluded. The
single-row scope uses one retained row per model; neither it nor 31 repeated batch
jobs is a production latency distribution. Retained models are never retrained.

The evidence delivery supplies `inputs/`, both raw runs and exact measured source
snapshots. `bench.py` requires explicitly trusted frozen sklearn receipts for its
reference arm; deployment never loads those pickles. `audit.py` uses only the
standard library and does not import benchmark aggregation or the native runtime.
It checks every feature byte, bound input/source file, timing row/order and summary;
it does not authenticate a maliciously replaced entire evidence packet.

```bash
python -m pytest tests/svm --confcutdir=tests/svm
python experiments/native_preprocessing/bench.py --inputs /path/to/inputs \
  --library /path/to/libspectra_svm.so --preprocessor /path/to/_spectra_preprocess.cpython-313-x86_64-linux-gnu.so \
  --out /new/run
python experiments/native_preprocessing/audit.py --run /new/run \
  --inputs /path/to/inputs --source /path/to/this/checkout
```

The explicit filename above is an example for CPython 3.13/Linux x86-64. Use the
actual path returned by `build_preprocessor` on the target interpreter. The installed
wheel check builds this extension from the wheel's shipped source, with no numerical
frameworks installed. No new package version or replacement of v0.7.1 is implied.
