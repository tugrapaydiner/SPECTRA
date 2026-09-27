# Optional tiled raw-input execution

`PipelineSession.predict_fused` combines the explicitly built preprocessing
extension with the existing SVM worker. It is opt-in: `predict`, `predict_many`,
`predict_buffer`, and certificate behavior retain their existing defaults.

```python
from spectra.svm import build_runtime
from spectra.svm_preprocess_native import build_preprocessor
from spectra.svm_pipeline import PreparedPipeline

svm = build_runtime('svm-build')
pre = build_preprocessor('pre-build')
with PreparedPipeline('exported-model', svm, preprocessor_library=pre) as model:
    with model.session() as session:
        labels = session.predict_fused(raw_rows, columns=model.columns)
```

`raw_rows` must use the fitted column order and the same supported scalar/missing
value conventions as `predict_many`. Both build directories must be fresh. The
existing Python preprocessor remains available; no code compiles during import or
package installation. Rebuild the CPython-specific preprocessing extension: its
private operation ABI is now 4. This is not abi3 or a universal binary.

## Resource and numerical contract

For exact built-in lists/tuples and supported built-in scalar values, the engine
uses a private feature tile of at most 16,384 binary64 values (128 KiB). The row
limit is also at most 128; callers can lower it with `tile_rows=1..128`. Smaller
tiles trade call overhead for responsiveness. Whole-batch input and result-list
storage, model data, worker scratch, and interpreter memory are additional.
The original row/eight-million-element bounds still apply. Custom containers,
scalars and iterators fall back to the existing full-batch path; the 128 KiB
feature-tile bound does not describe that fallback's allocation.

No preprocessing algebra or SVM arithmetic is changed. The private tile is
validated by the same existing native worker before it is evaluated. Every row
resets its own caches. The vote contract is fidelity to computed pair outcomes,
not accuracy of real-world labels or exact-real exponential arithmetic.

The call returns a fresh complete label list, or raises without returning partial
labels. A later invalid row or interrupt can occur after earlier tiles have run;
there is no rollback guarantee for internal counters or computation already done.
Successful batches do not expose a last-row certificate. Use the existing
`predict_with_certificate` API for that interface.

## Lifetime, threads and input ownership

Do not mutate any raw container or value until the call returns. The implementation
checks container sizes/types again after releasing the GIL, preventing unchecked
indexing of resized rows; this is not snapshot semantics for concurrent mutation.
It accesses Python objects only while holding the GIL. Only SVM computation over
private C++ tile storage runs with the GIL released. This is not a claim of
parallel preprocessing or measured multiworker throughput.

A worker lock covers the entire request and blocks other-thread close/calls.
Same-thread signal handlers can re-enter an RLock, so explicit close is deferred
until the active fused request exits, and reentrant inference raises. Interrupts
are serviced between tiles; the worker can be reused after an interrupt unless
close was requested. Existing workers remain usable after their prepared owner
closes, as before.

The extension's `_bind_worker` capsule is a private integration of trusted native
function/worker pointers, not a supported public interface or safe wire format.
The Python wrapper binds only a matching live stock worker, owns its CDLL, and
checks lifetime under the lock. Invalid raw pointers, arbitrary extension libraries,
or private-state tampering can crash the process and are outside the contract.

## Evidence boundaries

See `experiments/fused_pipeline/PROTOCOL.md` and `bench.py` for the fixed mixed-size
trace. Those requests replay all retained examples in deterministic shuffled
order; they are not live production traffic, a new accuracy test, or independent
researcher replication. Latency quantiles are local replay observations only.
`memory_probe.py` uses a trivial synthetic model and repeated raw row objects to
isolate feature materialization; its memory savings must not be generalized to
all deployment workloads. No default backend, model, release version or numerical
tolerance is changed by this optional interface.

CPython threading reference: https://docs.python.org/3.13/c-api/init.html
Capsule integration reference: https://docs.python.org/3/c-api/capsule.html
