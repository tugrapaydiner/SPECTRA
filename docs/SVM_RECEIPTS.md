# Replayable SVM decision receipts

`spectra.svm_receipt` binds a decision to the exact model bytes and transformed
input. Its verifier separately decodes the model, recomputes each disclosed
pairwise decision with ordered Python binary64 arithmetic and system `exp`, then
checks that no undisclosed comparisons can change the winner. It does not import
the native executor, NumPy, scikit-learn or PyTorch.

```python
from spectra.svm_receipt import create_receipt, verify_receipt

# worker is a stock SharedSession. Features are already preprocessed, in order.
receipt = create_receipt(worker, transformed_features, schedule="binary_stream")
check = verify_receipt("model.srt", receipt,
                       expected_input=transformed_features,
                       input_dtype="float64")
assert check["verified"]
```

The verifier's expected input must come from the caller's intended request, not
be copied unquestioningly from the receipt. Likewise, supply a trusted intended
model. `input_dtype` is explicit: the recorded values are the actual post-rounding
binary64 values passed to the engine. Signed zero and class-label types/order are
preserved. Both supported model formats are understood by the independent parser.

A receipt is an ordinary JSON object. Use `load_receipt(path)` for a size-bounded,
duplicate-key-rejecting reader. No executable pickle is loaded. Receipts contain
feature values, which may be sensitive; sharing them shares those values.

## What is stronger than a vote certificate?

`verify_certificate` answers a narrower question: **if** the supplied pair outcomes
are true, do they settle this winner? That predicate deliberately does not evaluate
the model. A forged set of pair signs can satisfy it. `verify_receipt` additionally
checks model identity, expected input, class labels, and every disclosed pair sign.
Unknown pairs need not be computed once the unchanged vote bounds settle the result.

This is independent implementation replay, not a cryptographic signature, exact-real
RBF proof, proof of correct training, real-world label correctness or external
researcher replication. A different libm/compiler environment can disagree near a
zero margin; replay then rejects. No numerical tolerance is applied to accept a
changed sign. Raw-schema feature extraction and preprocessing are outside this
SVM-level receipt. Existing raw preprocessing fidelity has separate tests.

## Cost and boundaries

Creation/verification are opt-in and are not included in ordinary prediction
latency. Replay is slower than native inference. The verifier defaults to at most
8 million evaluated kernel-feature operations and separately bounds model/receipt
bytes. It also scans stored model parameters and the vote inventory. This is not
a hard time limit or a process-memory security sandbox. Lower `max_kernel_work`
to impose a smaller replay budget; exhausted budgets raise, not accept.

## Unified native ownership

Legacy Session, PreparedModel, SharedSession and fused calls now use the same
request-lifetime implementation. An operation holds a strong owning reference
through native execution and result/certificate extraction. Same-thread reentry
is rejected; close requested during an operation is deferred until it unwinds.
Destruction detaches its pointer before entering C, preventing recursive close
from scheduling the same destruction twice. Fused calls use invocation-local
capsules that hold the owning resource, not only its integer address.

This fixes the tested gap where only fused execution had a busy guard. Other
threads remain serialized per worker; independent workers can still run in
parallel. This does not make arbitrary asynchronous exceptions during cleanup,
process termination, forged native addresses or private-state mutation safe.
The Python reference: https://docs.python.org/3/library/signal.html
and ctypes call semantics: https://docs.python.org/3/library/ctypes.html
