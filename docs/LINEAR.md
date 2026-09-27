# Optional fixed-model LinearSVC deployment

This adapter exports an already fitted, stock one-vs-rest `LinearSVC` into a small
model-specific C++ evaluator. It exists because the fixed linear control was a
better quality/cost choice than the tested RBF models on the subject-held-out HAR
workload. It does not change the SVM runner, choose a model for the caller, train a
new model or supply a new learning algorithm. It is ordinary dense linear inference.

## Export, explicitly compile, then predict

In the original training environment:

```python
from spectra.linear import export_linear_svc
export_linear_svc(fitted_linear_svc, 'linear-model')
```

Export writes original binary64 parameters, inert metadata and generated C++ source
to a NEW directory. Supported labels are unique bounded integers or UTF-8 strings;
feature dimensions are 1–4096 and class count 2–128. Crammer-Singer, arbitrary
estimators and embedded feature processing are not supported.

On the deployment/build machine:

```python
from array import array
from spectra.linear import build_linear, CompiledLinear

library = build_linear('linear-model', 'new-linear-build')
model = CompiledLinear('linear-model', library)
labels = model.predict_buffer(array('d', [0.0] * model.features))
```

The zero row illustrates the API only; real inputs require the SAME feature
extraction and preprocessing used for fitting. No sklearn/NumPy/PyTorch import is
required for compilation or prediction. `predict_many` accepts row iterables;
`predict_buffer` borrows writable, naturally aligned, contiguous binary64 storage
without casting. The caller must not mutate it concurrently. Both return fresh
lists of original labels. Empty batches are accepted; existing 65,536-row and
8-million-element caps apply. These caps are not a sandbox or hard deadline.

One compiled library belongs to one fixed model, unlike the generic RBF executor.
Compilation, generated source size, original parameter bytes and binary size are
separate costs. Build output directories must be new. The existing strict platform
builder records compiler errors and uses noncontracted arithmetic. Windows and ARM
acceptance is the outcome of their actual CI, not inferred from this document.
The native model state is immutable and each invocation owns its small score buffer;
no inter-request cache or owning mutable worker is needed. No explicit library
unload is offered. No parallel-throughput measurement is implied.

## Numerical contract

Each class score starts at zero, adds original products in increasing feature
order, then adds its original bias. The implementation does not reassociate sums,
use FMA, quantize parameters or normalize weights. Multiclass ties choose the first
class index; binary zero chooses the first class, matching the LinearSVC convention.

An optimized BLAS may use a different summation order. Thus identical stored weights
DO NOT by themselves prove identical decisions on every possible boundary input.
The held-out experiment separately reports observed label agreement and ordered
score equality. `scores_buffer` returns flat raw scores for inspection, not
probabilities, certified real-number intervals or SVM tournament certificates.
A nonfinite intermediate/final score raises rather than manufacturing a prediction.
Late native computation errors can follow earlier internal work, but no partial
Python result is returned. No constant-time or allocation-free whole-API claim.

Generated C++ and supplied libraries are trusted executable code. Hashes and the
embedded model identity bind metadata/weights to a particular declared model, not
authorship or the honesty of arbitrary code containing that identifier. Modifying
private files/pointers concurrently is unsupported. This is not safe execution of
untrusted uploaded source. Installation/import never invokes a compiler.

## Evidence and application example

`experiments/har_execution/` retains the failed RBF packing proposals, fixed model
comparison and the post-test linear-deployment amendment. Its `application.py`
processes the supplied 561-dimensional feature JSONL through the same bounded
parser/writer for all compared backends. This is an example harness, not support
for linear bundles in `spectra svm run`. HAR signal processing is outside its timer.
The strongest NumPy linear control uses inert parameters and needs no sklearn
model deserialization; batch wins and losses are both reported.

Primary API reference: https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
