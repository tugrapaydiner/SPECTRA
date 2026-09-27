# Secondary: avoid the remaining built-in row snapshot

The first complete run is retained, including its failed 1.20x-on-each-task gate
(Titanic candidate/reference point ratio 0.835939). No threshold is changed.

Test a second, additive implementation that directly traverses exact built-in
list/tuple rows while retaining the CPython GIL. No Python callbacks are performed
on this fast path. Custom containers/scalars use bounded reference materialization
and reference semantics. Outputs remain fresh and request-private. This removes
intermediate row tuples, not the required output feature buffer or SVM work.

Before the second timing run, freeze this implementation and test scalar subclasses,
container subclasses, single-use iterators, rejected late inputs, output lifetime,
all original feature bytes and predictions. Compare both compiled variants in the
SAME run with the original Python, specialized NumPy/native and fitted sklearn.
Keep all eighteen models, both prior sizes and all31 repeats; new random-order seed
2026092702. All costs begin at the same raw Python rows. No new model fitting,
future-outcome access or hidden preprocessing/cross-request caching.

This is adaptive engineering on already consumed inputs, not independent confirmation.
Report the original failed gate separately; do not overwrite its observations or
claim the second matrix was preregistered before the first result. No automatic
runtime-default change or universal batch/tail performance claim.
