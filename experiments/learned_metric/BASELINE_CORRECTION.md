# Warm-baseline correction before accepted timing

Review during the first timing attempt found that `NpzFile` accesses inside the
sklearn request function reloaded model kernel tables on each call. This charged
model decompression despite the declared warm-request scope. That complete attempt
was terminated and retained as rejected, not used for any promoted runtime ratio.
No candidate is changed or selected from those timings.

The corrected baseline loads table and weights once, alongside its fitted model,
before warmup. Query conversion, kernel construction, model inference and fresh
labels remain timed. The entire fixed eleven-arm, seven-repeat, three-batch matrix
is rerun. Final model/label selection and all evaluation predictions are unchanged.
