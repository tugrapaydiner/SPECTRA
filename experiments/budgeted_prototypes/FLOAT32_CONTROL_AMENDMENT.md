# Ordinary FP32 neural deployment control

The first1176-cell and stronger1344-cell comparisons are both complete and retained.
A remaining memory/precision asymmetry is important: prototype geometry is compact
integer data, whereas the selected MLP was deployed with its original float64
parameters. Before interpreting compactness, mechanically round the SAME selected
MLP/scaler arrays to float32, with no training or model choice. Store them in a
new SPNF0001 format. Native single-thread OpenBLAS SGEMM executes the full network
from raw integer codes; normalization, scaling, allocation, layers and fresh labels
are charged. This is a distinct numerical deployment of a frozen selected model,
not bit-identical float64 scores or quantization-aware retraining.

Write a conversion/source/hash lock before checking any converted output. Compare
all converted predictions to the original model and to a separately expressed
NumPy-float32 forward pass. Any disagreement must remain and receive its actual
accuracy, not a removed example. No candidate classifier, original selected model,
threshold or prototype layout is changed. This control is not INT8 and cannot
establish superiority over unmeasured quantized MLPs or ProtoNN.

Run every previous16 arm again plus mlp_float32 (17 arms ×4 tasks ×3 batch sizes
×7 repetitions =1428 cells), one pinned core, seed2026092917, one BLAS thread.
This complete final comparison replaces neither earlier raw run. Include setup,
model-byte and fresh-process memory for all four FP32 variants separately. The
learning gate and layout gate stay exactly as frozen. Report this amendment as
comparison strengthening after evaluation, not prospective model-quality selection.
