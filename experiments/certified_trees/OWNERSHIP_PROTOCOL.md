# Fused owning refinement interface after complete tiled comparison

Retain both completed native matrices: the original1092 cells and tiled1260
cells. They use the same frozen sources, integer leaves and certificate bounds.
Tiling improves the compact execution path substantially, but full-coverage
refinement still pays for two separate Python owning-handle leases per call.

Implement one native owning pipeline containing the compact model and official
CatBoost model. Construct and source-bind both once, preserve safe destruction,
then expose one synchronous raw-uint8-to-class-index call. This removes duplicated
interface/ownership overhead, not fallback work, validations or model storage.
Every ambiguous row still reaches the unmodified official C API. Keep both models
and the upstream library in the deployment inventory; do not claim that the
fully resolving pipeline has the compact-only footprint.

Freeze this interface change before timing. Check all four complete model panels,
all retained statuses, input failures and concurrent/close behavior. Rerun the
entire15-arm tiled matrix plus fused owning refinement:16 arms, four models,
batches1/32/256, seven shuffled repetitions, seed2026092920. No candidate
quantization, arithmetic bound, model setting or source classifier changes.
Prior results remain separate, with no pooled or selectively retimed cells.
This is the final planned interface refinement for this continuation, not another
model-quality experiment or evidence of a universal performance advantage.
