# Additional feasible code-generation controls — before runtime timing

The original unmodified m2cgen ISOLET export exceeded its fixed120-second generation cap. Gas export completed but the O3 compiler was killed (the memory controller recorded an OOM kill). These are retained failed primary controls; no native runtime timing has begun.

To avoid benefiting from a previously diagnosed export limitation, add a separate ISOLET export using the already tested PR36 single-lookup patch at afcf25a, applied to an isolated pinned m2cgen0.10.0 package. Keep the same120-second generation and180-second compiler caps and model bytes. This is PATCHED m2cgen, not an unmodified-upstream success. When both original and patched sources exist their bytes must match. No new exporter design/tuning.

For an already generated source whose O3 compilation fails, try exactly one O1 compilation with the same strict floating-point/no-contraction/AVX2 options and180-second limit. This deliberately changes optimization level to give the competitor a feasible build, not to rerun until fast. Retain O3 failure; label O1 separately. Prefer original/O3, then original/O1, then patched/O3, then patched/O1 by this fixed feasibility order, never measured prediction speed. If none builds, the comparison remains incomplete.

Also include the fitted scikit-learn linear and all three selected MLPs as same-model runtime controls in addition to our strict native controls: a custom native loop is not assumed to be the fastest implementation. Use one numerical thread and the same input/output boundary. The hist-gradient-boosting timing is explicitly a framework control, not an optimized native-tree benchmark.

Final models, parameters, test predictions and task-admission criteria are frozen and unchanged. This is a stronger implementation comparison after build failure, not fresh model selection, an accuracy study, or authority to claim the missing original baseline was defeated.
