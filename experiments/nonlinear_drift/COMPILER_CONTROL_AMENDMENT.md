# Additional generated-C compiler controls, before performance measurement

All model choices and exports remain frozen by PRETEST_FREEZE.json and EXPORT_FREEZE.json. The one final chronological quality evaluation has completed; no fitting, model selection or test-driven parameter changes follow it.

The selected full SVC's original m2cgen0.10.0 source generated successfully (1,492,387 bytes). The fixed GCC O3/AVX2/strict-FP build was killed after17.70 seconds. Its complete command, compiler output and generation source remain; it is a failed build, not a zero-time result.

Before any inference timing, add two separately labelled native generated-C controls on exactly the same source: GCC O1/AVX2/no-fast-math/no-contraction, and Clang O3/AVX2/no-fast-math/no-contraction with a4096 bracket-depth parser cap. Each has its own180-second/4-GiB address-space budget. Do not replace the failed O3 entry or choose only a favorable timing. If either builds, preserve every original class output and include it as a separate comparator. Different compiler/optimization settings remain named; no claim of identical compilation configuration across these controls.

The purpose is to avoid treating a failed compiler attempt as a defeated deployment alternative. The compact landmark model is a different frozen classifier, not bitwise equivalent to the full SVC. Its own fused/unfused native controls continue to share the same C++ implementation, compiler and stored original model. No runtime default, research gate or numerical tolerance changes. Any missing comparator remains missing.
