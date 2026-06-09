# SPECTRA fused sparse ternary kernel (Phase 12)

`spectra_kernel.cpp` implements the fused **B=1** sparse ternary GEMV of
BLUEPRINT section 26.3: active-token gather → packed W1.58 ternary GEMV → scale →
int8 requantize → scatter, in one cache-resident streaming pass.

## Status

This is the deployment kernel deliverable. It is **portable C++** with a correct
scalar reference and an AVX2 fast path (`#ifdef __AVX2__`). It is **not compiled
on the Windows dev box / in CI** — the PyTorch fake-quant path (`model/bitlinear.py`,
`model/fake_quant.py`) is the source of truth for correctness, and this kernel is
validated against it on the Linux/x86 eval box.

Per BLUEPRINT section 1 (build principles) and 36, the kernel is only worth
enabling once lazy-routing active-token density is proven low (< 30–40%, the
Phase 7 gate — SPECTRA measures ~0.25), otherwise dense `bitnet.cpp` wins.

## Build (Linux/x86 eval box)

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
# -> build/libspectra_kernel.so, loadable via deploy/bitnet_cpp_adapter
```

## Validation protocol

1. Export a trained ternary core with `deploy/export_bitnet.save_export`.
2. Feed identical INT8 activations + active-token indices to (a) the PyTorch
   reference and (b) this kernel.
3. Assert int8 outputs match (scalar path is exact; AVX2 path matches the scalar
   path bit-for-bit by construction).
4. Only then benchmark latency/energy vs the dense baseline under cgroups + RAPL.
