# Checked sparse ternary kernel

`spectra_kernel.cpp` implements CPU sparse ternary GEMV and integer
requantization with a scalar path and an optional AVX2 path. The C ABI and
Python adapters validate shapes, buffer lengths, indices and numeric limits.

Builds are explicit. The PyTorch adapter's `deploy.torch_kernel.load_extension()`
requires the optional research dependencies and a compatible C++ compiler. It
selects scalar or AVX2 code for the host and reports build failures. Importing
the adapter does not compile it.

The standalone C ABI uses [CMakeLists.txt](CMakeLists.txt), with
`SPECTRA_AVX2=AUTO`, `ON` or `OFF`. Use `OFF` for a scalar build; `ON` requires
an AVX2-capable target. The NumPy adapter is in
[`deploy.bitnet_cpp_adapter`](../bitnet_cpp_adapter/__init__.py).

The [kernel contracts](../../tests/test_kernel_contract.py) compare native
outputs with an independent wide-integer reference and check malformed buffers,
indices, shifts and weights. They do not establish a universal speedup or a
bitnet.cpp integration. Benchmark the intended shapes and active-token density
before choosing a backend.

See the [runtime guide](../../docs/RUNTIME_GUIDE.md) for the maintained integrated
execution path and its measurement limits.
