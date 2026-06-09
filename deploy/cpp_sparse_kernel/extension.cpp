// PyTorch C++ extension for the SPECTRA sparse ternary kernel.
//
// WHY THIS EXISTS (the B=1 binding-overhead trap): a ctypes-per-GEMV binding pays
// Python interpreter overhead on EVERY matmul. At B=1 with deep recursion
// (T * n * N_sup * n_layers GEMVs) that overhead dwarfs the ~microsecond AVX2
// kernel. This extension (a) exposes the kernel as a native ``torch::Tensor`` op
// (no numpy<->ctypes marshaling), and (b) provides ``fused_ternary_ffn`` which
// runs BOTH FFN projections + the activation inside ONE C++ call -- so the hot
// recursion loop stays out of the Python interpreter entirely.
//
// Build:  python deploy/cpp_sparse_kernel/setup.py build_ext --inplace
//   (or JIT via deploy/torch_kernel.load_extension()). Requires a C++ toolchain
//   ABI-compatible with the installed torch: gcc on Linux, MSVC on Windows.

#include <torch/extension.h>

#include <vector>

#include "spectra_kernel.cpp"  // reuse the validated kernel (static ternary_dot, etc.)

namespace {

void check_inputs(const torch::Tensor& X, const torch::Tensor& active_idx) {
  TORCH_CHECK(X.dtype() == torch::kInt8, "X must be int8");
  TORCH_CHECK(X.is_contiguous(), "X must be contiguous");
  TORCH_CHECK(active_idx.dtype() == torch::kInt32, "active_idx must be int32");
}

}  // namespace

// Single fused sparse ternary GEMV exposed as a torch op (zero Python marshaling).
torch::Tensor sparse_ternary_gemv(
    torch::Tensor X, torch::Tensor active_idx, torch::Tensor W_packed,
    torch::Tensor requant_mult, int64_t shift, int64_t out_dim) {
  check_inputs(X, active_idx);
  const int hidden = static_cast<int>(X.size(1));
  auto Y = torch::zeros({X.size(0), out_dim}, torch::dtype(torch::kInt8));
  spectra_sparse_ternary_gemv(
      X.data_ptr<int8_t>(), active_idx.data_ptr<int32_t>(),
      static_cast<int>(active_idx.size(0)),
      W_packed.data_ptr<uint8_t>(), requant_mult.data_ptr<int32_t>(),
      static_cast<int>(shift), hidden, static_cast<int>(out_dim),
      Y.data_ptr<int8_t>());
  return Y;
}

// Fused two-layer ternary FFN: up-projection -> int8 ReLU -> down-projection, all
// inside one native call. The deep recursion can call this once per block instead
// of bouncing to Python between every matmul -- the key to realising the AVX2 win
// at B=1.
torch::Tensor fused_ternary_ffn(
    torch::Tensor X, torch::Tensor active_idx,
    torch::Tensor W1_packed, torch::Tensor mult1,
    torch::Tensor W2_packed, torch::Tensor mult2,
    int64_t shift, int64_t inter_dim, int64_t out_dim) {
  check_inputs(X, active_idx);
  const int hidden = static_cast<int>(X.size(1));
  const int p1 = hidden / 4, p2 = static_cast<int>(inter_dim) / 4;

  (void)p1;
  (void)p2;
  auto Y = torch::zeros({X.size(0), out_dim}, torch::dtype(torch::kInt8));
  // Delegate to the validated extern "C" fused kernel (the loop stays in C++).
  spectra_fused_ternary_ffn(
      X.data_ptr<int8_t>(), active_idx.data_ptr<int32_t>(),
      static_cast<int>(active_idx.size(0)),
      W1_packed.data_ptr<uint8_t>(), mult1.data_ptr<int32_t>(),
      W2_packed.data_ptr<uint8_t>(), mult2.data_ptr<int32_t>(),
      static_cast<int>(shift), hidden, static_cast<int>(inter_dim),
      static_cast<int>(out_dim), Y.data_ptr<int8_t>());
  return Y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sparse_ternary_gemv", &sparse_ternary_gemv,
        "B=1 fused sparse ternary GEMV (native torch op)");
  m.def("fused_ternary_ffn", &fused_ternary_ffn,
        "Fused two-layer ternary FFN; recursion loop stays in C++");
}
