// PyTorch C++ extension for the checked SPECTRA native ternary kernels.
#include <torch/extension.h>

#include <climits>
#include <cstddef>
#include <cstdint>

#include "spectra_kernel.cpp"

namespace {

constexpr int64_t kMaxDotWidth = INT32_MAX / 128;

void check_cpu_contiguous(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.device().is_cpu(), name, " must be a CPU tensor, got ", t.device());
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

void check_tensor(
    const torch::Tensor& t, torch::ScalarType dtype, int64_t dim, const char* name) {
  check_cpu_contiguous(t, name);
  TORCH_CHECK(t.scalar_type() == dtype, name, " has wrong dtype: ", t.scalar_type());
  TORCH_CHECK(t.dim() == dim, name, " must have rank ", dim, ", got rank ", t.dim());
}

int64_t packed_row_bytes(int64_t width) { return (width + 3) / 4; }

void check_shift(int64_t shift) {
  TORCH_CHECK(shift >= 0 && shift <= 62, "shift must be in [0, 62], got ", shift);
}

void check_width(int64_t width, const char* name) {
  TORCH_CHECK(width > 0 && width <= kMaxDotWidth,
              name, " must be in [1, ", kMaxDotWidth, "], got ", width);
}

void check_positive_int(int64_t value, const char* name) {
  TORCH_CHECK(value > 0 && value <= INT_MAX, name, " must be in [1, INT_MAX], got ", value);
}

void check_active_indices(const torch::Tensor& active_idx, int64_t num_tokens) {
  const auto n = active_idx.numel();
  const auto* p = active_idx.data_ptr<int32_t>();
  for (int64_t i = 0; i < n; ++i) {
    TORCH_CHECK(p[i] >= 0 && p[i] < num_tokens,
                "active_idx[", i, "]=", p[i], " is outside [0, ", num_tokens, ")");
  }
  // Duplicate indices are valid by contract: the same output row is recomputed
  // and overwritten deterministically; no accumulation occurs.
}

void check_native_status(int status, const char* op) {
  TORCH_CHECK(status == SPECTRA_OK, op, " rejected input: ", spectra_status_string(status),
              " (status=", status, ")");
}

}  // namespace

torch::Tensor sparse_ternary_gemv(
    torch::Tensor X, torch::Tensor active_idx, torch::Tensor W_packed,
    torch::Tensor requant_mult, int64_t shift, int64_t out_dim) {
  check_tensor(X, torch::kInt8, 2, "X");
  check_tensor(active_idx, torch::kInt32, 1, "active_idx");
  check_tensor(W_packed, torch::kUInt8, 1, "W_packed");
  check_tensor(requant_mult, torch::kInt32, 1, "requant_mult");
  check_shift(shift);
  check_positive_int(out_dim, "out_dim");

  const int64_t num_tokens = X.size(0);
  const int64_t hidden = X.size(1);
  TORCH_CHECK(num_tokens > 0 && num_tokens <= INT_MAX,
              "X.size(0) must be in [1, INT_MAX], got ", num_tokens);
  check_width(hidden, "hidden dimension");
  check_active_indices(active_idx, num_tokens);

  const int64_t expected_w = out_dim * packed_row_bytes(hidden);
  TORCH_CHECK(W_packed.numel() == expected_w,
              "W_packed must contain exactly out_dim*ceil(hidden/4)=", expected_w,
              " bytes, got ", W_packed.numel());
  TORCH_CHECK(requant_mult.numel() == out_dim,
              "requant_mult must contain exactly out_dim=", out_dim,
              " entries, got ", requant_mult.numel());

  auto Y = torch::zeros({num_tokens, out_dim}, X.options().dtype(torch::kInt8));
  const int status = spectra_sparse_ternary_gemv(
      X.data_ptr<int8_t>(), static_cast<size_t>(X.numel()),
      active_idx.data_ptr<int32_t>(), static_cast<size_t>(active_idx.numel()),
      W_packed.data_ptr<uint8_t>(), static_cast<size_t>(W_packed.numel()),
      requant_mult.data_ptr<int32_t>(), static_cast<size_t>(requant_mult.numel()),
      static_cast<int>(shift), static_cast<int>(num_tokens), static_cast<int>(hidden),
      static_cast<int>(out_dim), Y.data_ptr<int8_t>(), static_cast<size_t>(Y.numel()));
  check_native_status(status, "spectra_sparse_ternary_gemv");
  return Y;
}

torch::Tensor fused_ternary_ffn(
    torch::Tensor X, torch::Tensor active_idx,
    torch::Tensor W1_packed, torch::Tensor mult1,
    torch::Tensor W2_packed, torch::Tensor mult2,
    int64_t shift, int64_t inter_dim, int64_t out_dim) {
  check_tensor(X, torch::kInt8, 2, "X");
  check_tensor(active_idx, torch::kInt32, 1, "active_idx");
  check_tensor(W1_packed, torch::kUInt8, 1, "W1_packed");
  check_tensor(mult1, torch::kInt32, 1, "mult1");
  check_tensor(W2_packed, torch::kUInt8, 1, "W2_packed");
  check_tensor(mult2, torch::kInt32, 1, "mult2");
  check_shift(shift);
  check_width(inter_dim, "inter_dim");
  check_positive_int(out_dim, "out_dim");

  const int64_t num_tokens = X.size(0);
  const int64_t hidden = X.size(1);
  TORCH_CHECK(num_tokens > 0 && num_tokens <= INT_MAX,
              "X.size(0) must be in [1, INT_MAX], got ", num_tokens);
  check_width(hidden, "hidden dimension");
  check_active_indices(active_idx, num_tokens);

  const int64_t expected_w1 = inter_dim * packed_row_bytes(hidden);
  const int64_t expected_w2 = out_dim * packed_row_bytes(inter_dim);
  TORCH_CHECK(W1_packed.numel() == expected_w1,
              "W1_packed must contain exactly ", expected_w1, " bytes, got ", W1_packed.numel());
  TORCH_CHECK(mult1.numel() == inter_dim,
              "mult1 must contain exactly inter_dim=", inter_dim, " entries, got ", mult1.numel());
  TORCH_CHECK(W2_packed.numel() == expected_w2,
              "W2_packed must contain exactly ", expected_w2, " bytes, got ", W2_packed.numel());
  TORCH_CHECK(mult2.numel() == out_dim,
              "mult2 must contain exactly out_dim=", out_dim, " entries, got ", mult2.numel());

  auto Y = torch::zeros({num_tokens, out_dim}, X.options().dtype(torch::kInt8));
  const int status = spectra_fused_ternary_ffn(
      X.data_ptr<int8_t>(), static_cast<size_t>(X.numel()),
      active_idx.data_ptr<int32_t>(), static_cast<size_t>(active_idx.numel()),
      W1_packed.data_ptr<uint8_t>(), static_cast<size_t>(W1_packed.numel()),
      mult1.data_ptr<int32_t>(), static_cast<size_t>(mult1.numel()),
      W2_packed.data_ptr<uint8_t>(), static_cast<size_t>(W2_packed.numel()),
      mult2.data_ptr<int32_t>(), static_cast<size_t>(mult2.numel()),
      static_cast<int>(shift), static_cast<int>(num_tokens), static_cast<int>(hidden),
      static_cast<int>(inter_dim), static_cast<int>(out_dim),
      Y.data_ptr<int8_t>(), static_cast<size_t>(Y.numel()));
  check_native_status(status, "spectra_fused_ternary_ffn");
  return Y;
}

const char* backend_name() {
  return spectra_compiled_with_avx2() ? "avx2" : "scalar";
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sparse_ternary_gemv", &sparse_ternary_gemv,
        "Checked B=1 fused sparse ternary GEMV");
  m.def("fused_ternary_ffn", &fused_ternary_ffn,
        "Checked fused two-layer ternary FFN");
  m.def("backend_name", &backend_name, "Compiled kernel backend: avx2 or scalar");
}
