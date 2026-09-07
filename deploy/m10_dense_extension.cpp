// Milestone 10 correctness-first packed ternary FP32 linear.
//
// This operator is intentionally separate from the historical W1.58A8 integer
// GEMV/FFN kernels. It consumes the same row-padded 2-bit ternary weight encoding,
// but keeps input activations, accumulation, scales, bias, and outputs in FP32 so
// the trained GELU/RMSNorm/attention graph can be reproduced faithfully.
#include <torch/extension.h>

#include <cstddef>
#include <cstdint>

namespace {

inline int ternary_value(uint8_t code) {
  switch (code & 0x3u) {
    case 0x0u: return 0;
    case 0x1u: return 1;
    case 0x2u: return -1;
    default: return 2;  // reserved / invalid sentinel
  }
}

inline int64_t row_bytes(int64_t hidden) { return (hidden + 3) / 4; }

void check_cpu_contiguous(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.device().is_cpu(), name, " must be CPU, got ", t.device());
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

void validate_packed(const torch::Tensor& packed, int64_t out_dim, int64_t hidden) {
  const int64_t rb = row_bytes(hidden);
  TORCH_CHECK(packed.numel() == out_dim * rb,
              "packed_weight must contain exactly ", out_dim * rb,
              " bytes, got ", packed.numel());
  const uint8_t* p = packed.data_ptr<uint8_t>();
  const int tail = static_cast<int>(hidden & 3);
  for (int64_t o = 0; o < out_dim; ++o) {
    const uint8_t* row = p + o * rb;
    for (int64_t b = 0; b < rb; ++b) {
      const int valid = (b == rb - 1 && tail != 0) ? tail : 4;
      const uint8_t byte = row[b];
      for (int q = 0; q < valid; ++q) {
        TORCH_CHECK(((byte >> (2 * q)) & 0x3u) != 0x3u,
                    "packed_weight contains reserved ternary code 11 at row ",
                    o, ", byte ", b, ", lane ", q);
      }
      for (int q = valid; q < 4; ++q) {
        TORCH_CHECK(((byte >> (2 * q)) & 0x3u) == 0u,
                    "packed_weight row padding must use zero code at row ", o);
      }
    }
  }
}

}  // namespace

torch::Tensor dense_ternary_linear_fp32(
    torch::Tensor x,
    torch::Tensor packed_weight,
    torch::Tensor row_scale,
    torch::Tensor bias,
    int64_t out_dim) {
  check_cpu_contiguous(x, "x");
  check_cpu_contiguous(packed_weight, "packed_weight");
  check_cpu_contiguous(row_scale, "row_scale");
  check_cpu_contiguous(bias, "bias");

  TORCH_CHECK(x.scalar_type() == torch::kFloat32,
              "x must be float32, got ", x.scalar_type());
  TORCH_CHECK(packed_weight.scalar_type() == torch::kUInt8,
              "packed_weight must be uint8");
  TORCH_CHECK(row_scale.scalar_type() == torch::kFloat32,
              "row_scale must be float32");
  TORCH_CHECK(bias.scalar_type() == torch::kFloat32,
              "bias must be float32");
  TORCH_CHECK(x.dim() == 2, "x must have shape [vectors, hidden]");
  TORCH_CHECK(packed_weight.dim() == 1, "packed_weight must be rank 1");
  TORCH_CHECK(row_scale.dim() == 1, "row_scale must be rank 1");
  TORCH_CHECK(bias.dim() == 1, "bias must be rank 1");
  TORCH_CHECK(x.size(0) > 0 && x.size(1) > 0,
              "x dimensions must be positive");
  TORCH_CHECK(out_dim > 0, "out_dim must be positive");
  TORCH_CHECK(row_scale.numel() == out_dim,
              "row_scale must have out_dim entries");
  TORCH_CHECK(bias.numel() == 0 || bias.numel() == out_dim,
              "bias must have zero or out_dim entries");
  TORCH_CHECK(torch::isfinite(row_scale).all().item<bool>(),
              "row_scale must be finite");
  TORCH_CHECK((row_scale >= 0).all().item<bool>(),
              "row_scale must be non-negative");
  TORCH_CHECK(bias.numel() == 0 || torch::isfinite(bias).all().item<bool>(),
              "bias must be finite");

  const int64_t vectors = x.size(0);
  const int64_t hidden = x.size(1);
  validate_packed(packed_weight, out_dim, hidden);

  auto y = torch::empty({vectors, out_dim}, x.options());
  const float* xp = x.data_ptr<float>();
  const uint8_t* wp = packed_weight.data_ptr<uint8_t>();
  const float* sp = row_scale.data_ptr<float>();
  const float* bp = bias.numel() ? bias.data_ptr<float>() : nullptr;
  float* yp = y.data_ptr<float>();
  const int64_t rb = row_bytes(hidden);

  // Correctness-first scalar FP32 accumulation in increasing hidden-index order.
  // The per-row ternary scale is applied to every non-zero product so the
  // floating boundary is explicit and mirrors the hard-ternary effective weight.
  for (int64_t v = 0; v < vectors; ++v) {
    const float* xv = xp + v * hidden;
    for (int64_t o = 0; o < out_dim; ++o) {
      const uint8_t* wr = wp + o * rb;
      const float scale = sp[o];
      float acc = 0.0f;
      for (int64_t d = 0; d < hidden; ++d) {
        const uint8_t code = (wr[d >> 2] >> (2 * (d & 3))) & 0x3u;
        const int tv = ternary_value(code);
        // validate_packed() already rejects the reserved code.
        if (tv == 1) {
          acc += xv[d] * scale;
        } else if (tv == -1) {
          acc -= xv[d] * scale;
        }
      }
      if (bp) acc += bp[o];
      yp[v * out_dim + o] = acc;
    }
  }
  return y;
}

const char* m10_operator_identity() {
  return "packed_ternary_fp32_scalar_linear_v1";
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dense_ternary_linear_fp32", &dense_ternary_linear_fp32,
        "M10 packed ternary FP32 dense linear");
  m.def("operator_identity", &m10_operator_identity,
        "M10 dense operator identity");
}
