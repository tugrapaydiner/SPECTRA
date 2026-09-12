// Independent-vector blocking; every output retains increasing hidden-index order.
// No fast-math, reassociation, predequantization or horizontal reduction.
#include <torch/extension.h>
#include <cstdint>
#include <limits>
#include <vector>

namespace {
void require_tensor(const torch::Tensor& t, torch::ScalarType dtype,
                    int64_t rank, const char* name) {
  TORCH_CHECK(t.device().is_cpu() && t.is_contiguous(), name, " must be contiguous CPU");
  TORCH_CHECK(t.scalar_type() == dtype && t.dim() == rank, name, " has invalid dtype/rank");
}

class BlockedPackedLinear {
 public:
  BlockedPackedLinear(torch::Tensor packed, torch::Tensor scale,
                      torch::Tensor bias, int64_t outputs, int64_t inputs)
      : outputs_(outputs), inputs_(inputs) {
    TORCH_CHECK(inputs > 0 && outputs > 0, "dimensions must be positive");
    require_tensor(packed, torch::kUInt8, 1, "packed");
    require_tensor(scale, torch::kFloat32, 1, "scale");
    require_tensor(bias, torch::kFloat32, 1, "bias");
    row_bytes_ = inputs / 4 + (inputs % 4 != 0);
    TORCH_CHECK(outputs <= std::numeric_limits<int64_t>::max() / row_bytes_,
                "packed geometry overflows int64");
    TORCH_CHECK(packed.numel() == outputs * row_bytes_, "packed shape mismatch");
    TORCH_CHECK(scale.numel() == outputs, "scale shape mismatch");
    TORCH_CHECK(bias.numel() == 0 || bias.numel() == outputs, "bias shape mismatch");
    // Validate owned copies, never aliases of caller-owned tensors.
    auto p = packed.clone(); auto s = scale.clone(); auto b = bias.clone();
    TORCH_CHECK(torch::isfinite(s).all().item<bool>() && (s >= 0).all().item<bool>(),
                "scale must be finite and non-negative");
    TORCH_CHECK(b.numel() == 0 || torch::isfinite(b).all().item<bool>(), "bias must be finite");
    const uint8_t* bytes = p.data_ptr<uint8_t>();
    for (int64_t o = 0; o < outputs; ++o) {
      for (int64_t byte = 0; byte < row_bytes_; ++byte) {
        for (int lane = 0; lane < 4; ++lane) {
          const auto code = (bytes[o * row_bytes_ + byte] >> (2 * lane)) & 3u;
          const bool padding = byte == row_bytes_ - 1 && inputs % 4 != 0 && lane >= inputs % 4;
          TORCH_CHECK(padding ? code == 0u : code != 3u, "reserved code or nonzero row padding");
        }
      }
    }
    packed_.assign(bytes, bytes + p.numel());
    scale_.assign(s.data_ptr<float>(), s.data_ptr<float>() + s.numel());
    if (b.numel()) bias_.assign(b.data_ptr<float>(), b.data_ptr<float>() + b.numel());
  }

  torch::Tensor forward(torch::Tensor x) const {
    require_tensor(x, torch::kFloat32, 2, "x");
    TORCH_CHECK(x.size(0) > 0 && x.size(1) == inputs_, "input shape mismatch");
    const int64_t vectors = x.size(0);
    auto result = torch::empty({vectors, outputs_}, x.options());
    const float* xp = x.data_ptr<float>();
    float* yp = result.data_ptr<float>();
    int64_t v = 0;
    for (; v <= vectors - 4; v += 4) {
      const float* x0 = xp + v * inputs_;
      const float* x1 = x0 + inputs_;
      const float* x2 = x1 + inputs_;
      const float* x3 = x2 + inputs_;
      for (int64_t o = 0; o < outputs_; ++o) {
        const uint8_t* w = packed_.data() + o * row_bytes_;
        const float scale = scale_[o];
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        for (int64_t d = 0; d < inputs_; ++d) {
          const auto code = (w[d >> 2] >> (2 * (d & 3))) & 3u;
          if (code == 1u) {
            a0 += x0[d] * scale; a1 += x1[d] * scale;
            a2 += x2[d] * scale; a3 += x3[d] * scale;
          } else if (code == 2u) {
            a0 -= x0[d] * scale; a1 -= x1[d] * scale;
            a2 -= x2[d] * scale; a3 -= x3[d] * scale;
          }
        }
        if (!bias_.empty()) {
          a0 += bias_[o]; a1 += bias_[o]; a2 += bias_[o]; a3 += bias_[o];
        }
        yp[v * outputs_ + o] = a0; yp[(v + 1) * outputs_ + o] = a1;
        yp[(v + 2) * outputs_ + o] = a2; yp[(v + 3) * outputs_ + o] = a3;
      }
    }
    // One to three remaining vectors use the same ordered scalar reduction.
    for (; v < vectors; ++v) {
      for (int64_t o = 0; o < outputs_; ++o) {
        const uint8_t* w = packed_.data() + o * row_bytes_;
        const float scale = scale_[o];
        float acc = 0.0f;
        for (int64_t d = 0; d < inputs_; ++d) {
          const auto code = (w[d >> 2] >> (2 * (d & 3))) & 3u;
          if (code == 1u) acc += xp[v * inputs_ + d] * scale;
          else if (code == 2u) acc -= xp[v * inputs_ + d] * scale;
        }
        if (!bias_.empty()) acc += bias_[o];
        yp[v * outputs_ + o] = acc;
      }
    }
    return result;
  }

  int64_t storage_bytes() const {
    return packed_.size() + sizeof(float) * (scale_.size() + bias_.size());
  }
 private:
  const int64_t outputs_, inputs_;
  int64_t row_bytes_;
  std::vector<uint8_t> packed_;
  std::vector<float> scale_, bias_;
};
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  pybind11::class_<BlockedPackedLinear>(m, "BlockedPackedLinear")
      .def(pybind11::init<torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t>())
      .def("forward", &BlockedPackedLinear::forward)
      .def("storage_bytes", &BlockedPackedLinear::storage_bytes);
}
