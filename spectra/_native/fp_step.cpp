// Prepared FP recurrence: the same ATen operators as the supported eager graph.
// Dispatch elimination, not a new GEMM/attention algorithm or quantizer.
#include <torch/extension.h>
#include <c10/core/InferenceMode.h>
#include <vector>
#include <tuple>
#include <utility>
namespace {
using torch::Tensor;
void tensor_shape(const Tensor& t, at::IntArrayRef shape, const char* name) {
  TORCH_CHECK(t.device().is_cpu() && t.scalar_type() == at::kFloat &&
              t.layout() == at::kStrided && t.is_contiguous(), name, " must be contiguous CPU FP32");
  TORCH_CHECK(t.sizes() == shape, name, " has an unsupported shape");
}
Tensor own(const Tensor& t) {
  TORCH_CHECK(t.device().is_cpu() && t.scalar_type() == at::kFloat &&
              t.layout() == at::kStrided, "weights must be CPU FP32 strided tensors");
  auto result = t.detach().clone().contiguous();
  TORCH_CHECK(at::isfinite(result).all().item<bool>(), "weights must be finite");
  return result;
}
class PreparedFPStep {
 public:
  PreparedFPStep(std::vector<Tensor> common, std::vector<std::vector<Tensor>> blocks,
                 std::vector<int64_t> heads) : heads_(std::move(heads)) {
    c10::InferenceMode guard;
    TORCH_CHECK(common.size() == 8, "expected eight common tensors");
    TORCH_CHECK(!blocks.empty() && blocks.size() <= 32 && heads_.size() == blocks.size(), "invalid block/head count");
    for (const auto& t : common) common_.push_back(own(t));
    TORCH_CHECK(common_[0].dim() == 2, "token table must be a matrix");
    dim_ = common_[0].size(1);
    TORCH_CHECK(dim_ >= 2 && dim_ <= 4096, "unsupported hidden dimension");
    tensor_shape(common_[0], {5, dim_}, "token table");
    tensor_shape(common_[1], {1, 16, dim_}, "position table");
    tensor_shape(common_[2], {dim_}, "norm_y");
    tensor_shape(common_[3], {dim_}, "norm_z");
    tensor_shape(common_[4], {}, "alpha_y");
    tensor_shape(common_[5], {}, "alpha_z");
    tensor_shape(common_[6], {5, dim_}, "output weight");
    tensor_shape(common_[7], {5}, "output bias");
    for (size_t i = 0; i < blocks.size(); ++i) {
      TORCH_CHECK(blocks[i].size() == 10, "expected ten tensors per block");
      TORCH_CHECK(heads_[i] > 0 && heads_[i] % 2 == 0 && dim_ % heads_[i] == 0, "expected even head count dividing hidden dimension");
      std::vector<Tensor> b;
      for (const auto& t : blocks[i]) b.push_back(own(t));
      tensor_shape(b[0], {dim_}, "attention norm");
      tensor_shape(b[1], {3*dim_, dim_}, "QKV weight");
      tensor_shape(b[2], {3*dim_}, "QKV bias");
      tensor_shape(b[3], {dim_, dim_}, "projection weight");
      tensor_shape(b[4], {dim_}, "projection bias");
      tensor_shape(b[5], {dim_}, "FFN norm");
      TORCH_CHECK(b[6].dim() == 2 && b[6].size(0) > 0, "FFN weight must be a matrix");
      const auto inner = b[6].size(0);
      tensor_shape(b[6], {inner, dim_}, "FFN input weight");
      tensor_shape(b[7], {inner}, "FFN input bias");
      tensor_shape(b[8], {dim_, inner}, "FFN output weight");
      tensor_shape(b[9], {dim_}, "FFN output bias");
      blocks_.push_back(std::move(b));
    }
  }
  Tensor encode(const Tensor& x) const {
    c10::InferenceMode guard;
    TORCH_CHECK(x.device().is_cpu() && x.scalar_type() == at::kLong && x.layout() == at::kStrided &&
                x.is_contiguous() && x.sizes() == at::IntArrayRef({1,16}), "input must be CPU int64 [1,16]");
    TORCH_CHECK(x.min().item<int64_t>() >= 0 && x.max().item<int64_t>() <= 4, "input symbols must be in [0,4]");
    return at::embedding(common_[0], x) + common_[1];
  }
  std::tuple<Tensor, Tensor, Tensor> step(const Tensor& x, const Tensor& y, const Tensor& z) const {
    c10::InferenceMode guard;
    tensor_shape(x, {1,16,dim_}, "x_emb");
    tensor_shape(y, {1,16,dim_}, "y");
    tensor_shape(z, {1,16,dim_}, "z");
    auto uz = f((x + y) + z);
    auto nz = at::rms_norm(z + common_[5] * uz, {dim_}, common_[3], std::nullopt);
    auto uy = f(y + nz);
    auto ny = at::rms_norm(y + common_[4] * uy, {dim_}, common_[2], std::nullopt);
    return {ny, nz, at::linear(ny, common_[6], common_[7]).contiguous()};
  }
  int64_t owned_bytes() const {
    int64_t bytes = 0;
    for (const auto& t : common_) bytes += t.nbytes();
    for (const auto& b : blocks_) for (const auto& t : b) bytes += t.nbytes();
    return bytes;
  }
 private:
  Tensor f(Tensor h) const {
    for (size_t i = 0; i < blocks_.size(); ++i) {
      const auto& b = blocks_[i];
      auto q = at::rms_norm(h, {dim_}, b[0], std::nullopt);
      auto a = std::get<0>(at::_native_multi_head_attention(q, q, q, dim_, heads_[i],
                             b[1], b[2], b[3], b[4], std::nullopt, false, true, std::nullopt));
      h = h + a;
      auto normalized = at::rms_norm(h, {dim_}, b[5], std::nullopt);
      auto ff = at::linear(at::gelu(at::linear(normalized, b[6], b[7]), "none"), b[8], b[9]);
      h = h + ff;
    }
    return h;
  }
  int64_t dim_;
  std::vector<Tensor> common_;
  std::vector<std::vector<Tensor>> blocks_;
  std::vector<int64_t> heads_;
};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  pybind11::class_<PreparedFPStep>(m, "PreparedFPStep")
    .def(pybind11::init<std::vector<Tensor>, std::vector<std::vector<Tensor>>, std::vector<int64_t>>())
    .def("encode", &PreparedFPStep::encode, pybind11::call_guard<pybind11::gil_scoped_release>())
    .def("step", &PreparedFPStep::step, pybind11::call_guard<pybind11::gil_scoped_release>())
    .def("owned_bytes", &PreparedFPStep::owned_bytes);
}
