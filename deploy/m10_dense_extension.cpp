// Milestone 10 correctness-first packed ternary FP32 linear.
//
// This operator is intentionally separate from the historical W1.58A8 integer
// GEMV/FFN kernels. It consumes the same row-padded 2-bit ternary weight encoding,
// but keeps input activations, accumulation, scales, bias, and outputs in FP32 so
// the trained GELU/RMSNorm/attention graph can be reproduced faithfully.
#include <torch/extension.h>

#include <cstddef>
#include <cstdint>
#include <vector>
#include <cmath>
#include <algorithm>

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

torch::Tensor ordered_linear(const torch::Tensor& x, const uint8_t* wp,
                             const float* sp, const float* bp,
                             int64_t out_dim, int64_t hidden) {
  const int64_t vectors = x.size(0);
  auto y = torch::empty({vectors, out_dim}, x.options());
  const float* xp = x.data_ptr<float>();
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

  return ordered_linear(x, packed_weight.data_ptr<uint8_t>(),
                        row_scale.data_ptr<float>(),
                        bias.numel() ? bias.data_ptr<float>() : nullptr,
                        out_dim, hidden);
}

const char* m10_operator_identity() {
  return "packed_ternary_fp32_scalar_linear_v1";
}

// Owning immutable copies: caller tensor mutation cannot invalidate validation.
// The only exposed operation is forward; no writable weight alias is returned.
class ValidatedPackedLinear {
 public:
  ValidatedPackedLinear(torch::Tensor packed, torch::Tensor scale,
                        torch::Tensor bias, int64_t out_dim, int64_t hidden)
      : out_dim_(out_dim), hidden_(hidden) {
    TORCH_CHECK(hidden > 0 && out_dim > 0, "dimensions must be positive");
    check_cpu_contiguous(packed, "packed");
    check_cpu_contiguous(scale, "scale");
    check_cpu_contiguous(bias, "bias");
    TORCH_CHECK(packed.scalar_type() == torch::kUInt8 && packed.dim() == 1,
                "packed must be rank-1 uint8");
    TORCH_CHECK(scale.scalar_type() == torch::kFloat32 && scale.dim() == 1,
                "scale must be rank-1 float32");
    TORCH_CHECK(bias.scalar_type() == torch::kFloat32 && bias.dim() == 1,
                "bias must be rank-1 float32");
    TORCH_CHECK(scale.numel() == out_dim, "scale shape mismatch");
    TORCH_CHECK(bias.numel() == 0 || bias.numel() == out_dim, "bias shape mismatch");
    // Clone BEFORE validation, so the object only trusts the copies it owns.
    auto p = packed.clone(); auto s = scale.clone(); auto b = bias.clone();
    validate_packed(p, out_dim, hidden);
    TORCH_CHECK(torch::isfinite(s).all().item<bool>() && (s >= 0).all().item<bool>(),
                "scale must be finite and non-negative");
    TORCH_CHECK(b.numel() == 0 || torch::isfinite(b).all().item<bool>(),
                "bias must be finite");
    packed_.assign(p.data_ptr<uint8_t>(), p.data_ptr<uint8_t>() + p.numel());
    scale_.assign(s.data_ptr<float>(), s.data_ptr<float>() + s.numel());
    if (b.numel()) bias_.assign(b.data_ptr<float>(), b.data_ptr<float>() + b.numel());
  }

  torch::Tensor forward(torch::Tensor x) const {
    check_cpu_contiguous(x, "x");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32 && x.dim() == 2,
                "x must be rank-2 float32");
    TORCH_CHECK(x.size(0) > 0 && x.size(1) == hidden_, "input shape mismatch");
    return ordered_linear(x, packed_.data(), scale_.data(),
                          bias_.empty() ? nullptr : bias_.data(), out_dim_, hidden_);
  }

  int64_t storage_bytes() const {
    return packed_.size() + sizeof(float) * (scale_.size() + bias_.size());
  }

 private:
  const int64_t out_dim_, hidden_;
  std::vector<uint8_t> packed_;
  std::vector<float> scale_, bias_;
};

// Exact CPU Sudoku checker. Input validity is computed once per owned puzzle.
// No reference answer is accepted; every decoded candidate still gets checked.
class SudokuProblem {
 public:
  SudokuProblem(torch::Tensor puzzle, int64_t box) : box_(box), valid_(true) {
    TORCH_CHECK(box > 0 && box <= 8, "box must be in [1,8]");
    n_ = box * box;
    check_cpu_contiguous(puzzle, "puzzle");
    TORCH_CHECK(puzzle.scalar_type() == torch::kInt64 && puzzle.dim() == 2 &&
                puzzle.size(0) == 1 && puzzle.size(1) == n_ * n_,
                "puzzle must be int64 [1,N*N]");
    const int64_t* data = puzzle.data_ptr<int64_t>();
    givens_.assign(data, data + n_ * n_);
    std::vector<uint64_t> rows(n_, 0), cols(n_, 0), boxes(n_, 0);
    for (int64_t i = 0; i < n_ * n_; ++i) {
      const int64_t v = givens_[i];
      if (v == 0) continue;
      if (v < 1 || v > n_) { valid_ = false; continue; }
      const int64_t r = i / n_, c = i % n_, b = (r / box_) * box_ + c / box_;
      const uint64_t bit = uint64_t(1) << (v - 1);
      if ((rows[r] | cols[c] | boxes[b]) & bit) valid_ = false;
      rows[r] |= bit; cols[c] |= bit; boxes[b] |= bit;
    }
  }

  bool check(torch::Tensor answer) const {
    check_cpu_contiguous(answer, "answer");
    TORCH_CHECK(answer.scalar_type() == torch::kInt64 && answer.dim() == 2 &&
                answer.size(0) == 1 && answer.size(1) == n_ * n_,
                "answer must be int64 [1,N*N]");
    if (!valid_) return false;
    const int64_t* a = answer.data_ptr<int64_t>();
    std::vector<uint64_t> rows(n_, 0), cols(n_, 0), boxes(n_, 0);
    for (int64_t i = 0; i < n_ * n_; ++i) {
      const int64_t v = a[i];
      if (v < 1 || v > n_ || (givens_[i] != 0 && givens_[i] != v)) return false;
      const int64_t r = i / n_, c = i % n_, b = (r / box_) * box_ + c / box_;
      const uint64_t bit = uint64_t(1) << (v - 1);
      if ((rows[r] | cols[c] | boxes[b]) & bit) return false;
      rows[r] |= bit; cols[c] |= bit; boxes[b] |= bit;
    }
    return true;
  }

  std::tuple<torch::Tensor, bool> decode(torch::Tensor logits) const {
    check_cpu_contiguous(logits, "logits");
    TORCH_CHECK(logits.scalar_type() == torch::kFloat32 && logits.dim() == 3 &&
                logits.size(0) == 1 && logits.size(1) == n_ * n_ && logits.size(2) == n_ + 1,
                "logits must be float32 [1,N*N,N+1]");
    auto answer = torch::empty({1, n_ * n_}, logits.options().dtype(torch::kInt64));
    const float* lp = logits.data_ptr<float>(); int64_t* ap = answer.data_ptr<int64_t>();
    for (int64_t i = 0; i < n_ * n_; ++i) {
      const float* row = lp + i * (n_ + 1);
      int64_t best = 0;
      for (int64_t v = 0; v <= n_; ++v) {
        TORCH_CHECK(std::isfinite(row[v]), "logits must be finite");
        if (row[v] > row[best]) best = v; // first maximum, like torch.argmax
      }
      ap[i] = givens_[i] == 0 ? best : givens_[i];
    }
    return {answer, check(answer)};
  }

 private:
  int64_t box_, n_;
  bool valid_;
  std::vector<int64_t> givens_;
};


// Owning exact maze contract. BFS preprocessing is part of construction and
// therefore charged inside each complete solve. No stored reference is accepted.
// The selected cells must form a single simple four-neighbour path; checking
// just connectivity would incorrectly accept branches or detached cycles.
class MazeProblem {
 public:
  MazeProblem(torch::Tensor input, int64_t height, int64_t width, bool optimal)
      : h_(height), w_(width), optimal_(optimal) {
    TORCH_CHECK(h_ > 0 && w_ > 0 && h_ <= 512 && w_ <= 512,
                "maze dimensions must be in [1,512]");
    cells_ = h_ * w_;
    check_cpu_contiguous(input, "input");
    TORCH_CHECK(input.scalar_type() == torch::kInt64 && input.dim() == 2 &&
                input.size(0) == 1 && input.size(1) == cells_,
                "maze input must be int64 [1,H*W]");
    const int64_t* src = input.data_ptr<int64_t>();
    input_.assign(src, src + cells_);
    int starts = 0, goals = 0;
    valid_ = true;
    for (int64_t i = 0; i < cells_; ++i) {
      if (input_[i] < 0 || input_[i] > 3) valid_ = false;
      if (input_[i] == 2) { start_ = i; ++starts; }
      if (input_[i] == 3) { goal_ = i; ++goals; }
    }
    valid_ = valid_ && starts == 1 && goals == 1 && start_ != goal_;
    if (!valid_) return;
    previous_.assign(cells_, -2);
    previous_[start_] = -1;
    std::vector<int64_t> queue{start_};
    std::vector<int64_t> distance(cells_, -1);
    distance[start_] = 0;
    for (size_t head = 0; head < queue.size(); ++head) {
      const int64_t cell = queue[head];
      if (cell == goal_) { distance_ = distance[cell]; break; }
      neighbours(cell, [&](int64_t next) {
        if (input_[next] != 0 && previous_[next] == -2) {
          previous_[next] = cell;
          distance[next] = distance[cell] + 1;
          queue.push_back(next);
        }
      });
    }
  }

  bool check(torch::Tensor answer) const {
    check_cpu_contiguous(answer, "answer");
    TORCH_CHECK(answer.scalar_type() == torch::kInt64 && answer.dim() == 2 &&
                answer.size(0) == 1 && answer.size(1) == cells_,
                "maze answer must be int64 [1,H*W]");
    if (!valid_ || distance_ < 0) return false;
    const int64_t* a = answer.data_ptr<int64_t>();
    std::vector<uint8_t> selected(cells_, 0);
    int64_t count = 0;
    for (int64_t i = 0; i < cells_; ++i) {
      if (input_[i] == 1) {
        if (a[i] != 1 && a[i] != 4) return false;
      } else if (a[i] != input_[i]) {
        return false;
      }
      selected[i] = a[i] == 2 || a[i] == 3 || a[i] == 4;
      count += selected[i];
    }
    if (count < 2 || (optimal_ && count != distance_ + 1)) return false;
    for (int64_t i = 0; i < cells_; ++i) {
      if (!selected[i]) continue;
      int degree = 0;
      neighbours(i, [&](int64_t next) { degree += selected[next]; });
      if (degree != ((i == start_ || i == goal_) ? 1 : 2)) return false;
    }
    std::vector<uint8_t> seen(cells_, 0);
    std::vector<int64_t> queue{start_};
    seen[start_] = 1;
    for (size_t head = 0; head < queue.size(); ++head) {
      neighbours(queue[head], [&](int64_t next) {
        if (selected[next] && !seen[next]) {
          seen[next] = 1;
          queue.push_back(next);
        }
      });
    }
    return seen[goal_] && static_cast<int64_t>(queue.size()) == count;
  }

  std::tuple<torch::Tensor, bool> decode(torch::Tensor logits) const {
    check_cpu_contiguous(logits, "logits");
    TORCH_CHECK(logits.scalar_type() == torch::kFloat32 && logits.dim() == 3 &&
                logits.size(0) == 1 && logits.size(1) == cells_ && logits.size(2) == 5,
                "maze logits must be float32 [1,H*W,5]");
    auto answer = torch::empty({1, cells_}, logits.options().dtype(torch::kInt64));
    const float* lp = logits.data_ptr<float>();
    int64_t* ap = answer.data_ptr<int64_t>();
    for (int64_t i = 0; i < cells_; ++i) {
      int64_t best = 0;
      for (int64_t v = 0; v < 5; ++v) {
        TORCH_CHECK(std::isfinite(lp[i*5+v]), "logits must be finite");
        if (lp[i*5+v] > lp[i*5+best]) best = v;
      }
      ap[i] = input_[i] == 1 ? (best == 4 ? 4 : 1) : input_[i];
    }
    return {answer, check(answer)};
  }

  // Additional strong classical context. The constructor already performed BFS;
  // it must not be constructed outside the timed window for a solve benchmark.
  // On malformed/unreachable inputs this returns the input, which check rejects.
  torch::Tensor shortest_solution() const {
    auto answer = torch::empty({1, cells_}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
    int64_t* a = answer.data_ptr<int64_t>();
    std::copy(input_.begin(), input_.end(), a);
    if (valid_ && distance_ >= 0) {
      for (int64_t cell = goal_; cell != -1; cell = previous_[cell]) {
        if (cell != start_ && cell != goal_) a[cell] = 4;
      }
    }
    return answer;
  }

 private:
  template <typename Function>
  void neighbours(int64_t cell, Function&& visit) const {
    const int64_t row = cell / w_, col = cell % w_;
    // Same deterministic order as data.maze.shortest_path.
    if (row > 0) visit(cell - w_);
    if (row + 1 < h_) visit(cell + w_);
    if (col > 0) visit(cell - 1);
    if (col + 1 < w_) visit(cell + 1);
  }
  int64_t h_, w_, cells_, start_ = -1, goal_ = -1, distance_ = -1;
  bool optimal_, valid_ = false;
  std::vector<int64_t> input_, previous_;
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  pybind11::class_<MazeProblem>(m, "MazeProblem")
      .def(pybind11::init<torch::Tensor, int64_t, int64_t, bool>())
      .def("check", &MazeProblem::check)
      .def("decode", &MazeProblem::decode)
      .def("shortest_solution", &MazeProblem::shortest_solution);
  pybind11::class_<SudokuProblem>(m, "SudokuProblem")
      .def(pybind11::init<torch::Tensor, int64_t>())
      .def("check", &SudokuProblem::check)
      .def("decode", &SudokuProblem::decode);
  pybind11::class_<ValidatedPackedLinear>(m, "ValidatedPackedLinear")
      .def(pybind11::init<torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t>())
      .def("forward", &ValidatedPackedLinear::forward)
      .def("storage_bytes", &ValidatedPackedLinear::storage_bytes);
  m.def("dense_ternary_linear_fp32", &dense_ternary_linear_fp32,
        "M10 packed ternary FP32 dense linear");
  m.def("operator_identity", &m10_operator_identity,
        "M10 dense operator identity");
}
