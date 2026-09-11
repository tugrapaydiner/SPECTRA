// Historical M17 numerical replay, NOT a trained/deployed speedup claim.
// Explicit FP32 fused multiply-add in increasing k order. No reassociation.
#include <torch/extension.h>
#include <immintrin.h>
#include <cmath>
#include <cstdint>

static void tensor_contract(const torch::Tensor& t, int rank) {
  TORCH_CHECK(t.device().is_cpu() && t.scalar_type() == torch::kFloat32,
              "ordered replay requires CPU FP32 tensors");
  TORCH_CHECK(t.dim() == rank && t.is_contiguous(), "ordered replay requires contiguous declared-rank tensors");
  TORCH_CHECK(!t.requires_grad(), "ordered replay is inference-only");
  for (int d = 0; d < rank; ++d) TORCH_CHECK(t.size(d) > 0, "ordered replay rejects empty axes");
}

torch::Tensor ordered_bmm(const torch::Tensor& a, const torch::Tensor& b) {
  tensor_contract(a, 3); tensor_contract(b, 3);
  TORCH_CHECK(a.size(0) == b.size(0) && a.size(2) == b.size(1), "ordered BMM geometry mismatch");
  const int64_t B=a.size(0), M=a.size(1), K=a.size(2), N=b.size(2);
  auto result=torch::empty({B,M,N}, a.options());
  const auto* ap=a.data_ptr<float>(); const auto* bp=b.data_ptr<float>(); auto* out=result.data_ptr<float>();
  for (int64_t batch=0;batch<B;++batch) for (int64_t m=0;m<M;++m) {
    const float* x=ap+(batch*M+m)*K; const float* w=bp+batch*K*N;
    float* o=out+(batch*M+m)*N;
    int64_t n=0;
    for (;n+8<=N;n+=8) {
      auto s=_mm256_setzero_ps();
      for (int64_t k=0;k<K;++k) s=_mm256_fmadd_ps(_mm256_set1_ps(x[k]),_mm256_loadu_ps(w+k*N+n),s);
      _mm256_storeu_ps(o+n,s);
    }
    for (;n<N;++n) {
      float s=0.0f;
      for (int64_t k=0;k<K;++k) s=std::fma(x[k],w[k*N+n],s);
      o[n]=s;
    }
  }
  return result;
}

torch::Tensor blocked_linear(const torch::Tensor& a, const torch::Tensor& wt,
                             const torch::Tensor& bias, int64_t block) {
  tensor_contract(a,2); tensor_contract(wt,2); tensor_contract(bias,1);
  TORCH_CHECK(a.size(1)==wt.size(0) && wt.size(1)==bias.size(0), "ordered linear geometry mismatch");
  TORCH_CHECK(block>0 && block<=a.size(1), "invalid ordered linear reduction block");
  const int64_t M=a.size(0), K=a.size(1), N=wt.size(1);
  auto result=torch::empty({M,N},a.options());
  const auto* ap=a.data_ptr<float>(); const auto* wp=wt.data_ptr<float>();
  const auto* bp=bias.data_ptr<float>(); auto* out=result.data_ptr<float>();
  for (int64_t m=0;m<M;++m) {
    const float* x=ap+m*K; float* o=out+m*N;
    int64_t n=0;
    for (;n+8<=N;n+=8) {
      auto total=_mm256_loadu_ps(bp+n);
      for(int64_t begin=0;begin<K;) {
        auto partial=_mm256_setzero_ps();
        int64_t end=begin+std::min(block,K-begin);
        for(int64_t k=begin;k<end;++k) partial=_mm256_fmadd_ps(_mm256_set1_ps(x[k]),_mm256_loadu_ps(wp+k*N+n),partial);
        total=_mm256_add_ps(total,partial); begin=end;
      }
      _mm256_storeu_ps(o+n,total);
    }
    for (;n<N;++n) {
      float total=bp[n];
      for(int64_t begin=0;begin<K;) {
        float partial=0.0f;
        int64_t end=begin+std::min(block,K-begin);
        for(int64_t k=begin;k<end;++k) partial=std::fma(x[k],wp[k*N+n],partial);
        total=total+partial; begin=end;
      }
      o[n]=total;
    }
  }
  return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
  m.def("ordered_bmm", &ordered_bmm);
  m.def("blocked_linear", &blocked_linear);
}
