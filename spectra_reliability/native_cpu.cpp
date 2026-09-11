// CPU-only primitives; no libtorch/CUDA dependency. AVX2 vectorizes output
// channels so every output preserves increasing-hidden-index FP32 accumulation.
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define SPECTRA_X86 1
#else
#define SPECTRA_X86 0
#endif
namespace {
constexpr int64_t MAX_ELEMENTS = int64_t(1) << 28;
struct Weight {
    int out, hidden;
    std::vector<uint8_t> packed;
    std::vector<int8_t> transposed;
    std::vector<float> scales, bias;
};
bool dimensions(int out, int hidden) {
    return out>0 && hidden>0 && out<=100000 && hidden<=100000 && int64_t(out)*hidden<=MAX_ELEMENTS;
}
bool validate(const uint8_t* w, size_t wn, const float* s, size_t sn,
              const float* b, size_t bn, int out, int hidden) {
    if (!dimensions(out,hidden) || !w || !s || sn!=size_t(out) ||
        wn!=size_t(out)*((size_t(hidden)+3)/4) || (bn!=0 && bn!=size_t(out)) || (bn && !b)) return false;
    const int rb=(hidden+3)/4;
    for(int o=0;o<out;++o) {
        if(!std::isfinite(s[o]) || s[o]<0 || (bn && !std::isfinite(b[o]))) return false;
        for(int d=0;d<rb*4;++d) {
            const unsigned c=(w[o*rb+d/4]>>(2*(d%4)))&3u;
            if((d<hidden && c==3) || (d>=hidden && c!=0)) return false;
        }
    }
    return true;
}
void scalar(const float* x,int vectors,const uint8_t* w,const float* s,const float* b,int out,int hidden,float* y) {
    const int rb=(hidden+3)/4;
    for(int v=0;v<vectors;++v) for(int o=0;o<out;++o) {
        float acc=0.f;
        for(int d=0;d<hidden;++d) {
            const unsigned c=(w[o*rb+d/4]>>(2*(d%4)))&3u;
            if(c==1) acc+=x[v*hidden+d]*s[o];
            else if(c==2) acc-=x[v*hidden+d]*s[o];
        }
        if(b) acc+=b[o];
        y[v*out+o]=acc;
    }
}
#if SPECTRA_X86
template<int V>
__attribute__((target("avx2")))
void avx_tile(const float* x,int first,const Weight& w,float* y) {
    const int out=w.out,hidden=w.hidden;
    const __m256i one=_mm256_set1_epi32(1),neg=_mm256_set1_epi32(-1);
    int o=0;
    for(;o+8<=out;o+=8) {
        __m256 acc[V];
        for(int j=0;j<V;++j)acc[j]=_mm256_setzero_ps();
        const __m256 scale=_mm256_loadu_ps(w.scales.data()+o);
        for(int d=0;d<hidden;++d) {
            // These V inputs already exist in one linear call. No future
            // recurrent-step activation is assumed to be materialized.
            const __m128i small=_mm_loadl_epi64(reinterpret_cast<const __m128i*>(w.transposed.data()+d*out+o));
            const __m256i codes=_mm256_cvtepi8_epi32(small);
            const __m256 positive=_mm256_castsi256_ps(_mm256_cmpeq_epi32(codes,one));
            const __m256 negative=_mm256_castsi256_ps(_mm256_cmpeq_epi32(codes,neg));
            for(int j=0;j<V;++j) {
                const __m256 prod=_mm256_mul_ps(_mm256_set1_ps(x[(first+j)*hidden+d]),scale);
                const __m256 add=_mm256_add_ps(acc[j],prod),sub=_mm256_sub_ps(acc[j],prod);
                acc[j]=_mm256_blendv_ps(acc[j],add,positive);
                acc[j]=_mm256_blendv_ps(acc[j],sub,negative);
            }
        }
        for(int j=0;j<V;++j) {
            if(!w.bias.empty())acc[j]=_mm256_add_ps(acc[j],_mm256_loadu_ps(w.bias.data()+o));
            _mm256_storeu_ps(y+(first+j)*out+o,acc[j]);
        }
    }
    for(;o<out;++o)for(int j=0;j<V;++j) {
        float acc=0.f;
        for(int d=0;d<hidden;++d) {
            const int8_t c=w.transposed[d*out+o];
            if(c==1)acc+=x[(first+j)*hidden+d]*w.scales[o];
            else if(c==-1)acc-=x[(first+j)*hidden+d]*w.scales[o];
        }
        if(!w.bias.empty())acc+=w.bias[o];
        y[(first+j)*out+o]=acc;
    }
}
__attribute__((target("avx2")))
void avx_outputs(const float* x,int vectors,const Weight& w,float* y) {
    int v=0;
    for(;v+4<=vectors;v+=4)avx_tile<4>(x,v,w,y);
    for(;v<vectors;++v)avx_tile<1>(x,v,w,y);
}
#endif
}
extern "C" {
int spectra_abi_version() { return 1; }
int spectra_has_avx2() {
#if SPECTRA_X86
    return __builtin_cpu_supports("avx2") ? 1 : 0;
#else
    return 0;
#endif
}
void* spectra_weight_create(const uint8_t* w,size_t wn,const float* s,size_t sn,
                           const float* b,size_t bn,int out,int hidden) {
    if(!validate(w,wn,s,sn,b,bn,out,hidden)) return nullptr;
    try {
        auto* p=new Weight; p->out=out; p->hidden=hidden;
        try {
            p->packed.assign(w,w+wn); p->scales.assign(s,s+sn);
            if(bn) p->bias.assign(b,b+bn);
            p->transposed.resize(size_t(out)*hidden);
            const int rb=(hidden+3)/4;
            for(int d=0;d<hidden;++d) for(int o=0;o<out;++o) {
                const unsigned c=(w[o*rb+d/4]>>(2*(d%4)))&3u;
                p->transposed[d*out+o]=c==2 ? -1 : int8_t(c);
            }
        } catch(...) { delete p; return nullptr; }
        return p;
    } catch(...) { return nullptr; }
}
void spectra_weight_destroy(void* p) { delete static_cast<Weight*>(p); }
int spectra_weight_linear(void* ptr,const float* x,int vectors,int hidden,float* y,int avx) {
    if(!ptr || !x || !y || vectors<=0 || vectors>100000 || hidden<=0) return -1;
    const auto& w=*static_cast<Weight*>(ptr);
    if(hidden!=w.hidden || (avx!=0 && avx!=1) || int64_t(vectors)*hidden>MAX_ELEMENTS || int64_t(vectors)*w.out>MAX_ELEMENTS) return -1;
    if(avx) {
#if SPECTRA_X86
        if(!spectra_has_avx2()) return -2;
        avx_outputs(x,vectors,w,y); return 0;
#else
        return -2;
#endif
    }
    scalar(x,vectors,w.packed.data(),w.scales.data(),w.bias.empty()?nullptr:w.bias.data(),w.out,w.hidden,y); return 0;
}
int spectra_checked_linear(const float* x,int vectors,const uint8_t* w,size_t wn,
                           const float* s,size_t sn,const float* b,size_t bn,int out,int hidden,float* y) {
    if(!x || !y || vectors<=0 || vectors>100000 || int64_t(vectors)*hidden>MAX_ELEMENTS || int64_t(vectors)*out>MAX_ELEMENTS ||
       !validate(w,wn,s,sn,b,bn,out,hidden)) return -1;
    scalar(x,vectors,w,s,bn?b:nullptr,out,hidden,y); return 0;
}
int spectra_sudoku_valid(const int64_t* x,const int64_t* y,size_t length,int box) {
    if(!x || !y || box<1 || box>7) return -1;
    const int n=box*box;
    if(length!=size_t(n)*n) return -1;
    uint64_t rows[64]={},cols[64]={},boxes[64]={};
    for(int i=0;i<n*n;++i) {
        const int64_t v=y[i];
        if(v<1 || v>n || (x[i]!=0 && x[i]!=v)) return 0;
        const int r=i/n,c=i%n,b=(r/box)*box+c/box;
        const uint64_t bit=uint64_t(1)<<(v-1);
        if((rows[r]|cols[c]|boxes[b])&bit) return 0;
        rows[r]|=bit; cols[c]|=bit; boxes[b]|=bit;
    }
    return 1;
}
}
