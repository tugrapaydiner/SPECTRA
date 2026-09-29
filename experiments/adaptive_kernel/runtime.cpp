// Experimental adaptive precision; original SPECTRA native sources unchanged.
#ifndef AK_BASE_RUNTIME
#error Define AK_BASE_RUNTIME to the reviewed spectra/_native/ovo/runtime.cpp
#endif
#include AK_BASE_RUNTIME
#include <array>
#include <cfloat>
#if defined(__SSE__)
#include <xmmintrin.h>
#endif

namespace ak {
constexpr double EXP_ABS_ERROR=0x1p-19;
constexpr double LIBM_ABS_ERROR=0x1p-48; // Explicit reference-library assumption.
constexpr double FLOOR_ERROR=0x1p-60;
constexpr double MAX_COORD=0x1p20;
static_assert(sizeof(float)==4 && std::numeric_limits<float>::is_iec559,"IEEE binary32 required");
inline double gamma_n(int n,double u){return (n*u)/(1.-n*u);}
inline void environment(){
    lo::require(std::fegetround()==FE_TONEAREST,"round-to-nearest required");
#if defined(__SSE__)
    lo::require((_mm_getcsr()&0xe040)==0,"SIMD round-to-nearest and gradual underflow required");
#endif
}
inline float exp_scalar(float t){
    if(t>=16.f)return 0.f;
    float k=std::nearbyint(t*1.44269504088896341f);
    float r=(t-k*.693359375f)-k*(-.00021219444005469058f);
    float p=1.f/40320.f;
    p=p*r-1.f/5040.f;p=p*r+1.f/720.f;p=p*r-1.f/120.f;
    p=p*r+1.f/24.f;p=p*r-1.f/6.f;p=p*r+.5f;p=p*r-1.f;p=p*r+1.f;
    return std::ldexp(p,-int(k));
}
#if defined(__AVX2__)
inline __m256 exp8(__m256 input){
    __m256 t=_mm256_min_ps(input,_mm256_set1_ps(16.f));
    __m256 k=_mm256_round_ps(_mm256_mul_ps(t,_mm256_set1_ps(1.44269504088896341f)),
                            _MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
    __m256 r=_mm256_sub_ps(_mm256_sub_ps(t,_mm256_mul_ps(k,_mm256_set1_ps(.693359375f))),
                          _mm256_mul_ps(k,_mm256_set1_ps(-.00021219444005469058f)));
    __m256 p=_mm256_set1_ps(1.f/40320.f);
#define AK_STEP(c) p=_mm256_add_ps(_mm256_mul_ps(p,r),_mm256_set1_ps(c))
    AK_STEP(-1.f/5040.f);AK_STEP(1.f/720.f);AK_STEP(-1.f/120.f);AK_STEP(1.f/24.f);
    AK_STEP(-1.f/6.f);AK_STEP(.5f);AK_STEP(-1.f);AK_STEP(1.f);
#undef AK_STEP
    __m256i ki=_mm256_cvtps_epi32(k);
    __m256 scale=_mm256_castsi256_ps(_mm256_slli_epi32(_mm256_sub_epi32(_mm256_set1_epi32(127),ki),23));
    return _mm256_and_ps(_mm256_mul_ps(p,scale),_mm256_cmp_ps(input,_mm256_set1_ps(16.f),_CMP_LT_OQ));
}
#endif
struct Engine {
    spm::Owner model;
    spm::Worker exact;
    std::vector<float> bank,x32;
    std::vector<double> fast_kernel,magnitude;
    std::vector<uint64_t> epoch;
    std::vector<uint32_t> packed_to_original,original_to_packed;
    bool permuted=false;
    uint64_t serial=0,approximate_kernels=0,guarded_pairs=0,fallback_pairs=0,domain_fallback=0;
    double support_error=0,kernel_error=0;
    float gamma32=0;
    int stride=0,variant=1;
    bool admitted=true;
    Engine(const unsigned char* bytes,size_t count):model(std::make_shared<const spm::Model>(bytes,count,false)),exact(model){
        environment();
        admitted=model->gamma>=0x1p-20 && model->gamma<=0x1p20;
        for(double v:model->sv)admitted=admitted && std::abs(v)<=MAX_COORD;
        stride=((model->nsv+7)/8)*8;
        x32.resize(model->d);fast_kernel.resize(model->nsv);epoch.resize(model->nsv,0);
        magnitude.reserve(model->pairs.size());
        for(const auto& pair:model->pairs){
            double a=0;for(double v:pair.values)a=std::nextafter(a+std::abs(v),INFINITY);
            magnitude.push_back(a);
        }
        if(!admitted)return;
        gamma32=float(model->gamma);bank.resize(size_t(model->d)*stride,0.f);
#if !defined(AK_GATHER) && !defined(AK_SIMPLE_BLOCKS)
        if(model->c>2){
            // Model-only adjacency ordering: no data, labels, or timing is read.
            std::vector<std::vector<uint16_t>> uses(model->nsv);
            for(size_t p=0;p<model->pairs.size();++p)
                for(uint32_t id:model->pairs[p].ids)uses[id].push_back(uint16_t(p));
            packed_to_original.resize(model->nsv);original_to_packed.resize(model->nsv);
            for(uint32_t id=0;id<uint32_t(model->nsv);++id)packed_to_original[id]=id;
            std::sort(packed_to_original.begin(),packed_to_original.end(),[&](uint32_t a,uint32_t b){
                if(uses[a]!=uses[b])return uses[a]<uses[b];return a<b;
            });
            for(uint32_t k=0;k<uint32_t(model->nsv);++k)original_to_packed[packed_to_original[k]]=k;
            permuted=true;
        }
#endif
        double largest=0;
        for(int i=0;i<model->nsv;++i){
            double sum=0;
            for(int j=0;j<model->d;++j){
                double v=model->sv[size_t(i)*model->d+j];float q=float(v);
#if defined(AK_GATHER)
                bank[size_t(j)*stride+i]=q;
#else
                const uint32_t slot=permuted?original_to_packed[i]:uint32_t(i);
                bank[(size_t(slot/8)*model->d+j)*8+slot%8]=q;
#endif
                double e=v-double(q);sum+=e*e;
            }
            largest=std::max(largest,sum);
        }
        support_error=2.*std::sqrt(largest)+FLOOR_ERROR;
    }
    uint64_t extra_bytes()const{
        // Includes embedded exact Worker object, but NOT its vector capacities.
        return sizeof(*this)+4*(bank.capacity()+x32.capacity())+
               8*(fast_kernel.capacity()+magnitude.capacity()+epoch.capacity())+
               4*(packed_to_original.capacity()+original_to_packed.capacity());
    }
    bool begin(const double* x,int mode){
        variant=mode;approximate_kernels=guarded_pairs=fallback_pairs=domain_fallback=0;
        exact.begin(x);
        if(++serial==0){std::fill(epoch.begin(),epoch.end(),0);serial=1;}
        if(!admitted){domain_fallback=1;return false;}
        double sum=0;
        for(int j=0;j<model->d;++j){
            if(std::abs(x[j])>MAX_COORD){domain_fallback=1;return false;}
            float v=float(x[j]);x32[j]=v;double e=x[j]-double(v);sum+=e*e;
        }
        double ex=2.*std::sqrt(sum)+FLOOR_ERROR;
        double qerr=2.*std::sqrt(model->gamma)*(ex+support_error);
        double gerr=std::abs(double(gamma32)-model->gamma)/std::min(double(gamma32),model->gamma);
        double e32=gamma_n(2*model->d+16,0x1p-24),e64=gamma_n(2*model->d+16,0x1p-53);
        double approximation=mode==3?LIBM_ABS_ERROR:EXP_ABS_ERROR;
        kernel_error=qerr+gerr+e32/(1.-e32)+e64/(1.-e64)+approximation+LIBM_ABS_ERROR+FLOOR_ERROR;
        return std::isfinite(kernel_error);
    }
#if defined(AK_GATHER)
    void prepare(const std::vector<uint32_t>& ids){
        uint32_t todo[8];int count=0;
        auto flush=[&]{
            float dist[8]={};
#if defined(__AVX2__)
            if(count==8){
                __m256i idx=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(todo));
                __m256 sum=_mm256_setzero_ps();
                for(int j=0;j<model->d;++j){
                    __m256 s=_mm256_i32gather_ps(bank.data()+size_t(j)*stride,idx,4);
                    __m256 delta=_mm256_sub_ps(_mm256_set1_ps(x32[j]),s);
                    sum=_mm256_add_ps(sum,_mm256_mul_ps(delta,delta));
                }
                __m256 t=_mm256_mul_ps(sum,_mm256_set1_ps(gamma32));
                if(variant==3)_mm256_storeu_ps(dist,t);
                else _mm256_storeu_ps(dist,exp8(t));
            }else
#endif
            {
                for(int k=0;k<count;++k){
                    float sum=0;
                    for(int j=0;j<model->d;++j){float d=x32[j]-bank[size_t(j)*stride+todo[k]];sum+=d*d;}
                    float t=gamma32*sum;dist[k]=variant==3?t:exp_scalar(t);
                }
            }
            for(int k=0;k<count;++k){
                double value=variant==3?std::exp(-double(dist[k])):double(dist[k]);
                lo::require(std::isfinite(value)&&value>=0&&value<=1.+EXP_ABS_ERROR,"invalid proxy kernel");
                fast_kernel[todo[k]]=value;epoch[todo[k]]=serial;++approximate_kernels;
            }
            count=0;
        };
        for(uint32_t id:ids)if(epoch[id]!=serial){todo[count++]=id;if(count==8)flush();}
        if(count)flush();
    }
#else
    void prepare(const std::vector<uint32_t>& ids){
        // Pack fixed groups of support vectors, not one scattered SIMD gather
        // per feature. Every computed lane is cached for this input only.
        for(uint32_t requested:ids){
            if(epoch[requested]==serial)continue;
            const uint32_t slot=permuted?original_to_packed[requested]:requested;
            const uint32_t first=(slot/8)*8;
            const int count=std::min(8,model->nsv-int(first));
            const float* block=bank.data()+size_t(first/8)*model->d*8;
            float dist[8]={};
#if defined(__AVX2__)
            __m256 sum=_mm256_setzero_ps();
            for(int j=0;j<model->d;++j){
                __m256 delta=_mm256_sub_ps(_mm256_set1_ps(x32[j]),_mm256_loadu_ps(block+size_t(j)*8));
                sum=_mm256_add_ps(sum,_mm256_mul_ps(delta,delta));
            }
            __m256 t=_mm256_mul_ps(sum,_mm256_set1_ps(gamma32));
            _mm256_storeu_ps(dist,variant==3?t:exp8(t));
#else
            for(int k=0;k<count;++k){
                float sum=0;
                for(int j=0;j<model->d;++j){float delta=x32[j]-block[size_t(j)*8+k];sum+=delta*delta;}
                float t=gamma32*sum;dist[k]=variant==3?t:exp_scalar(t);
            }
#endif
            for(int k=0;k<count;++k){
                double value=variant==3?std::exp(-double(dist[k])):double(dist[k]);
                lo::require(std::isfinite(value)&&value>=0&&value<=1.+EXP_ABS_ERROR,"invalid proxy kernel");
                const uint32_t id=permuted?packed_to_original[first+k]:first+k;
                fast_kernel[id]=value;epoch[id]=serial;++approximate_kernels;
            }
        }
    }
#endif
    double proxy_score(size_t index){
        const auto& pair=model->pairs[index];prepare(pair.ids);
        double sum=0;
        for(size_t t=0;t<pair.ids.size();++t)sum+=pair.values[t]*fast_kernel[pair.ids[t]];
        return sum+pair.bias;
    }
    double error(size_t index)const{
        const auto& p=model->pairs[index];double a=magnitude[index];
        double g=gamma_n(int(2*p.ids.size()+8),0x1p-53);
        // Factor two gives conservative slack for evaluation of this positive bound.
        return 2.*(a*kernel_error+g*(a*(2.+kernel_error)+2.*std::abs(p.bias)))+FLOOR_ERROR;
    }
    int pair(int i,int j,const double* x){
        if(i>j)std::swap(i,j);size_t index=model->pair_index(i,j);
        double sum=proxy_score(index),bound=error(index);
        if(std::isfinite(sum)&&(variant==2||std::abs(sum)>bound)){
            ++guarded_pairs;return model->c==2?(sum>=0?1:0):(sum>0?i:j);
        }
        ++fallback_pairs;return exact.edge(i,j,x);
    }
    int run(const double* x,int mode){
        if(!begin(x,mode))return exact.run(x,5,-1);
        if(model->c==2){
            int result=pair(0,1,x);exact.vote.add(0,1,result);
            lo::require(exact.vote.certificate()==result,"binary proxy certificate inconsistent");
            return result;
        }
        std::fill(exact.priority.begin(),exact.priority.end(),0.);
        for(int i=0;i<model->c;++i){
            double sum=0;
            for(int j=0;j<model->d;++j){double v=x[j]-model->centers[size_t(i)*model->d+j];sum+=v*v;}
            exact.priority[i]=std::isfinite(sum)?sum:INFINITY;
        }
        auto edge=[&](int i,int j){return pair(i,j,x);};
        int result=et::beretta(exact.vote,edge,exact.priority,5,exact.rounds);
        lo::require(exact.vote.certificate()==result,"proxy vote certificate inconsistent");
        return result;
    }
    double original_score(int i,int j,const double* x){
        const auto& p=model->pairs[model->pair_index(i,j)];exact.prepare(p.ids,x);double sum=0;
        for(size_t t=0;t<p.ids.size();++t)sum+=p.values[t]*exact.kernel[p.ids[t]];
        return sum+p.bias;
    }
};
}
extern "C" {
int ak_abi(){return 1;}
void* ak_create(const unsigned char* data,uint64_t bytes){
    void* out=nullptr;lo::protect([&]{lo::require(data&&bytes<=lo::MAX_BYTES,"invalid model bytes");out=new ak::Engine(data,size_t(bytes));});return out;
}
void ak_destroy(void* p){delete static_cast<ak::Engine*>(p);}
int ak_info(void* p,uint64_t* out,int size){return lo::protect([&]{
    lo::require(p&&out&&size==6,"invalid info buffer");auto& e=*static_cast<ak::Engine*>(p);
    uint64_t values[]={uint64_t(e.model->d),uint64_t(e.model->c),uint64_t(e.model->nsv),
                       uint64_t(e.admitted),e.model->storage(),e.extra_bytes()+e.exact.scratch()-sizeof(spm::Worker)};
    std::copy(values,values+6,out);
});}
int ak_run(void* p,const double* input,int rows,int d,int mode,int* out,uint64_t* stats,int stat_size){return lo::protect([&]{
    lo::require(p&&rows>=0&&rows<=65536&&uint64_t(rows)*d<=8000000&&stats&&stat_size==6,"invalid batch");
    auto& e=*static_cast<ak::Engine*>(p);lo::require(d==e.model->d&&mode>=1&&mode<=3,"invalid geometry/mode");
    lo::require(!rows||(input&&out),"null input/output");ak::environment();
    for(size_t i=0;i<size_t(rows)*d;++i)lo::require(std::isfinite(input[i]),"finite inputs required");
    uint64_t totals[6]={};
    for(int i=0;i<rows;++i){
        out[i]=e.run(input+size_t(i)*d,mode);
        totals[0]+=e.approximate_kernels;totals[1]+=e.exact.work.kernels;
        totals[2]+=e.guarded_pairs;totals[3]+=e.fallback_pairs;
        totals[4]+=e.domain_fallback;totals[5]+=e.exact.work.pairs;
    }
    std::copy(totals,totals+6,stats);
});}
// Observes EVERY pair, separate from timed adaptive execution. Per pair:
// proxy margin, error bound, original margin, accepted (1) / fallback (0).
int ak_probe(void* p,const double* input,int rows,int d,double* output,uint64_t cells){return lo::protect([&]{
    lo::require(p&&rows>=0&&rows<=65536&&uint64_t(rows)*d<=8000000,"invalid probe shape");
    auto& e=*static_cast<ak::Engine*>(p);uint64_t pairs=e.model->pairs.size();
    lo::require(d==e.model->d&&cells==uint64_t(rows)*pairs*4&&(!rows||(input&&output)),"invalid probe output");
    ak::environment();for(size_t k=0;k<size_t(rows)*d;++k)lo::require(std::isfinite(input[k]),"finite inputs required");
    for(int r=0;r<rows;++r){
        const double* x=input+size_t(r)*d;bool allowed=e.begin(x,1);size_t k=0;
        for(int i=0;i<e.model->c;++i)for(int j=i+1;j<e.model->c;++j,++k){
            double orig=e.original_score(i,j,x),value=allowed?e.proxy_score(k):orig,bound=allowed?e.error(k):INFINITY;
            double* o=output+(size_t(r)*pairs+k)*4;o[0]=value;o[1]=bound;o[2]=orig;
            o[3]=(allowed&&std::isfinite(value)&&std::abs(value)>bound)?1.:0.;
        }
    }
});}
int ak_exp_probe(const float* t,float* out,int count){return lo::protect([&]{
    lo::require(count>=0&&count<=16000000&&(!count||(t&&out)),"invalid exp probe");ak::environment();
    for(int k=0;k<count;++k)lo::require(std::isfinite(t[k])&&t[k]>=0,"nonnegative finite exponent required");
    int k=0;
#if defined(__AVX2__)
    for(;k+8<=count;k+=8)_mm256_storeu_ps(out+k,ak::exp8(_mm256_loadu_ps(t+k)));
#endif
    for(;k<count;++k)out[k]=ak::exp_scalar(t[k]);
});}
}
