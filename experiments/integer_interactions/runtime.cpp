// Experimental integer feature interactions; no change to the stock executor.
#ifndef II_BASE_RUNTIME
#error Define II_BASE_RUNTIME to the original spectra/_native/ovo/runtime.cpp
#endif
#include II_BASE_RUNTIME
#include <array>
#if defined(__SSE__)
#include <xmmintrin.h>
#endif
namespace ii {
constexpr uint64_t MAX_ENTRIES=1048576;
inline void environment(){
    lo::require(std::fegetround()==FE_TONEAREST,"round-to-nearest required");
#if defined(__SSE__)
    lo::require((_mm_getcsr()&0xe040)==0,"SIMD round-to-nearest and gradual underflow required");
#endif
}

struct Term {uint16_t column; int16_t value;};
struct Engine {
    spm::Owner model;
    std::unique_ptr<spm::Worker> state;
    int raw_d=0,rank=0,maximum=0,bits=0;
    uint64_t bound=0,mask=0,matrix_terms=0;
    bool short_sum=false,diagonal=false;
    std::vector<uint32_t> diagonal_weights;
    std::vector<uint8_t> raw_supports,raw_query;
    std::vector<double> raw_centers;
    uint64_t projection_ops=0;
    std::vector<std::vector<Term>> transform;
    std::vector<int32_t> shifts;
    std::vector<int16_t> supports,query;
    std::vector<double> high,low,priority;
    int mode=1;
    uint64_t kernel_count=0,pair_count=0,term_count=0;
    Engine(const unsigned char* raw,size_t bytes) {
        lo::require(raw && bytes>=44 && bytes<=lo::MAX_BYTES,"invalid interaction model length");
        environment();
        lo::require(!std::memcmp(raw,"SPINT001",8),"wrong integer-interaction model magic");
        auto u32=[&](size_t offset){uint32_t v;std::memcpy(&v,raw+offset,4);return v;};
        uint32_t d=u32(8),m=u32(12),D=u32(16),c=u32(20),ns=u32(24),b=u32(28),meta=u32(32),payload=u32(36),crc=u32(40);
        lo::require(d>=1&&d<=64&&m>=1&&m<=128&&D>=1&&D<=255&&c>=2&&c<=128&&ns>=1&&ns<=100000&&uint64_t(m)*ns<=8000000,"unsupported interaction geometry");
        uint64_t expected=8+2ull*m*d+uint64_t(ns)*d+4ull*c+8ull*((c-1ull)*ns+c*(c-1ull)/2)+meta;
        lo::require(meta<=65536&&payload==expected&&bytes==44+expected,"interaction inventory mismatch");
        lo::require(lo::crc32(raw+44,bytes-44)==crc,"interaction CRC mismatch");
        raw_d=d;rank=m;maximum=D;
        double alpha;std::memcpy(&alpha,raw+44,8);
        lo::require(std::isfinite(alpha)&&alpha>0,"invalid kernel coefficient");
        transform.resize(m);shifts.resize(m,0);query.resize(m);priority.resize(c);
        size_t offset=52;
        for(uint32_t r=0;r<m;++r){int length=0;
            for(uint32_t j=0;j<d;++j){int16_t a;std::memcpy(&a,raw+offset,2);offset+=2;
                lo::require(a>=-31&&a<=31,"projection coefficient outside supported range");
                if(a){transform[r].push_back(Term{uint16_t(j),a});++matrix_terms;}
                length+=std::abs(int(a))*D;if(a<0)shifts[r]-=int(a)*int(D);
            }
            lo::require(length>0&&length<=32767,"projection row outside int16 range");
            bound+=uint64_t(length)*length;
        }
        int width=0;for(uint64_t x=bound;x;x>>=1)++width;
        bits=(width+1)/2;mask=(uint64_t(1)<<bits)-1;
        lo::require(b==uint32_t(bits)&&bound<(uint64_t(1)<<48)&&(mask+1)+(bound>>bits)+1<=MAX_ENTRIES,"kernel-table geometry mismatch");
        short_sum=bound<=uint64_t(INT32_MAX);
        // Do not handicap uniform/diagonal controls with redundant feature mixing.
        // Recognize A.T*A exactly using model-only integer arithmetic.
        std::vector<int64_t> metric(size_t(d)*d,0);
        for(const auto& row:transform)for(const auto&a:row)for(const auto&b:row)
            metric[size_t(a.column)*d+b.column]+=int64_t(a.value)*b.value;
        diagonal=true;
        for(uint32_t i=0;i<d;++i)for(uint32_t j=0;j<d;++j)if(i!=j&&metric[size_t(i)*d+j]!=0)diagonal=false;
        if(diagonal){
            diagonal_weights.resize(d);raw_query.resize(d);
            for(uint32_t j=0;j<d;++j)diagonal_weights[j]=uint32_t(metric[size_t(j)*d+j]);
            raw_supports.assign(raw+offset,raw+offset+size_t(ns)*d);
        }
        high.resize(size_t(bound>>bits)+1);low.resize(size_t(mask)+1);
        for(size_t k=0;k<high.size();++k)high[k]=std::exp(-alpha*double(uint64_t(k)<<bits));
        for(size_t k=0;k<low.size();++k)low[k]=std::exp(-alpha*double(k));
        supports.resize(size_t(ns)*m);
        std::vector<double> original_svs(size_t(ns)*m);
        for(uint32_t k=0;k<ns;++k){const uint8_t* q=raw+offset+size_t(k)*d;
            for(uint32_t j=0;j<d;++j)lo::require(q[j]<=D,"raw support outside grid");
            for(uint32_t r=0;r<m;++r){int32_t sum=shifts[r];for(const auto&t:transform[r])sum+=int32_t(t.value)*q[t.column];
                lo::require(sum>=0&&sum<=32767,"support projection overflow");
                supports[size_t(k)*m+r]=int16_t(sum);original_svs[size_t(k)*m+r]=double(sum);
            }
        }
        offset+=size_t(ns)*d;
        const size_t counts=offset;offset+=4*c;
        if(diagonal){
            raw_centers.resize(size_t(c)*d,0.);uint32_t start=0;
            for(uint32_t i=0;i<c;++i){uint32_t n;std::memcpy(&n,raw+counts+4*i,4);
                lo::require(n>0&&n<=ns-start,"invalid class support count");
                for(uint32_t j=0;j<d;++j)for(uint32_t k=start;k<start+n;++k)
                    raw_centers[size_t(i)*d+j]+=double(raw_supports[size_t(k)*d+j])/double(n);
                start+=n;
            }
            lo::require(start==ns,"class support inventory mismatch");
        }
        const size_t coefficients=offset;
        const size_t number_bytes=8ull*((c-1ull)*ns+c*(c-1ull)/2);offset+=number_bytes;
        lo::require(offset+meta==bytes,"unused interaction bytes");
        // Feed unchanged model validation/sparse coefficient construction/voting.
        size_t inner_payload=8+4*c+original_svs.size()*8+number_bytes;
        std::vector<unsigned char> inner(32+meta+inner_payload);
        std::memcpy(inner.data(),"SPCSVM02",8);
        auto put32=[&](size_t p,uint32_t v){std::memcpy(inner.data()+p,&v,4);};
        put32(8,c);put32(12,ns);put32(16,m);put32(20,meta);put32(24,uint32_t(inner_payload));
        std::memcpy(inner.data()+32,raw+offset,meta);size_t out=32+meta;
        std::memcpy(inner.data()+out,&alpha,8);out+=8;
        std::memcpy(inner.data()+out,raw+counts,4*c);out+=4*c;
        std::memcpy(inner.data()+out,original_svs.data(),original_svs.size()*8);out+=original_svs.size()*8;
        std::memcpy(inner.data()+out,raw+coefficients,number_bytes);
        put32(28,lo::crc32(inner.data()+32,inner.size()-32));
        auto prepared=std::make_shared<spm::Model>(inner.data(),inner.size(),false);
        // All exported modes use the checked integer support bank. Retain the
        // original model's validated pair coefficients and centers, not an unused
        // duplicate binary64 support-feature bank. No stock Worker::edge/run is
        // called by this experimental engine; numerical reference lives separately.
        std::vector<double>().swap(prepared->sv);
        model=std::move(prepared);
        state=std::make_unique<spm::Worker>(model);
    }
    void project(const uint8_t* raw){
        if(diagonal&&mode!=4){std::copy(raw,raw+raw_d,raw_query.begin());projection_ops=0;return;}
        projection_ops=matrix_terms;
        for(int r=0;r<rank;++r){int32_t sum=shifts[r];for(const auto&t:transform[r])sum+=int32_t(t.value)*raw[t.column];query[r]=int16_t(sum);}
    }
    uint64_t distance(uint32_t id) const {
        if(diagonal&&mode!=4){
            const uint8_t* s=raw_supports.data()+size_t(id)*raw_d;uint64_t result=0;int f=0;
#if defined(__AVX2__)
            if(mode!=2&&short_sum){
                __m256i acc=_mm256_setzero_si256();
                for(;f+8<=raw_d;f+=8){
                    auto a=_mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(raw_query.data()+f)));
                    auto b=_mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(s+f)));
                    auto delta=_mm256_sub_epi32(a,b);
                    auto terms=_mm256_mullo_epi32(_mm256_mullo_epi32(delta,delta),_mm256_loadu_si256(reinterpret_cast<const __m256i*>(diagonal_weights.data()+f)));
                    acc=_mm256_add_epi32(acc,terms);
                }
                uint32_t v[8];_mm256_storeu_si256(reinterpret_cast<__m256i*>(v),acc);for(auto x:v)result+=x;
            }
#endif
            for(;f<raw_d;++f){int delta=int(raw_query[f])-s[f];result+=uint64_t(diagonal_weights[f])*uint32_t(delta*delta);}
            return result;
        }
        const int16_t* s=supports.data()+size_t(id)*rank;int f=0;uint64_t sum=0;
#if defined(__AVX2__)
        if(mode!=2){
            if(short_sum){
                __m256i acc=_mm256_setzero_si256();
                for(;f+16<=rank;f+=16){
                    auto v=_mm256_sub_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(query.data()+f)),_mm256_loadu_si256(reinterpret_cast<const __m256i*>(s+f)));
                    acc=_mm256_add_epi32(acc,_mm256_madd_epi16(v,v));
                }
                uint32_t a[8];_mm256_storeu_si256(reinterpret_cast<__m256i*>(a),acc);for(auto z:a)sum+=z;
            }else{
                __m256i lo=_mm256_setzero_si256(),hi=_mm256_setzero_si256();
                for(;f+16<=rank;f+=16){
                    auto v=_mm256_sub_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(query.data()+f)),_mm256_loadu_si256(reinterpret_cast<const __m256i*>(s+f)));
                    auto p=_mm256_madd_epi16(v,v);
                    lo=_mm256_add_epi64(lo,_mm256_cvtepu32_epi64(_mm256_castsi256_si128(p)));
                    hi=_mm256_add_epi64(hi,_mm256_cvtepu32_epi64(_mm256_extracti128_si256(p,1)));
                }
                uint64_t a[4],b[4];_mm256_storeu_si256(reinterpret_cast<__m256i*>(a),lo);_mm256_storeu_si256(reinterpret_cast<__m256i*>(b),hi);
                for(int i=0;i<4;++i)sum+=a[i]+b[i];
            }
        }
#endif
        for(;f<rank;++f){int32_t delta=int32_t(query[f])-s[f];sum+=uint64_t(delta*delta);}
        return sum;
    }
    double score(int i,int j){
        auto&w=*state;const auto&p=model->pairs[model->pair_index(i,j)];double sum=0;
        for(size_t t=0;t<p.ids.size();++t){auto id=p.ids[t];
            if(w.epoch[id]!=w.serial){auto S=distance(id);lo::require(S<=bound,"signature out of bounds");
                w.kernel[id]=mode==3?std::exp(-model->gamma*double(S)):high[S>>bits]*low[S&mask];
                w.epoch[id]=w.serial;++kernel_count;
            }
            sum+=p.values[t]*w.kernel[id];
        }
        term_count+=p.ids.size();sum+=p.bias;lo::require(std::isfinite(sum),"nonfinite pair score");return sum;
    }
    int edge(int i,int j){if(i>j)std::swap(i,j);++pair_count;double s=score(i,j);return model->c==2?(s>=0?1:0):(s>0?i:j);}
    int run(const uint8_t*x,int selected){
        mode=selected;kernel_count=pair_count=term_count=0;project(x);auto&w=*state;w.begin(nullptr);
        if(model->c==2){int result=edge(0,1);w.vote.add(0,1,result);return result;}
        auto oracle=[&](int i,int j){return edge(i,j);};
        if(mode==0)return w.vote.run(oracle,priority,1);
        for(int c=0;c<model->c;++c){double v=0;
            if(diagonal&&mode!=4){for(int f=0;f<raw_d;++f){double delta=double(raw_query[f])-raw_centers[size_t(c)*raw_d+f];v+=diagonal_weights[f]*delta*delta;}}
            else {for(int f=0;f<rank;++f){double delta=double(query[f])-model->centers[size_t(c)*rank+f];v+=delta*delta;}}
            priority[c]=v;
        }
        int result=et::beretta(w.vote,oracle,priority,5,w.rounds);
        lo::require(w.vote.certificate()==result,"interaction vote certificate mismatch");return result;
    }
    uint64_t storage()const{
        uint64_t n=sizeof(*this)+model->storage()+state->scratch()+2*(supports.capacity()+query.capacity())+4*shifts.capacity()+8*(high.capacity()+low.capacity()+priority.capacity()+raw_centers.capacity())+4*diagonal_weights.capacity()+raw_supports.capacity()+raw_query.capacity();
        n+=transform.capacity()*sizeof(std::vector<Term>);for(auto&t:transform)n+=t.capacity()*sizeof(Term);return n;
    }
};
}
extern "C" {
int ii_abi(){return 1;}
void* ii_create(const unsigned char* raw,uint64_t n){void*out=nullptr;lo::protect([&]{lo::require(n<=lo::MAX_BYTES,"model cap");out=new ii::Engine(raw,size_t(n));});return out;}
void ii_destroy(void*p){delete static_cast<ii::Engine*>(p);}
int ii_info(void*p,uint64_t*out,int count){return lo::protect([&]{lo::require(p&&out&&count==9,"invalid info");auto&e=*static_cast<ii::Engine*>(p);
uint64_t v[]={uint64_t(e.raw_d),uint64_t(e.rank),uint64_t(e.model->c),uint64_t(e.model->nsv),uint64_t(e.maximum),e.high.size()+e.low.size(),e.bound+1,e.storage(),e.matrix_terms};std::copy(v,v+9,out);});}
int ii_run(void*p,const uint8_t*x,int rows,int d,int mode,int*out,uint64_t*stats,int count){return lo::protect([&]{
lo::require(p,"null interaction engine");auto&e=*static_cast<ii::Engine*>(p);ii::environment();
lo::require(rows>=0&&rows<=65536&&d==e.raw_d&&uint64_t(rows)*d<=8000000&&mode>=0&&mode<=4&&count==4&&stats&&(!rows||(x&&out)),"invalid batch geometry");
for(size_t i=0;i<size_t(rows)*d;++i)lo::require(x[i]<=e.maximum,"input outside integer domain");std::fill(stats,stats+4,0);
for(int r=0;r<rows;++r){out[r]=e.run(x+size_t(r)*d,mode);stats[0]+=e.kernel_count;stats[1]+=e.pair_count;stats[2]+=e.term_count;stats[3]+=e.projection_ops;}
});}
int ii_probe(void*p,const uint8_t*x,int rows,int d,double*out,uint64_t count){return lo::protect([&]{
lo::require(p,"null engine");auto&e=*static_cast<ii::Engine*>(p);ii::environment();
lo::require(rows>=0&&rows<=65536&&d==e.raw_d&&count==uint64_t(rows)*e.model->pairs.size()&&count<=8000000&&(!rows||(x&&out)),"invalid probe geometry");
for(size_t i=0;i<size_t(rows)*d;++i)lo::require(x[i]<=e.maximum,"probe input outside grid");
for(int r=0;r<rows;++r){e.mode=1;e.project(x+size_t(r)*d);e.state->begin(nullptr);for(int i=0;i<e.model->c;++i)for(int j=i+1;j<e.model->c;++j)*out++=e.score(i,j);}
});}
}
