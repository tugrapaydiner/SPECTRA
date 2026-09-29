// Total source-arithmetic evaluator: unchanged compact proof + lossless leaves.
// Exact means the original sequential binary64 operations, not a real-number sum.
#ifndef TT_BASE_RUNTIME
#error TT_BASE_RUNTIME must name the retained certified_trees/runtime.cpp
#endif
#include TT_BASE_RUNTIME
namespace tt {
struct Model {
    std::unique_ptr<tc::Compact> base;
    std::vector<uint16_t> ids16;
    std::vector<uint32_t> ids32;
    std::vector<double> bank, biases;
    double scale=1;
    uint32_t rows=0,unique=0,width=0;
    Model(const uint8_t*raw,size_t bytes) {
        tc::environment();
        tc::need(raw && bytes>=104 && bytes<=64ull*1024*1024,"total model byte cap");
        tc::need(!std::memcmp(raw,"SPCTOT01",8),"unsupported total magic");
        tc::Reader r{raw,bytes,8};
        uint32_t nb=r.value<uint32_t>(),c=r.value<uint32_t>();
        rows=r.value<uint32_t>();unique=r.value<uint32_t>();width=r.value<uint32_t>();
        uint32_t body=r.value<uint32_t>(),crc=r.value<uint32_t>(),flags=r.value<uint32_t>();
        tc::need(c>=2 && c<=64 && unique>=1 && unique<=rows && uint64_t(rows)*c<=500000,"lossless geometry");
        tc::need((flags==0 && width==0 && unique==rows) ||
                 (flags==1 && width==(unique<=65536?2u:4u)),"lossless index width");
        tc::need(body==uint64_t(nb)+8ull*(c+1)+uint64_t(width)*rows+8ull*unique*c && bytes==104ull+body,
                 "total model inventory mismatch");
        tc::need(tc::crc32(raw+104,bytes-104)==crc,"total model CRC mismatch");
        base=std::make_unique<tc::Compact>(raw+104,nb);const auto&m=*base;
        tc::need(m.bits==16 && !(m.flags&2) && m.c==c &&
                 m.trees.back().leaf+(1u<<m.trees.back().depth)==rows,"proof/source shape mismatch");
        tc::need(!std::memcmp(raw+40,m.source.data(),32),"total source binding mismatch");
        // Mathematical verification and both SHA256 bindings occur before loading
        // via VerifiedTotal. Native structural validation alone is not the proof.
        r.offset=104+nb;scale=r.value<double>();r.array(biases,c);
        tc::need(std::isfinite(scale) && scale>0,"invalid original scale");
        for(double b:biases)tc::need(std::isfinite(b),"nonfinite original bias");
        if(width==2) {r.array(ids16,rows);for(auto id:ids16)tc::need(id<unique,"leaf index outside bank");}
        if(width==4) {r.array(ids32,rows);for(auto id:ids32)tc::need(id<unique,"leaf index outside bank");}
        r.array(bank,size_t(unique)*c);tc::need(r.offset==bytes,"unused total model bytes");
        for(double v:bank)tc::need(std::isfinite(v),"nonfinite source leaf");
    }
    uint32_t index(uint32_t leaf)const {
        return width==2 ? ids16[leaf] : width==4 ? ids32[leaf] : leaf;
    }
    void exact_scores(const uint16_t*routes,uint32_t count,uint32_t row,double*out)const {
        const auto&m=*base;std::fill(out,out+m.c,0.);
        // Never sum contrasts here, skip zero leaves or reassociate across trees.
        // Common-mode source roundoff and source tie semantics are preserved.
        for(uint32_t k=0;k<m.t;++k) {
            const uint32_t leaf=m.trees[k].leaf+routes[size_t(k)*count+row];
            const double*values=bank.data()+size_t(index(leaf))*m.c;
            for(uint32_t j=0;j<m.c;++j)out[j]+=values[j];
        }
        for(uint32_t j=0;j<m.c;++j) {
            out[j]=scale*out[j]+biases[j];
            tc::need(std::isfinite(out[j]),"source arithmetic overflow");
        }
    }
    int exact(const uint16_t*routes,uint32_t count,uint32_t row)const {
        double scores[64];exact_scores(routes,count,row,scores);
        return int(std::max_element(scores,scores+base->c)-scores);
    }
    int coarse(const uint16_t*routes,uint32_t count,uint32_t row)const {
        const auto&m=*base;
#if defined(TC_FAST_END) && defined(__AVX2__)
        return m.fast_end(routes,count,row);
#else
        int32_t sums[64];std::copy(m.bias.begin(),m.bias.end(),sums);
        for(uint32_t k=0;k<m.t;++k)m.add(sums,m.trees[k].leaf+routes[size_t(k)*count+row],true);
        return m.settle(sums,m.t);
#endif
    }
    void route(const uint8_t* q,uint32_t count,bool vectorized,std::vector<uint16_t>& routes)const {
        const auto& m=*base;routes.resize(size_t(count)*m.t);
#if defined(__AVX2__)
        if(vectorized){
            constexpr uint32_t pitch=32;
            std::vector<uint8_t> columns(size_t(m.d)*pitch,0),pred(m.predicates.size()*pitch);
            for(uint32_t f=0;f<m.d;++f)for(uint32_t r=0;r<count;++r)columns[size_t(f)*pitch+r]=q[size_t(r)*m.d+f];
            for(size_t k=0;k<m.predicates.size();++k){auto& p=m.predicates[k];__m256i value;
                if(p.cutoff<0)value=_mm256_set1_epi8(1);
                else if(p.cutoff>=int(m.D))value=_mm256_setzero_si256();
                else {
                    auto x=_mm256_xor_si256(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(columns.data()+size_t(p.feature)*pitch)),_mm256_set1_epi8(char(0x80)));
                    auto cutoff=_mm256_set1_epi8(char(p.cutoff^0x80));
                    value=_mm256_and_si256(_mm256_cmpgt_epi8(x,cutoff),_mm256_set1_epi8(1));
                }
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(pred.data()+k*pitch),value);
            }
            for(uint32_t k=0;k<m.t;++k){auto& tree=m.trees[k];
                if(tree.depth<=8){auto leaves=_mm256_setzero_si256();
                    for(uint32_t b=0;b<tree.depth;++b){auto bits=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(pred.data()+size_t(m.refs[tree.split+b])*pitch));
                        leaves=_mm256_or_si256(leaves,_mm256_sll_epi16(bits,_mm_cvtsi32_si128(int(b))));}
                    uint8_t data[32];_mm256_storeu_si256(reinterpret_cast<__m256i*>(data),leaves);
                    for(uint32_t r=0;r<count;++r)routes[size_t(k)*count+r]=data[r];
                }else for(uint32_t r=0;r<count;++r){uint16_t leaf=0;
                    for(uint32_t b=0;b<tree.depth;++b)leaf|=uint16_t(pred[size_t(m.refs[tree.split+b])*pitch+r])<<b;
                    routes[size_t(k)*count+r]=leaf;}
            }
            return;
        }
#endif
        for(uint32_t k=0;k<m.t;++k)for(uint32_t r=0;r<count;++r)routes[size_t(k)*count+r]=m.route(q+size_t(r)*m.d,k);
    }

    void run(const uint8_t*q,uint32_t n,int policy,bool tiled,std::vector<int32_t>&out,uint64_t*stats)const {
        const auto&m=*base;out.resize(n);std::fill(stats,stats+6,0);
        for(uint32_t start=0;start<n;start+=32) {
            uint32_t count=std::min(32u,n-start);std::vector<uint16_t> routes;
            route(q+size_t(start)*m.d,count,tiled,routes);stats[3]+=uint64_t(count)*m.t;
            for(uint32_t row=0;row<count;++row) {
                int winner=-1;
                if(policy!=2) {winner=coarse(routes.data(),count,row);stats[4]+=m.t;if(winner>=0)++stats[0];}
                if((winner<0 && policy!=1) || policy==3) {
                    int original=exact(routes.data(),count,row);stats[5]+=m.t;
                    if(policy==3 && winner>=0)tc::need(winner==original,"certificate/source mismatch");
                    if(winner<0) {winner=original;++stats[1];}
                }
                if(winner<0)++stats[2];out[start+row]=winner;
            }
        }
    }
    void scores(const uint8_t*q,uint32_t n,bool tiled,std::vector<double>&out)const {
        const auto&m=*base;out.resize(size_t(n)*m.c);
        for(uint32_t start=0;start<n;start+=32) {
            uint32_t count=std::min(32u,n-start);std::vector<uint16_t> routes;
            route(q+size_t(start)*m.d,count,tiled,routes);
            for(uint32_t row=0;row<count;++row)exact_scores(routes.data(),count,row,out.data()+size_t(start+row)*m.c);
        }
    }
    uint64_t storage()const {
        return sizeof(*this)+base->storage()+2ull*ids16.capacity()+4ull*ids32.capacity()+8ull*(bank.capacity()+biases.capacity());
    }
};
}
extern "C" {
int tt_abi(){return 1;}
void*tt_create(const uint8_t*raw,uint64_t n){void*out=nullptr;tc::protect([&]{out=new tt::Model(raw,size_t(n));});return out;}
void tt_destroy(void*p){delete static_cast<tt::Model*>(p);}
int tt_info(void*p,uint64_t*out,int n){return tc::protect([&]{tc::need(p&&out&&n==9,"invalid total info");auto&m=*static_cast<tt::Model*>(p);auto&b=*m.base;
    uint64_t v[]={b.d,b.D,b.c,b.t,m.rows,m.unique,m.width,m.bank.size()*8ull,m.storage()};std::copy(v,v+9,out);});}
int tt_run(void*p,const uint8_t*q,int n,int d,int policy,int tiled,int32_t*out,uint64_t*stats,int count){return tc::protect([&]{
    tc::need(p,"closed total engine");auto&m=*static_cast<tt::Model*>(p);auto&b=*m.base;tc::environment();
    tc::need(n>=0&&n<=65536&&d==int(b.d)&&uint64_t(n)*d<=8000000&&policy>=0&&policy<=3&&(tiled==0||tiled==1)&&stats&&count==6&&(!n||(q&&out)),"invalid total buffers");
    for(size_t k=0;k<size_t(n)*d;++k)tc::need(q[k]<=b.D,"input outside integer domain");
    std::vector<int32_t> values;uint64_t work[6];m.run(q,n,policy,tiled,values,work);
    std::copy(values.begin(),values.end(),out);std::copy(work,work+6,stats);
});}
int tt_scores(void*p,const uint8_t*q,int n,int d,int tiled,double*out,uint64_t cells){return tc::protect([&]{
    tc::need(p,"closed total engine");auto&m=*static_cast<tt::Model*>(p);auto&b=*m.base;tc::environment();
    tc::need(n>=0&&n<=65536&&d==int(b.d)&&uint64_t(n)*d<=8000000&&cells==uint64_t(n)*b.c&&cells<=8000000&&(tiled==0||tiled==1)&&(!n||(q&&out)),"invalid score buffers");
    for(size_t k=0;k<size_t(n)*d;++k)tc::need(q[k]<=b.D,"input outside integer domain");
    std::vector<double> values;m.scores(q,n,tiled,values);std::copy(values.begin(),values.end(),out);
});}
}
