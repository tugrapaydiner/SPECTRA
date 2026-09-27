// Additive generic/shared runtime. Historical numerical sources stay unchanged.
// Prepared data is immutable; each worker owns every input-dependent value.
#pragma once
#include <atomic>
#include <cfenv>
#include <memory>

namespace spm {
constexpr size_t MAX_DIM = 4096;
struct Model {
    int c = 0, nsv = 0, d = 0;
    double gamma = 0;
    uint64_t id;
    std::vector<double> sv, centers, values;
    std::vector<uint32_t> offsets, active;
    std::vector<uint8_t> codes;
    std::vector<lo::Pair> pairs;
    bool tables = false, global_codes = false;
    inline static std::atomic<uint64_t> next_id{1};

    Model(const unsigned char* bytes, size_t size, bool requested) : id(next_id++) {
        lo::require(bytes && size >= 24 && size <= lo::MAX_BYTES, "invalid model size");
        const uint16_t one = 1;
        lo::require(*reinterpret_cast<const uint8_t*>(&one) == 1 &&
                    std::numeric_limits<double>::is_iec559 && sizeof(double) == 8,
                    "little-endian IEEE binary64 required");
        auto u32 = [&](size_t offset) { uint32_t v; std::memcpy(&v, bytes + offset, 4); return v; };
        bool v1 = std::memcmp(bytes, "SPCSVM01", 8) == 0;
        lo::require(v1 || std::memcmp(bytes, "SPCSVM02", 8) == 0, "unsupported model format");
        size_t header = v1 ? 24 : 32;
        lo::require(size >= header, "truncated model header");
        uint32_t cc = u32(8), ns = u32(12), dims = v1 ? 16 : u32(16);
        uint32_t labels = v1 ? 0 : u32(20), payload = u32(v1 ? 16 : 24);
        uint32_t crc = u32(v1 ? 20 : 28);
        lo::require(cc >= 2 && cc <= 128 && ns > 0 && ns <= 100000 &&
                    dims > 0 && dims <= MAX_DIM && uint64_t(ns) * dims <= 8000000,
                    "unsupported model geometry");
        uint64_t doubles = 1 + uint64_t(dims)*ns + (cc-1ull)*ns + cc*(cc-1ull)/2;
        lo::require(labels <= 65536 && payload == 8*doubles + 4ull*cc &&
                    size == header + uint64_t(labels) + payload, "model inventory mismatch");
        lo::require(lo::crc32(bytes + header, size - header) == crc, "model CRC mismatch");
        c = int(cc); nsv = int(ns); d = int(dims);
        size_t off = header + labels;
        auto take = [&](void* out, size_t n) {
            lo::require(n <= size - off, "model bounds");
            std::memcpy(out, bytes + off, n); off += n;
        };
        take(&gamma, 8);
        lo::require(std::isfinite(gamma) && gamma > 0, "invalid RBF gamma");
        std::vector<uint32_t> counts(c), starts(c);
        take(counts.data(), 4*c);
        uint64_t total = 0;
        for (int i=0; i<c; ++i) {
            starts[i] = uint32_t(total);
            lo::require(counts[i] > 0 && counts[i] <= ns, "invalid support count");
            total += counts[i];
        }
        lo::require(total == ns, "support count sum mismatch");
        sv.resize(size_t(d)*ns);
        std::vector<double> dense((c-1ull)*ns), bias(c*(c-1)/2);
        for (auto* a : {&sv, &dense, &bias}) {
            take(a->data(), a->size()*8);
            for (double x : *a) lo::require(std::isfinite(x), "nonfinite parameter");
        }
        lo::require(off == size, "unused model bytes");
        std::vector<uint8_t> used(ns, 0);
        pairs.reserve(bias.size());
        int p = 0;
        for (int i=0; i<c; ++i) for (int j=i+1; j<c; ++j) {
            lo::Pair a; a.bias = bias[p++];
            for (auto half : {std::pair<int,int>{i,j-1}, {j,i}})
                for (uint32_t k=starts[half.first]; k<starts[half.first]+counts[half.first]; ++k) {
                    double value = dense[size_t(half.second)*ns+k];
                    if (value != 0.0) { a.ids.push_back(k); a.values.push_back(value); used[k] = 1; }
                }
            double magnitude = std::abs(a.bias), limit = std::numeric_limits<double>::max()/4;
            lo::require(magnitude < limit, "pair bound too large");
            for (double value : a.values) {
                lo::require(std::abs(value) < limit-magnitude, "pair bound too large");
                magnitude += std::abs(value);
            }
            a.ids.shrink_to_fit(); a.values.shrink_to_fit(); pairs.push_back(std::move(a));
        }
        for (uint32_t k=0; k<ns; ++k) if (used[k]) active.push_back(k);
        active.shrink_to_fit();
        centers.resize(size_t(c)*d);
        for (int i=0; i<c; ++i) for (int f=0; f<d; ++f) {
            double sum = 0;
            for (uint32_t k=starts[i]; k<starts[i]+counts[i]; ++k)
                sum += sv[size_t(d)*k+f]/double(counts[i]);
            centers[size_t(i)*d+f] = sum;
        }
        et::FeatureTables trial(d, nsv, sv.data(), requested, false);
        tables = trial.enabled; global_codes = trial.global_codes;
        offsets = std::move(trial.offsets); values = std::move(trial.values); codes = std::move(trial.codes);
        if (tables) std::vector<double>().swap(sv);
    }
    size_t pair_index(int i, int j) const {
        if (i>j) std::swap(i,j);
        return size_t(i)*(2*c-i-1)/2+j-i-1;
    }
    uint64_t storage() const {
        uint64_t n = sizeof(*this) + 8*(sv.capacity()+centers.capacity()+values.capacity()) +
                     4*(offsets.capacity()+active.capacity()) + codes.capacity() + sizeof(lo::Pair)*pairs.capacity();
        for (const auto& p : pairs) n += 4*p.ids.capacity()+8*p.values.capacity();
        return n;
    }
};
using Owner = std::shared_ptr<const Model>;

struct Worker {
    Owner model;
    std::vector<double> kernel, squared, priority;
    std::vector<uint64_t> epoch;
    uint64_t serial = 0, rounds = 0, cost_terms = 0;
    lo::Votes vote;
    lo::Work work;
    bool valid_certificate = false;
    explicit Worker(Owner prepared) : model(std::move(prepared)), kernel(model->nsv),
        squared(model->values.size()), priority(model->c), epoch(model->nsv,0), vote(model->c) {}
    uint64_t scratch() const {
        return sizeof(*this)+8*(kernel.capacity()+squared.capacity()+priority.capacity()+epoch.capacity())+
               vote.known.capacity()+sizeof(int)*(vote.low.capacity()+vote.remain.capacity());
    }
    void begin(const double* x) {
        valid_certificate = false;
        work = lo::Work{}; work.classes = model->c; work.supports = model->nsv;
        rounds = cost_terms = 0;
        if (++serial == 0) { std::fill(epoch.begin(),epoch.end(),0); serial = 1; }
        vote.reset(work);
        if (model->tables) for (int d=0; d<model->d; ++d)
            for (uint32_t k=model->offsets[d]; k<model->offsets[d+1]; ++k) {
                double t = x[d]-model->values[k]; squared[k] = t*t;
            }
    }
    template<int D, bool Tables, bool Global>
    void distances(const uint32_t* ids, int count, const double* x, double* out) const {
        const int dimensions = D ? D : model->d;
#if defined(__AVX2__)
        if (count == 4) {
            __m256d sum = _mm256_setzero_pd();
            for (int d=0; d<dimensions; ++d) {
                __m256d term;
                if constexpr (Tables) {
                    const double* bank = squared.data()+(Global ? 0 : model->offsets[d]);
                    term = _mm256_setr_pd(bank[model->codes[size_t(ids[0])*dimensions+d]],
                        bank[model->codes[size_t(ids[1])*dimensions+d]],
                        bank[model->codes[size_t(ids[2])*dimensions+d]],
                        bank[model->codes[size_t(ids[3])*dimensions+d]]);
                } else {
                    __m256d diff = _mm256_sub_pd(_mm256_set1_pd(x[d]),
                        _mm256_setr_pd(model->sv[size_t(ids[0])*dimensions+d], model->sv[size_t(ids[1])*dimensions+d],
                            model->sv[size_t(ids[2])*dimensions+d], model->sv[size_t(ids[3])*dimensions+d]));
                    term = _mm256_mul_pd(diff,diff);
                }
                sum = _mm256_add_pd(sum,term);
            }
            _mm256_storeu_pd(out,sum); return;
        }
#endif
        for (int k=0; k<count; ++k) {
            double sum = 0;
            for (int d=0; d<dimensions; ++d) {
                if constexpr (Tables) sum += squared[(Global ? 0 : model->offsets[d])+model->codes[size_t(ids[k])*dimensions+d]];
                else { double t=x[d]-model->sv[size_t(ids[k])*dimensions+d]; sum+=t*t; }
            }
            out[k]=sum;
        }
    }
    template<int D> void distance_kind(const uint32_t* ids, int count, const double* x, double* out) const {
        if (!model->tables) distances<D,false,false>(ids,count,x,out);
        else if (model->global_codes) distances<D,true,true>(ids,count,x,out);
        else distances<D,true,false>(ids,count,x,out);
    }
    void prepare(const std::vector<uint32_t>& ids, const double* x) {
        uint32_t todo[4]; int size=0;
        auto flush = [&] {
            double dist[4];
            if (model->d==16) distance_kind<16>(todo,size,x,dist);
            else distance_kind<0>(todo,size,x,dist);
            for (int k=0;k<size;++k) {
                double v=std::exp(-model->gamma*dist[k]);
                lo::require(std::isfinite(v)&&v>=0&&v<=1,"invalid kernel");
                kernel[todo[k]]=v; epoch[todo[k]]=serial; ++work.kernels;
            }
            size=0;
        };
        for (uint32_t id:ids) if (epoch[id]!=serial) { todo[size++]=id; if(size==4) flush(); }
        if(size) flush();
    }
    int edge(int i,int j,const double* x) {
        if(i>j)std::swap(i,j);
        const auto& p=model->pairs[model->pair_index(i,j)];
        prepare(p.ids,x);
        double sum=0;
        for(size_t t=0;t<p.ids.size();++t)sum+=p.values[t]*kernel[p.ids[t]];
        sum+=p.bias; work.terms+=p.ids.size(); lo::require(std::isfinite(sum),"nonfinite pair decision");
        return model->c==2?(sum>=0?1:0):(sum>0?i:j);
    }

    // Optional binary specialization. There is only one classifier, hence no
    // within-request cross-pair kernel reuse. Preserve feature/term order and
    // scalar libm exp; do not turn a sum of products into an FMA/reduction tree.
    double binary_stream_score(const double* x) {
        const auto& p=model->pairs[0];
        double sum=0;
        for(size_t first=0;first<p.ids.size();first+=4) {
            int count=int(std::min(size_t(4),p.ids.size()-first));
            double dist[4];
            if(model->d==16)distance_kind<16>(p.ids.data()+first,count,x,dist);
            else distance_kind<0>(p.ids.data()+first,count,x,dist);
            for(int k=0;k<count;++k) {
                double value=std::exp(-model->gamma*dist[k]);
                lo::require(std::isfinite(value)&&value>=0&&value<=1,"invalid streamed kernel");
                sum+=p.values[first+size_t(k)]*value;
            }
        }
        sum+=p.bias;
        lo::require(std::isfinite(sum),"nonfinite streamed pair decision");
        return sum;
    }
    int finish_binary(double score) {
        const auto terms=model->pairs[0].ids.size();
        work.kernels=terms;work.terms=terms;
        const int winner=score>=0?1:0; // The original binary SVC zero convention.
        vote.add(0,1,winner);
        lo::require(vote.certificate()==winner,"binary vote certificate mismatch");
        valid_certificate=true;return winner;
    }
#if defined(__AVX2__)
    // Each SIMD lane is one independent input, NOT a partial floating-point sum.
    // Input feature packing is caller-owned temporary storage (4*d doubles).
    // This direct-distance path never reads/writes the request kernel cache.
    void binary_four_scores(const double* x,double* packed,double* out) const {
        const auto& p=model->pairs[0];const auto d=model->d;
        for(int f=0;f<d;++f) {
            packed[4*f]=x[f];packed[4*f+1]=x[d+f];
            packed[4*f+2]=x[2*d+f];packed[4*f+3]=x[3*d+f];
        }
        __m256d score=_mm256_setzero_pd();
        for(size_t t=0;t<p.ids.size();++t) {
            const double* sv=model->sv.data()+size_t(p.ids[t])*d;
            __m256d distance=_mm256_setzero_pd();
            for(int f=0;f<d;++f) {
                const __m256d diff=_mm256_sub_pd(_mm256_loadu_pd(packed+4*f),_mm256_set1_pd(sv[f]));
                distance=_mm256_add_pd(distance,_mm256_mul_pd(diff,diff));
            }
            double terms[4];_mm256_storeu_pd(terms,distance);
            for(int lane=0;lane<4;++lane) {
                terms[lane]=std::exp(-model->gamma*terms[lane]);
                lo::require(std::isfinite(terms[lane])&&terms[lane]>=0&&terms[lane]<=1,
                            "invalid row-lane kernel");
            }
            score=_mm256_add_pd(score,_mm256_mul_pd(_mm256_set1_pd(p.values[t]),_mm256_loadu_pd(terms)));
        }
        score=_mm256_add_pd(score,_mm256_set1_pd(p.bias));
        _mm256_storeu_pd(out,score);
        for(int lane=0;lane<4;++lane)lo::require(std::isfinite(out[lane]),"nonfinite row-lane decision");
    }
#endif
    // Experimental: selection only. This score never participates in acceptance.
    int cost_aware(const double* x) {
        const int c=model->c;
        for(int step=0;step<c*(c-1)/2;++step) {
            int pivot=-1;
            for(int i=0;i<c;++i)if(vote.remain[i]>0) {
                if(pivot<0 || vote.low[i]+vote.remain[i]>vote.low[pivot]+vote.remain[pivot] ||
                   (vote.low[i]+vote.remain[i]==vote.low[pivot]+vote.remain[pivot] &&
                    (priority[i]<priority[pivot] || (priority[i]==priority[pivot]&&i<pivot))))pivot=i;
            }
            lo::require(pivot>=0,"no unresolved pivot");
            int next=-1; uint64_t best=std::numeric_limits<uint64_t>::max();
            for(int j=0;j<c;++j)if(j!=pivot && vote.known[pivot*c+j]<0) {
                const auto& p=model->pairs[model->pair_index(pivot,j)];
                uint64_t missing=0;
                for(uint32_t id:p.ids) {missing+=epoch[id]!=serial;++cost_terms;}
                uint64_t score=missing*(uint64_t(model->d)+16)+p.ids.size();
                if(score<best || (score==best&&(next<0||j<next))) {best=score;next=j;}
            }
            lo::require(next>=0,"no unresolved comparison");
            vote.add(pivot,next,edge(pivot,next,x));
            int found=vote.certificate(); if(found>=0)return found;
        }
        throw std::logic_error("cost scheduler failed to terminate");
    }
    int run(const double* x,int mode,int hint) {
        begin(x);
        if(mode==7) {
            if(model->c==2)return finish_binary(binary_stream_score(x));
            mode=5; // Explicitly preserve the old scheduler for multiclass inputs.
        }
        std::fill(priority.begin(),priority.end(),0);
        if(mode==1||mode>=4||hint>=0)for(int i=0;i<model->c;++i) {
            double sum=0;
            for(int d=0;d<model->d;++d){double t=x[d]-model->centers[size_t(i)*model->d+d];sum+=t*t;}
            priority[i]=std::isfinite(sum)?sum:std::numeric_limits<double>::infinity();
        }
        if(hint>=0)priority[hint]=-1;
        if(mode==0)prepare(model->active,x);
        auto pair=[&](int i,int j){return edge(i,j,x);};
        int result=mode==6?cost_aware(x):(mode>=3?et::beretta(vote,pair,priority,mode,rounds):
            vote.run(pair,priority,mode==0?1:(mode==1?2:6)));
        valid_certificate=true;return result;
    }
};
}

extern "C" {
int sp_shared_abi(){return 1;}
void* sp_model_create(const unsigned char* bytes,uint64_t size,int tables) {
    void* result=nullptr;
    lo::protect([&]{lo::require(tables==0||tables==1,"invalid table option");
        lo::require(std::fegetround()==FE_TONEAREST,"round-to-nearest required");
        auto model=std::make_shared<const spm::Model>(bytes,size,bool(tables));
        result=new spm::Owner(std::move(model));});
    return result;
}
void sp_model_destroy(void* p){delete static_cast<spm::Owner*>(p);}
void* sp_worker_create(void* p) {
    void* result=nullptr;lo::protect([&]{lo::require(p,"null model");
        result=new spm::Worker(*static_cast<spm::Owner*>(p));});return result;
}
void sp_worker_destroy(void* p){delete static_cast<spm::Worker*>(p);}
int sp_model_info(void* p,uint64_t* out,int count){return lo::protect([&]{
    lo::require(p&&out&&count==7,"invalid model info");const auto&m=**static_cast<spm::Owner*>(p);
    uint64_t values[]={m.id,uint64_t(m.d),uint64_t(m.c),uint64_t(m.nsv),m.storage(),uint64_t(m.tables),m.values.size()};
    std::copy(values,values+7,out);});}
int sp_worker_info(void* p,uint64_t* out,int count){return lo::protect([&]{
    lo::require(p&&out&&count==3,"invalid worker info");auto&w=*static_cast<spm::Worker*>(p);
    out[0]=w.model->id;out[1]=w.model->storage();out[2]=w.scratch();});}
int sp_worker_run(void* p,const double* inputs,int rows,int features,int mode,int hint,
                  int* output,int capacity,uint64_t* stats,int stats_count,int certificate) {
    if(p)static_cast<spm::Worker*>(p)->valid_certificate=false;
    return lo::protect([&]{
        lo::require(p,"null worker");auto&w=*static_cast<spm::Worker*>(p);
        lo::require(rows>=0&&rows<=65536&&features==w.model->d&&uint64_t(rows)*features<=8000000&&capacity==rows&&stats&&stats_count==5&&
                    (certificate==0||(certificate==1&&rows==1)),
                    "invalid worker geometry");
        lo::require(mode>=0&&mode<=7&&hint>=-1&&hint<w.model->c,"invalid worker schedule/hint");
        lo::require(std::fegetround()==FE_TONEAREST,"round-to-nearest required");
        lo::require(!rows||(inputs&&output),"null worker buffers");
        for(size_t i=0;i<size_t(rows)*features;++i)lo::require(std::isfinite(inputs[i]),"nonfinite input");
        uint64_t totals[5]={};
        try {
            int row=0;
#if defined(__AVX2__)
            if(mode==7 && w.model->c==2 && !w.model->tables && rows>=4) {
                // Allocate once per batch, never once per support or prediction.
                // At d<=4096, this scratch is bounded by 128 KiB.
                std::vector<double> packed(size_t(features)*4);
                for(;row+4<=rows;row+=4) {
                    double scores[4];
                    w.binary_four_scores(inputs+size_t(row)*features,packed.data(),scores);
                    for(int lane=0;lane<4;++lane) {
                        w.begin(inputs+size_t(row+lane)*features);
                        output[row+lane]=w.finish_binary(scores[lane]);
                        totals[0]+=w.work.kernels;totals[1]+=w.work.pairs;totals[2]+=w.work.terms;
                        totals[3]+=w.work.cert_checks;totals[4]+=w.cost_terms;
                    }
                }
            }
#endif
            for(;row<rows;++row){output[row]=w.run(inputs+size_t(row)*features,mode,hint);
                totals[0]+=w.work.kernels;totals[1]+=w.work.pairs;totals[2]+=w.work.terms;
                totals[3]+=w.work.cert_checks;totals[4]+=w.cost_terms;}
        } catch(...) {w.valid_certificate=false;throw;}
        if(!certificate)w.valid_certificate=false;
        std::copy(totals,totals+5,stats);
    });
}
int sp_worker_certificate(void* p,int8_t* out,int count){return lo::protect([&]{
    lo::require(p&&out,"invalid certificate");auto&w=*static_cast<spm::Worker*>(p);int c=w.model->c;
    lo::require(w.valid_certificate&&count==c*(c-1)/2,"no certificate or size mismatch");
    for(int i=0,k=0;i<c;++i)for(int j=i+1;j<c;++j)out[k++]=w.vote.known[i*c+j];});}
}
