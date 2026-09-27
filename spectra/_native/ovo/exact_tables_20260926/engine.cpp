// Optional additive engine: historical loader, coefficient order, exp and votes
// are included unchanged. Only distance computation and schedule selection vary.
#include "../lazy_ovo_20260926/engine.cpp"
#include "exact_tables.hpp"
#include "schedulers.hpp"
namespace et {
struct Engine:lo::Engine {
    FeatureTables table;uint64_t rounds=0;
    std::vector<uint8_t> stamps;uint8_t token=0;bool compact=false;
    Engine(const char*path,int backend):lo::Engine(path,false),table(16,nsv,sv.data(),backend%3>0,backend%3==2),compact(backend>=3){
        if(table.enabled)std::vector<double>().swap(sv);
        if(compact){stamps.resize(nsv,0);std::vector<uint64_t>().swap(epoch);}
    }
    bool seen(uint32_t id)const{return compact?stamps[id]==token:epoch[id]==serial;}
    void mark(uint32_t id){if(compact)stamps[id]=token;else epoch[id]=serial;}
    void distances(const uint32_t*ids,int count,const float*x,double*out){
        if(table.enabled){table.batch(ids,count,out);return;}
#if defined(__AVX2__)
        if(count==4){
            __m256d sum=_mm256_setzero_pd();
            for(int d=0;d<16;++d){
                __m256d a=_mm256_sub_pd(_mm256_set1_pd(double(x[d])),
                    _mm256_setr_pd(sv[16*ids[0]+d],sv[16*ids[1]+d],sv[16*ids[2]+d],sv[16*ids[3]+d]));
                sum=_mm256_add_pd(sum,_mm256_mul_pd(a,a));
            }
            _mm256_storeu_pd(out,sum);return;
        }
#endif
        for(int k=0;k<count;++k){double sum=0;for(int d=0;d<16;++d){double a=double(x[d])-sv[16*ids[k]+d];sum+=a*a;}out[k]=sum;}
    }
    void prepare(const std::vector<uint32_t>&ids,const float*x){
        if(!table.enabled&&!compact){prepare_kernels(ids,x,true);return;}
        uint32_t todo[4];int m=0;
        auto flush=[&](){double distance[4];distances(todo,m,x,distance);
            for(int k=0;k<m;++k){double v=std::exp(-gamma*distance[k]);lo::require(std::isfinite(v)&&v>=0&&v<=1,"invalid table kernel");
                kernel[todo[k]]=v;mark(todo[k]);++work.kernels;}m=0;};
        for(uint32_t id:ids){if(!seen(id)){todo[m++]=id;if(m==4)flush();}}
        if(m)flush();
    }
    int edge2(int i,int j,const float*x){
        if(i>j)std::swap(i,j);
        auto&p=pair[pair_index(i,j)];prepare(p.ids,x);
        double sum=0;for(size_t t=0;t<p.ids.size();++t)sum+=p.values[t]*kernel[p.ids[t]];
        sum+=p.bias;work.terms+=p.ids.size();lo::require(std::isfinite(sum),"nonfinite table pair");
        return c==2?(sum>=0?1:0):(sum>0?i:j);
    }
    void begin(const float*x,int n){
        valid_certificate=false;lo::require(x&&n==16,"sixteen features required");
        for(int d=0;d<16;++d)lo::require(std::isfinite(x[d]),"nonfinite input");
        work=lo::Work{};rounds=0;work.classes=c;work.supports=nsv;
        if(++serial==0){std::fill(epoch.begin(),epoch.end(),0);serial=1;}
        if(compact&&++token==0){std::fill(stamps.begin(),stamps.end(),0);token=1;}
        vote.reset(work);table.prepare(x);
    }
    int run2(const float*x,int n,int schedule,int hint){
        lo::require(schedule>=0&&schedule<=5&&hint>=-1&&hint<c,"invalid schedule/hint");begin(x,n);
        std::fill(priority.begin(),priority.end(),0);
        if(schedule==1||schedule>=4||hint>=0){
            for(int i=0;i<c;++i){double q=0;for(int d=0;d<16;++d){double t=double(x[d])-centers[i*16+d];q+=t*t;}
                priority[i]=std::isfinite(q)?q:std::numeric_limits<double>::infinity();}
        }
        if(hint>=0)priority[hint]=-1.0;
        if(schedule==0)prepare(active,x);
        auto edge=[&](int i,int j){return edge2(i,j,x);};
        int result=schedule>=3?beretta(vote,edge,priority,schedule,rounds):vote.run(edge,priority,schedule==0?1:(schedule==1?2:6));
        valid_certificate=true;return result;
    }
};
}
extern "C" {
void* et_create(const char*path,int enabled){void*out=nullptr;lo::protect([&]{lo::require(enabled>=0&&enabled<=5,"invalid table option");out=new et::Engine(path,enabled);});return out;}
void et_destroy(void*p){delete static_cast<et::Engine*>(p);}
const char*et_error(){return lo::error.c_str();}
int et_run(void*p,const float*x,int n,int schedule,int hint,int*out,uint64_t*stats,int cap){return lo::protect([&]{
    lo::require(p&&out&&stats&&cap==10,"invalid call outputs");auto&e=*static_cast<et::Engine*>(p);int r=e.run2(x,n,schedule,hint);
    uint64_t s[]={e.work.kernels,e.work.pairs,e.work.terms,e.work.cert_checks,e.table.enabled?e.table.values.size():0,e.rounds,uint64_t(e.c),uint64_t(e.nsv),uint64_t(e.table.enabled),uint64_t(schedule)};
    std::copy(s,s+10,stats);*out=r;
});}
int et_info(void*p,uint64_t*out,int n){return lo::protect([&]{lo::require(p&&out&&n==8,"invalid info");auto&e=*static_cast<et::Engine*>(p);
 uint64_t data[]={uint64_t(e.c),uint64_t(e.nsv),uint64_t(e.table.enabled),e.table.values.size(),
 e.resident_payload()+sizeof(et::Engine)-sizeof(lo::Engine)+e.table.resident()-sizeof(et::FeatureTables),e.scratch_payload()+e.table.scratch()+e.stamps.capacity(),e.table.codes.size(),8*e.table.values.size()};
 std::copy(data,data+8,out);
});}
int et_cache_info(void*p,uint64_t*out,int cap){return lo::protect([&]{lo::require(p&&out&&cap==2,"invalid cache info");auto&e=*static_cast<et::Engine*>(p);out[0]=e.compact;out[1]=e.stamps.capacity();});}
int et_table_info(void*p,uint64_t*out,int cap){return lo::protect([&]{lo::require(p&&out&&cap==4,"invalid table info");auto&e=*static_cast<et::Engine*>(p);
 uint64_t values[]={uint64_t(e.table.global_codes),e.table.prefix_values.size(),e.table.prefix_codes.size(),2*e.table.prefix_values.size()};std::copy(values,values+4,out);
});}
int et_certificate(void*p,int8_t*out,int count){return lo::protect([&]{lo::require(p&&out,"invalid certificate");auto&e=*static_cast<et::Engine*>(p);
 lo::require(e.valid_certificate&&count==e.c*(e.c-1)/2,"no certificate or size mismatch");
 for(int i=0,k=0;i<e.c;++i)for(int j=i+1;j<e.c;++j)out[k++]=e.vote.known[i*e.c+j];
});}
int et_kernels(void*p,const float*x,int n,double*out,int cap){return lo::protect([&]{lo::require(p&&out,"invalid kernel probe");auto&e=*static_cast<et::Engine*>(p);
 lo::require(cap==e.nsv,"invalid kernel capacity");e.begin(x,n);
 // Only the diagnostic path allocates this complete inventory.
 std::vector<uint32_t>ids(e.nsv);for(int i=0;i<e.nsv;++i)ids[i]=i;e.prepare(ids,x);std::copy(e.kernel.begin(),e.kernel.end(),out);
});}
int et_tournament(int c,const int8_t*all,int n,int seed,int mode,int*out,int8_t*known,int*queries){return lo::protect([&]{
 lo::require(c>=2&&c<=128&&all&&out&&known&&queries&&n==c*(c-1)/2&&seed>=0&&seed<c&&mode>=0&&mode<=5,"invalid tournament arguments");
 lo::Votes v(c);lo::Work work;v.reset(work);std::vector<double>priority(c,1);priority[seed]=0;
 for(int i=0,k=0;i<c;++i)for(int j=i+1;j<c;++j,++k)lo::require(all[k]==i||all[k]==j,"invalid tournament edge");
 auto edge=[&](int i,int j){if(i>j)std::swap(i,j);return int(all[i*(2*c-i-1)/2+j-i-1]);};uint64_t rounds=0;
 *out=mode>=3?et::beretta(v,edge,priority,mode,rounds):v.run(edge,priority,mode==0?1:(mode==1?2:6));
 *queries=int(work.pairs);for(int i=0,k=0;i<c;++i)for(int j=i+1;j<c;++j)known[k++]=v.known[i*c+j];
});}
int et_feature_probe(int d,int n,const double*raw,const float*x,double*out,int*enabled,uint64_t*size){return lo::protect([&]{
 lo::require(d>0&&d<=4096&&n>0&&n<=100000&&uint64_t(d)*n<=8000000&&raw&&x&&out&&enabled&&size,"invalid feature probe");
 for(int j=0;j<d;++j)lo::require(std::isfinite(x[j]),"nonfinite feature probe input");
 for(uint64_t k=0;k<uint64_t(d)*n;++k)lo::require(std::isfinite(raw[k]),"nonfinite feature bank");
 et::FeatureTables t(d,n,raw,true,true);*enabled=t.enabled;*size=t.resident()+t.scratch();t.prepare(x);
 for(int i=0;i<n;i+=4){int count=std::min(4,n-i);uint32_t ids[]={uint32_t(i),uint32_t(i+1),uint32_t(i+2),uint32_t(i+3)};
  if(t.enabled)t.batch(ids,count,out+i);else for(int j=0;j<count;++j){double s=0;for(int k=0;k<d;++k){double a=double(x[k])-raw[(i+j)*d+k];s+=a*a;}out[i+j]=s;}
 }
});}
}
