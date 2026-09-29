// Integer-metric RBF execution. A newly learned kernel, not a silent substitute
// for an old unweighted model. All per-feature weights are nonnegative integers.
#ifndef LM_BASE_RUNTIME
#error Define LM_BASE_RUNTIME to SPECTRA's original native runtime.cpp
#endif
#include LM_BASE_RUNTIME
namespace lm {
constexpr uint32_t MAX_TABLE = 4194304;
struct Engine {
 std::shared_ptr<const spm::Model> model;
 std::unique_ptr<spm::Worker> state;
 std::vector<uint16_t> weights;
 std::vector<uint8_t> codes;
 std::vector<double> table;
 int maximum=0,common=0;bool vector_safe=true;
 const uint8_t* query=nullptr;int mode=1;
 uint64_t kernels=0,pairs=0,terms=0;
 Engine(const unsigned char* bytes,size_t size) {
  lo::require(bytes&&size>=28&&size<=lo::MAX_BYTES,"invalid metric model size");
  lo::require(!std::memcmp(bytes,"SPLMET01",8),"wrong learned-metric magic");
  auto u32=[&](size_t p){uint32_t v;std::memcpy(&v,bytes+p,4);return v;};
  uint32_t limit=u32(8),d=u32(12),mass=u32(16),inner_size=u32(20),crc=u32(24);
  lo::require(limit>=1&&limit<=255&&d>=1&&d<=4096&&mass>0&&
    uint64_t(mass)*limit*limit+1<=MAX_TABLE,"unsupported metric/table geometry");
  lo::require(size==28+uint64_t(d)*2+inner_size&&inner_size>=32,"metric inventory mismatch");
  lo::require(lo::crc32(bytes+28,size-28)==crc,"metric CRC mismatch");
  weights.resize(d);std::memcpy(weights.data(),bytes+28,d*2);
  uint32_t total=0;
  for(uint16_t w:weights){lo::require(w<=255,"metric weight outside0..255");total+=w;vector_safe=vector_safe&&w<=127;}
  lo::require(total==mass,"metric mass mismatch");
  model=std::make_shared<const spm::Model>(bytes+28+d*2,inner_size,false);
  lo::require(model->d==int(d),"inner/outer feature mismatch");
  maximum=int(limit);codes.resize(model->sv.size());
  for(size_t i=0;i<codes.size();++i){double x=model->sv[i];
   lo::require(x>=0&&x<=maximum&&x==std::floor(x),"support is not an exact grid code");codes[i]=uint8_t(x);}
  common=weights[0];for(uint16_t w:weights)if(w!=common){common=0;break;}
  uint32_t table_max=(common?d:mass)*limit*limit;
  table.resize(size_t(table_max)+1);
  for(uint32_t s=0;s<=table_max;++s){
   // For a common weight retain the exact original integer product in argument.
   uint32_t signature=common?uint32_t(common)*s:s;
   table[s]=std::exp(-model->gamma*double(signature));
  }
  state=std::make_unique<spm::Worker>(model);
 }
 uint32_t signature(uint32_t id)const {
  const uint8_t* support=codes.data()+size_t(id)*model->d;
  uint32_t result=0;int f=0;
#if defined(__AVX2__)
  if(mode!=2&&vector_safe){
   __m256i sums=_mm256_setzero_si256();
   for(;f+16<=model->d;f+=16){
    auto q=_mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(query+f)));
    auto s=_mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(support+f)));
    auto delta=_mm256_sub_epi16(q,s);auto scaled=delta;
    if(!common)scaled=_mm256_mullo_epi16(delta,_mm256_loadu_si256(reinterpret_cast<const __m256i*>(weights.data()+f)));
    sums=_mm256_add_epi32(sums,_mm256_madd_epi16(delta,scaled));
   }
   uint32_t values[8];_mm256_storeu_si256(reinterpret_cast<__m256i*>(values),sums);
   for(auto x:values)result+=x;
  }
#endif
  for(;f<model->d;++f){int delta=int(query[f])-support[f];result+=(common?1:weights[f])*uint32_t(delta*delta);}
  return result;
 }
 double score(int i,int j){
  auto& w=*state;const auto& p=model->pairs[model->pair_index(i,j)];double total=0;
  for(size_t t=0;t<p.ids.size();++t){uint32_t id=p.ids[t];
   if(w.epoch[id]!=w.serial){uint32_t s=signature(id);
    lo::require(s<table.size(),"signature outside table");
    w.kernel[id]=mode==3?std::exp(-model->gamma*double(common?uint32_t(common)*s:s)):table[s];
    w.epoch[id]=w.serial;++kernels;}
   total+=p.values[t]*w.kernel[id];
  }
  terms+=p.ids.size();total+=p.bias;lo::require(std::isfinite(total),"nonfinite metric margin");return total;
 }
 int edge(int i,int j){if(i>j)std::swap(i,j);++pairs;double v=score(i,j);return model->c==2?(v>=0?1:0):(v>0?i:j);}
 int run(const uint8_t* x,int selected){
  query=x;mode=selected;kernels=pairs=terms=0;auto& w=*state;w.begin(nullptr);
  if(model->c==2){int winner=edge(0,1);w.vote.add(0,1,winner);return winner;}
  auto oracle=[&](int i,int j){return edge(i,j);};
  if(mode==0)return w.vote.run(oracle,w.priority,1);
  for(int i=0;i<model->c;++i){double sum=0;
   for(int f=0;f<model->d;++f){double v=double(query[f])-model->centers[size_t(i)*model->d+f];sum+=weights[f]*v*v;}
   w.priority[i]=std::isfinite(sum)?sum:INFINITY;
  }
  int winner=et::beretta(w.vote,oracle,w.priority,5,w.rounds);
  lo::require(w.vote.certificate()==winner,"metric vote certificate inconsistent");return winner;
 }
 uint64_t storage()const{return sizeof(*this)+model->storage()+state->scratch()+weights.capacity()*2+codes.capacity()+table.capacity()*8;}
};
}
extern "C" {
int lm_abi(){return 1;}
void* lm_create(const unsigned char* bytes,uint64_t size){void* out=nullptr;lo::protect([&]{
 lo::require(std::fegetround()==FE_TONEAREST,"round-to-nearest required");out=new lm::Engine(bytes,size_t(size));});return out;}
void lm_destroy(void* p){delete static_cast<lm::Engine*>(p);}
int lm_info(void* p,uint64_t* out,int count){return lo::protect([&]{
 lo::require(p&&out&&count==7,"invalid metric info");auto&e=*static_cast<lm::Engine*>(p);
 uint64_t v[]={uint64_t(e.model->d),uint64_t(e.model->c),uint64_t(e.model->nsv),uint64_t(e.maximum),e.table.size(),e.storage(),uint64_t(e.common)};std::copy(v,v+7,out);
});}
int lm_run(void* p,const uint8_t* x,int rows,int d,int mode,int* out,uint64_t* stats,int count){return lo::protect([&]{
 lo::require(p,"null metric engine");auto&e=*static_cast<lm::Engine*>(p);
 lo::require(std::fegetround()==FE_TONEAREST,"round-to-nearest required");
 lo::require(rows>=0&&rows<=65536&&d==e.model->d&&uint64_t(rows)*d<=8000000&&(!rows||(x&&out))&&stats&&count==4&&mode>=0&&mode<=3,"invalid metric batch");
 for(size_t i=0;i<size_t(rows)*d;++i)lo::require(x[i]<=e.maximum,"input outside declared integer grid");
 std::fill(stats,stats+4,0);
 for(int r=0;r<rows;++r){out[r]=e.run(x+size_t(r)*d,mode);stats[0]+=e.kernels;stats[1]+=e.pairs;stats[2]+=e.terms;stats[3]+=e.state->work.cert_checks;}
});}
int lm_probe(void* p,const uint8_t* x,int rows,int d,double* out,uint64_t count){return lo::protect([&]{
 lo::require(p,"null metric engine");auto&e=*static_cast<lm::Engine*>(p);
 lo::require(rows>=0&&rows<=65536&&d==e.model->d&&count==uint64_t(rows)*e.model->pairs.size()&&count<=8000000&&(!rows||(x&&out)),"invalid metric probe");
 lo::require(std::fegetround()==FE_TONEAREST,"round-to-nearest required");
 for(size_t k=0;k<size_t(rows)*d;++k)lo::require(x[k]<=e.maximum,"input outside declared integer grid");
 for(int r=0;r<rows;++r){e.query=x+size_t(r)*d;e.mode=1;e.state->begin(nullptr);
  for(int i=0;i<e.model->c;++i)for(int j=i+1;j<e.model->c;++j)*out++=e.score(i,j);}
});}
}
