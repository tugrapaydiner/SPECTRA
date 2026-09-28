// Learned radial mixture -> one finite-signature lookup. Training is not here.
// Same-model direct computation is retained; no reference-libm accuracy assumption
// is needed for dyadic-domain identity in the SAME runtime/FP environment.
#ifndef MK_BASE_RUNTIME
#error MK_BASE_RUNTIME required
#endif
#include MK_BASE_RUNTIME
namespace mk {
constexpr uint64_t MAX_TABLE=1048576;
inline void environment(){
 lo::require(std::fegetround()==FE_TONEAREST,"round-to-nearest required");
 volatile double t=std::numeric_limits<double>::denorm_min();volatile double two=2.;
 double r=t*two;lo::require(r>0.&&t!=0.,"gradual underflow required");
}
inline unsigned popcount(uint64_t x){
#if defined(__POPCNT__)
 return __builtin_popcountll(x);
#else
 x-=((x>>1)&0x5555555555555555ULL);x=(x&0x3333333333333333ULL)+((x>>2)&0x3333333333333333ULL);
 return unsigned((((x+(x>>4))&0x0f0f0f0f0f0f0f0fULL)*0x0101010101010101ULL)>>56);
#endif
}
struct Engine {
 spm::Owner model;spm::Worker worker;
 int power,D,words;bool admitted=false;
 std::vector<double> scales,weights,table,values;
 std::vector<uint8_t> codes,query;
 std::vector<uint32_t> counts;std::vector<int16_t> lane_counts;int budget=0;bool unit_counts=true;
 std::vector<uint64_t> bits,qbits,epochs;
 uint64_t serial=0,lookups=0,exps=0,edges=0,fallback_rows=0;
 Engine(const unsigned char* raw,size_t bytes,int p,int denominator,const double* g,const double* w,int components,const uint32_t* multiplicity,int width):
  model(std::make_shared<const spm::Model>(raw,bytes,false)),worker(model),power(p),D(denominator),words((model->d+63)/64){
   lo::require(p==1||p==2,"invalid distance power");lo::require(D>0&&D<=128&&(D&(D-1))==0,"invalid dyadic denominator");
   lo::require(components>=1&&components<=16&&g&&w,"invalid component inventory");double sum=0;
   for(int k=0;k<components;++k){lo::require(std::isfinite(g[k])&&g[k]>0&&std::isfinite(w[k])&&w[k]>=0&&w[k]<=1,"invalid component");scales.push_back(g[k]);weights.push_back(w[k]);sum+=w[k];}
   lo::require(sum>0&&sum<=1+0x1p-40,"invalid mixture mass");
   values.resize(model->nsv);epochs.resize(model->nsv,0);query.resize(model->d);
   lo::require(multiplicity&&width==model->d,"invalid feature counts");
   for(int f=0;f<width;++f){lo::require(multiplicity[f]<=4096,"invalid feature count");counts.push_back(multiplicity[f]);lane_counts.push_back(int16_t(multiplicity[f]));budget+=int(multiplicity[f]);unit_counts=unit_counts&&multiplicity[f]==1;}
   lo::require(budget>=1&&budget<=4096,"invalid feature budget");
   uint64_t maximum=uint64_t(budget)*D*(power==2?D:1);
   if(maximum+1>MAX_TABLE)return;
   codes.resize(model->sv.size());
   for(size_t i=0;i<model->sv.size();++i){double v=model->sv[i];if(!(v>=0&&v<=1))return;
    int q=int(std::nearbyint(v*D));if(v!=double(q)/D)return;codes[i]=uint8_t(q);}
   words=(budget+63)/64;admitted=true;table.resize(size_t(maximum)+1);
   for(size_t s=0;s<table.size();++s)table[s]=kernel(double(s)/double(power==2?D*D:D));
   if(D==1){bits.resize(size_t(model->nsv)*words,0);qbits.resize(words,0);
    for(int i=0;i<model->nsv;++i){int bit=0;for(int f=0;f<model->d;++f)for(uint32_t k=0;k<counts[f];++k,++bit)
     if(codes[size_t(i)*model->d+f])bits[size_t(i)*words+bit/64]|=uint64_t(1)<<(bit%64);}}
 }
 double kernel(double distance)const{
   double value=0;
   for(size_t k=0;k<scales.size();++k)if(weights[k]!=0){double term=std::exp(-scales[k]*distance);value+=weights[k]*term;}
   lo::require(std::isfinite(value)&&value>=0&&value<=1+0x1p-38,"invalid mixture kernel");return value;
 }
 bool encode(const double* x){
  if(!admitted)return false;
  for(int f=0;f<model->d;++f){double v=x[f];if(!(v>=0&&v<=1))return false;int q=int(std::nearbyint(v*D));if(v!=double(q)/D)return false;query[f]=uint8_t(q);}
  if(D==1){std::fill(qbits.begin(),qbits.end(),0);int bit=0;for(int f=0;f<model->d;++f)for(uint32_t k=0;k<counts[f];++k,++bit)if(query[f])qbits[bit/64]|=uint64_t(1)<<(bit%64);}
  return true;
 }
 uint32_t signature(uint32_t id)const{
  uint32_t sum=0;if(D==1){for(int k=0;k<words;++k)sum+=popcount(bits[size_t(id)*words+k]^qbits[k]);return sum;}
  const auto* s=codes.data()+size_t(id)*model->d;int f=0;

#if defined(__AVX2__)
  __m256i acc=_mm256_setzero_si256();const __m256i ones=_mm256_set1_epi16(1);
  for(;f+16<=model->d;f+=16){
   __m256i a=_mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(query.data()+f)));
   __m256i b=_mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(s+f)));
   __m256i delta=_mm256_sub_epi16(a,b);
   __m256i terms;
   if(unit_counts)terms=power==2?_mm256_madd_epi16(delta,delta):_mm256_madd_epi16(_mm256_abs_epi16(delta),ones);
   else{__m256i w=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(lane_counts.data()+f));
    __m256i u=power==2?_mm256_mullo_epi16(delta,delta):_mm256_abs_epi16(delta);terms=_mm256_madd_epi16(u,w);}
   acc=_mm256_add_epi32(acc,terms);
  }
  uint32_t vals[8];_mm256_storeu_si256(reinterpret_cast<__m256i*>(vals),acc);for(int k=0;k<8;++k)sum+=vals[k];
#endif
  for(;f<model->d;++f){int t=int(query[f])-s[f];sum+=counts[f]*(power==2?uint32_t(t*t):uint32_t(std::abs(t)));}return sum;
 }
 double direct(uint32_t id,const double* x){double distance=0;const double* s=model->sv.data()+size_t(id)*model->d;
  for(int f=0;f<model->d;++f){double t=x[f]-s[f];double term=power==2?t*t:std::abs(t);for(uint32_t j=0;j<counts[f];++j)distance+=term;}
  for(double w:weights)if(w!=0)++exps;return kernel(distance);
 }
 double score(size_t pair,const double* x,bool use_table){const auto& p=model->pairs[pair];double value=0;
  for(size_t k=0;k<p.ids.size();++k){auto id=p.ids[k];if(epochs[id]!=serial){values[id]=use_table?table[signature(id)]:direct(id,x);epochs[id]=serial;++lookups;}value+=p.values[k]*values[id];}
  value+=p.bias;lo::require(std::isfinite(value),"nonfinite pair score");return value;
 }
 void begin(const double* x){worker.begin(x);if(++serial==0){std::fill(epochs.begin(),epochs.end(),0);serial=1;}lookups=exps=edges=fallback_rows=0;}
 int run(const double* x,int mode){begin(x);bool grid=encode(x);bool fast=(mode==0||mode==2)&&grid;
  if((mode==0||mode==2)&&!grid)fallback_rows=1;
  auto edge=[&](int i,int j){if(i>j)std::swap(i,j);++edges;return score(model->pair_index(i,j),x,fast)>=0?j:i;};
  int result;
  if(mode>=2){result=worker.vote.run(edge,worker.priority,1);}else{
   for(int i=0;i<model->c;++i){double sum=0;for(int f=0;f<model->d;++f){double t=x[f]-model->centers[size_t(i)*model->d+f];for(uint32_t j=0;j<counts[f];++j)sum+=t*t;}worker.priority[i]=std::isfinite(sum)?sum:INFINITY;}
   result=et::beretta(worker.vote,edge,worker.priority,5,worker.rounds);
  }
  lo::require(worker.vote.certificate()==result,"vote certificate failed");return result;
 }
 uint64_t bytes()const{return sizeof(*this)+model->storage()+worker.scratch()-sizeof(spm::Worker)+8*(scales.capacity()+weights.capacity()+table.capacity()+values.capacity()+bits.capacity()+qbits.capacity()+epochs.capacity())+codes.capacity()+query.capacity()+4*counts.capacity()+2*lane_counts.capacity();}
};
}
extern "C" {
int mk_abi(){return 1;}
void* mk_create(const unsigned char* p,uint64_t n,int power,int D,const double* g,const double* w,int components,const uint32_t* counts,int width){void* out=nullptr;lo::protect([&]{lo::require(p&&n<=lo::MAX_BYTES,"invalid model bytes");mk::environment();out=new mk::Engine(p,n,power,D,g,w,components,counts,width);});return out;}
void mk_destroy(void* p){delete static_cast<mk::Engine*>(p);}
int mk_info(void* p,uint64_t* out,int count){return lo::protect([&]{lo::require(p&&out&&count==9,"invalid info");auto&e=*static_cast<mk::Engine*>(p);
 uint64_t a[]={uint64_t(e.model->d),uint64_t(e.model->c),uint64_t(e.model->nsv),uint64_t(e.D),uint64_t(e.power),e.scales.size(),e.table.size(),e.bytes(),uint64_t(e.admitted)};std::copy(a,a+9,out);});}
int mk_run(void* p,const double* x,int rows,int d,int mode,int* output,uint64_t* stats,int count){return lo::protect([&]{
 lo::require(p,"null engine");auto&e=*static_cast<mk::Engine*>(p);mk::environment();
 lo::require(rows>=0&&rows<=65536&&d==e.model->d&&uint64_t(rows)*d<=8000000&&mode>=0&&mode<=3&&stats&&count==4&&(!rows||(x&&output)),"invalid run geometry");
 for(size_t k=0;k<size_t(rows)*d;++k)lo::require(std::isfinite(x[k]),"nonfinite input");std::fill(stats,stats+4,0);
 for(int r=0;r<rows;++r){output[r]=e.run(x+size_t(r)*d,mode);stats[0]+=e.lookups;stats[1]+=e.exps;stats[2]+=e.edges;stats[3]+=e.fallback_rows;}
 });}
int mk_probe(void* p,const double* x,int rows,int d,double* output,uint64_t count){return lo::protect([&]{
 lo::require(p,"null engine");auto&e=*static_cast<mk::Engine*>(p);mk::environment();
 lo::require(rows>=0&&rows<=65536&&d==e.model->d&&uint64_t(rows)*d<=8000000&&count==uint64_t(rows)*e.model->pairs.size()*3&&count<=16000000&&(!rows||(x&&output)),"invalid probe geometry");
 for(size_t k=0;k<size_t(rows)*d;++k)lo::require(std::isfinite(x[k]),"nonfinite input");
 for(int r=0;r<rows;++r){const double* a=x+size_t(r)*d;bool grid=e.encode(a);e.begin(a);
  std::vector<double> direct(e.model->pairs.size());for(size_t j=0;j<direct.size();++j)direct[j]=e.score(j,a,false);
  e.begin(a);for(size_t j=0;j<direct.size();++j){*output++=direct[j];*output++=e.score(j,a,grid);*output++=grid?1.:0.;}
 }
 });}
}
