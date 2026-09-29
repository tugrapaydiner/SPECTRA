// Optional finite-domain RBF compilation; original stored model remains intact.
#ifndef FK_ADAPTIVE
#error Define FK_ADAPTIVE to the previous adaptive runtime.cpp
#endif
#include FK_ADAPTIVE
namespace fk {
constexpr int MAX_TABLE=262144;
constexpr int denominators[]={1,2,4,8,15,16,32,64,100,128,255};
inline int popcount(uint64_t x){
#if defined(__POPCNT__)
 return __builtin_popcountll(x);
#else
 x-=((x>>1)&0x5555555555555555ULL);
 x=(x&0x3333333333333333ULL)+((x>>2)&0x3333333333333333ULL);
 return int((((x+(x>>4))&0x0f0f0f0f0f0f0f0fULL)*0x0101010101010101ULL)>>56);
#endif
}
struct Engine {
 ak::Engine prior; const spm::Model& m;
 int denominator=0,encoding=0,words=0,kind=1;
 bool exact_grid=false;
 double epsilon=0;
 std::vector<uint8_t> codes,query;
 std::vector<uint64_t> bits,qbits,epochs;
 std::vector<double> table,kernel,bounds;
 uint64_t serial=0,lookups=0,accepted=0,fallbacks=0,domain_rows=0;
 Engine(const unsigned char* raw,size_t bytes):prior(raw,bytes),m(*prior.model){
  ak::environment();
  for(int D:denominators){
   if(int64_t(m.d)*D*D+1>MAX_TABLE)continue;
   for(int enc=0;enc<2;++enc){
    bool ok=true;
    for(double v:m.sv){
     if(!(v>=0.&&v<=1.)){ok=false;break;}
     int q=int(std::nearbyint(v*D));double a=enc?double(float(double(q)/D)):double(q)/D;
     if(v!=a){ok=false;break;}
    }
    if(ok){denominator=D;encoding=enc;break;}
   }
   if(denominator)break;
  }
  if(!denominator)return;
  exact_grid=(denominator&(denominator-1))==0;
  codes.resize(size_t(m.nsv)*m.d);query.resize(m.d);
  for(size_t i=0;i<codes.size();++i)codes[i]=uint8_t(std::nearbyint(m.sv[i]*denominator));
  words=(m.d+63)/64;
  if(denominator==1){
   bits.resize(size_t(m.nsv)*words,0);qbits.resize(words,0);
   for(int i=0;i<m.nsv;++i)for(int j=0;j<m.d;++j)
    if(codes[size_t(i)*m.d+j])bits[size_t(i)*words+j/64]|=uint64_t(1)<<(j%64);
  }
  int maxdist=m.d*denominator*denominator;table.resize(size_t(maxdist)+1);
  for(int i=0;i<=maxdist;++i)table[i]=std::exp(-m.gamma*(double(i)/(denominator*denominator)));
  if(!exact_grid){
   double e=0;
   for(int q=0;q<=denominator;++q){
    double v=encoding?double(float(double(q)/denominator)):double(q)/denominator;
    long double ideal=static_cast<long double>(q)/denominator;
    e=std::max(e,double(std::abs(static_cast<long double>(v)-ideal)));
   }
   e=2.*e+0x1p-52;double eta=ak::gamma_n(2*m.d+16,0x1p-53);
   double ed=double(m.d)*(4.*e+4.*e*e+2.*eta+0x1p-48);
   epsilon=std::min(2.,2.*m.gamma*ed+2.*ak::LIBM_ABS_ERROR+0x1p-60);
  }
  kernel.resize(m.nsv);epochs.resize(m.nsv,0);bounds.resize(m.pairs.size(),0.);
  if(!exact_grid)for(size_t i=0;i<m.pairs.size();++i){
   const auto& p=m.pairs[i];double a=prior.magnitude[i];double g=ak::gamma_n(int(2*p.ids.size()+8),0x1p-53);
   bounds[i]=2.*(a*epsilon+g*(a*(2.+epsilon)+2.*std::abs(p.bias)))+0x1p-60;
  }
 }
 bool encode(const double* x){
  if(!denominator)return false;
  for(int j=0;j<m.d;++j){
   double v=x[j];if(!(v>=0.&&v<=1.))return false;
   int q=int(std::nearbyint(v*denominator));
   double a=encoding?double(float(double(q)/denominator)):double(q)/denominator;
   if(v!=a)return false;query[j]=uint8_t(q);
  }
  if(denominator==1){std::fill(qbits.begin(),qbits.end(),0);for(int j=0;j<m.d;++j)if(query[j])qbits[j/64]|=uint64_t(1)<<(j%64);}
  return true;
 }
 uint64_t added_storage()const{
  return sizeof(*this)-sizeof(prior)+codes.capacity()+query.capacity()+8*(bits.capacity()+qbits.capacity()+epochs.capacity()+table.capacity()+kernel.capacity()+bounds.capacity());
 }
 uint32_t distance(uint32_t id)const{
  if(denominator==1&&kind!=2){uint32_t sum=0;for(int w=0;w<words;++w)sum+=popcount(qbits[w]^bits[size_t(id)*words+w]);return sum;}
  const uint8_t* s=codes.data()+size_t(id)*m.d;uint32_t sum=0;int j=0;
#if defined(__AVX2__)
  if(kind!=2){
   __m256i a=_mm256_setzero_si256();
   for(;j+16<=m.d;j+=16){
    __m256i u=_mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(query.data()+j)));
    __m256i v=_mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(s+j)));
    __m256i delta=_mm256_sub_epi16(u,v);a=_mm256_add_epi32(a,_mm256_madd_epi16(delta,delta));
   }
   uint32_t val[8];_mm256_storeu_si256(reinterpret_cast<__m256i*>(val),a);for(int k=0;k<8;++k)sum+=val[k];
  }
#endif
  for(;j<m.d;++j){int t=int(query[j])-s[j];sum+=uint32_t(t*t);}return sum;
 }
 double score(size_t index){
  const auto& p=m.pairs[index];double result=0;
  for(size_t t=0;t<p.ids.size();++t){uint32_t id=p.ids[t];
   if(epochs[id]!=serial){kernel[id]=table[distance(id)];epochs[id]=serial;++lookups;}
   result+=p.values[t]*kernel[id];
  }
  return result+p.bias;
 }
 int edge(int i,int j,const double* x){
  if(i>j)std::swap(i,j);size_t index=m.pair_index(i,j);double value=score(index);
  if(exact_grid||(std::isfinite(value)&&std::abs(value)>bounds[index])){
   ++accepted;return m.c==2?(value>=0?1:0):(value>0?i:j);
  }
  ++fallbacks;return prior.exact.edge(i,j,x);
 }
 int run(const double* x,int mode){
  kind=mode;lookups=accepted=fallbacks=domain_rows=0;
  if(mode==0||!encode(x)){domain_rows=1;return prior.run(x,1);}
  auto& w=prior.exact;w.begin(x);
  if(++serial==0){std::fill(epochs.begin(),epochs.end(),0);serial=1;}
  if(m.c==2){int result=edge(0,1,x);w.vote.add(0,1,result);return result;}
  if(mode==3){auto oracle=[&](int i,int j){return edge(i,j,x);};return w.vote.run(oracle,w.priority,1);}
  std::fill(w.priority.begin(),w.priority.end(),0.);
  for(int i=0;i<m.c;++i){double sum=0;for(int j=0;j<m.d;++j){double v=x[j]-m.centers[size_t(i)*m.d+j];sum+=v*v;}w.priority[i]=std::isfinite(sum)?sum:INFINITY;}
  auto pair=[&](int i,int j){return edge(i,j,x);};int result=et::beretta(w.vote,pair,w.priority,5,w.rounds);
  lo::require(w.vote.certificate()==result,"finite vote certificate inconsistent");return result;
 }
};
}
extern "C" {
int fk_abi(){return 1;}
void* fk_create(const unsigned char* data,uint64_t size){void* p=nullptr;lo::protect([&]{lo::require(data&&size<=lo::MAX_BYTES,"invalid model bytes");p=new fk::Engine(data,size_t(size));});return p;}
void fk_destroy(void* p){delete static_cast<fk::Engine*>(p);}
int fk_info(void* p,uint64_t* out,int count){return lo::protect([&]{
 lo::require(p&&out&&count==9,"invalid info");auto& e=*static_cast<fk::Engine*>(p);
 uint64_t v[]={uint64_t(e.m.d),uint64_t(e.m.c),uint64_t(e.m.nsv),uint64_t(e.denominator),uint64_t(e.encoding),uint64_t(e.exact_grid),e.table.size(),e.added_storage(),e.m.storage()+e.prior.extra_bytes()+e.prior.exact.scratch()-sizeof(spm::Worker)};std::copy(v,v+9,out);
});}
int fk_run(void* p,const double* x,int rows,int d,int mode,int* out,uint64_t* stats,int count){return lo::protect([&]{
 lo::require(p,"null engine");auto& e=*static_cast<fk::Engine*>(p);ak::environment();
 lo::require(rows>=0&&rows<=65536&&d==e.m.d&&uint64_t(rows)*d<=8000000&&(!rows||(x&&out))&&stats&&count==5&&mode>=0&&mode<=3,"invalid run buffers");
 for(size_t i=0;i<size_t(rows)*d;++i)lo::require(std::isfinite(x[i]),"nonfinite input");std::fill(stats,stats+5,0);
 for(int r=0;r<rows;++r){out[r]=e.run(x+size_t(r)*d,mode);stats[0]+=e.lookups;stats[1]+=e.accepted;stats[2]+=e.fallbacks;stats[3]+=e.domain_rows;stats[4]+=e.prior.exact.work.kernels;}
});}
int fk_probe(void* p,const double* x,int rows,int d,double* out,uint64_t capacity){return lo::protect([&]{
 lo::require(p,"null engine");auto& e=*static_cast<fk::Engine*>(p);ak::environment();
 lo::require(rows>=0&&rows<=65536&&d==e.m.d&&capacity==uint64_t(rows)*e.m.pairs.size()*4&&capacity<=16000000&&(!rows||(x&&out)),"invalid probe");
 for(size_t k=0;k<size_t(rows)*d;++k)lo::require(std::isfinite(x[k]),"nonfinite input");
 for(int r=0;r<rows;++r){const double* a=x+size_t(r)*d;bool valid=e.encode(a);e.kind=1;e.prior.exact.begin(a);
  if(++e.serial==0){std::fill(e.epochs.begin(),e.epochs.end(),0);e.serial=1;}
  size_t p=0;for(int i=0;i<e.m.c;++i)for(int j=i+1;j<e.m.c;++j,++p){
   double reference=e.prior.original_score(i,j,a);double value=valid?e.score(p):reference;
   *out++=reference;*out++=value;*out++=valid?e.bounds[p]:0.;*out++=valid?1.:0.;
  }
 }
});}
}
