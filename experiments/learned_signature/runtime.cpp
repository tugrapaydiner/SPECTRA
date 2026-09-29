// Finite learned-kernel model. The stored table, not a reconstructed libm call,
// defines each kernel value. Existing exact vote scheduling is reused unchanged.
#ifndef LS_BASE_RUNTIME
#error Define LS_BASE_RUNTIME to SPECTRA's existing runtime.cpp
#endif
#include LS_BASE_RUNTIME
#include <memory>
namespace ls {
constexpr uint64_t MAX_BYTES=128ull*1024*1024;
struct Pair {std::vector<uint32_t> ids;std::vector<double> coef;double bias=0;uint32_t profile=0;};
struct Engine {
 uint32_t d,c,n,cap,nt,ne;std::vector<uint32_t> weights,counts;
 std::vector<uint8_t> codes;std::vector<uint16_t> weights16;std::vector<uint32_t> norms;std::vector<double> tables,centers,priority,kernel;bool unit_metric=false;uint32_t query_norm=0;
 std::vector<Pair> pairs;std::vector<uint32_t> distances;std::vector<uint64_t> epochs;
 uint64_t serial=0,rounds=0,lookups=0;lo::Work work;lo::Votes vote;bool scalar=false;
 static uint32_t classes(const unsigned char* raw,uint64_t size){
  lo::require(raw&&size>=44&&size<=MAX_BYTES,"invalid learned model size");
  lo::require(std::memcmp(raw,"SPLKT001",8)==0,"invalid learned model magic");
  uint32_t value;std::memcpy(&value,raw+12,4);lo::require(value>=2&&value<=128,"invalid class count");return value;
 }
 Engine(const unsigned char* raw,uint64_t size):c(classes(raw,size)),vote(int(c)){
  const uint16_t one=1;lo::require(*reinterpret_cast<const uint8_t*>(&one)==1&&sizeof(double)==8&&std::numeric_limits<double>::is_iec559,"little-endian IEEE binary64 required");
  uint32_t fields[9];std::memcpy(fields,raw+8,36);
  d=fields[0];n=fields[2];cap=fields[3];nt=fields[4];ne=fields[5];uint32_t terms=fields[6],meta=fields[7],crc=fields[8];
  lo::require(d>=1&&d<=4096&&n>=1&&n<=100000&&cap>=1&&cap<=255&&nt>=1&&nt<=64&&ne>=1&&ne<=4000001&&terms<=8000000&&meta<=65536,"unsupported learned model geometry");
  uint64_t np=uint64_t(c)*(c-1)/2;
  uint64_t expected=44ull+meta+4ull*c+4ull*d+uint64_t(n)*d+16*np+12ull*terms+8ull*nt*ne;
  lo::require(size==expected,"learned model inventory differs");
  lo::require(lo::crc32(raw+44,size-44)==crc,"learned model CRC mismatch");
  size_t off=44+meta;
  auto take=[&](void* to,size_t bytes){lo::require(bytes<=size-off,"truncated model");if(bytes)std::memcpy(to,raw+off,bytes);off+=bytes;};
  counts.resize(c);take(counts.data(),4*c);uint64_t sum=0;for(uint32_t v:counts){lo::require(v>=1&&v<=n,"bad support count");sum+=v;}lo::require(sum==n,"support counts differ");
  weights.resize(d);take(weights.data(),4*d);sum=0;for(uint32_t v:weights){lo::require(v>=1&&v<=8,"invalid metric weight");sum+=v;}lo::require(sum*cap*cap+1==ne,"metric/table size differs");
  codes.resize(size_t(n)*d);take(codes.data(),codes.size());for(auto v:codes)lo::require(v<=cap,"stored coordinate outside domain");
  std::vector<uint32_t> sizes(np),profiles(np),allids(terms);std::vector<double> biases(np),coeff(terms);
  take(sizes.data(),4*np);take(profiles.data(),4*np);take(biases.data(),8*np);take(allids.data(),4ull*terms);take(coeff.data(),8ull*terms);
  sum=0;for(auto v:sizes)sum+=v;lo::require(sum==terms,"pair term counts differ");
  tables.resize(uint64_t(nt)*ne);take(tables.data(),tables.size()*8);lo::require(off==size,"unused model bytes");
  for(double v:tables)lo::require(std::isfinite(v)&&v>=0.&&v<=2.,"invalid stored kernel value");
  size_t start=0;double limit=std::numeric_limits<double>::max()/8;
  for(size_t k=0;k<np;++k){
   lo::require(profiles[k]<nt&&std::isfinite(biases[k]),"bad pair profile/bias");Pair p;p.bias=biases[k];p.profile=profiles[k];double magnitude=std::abs(p.bias);lo::require(magnitude<limit,"pair bound too large");
   for(size_t t=0;t<sizes[k];++t){auto id=allids[start+t];double a=coeff[start+t];lo::require(id<n&&std::isfinite(a)&&std::abs(a)<limit-magnitude,"bad coefficient");magnitude+=std::abs(a);p.ids.push_back(id);p.coef.push_back(a);}
   p.ids.shrink_to_fit();p.coef.shrink_to_fit();pairs.push_back(std::move(p));start+=sizes[k];
  }
  centers.assign(size_t(c)*d,0.);start=0;
  for(uint32_t i=0;i<c;++i){for(uint32_t k=0;k<counts[i];++k)for(uint32_t f=0;f<d;++f)centers[size_t(i)*d+f]+=double(codes[(start+k)*d+f])/counts[i];start+=counts[i];}
  priority.resize(c);distances.resize(n);epochs.assign(n,0);
  unit_metric=std::all_of(weights.begin(),weights.end(),[](uint32_t w){return w==1;});
  weights16.assign(weights.begin(),weights.end());
  if(nt==1)kernel.resize(n);
  if(unit_metric&&cap<=127){norms.resize(n);for(uint32_t i=0;i<n;++i){uint32_t norm=0;for(uint32_t j=0;j<d;++j){uint32_t v=codes[size_t(i)*d+j];norm+=v*v;}norms[i]=norm;}}
 }
 uint64_t storage()const{uint64_t bytes=sizeof(*this)+4*(weights.capacity()+counts.capacity()+distances.capacity()+norms.capacity())+2*weights16.capacity()+8*(tables.capacity()+centers.capacity()+priority.capacity()+epochs.capacity()+kernel.capacity())+codes.capacity()+sizeof(Pair)*pairs.capacity()+vote.known.capacity()+4*(vote.low.capacity()+vote.remain.capacity());for(const auto&p:pairs)bytes+=4*p.ids.capacity()+8*p.coef.capacity();return bytes;}
 uint32_t distance(const uint8_t* x,uint32_t id){
  const uint8_t* s=codes.data()+size_t(id)*d;uint32_t total=0,j=0;
#if defined(__AVX2__)
  if(!scalar&&unit_metric&&cap<=127){
   __m128i sums=_mm_setzero_si128();const __m128i ones=_mm_set1_epi16(1);
   for(;j+16<=d;j+=16){
    const __m128i xv=_mm_loadu_si128(reinterpret_cast<const __m128i*>(x+j));
    const __m128i sv=_mm_loadu_si128(reinterpret_cast<const __m128i*>(s+j));
    // Both operands <=127: pair sum <=32258, so maddubs never saturates.
    sums=_mm_add_epi32(sums,_mm_madd_epi16(_mm_maddubs_epi16(xv,sv),ones));
   }
   sums=_mm_hadd_epi32(sums,sums);sums=_mm_hadd_epi32(sums,sums);
   uint32_t dot=uint32_t(_mm_cvtsi128_si32(sums));for(;j<d;++j)dot+=uint32_t(x[j])*s[j];
   total=query_norm+norms[id]-2*dot;
  }else if(!scalar){
   __m256i sums=_mm256_setzero_si256();
   for(;j+16<=d;j+=16){
    __m256i xv=_mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(x+j)));
    __m256i sv=_mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(s+j)));
    __m256i delta=_mm256_sub_epi16(xv,sv);
    __m256i weighted=_mm256_mullo_epi16(delta,_mm256_loadu_si256(reinterpret_cast<const __m256i*>(weights16.data()+j)));
    // abs(delta*weight)<=255*8=2040 fits signed16; accumulated result <4,000,001.
    sums=_mm256_add_epi32(sums,_mm256_madd_epi16(delta,weighted));
   }
   __m128i sum=_mm_add_epi32(_mm256_castsi256_si128(sums),_mm256_extracti128_si256(sums,1));
   sum=_mm_hadd_epi32(sum,sum);sum=_mm_hadd_epi32(sum,sum);total=uint32_t(_mm_cvtsi128_si32(sum));
   for(;j<d;++j){int delta=int(x[j])-int(s[j]);total+=weights[j]*uint32_t(delta*delta);}
  }else
#endif
  {for(;j<d;++j){int delta=int(x[j])-int(s[j]);total+=weights[j]*uint32_t(delta*delta);}}
  lo::require(total<ne,"signature outside stored kernel table");++work.kernels;return total;
 }
 size_t pair_index(int i,int j)const{if(i>j)std::swap(i,j);return size_t(i)*(2*c-i-1)/2+j-i-1;}
 double margin(const uint8_t*x,size_t k){
  const auto&p=pairs[k];double result=0;const double* bank=tables.data()+size_t(p.profile)*ne;
  if(nt==1){
   for(size_t t=0;t<p.ids.size();++t){uint32_t id=p.ids[t];if(epochs[id]!=serial){distances[id]=distance(x,id);kernel[id]=bank[distances[id]];epochs[id]=serial;++lookups;}result+=p.coef[t]*kernel[id];}
  }else{
   for(size_t t=0;t<p.ids.size();++t){uint32_t id=p.ids[t];if(epochs[id]!=serial){distances[id]=distance(x,id);epochs[id]=serial;}result+=p.coef[t]*bank[distances[id]];++lookups;}
  }
  work.terms+=p.ids.size();result+=p.bias;lo::require(std::isfinite(result),"nonfinite margin");return result;
 }
 int edge(int i,int j,const uint8_t*x){if(i>j)std::swap(i,j);double value=margin(x,pair_index(i,j));return c==2?(value>=0?1:0):(value>0?i:j);}
 void begin(){if(++serial==0){std::fill(epochs.begin(),epochs.end(),0);serial=1;}work=lo::Work{};work.classes=c;work.supports=n;lookups=rounds=0;vote.reset(work);}
 int run(const uint8_t*x,int mode){
  begin();scalar=mode==2;query_norm=0;if(unit_metric&&cap<=127)for(uint32_t j=0;j<d;++j)query_norm+=uint32_t(x[j])*x[j];
  if(c==2){int winner=edge(0,1,x);vote.add(0,1,winner);return winner;}
  std::fill(priority.begin(),priority.end(),0.);
  for(uint32_t i=0;i<c;++i){double sum=0;for(uint32_t j=0;j<d;++j){double diff=x[j]-centers[size_t(i)*d+j];sum+=weights[j]*diff*diff;}priority[i]=sum;}
  auto oracle=[&](int i,int j){return edge(i,j,x);};int result=mode==0?vote.run(oracle,priority,1):et::beretta(vote,oracle,priority,5,rounds);
  lo::require(vote.certificate()==result,"invalid final vote certificate");return result;
 }
};
}
extern "C" {
int ls_abi(){return 1;}
void* ls_create(const unsigned char*raw,uint64_t bytes){void*out=nullptr;lo::protect([&]{lo::require(std::fegetround()==FE_TONEAREST,"round-to-nearest required");out=new ls::Engine(raw,bytes);});return out;}
void ls_destroy(void*p){delete static_cast<ls::Engine*>(p);}
int ls_info(void*p,uint64_t*out,int count){return lo::protect([&]{lo::require(p&&out&&count==7,"invalid info");auto&e=*static_cast<ls::Engine*>(p);uint64_t values[]={e.d,e.c,e.n,e.nt,e.ne,e.storage(),e.cap};std::copy(values,values+7,out);});}
int ls_run(void*p,const uint8_t*x,int rows,int d,int mode,int*out,uint64_t*stats,int count){return lo::protect([&]{
 lo::require(p&&rows>=0&&rows<=65536&&uint64_t(rows)*d<=8000000&&mode>=0&&mode<=2&&stats&&count==4,"invalid run geometry");auto&e=*static_cast<ls::Engine*>(p);lo::require(d==int(e.d)&&(!rows||(x&&out)),"bad input/output buffers");lo::require(std::fegetround()==FE_TONEAREST,"round-to-nearest required");
 for(size_t k=0;k<size_t(rows)*d;++k)lo::require(x[k]<=e.cap,"input outside learned integer domain");std::fill(stats,stats+4,0);
 for(int r=0;r<rows;++r){out[r]=e.run(x+size_t(r)*d,mode);stats[0]+=e.work.kernels;stats[1]+=e.lookups;stats[2]+=e.work.pairs;stats[3]+=e.work.cert_checks;}
 });}
int ls_probe(void*p,const uint8_t*x,int rows,int d,double*out,uint64_t capacity){return lo::protect([&]{
 lo::require(p&&rows>=0&&rows<=65536,"invalid probe");auto&e=*static_cast<ls::Engine*>(p);uint64_t columns=e.pairs.size();lo::require(d==int(e.d)&&capacity==uint64_t(rows)*columns&&capacity<=16000000&&(!rows||(x&&out)),"probe capacity");
 lo::require(std::fegetround()==FE_TONEAREST,"round-to-nearest required");for(size_t k=0;k<size_t(rows)*d;++k)lo::require(x[k]<=e.cap,"input outside domain");
 for(int r=0;r<rows;++r){e.begin();e.scalar=false;e.query_norm=0;if(e.unit_metric&&e.cap<=127)for(uint32_t j=0;j<e.d;++j)e.query_norm+=uint32_t(x[size_t(r)*d+j])*x[size_t(r)*d+j];for(size_t j=0;j<columns;++j)*out++=e.margin(x+size_t(r)*d,j);}
 });}
}

#include "dense_controls.hpp"
