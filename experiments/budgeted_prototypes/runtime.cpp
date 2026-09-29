// Experimental immutable integer-prototype classifier. No training or old-SVM
// equivalence claim. Strict ordered binary64 output accumulation, no fast-math.
#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#if defined(__AVX2__)
#include <immintrin.h>
#elif defined(__SSE__)
#include <xmmintrin.h>
#endif
namespace bp {
thread_local std::string error;
void require(bool x,const char*s){if(!x)throw std::invalid_argument(s);}
void environment(){
 require(std::fegetround()==FE_TONEAREST,"round-to-nearest required");
#if defined(__SSE__)
 require((_mm_getcsr()&0xe040)==0,"SIMD round-to-nearest and gradual underflow required");
#endif
}
template<class F> int protect(F f){try{f();return 0;}catch(const std::exception&e){error=e.what();return 1;}catch(...){error="unknown native failure";return 1;}}
uint32_t crc32(const unsigned char*p,size_t n){uint32_t c=~0u;for(size_t i=0;i<n;++i){c^=p[i];for(int k=0;k<8;++k)c=(c>>1)^(0xedb88320u&uint32_t(-int(c&1)));}return ~c;}
struct Model {
 uint32_t d,p,c,maximum,quarter,units,bits;uint64_t bound,mask;double alpha;
 bool uniform=true,narrow=true;
 std::vector<uint16_t> centers,weights;
#if defined(BP_PACKET) && defined(__AVX2__)
 std::vector<uint16_t> packet_centers,packet_weights;
#endif
 std::vector<double> head,bias,high,low;
 Model(const unsigned char*data,uint64_t size){
  environment();const uint16_t one=1;
  require(*reinterpret_cast<const uint8_t*>(&one)==1&&sizeof(double)==8&&std::numeric_limits<double>::is_iec559,"little-endian IEEE binary64 required");
  require(data&&size>=48&&size<=64ull*1024*1024,"model byte cap");
  require(!std::memcmp(data,"SPPRO001",8),"wrong prototype magic");
  auto u32=[&](size_t off){uint32_t v;std::memcpy(&v,data+off,4);return v;};
  d=u32(8);p=u32(12);c=u32(16);maximum=u32(20);quarter=u32(24);units=u32(28);bits=u32(32);
  uint32_t meta=u32(36),payload=u32(40),crc=u32(44);
  require(d>=1&&d<=256&&p>=1&&p<=4096&&c>=2&&c<=128&&maximum>=1&&maximum<=255&&quarter>=1&&quarter<=16&&units>=1&&units<=16,"invalid model dimensions");
  uint64_t expected=8+4ull*p*d+8ull*(uint64_t(p)*c+c)+meta;
  require(meta>=1&&meta<=65536&&payload==expected&&size==48+expected,"prototype inventory mismatch");
  require(crc32(data+48,size-48)==crc,"prototype CRC mismatch");
  bound=uint64_t(units)*d*(maximum*quarter)*(maximum*quarter);
  require(bound>0&&bound<(1ull<<48),"signature bound");
  int width=0;for(uint64_t x=bound;x;x>>=1)++width;
  require(bits==uint32_t((width+1)/2),"table bits mismatch");mask=(1ull<<bits)-1;
  require((mask+1)+(bound>>bits)+1<=1048576,"table allocation cap");narrow=bound<=uint64_t(INT32_MAX);
  std::memcpy(&alpha,data+48,8);require(std::isfinite(alpha)&&alpha>0,"invalid kernel coefficient");
  size_t off=56;centers.resize(size_t(p)*d);weights.resize(size_t(p)*d);
  for(auto*a:{&centers,&weights}){std::memcpy(a->data(),data+off,a->size()*2);off+=a->size()*2;}
  for(uint32_t j=0;j<p;++j){uint32_t mass=0;for(uint32_t f=0;f<d;++f){size_t i=size_t(j)*d+f;
    require(centers[i]<=maximum*quarter,"prototype center out of domain");require(weights[i]>=1,"zero local metric weight");mass+=weights[i];uniform=uniform&&weights[i]==units;
  }require(mass==units*d,"local metric mass mismatch");}
  head.resize(size_t(p)*c);bias.resize(c);
  for(auto*a:{&head,&bias}){std::memcpy(a->data(),data+off,a->size()*8);off+=a->size()*8;for(double v:*a)require(std::isfinite(v),"nonfinite output parameter");}
  require(off+meta==size,"unused model bytes");
  for(uint32_t k=0;k<c;++k){double magnitude=std::abs(bias[k]),limit=std::numeric_limits<double>::max()/4;
   require(magnitude<limit,"output magnitude too large");for(uint32_t j=0;j<p;++j){double a=std::abs(head[size_t(j)*c+k]);require(a<limit-magnitude,"output magnitude too large");magnitude+=a;}}
#if defined(BP_PACKET) && defined(__AVX2__)
  if(narrow){
   const size_t padded=(size_t(p)+7)/8*8;packet_centers.resize(padded*d,0);
   if(!uniform)packet_weights.resize(padded*d,0);
   for(uint32_t j=0;j<p;++j)for(uint32_t f=0;f<d;++f){
    size_t slot=(size_t(j/8)*d+f)*8+j%8;packet_centers[slot]=centers[size_t(j)*d+f];
    if(!uniform)packet_weights[slot]=weights[size_t(j)*d+f];
   }
  }
#endif
  high.resize(size_t(bound>>bits)+1);low.resize(size_t(mask)+1);
  for(size_t i=0;i<high.size();++i)high[i]=std::exp(-alpha*double(uint64_t(i)<<bits));
  for(size_t i=0;i<low.size();++i)low[i]=std::exp(-alpha*double(i));
 }
 uint64_t distance(const uint16_t*x,uint32_t j,bool scalar)const{
  const uint16_t*s=centers.data()+size_t(j)*d,*w=weights.data()+size_t(j)*d;uint64_t sum=0;uint32_t f=0;
#if defined(__AVX2__)
  if(!scalar&&narrow){__m256i acc=_mm256_setzero_si256();
   for(;f+8<=d;f+=8){auto a=_mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(x+f)));
    auto b=_mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(s+f)));
    auto delta=_mm256_sub_epi32(a,b);auto term=_mm256_mullo_epi32(delta,delta);
    if(!uniform)term=_mm256_mullo_epi32(term,_mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(w+f))));
    acc=_mm256_add_epi32(acc,term);
   }uint32_t lanes[8];_mm256_storeu_si256(reinterpret_cast<__m256i*>(lanes),acc);for(auto v:lanes)sum+=v;
  }
#endif
  for(;f<d;++f){int delta=int(x[f])-s[f];sum+=uint64_t(delta*delta)*(uniform?1:w[f]);}
  return uniform?sum*units:sum;
 }
 void score_original(const uint8_t*raw,int mode,double*out,uint16_t*x)const{
  for(uint32_t f=0;f<d;++f)x[f]=uint16_t(raw[f]*quarter);
  std::fill(out,out+c,0.);
  for(uint32_t j=0;j<p;++j){uint64_t S=distance(x,j,mode==1);require(S<=bound,"invalid native signature");
   double v=mode==2?std::exp(-alpha*double(S)):high[S>>bits]*low[S&mask];uint32_t k=0;
#if defined(__AVX2__)
   if(mode!=1){auto value=_mm256_set1_pd(v);for(;k+4<=c;k+=4){auto score=_mm256_loadu_pd(out+k);auto coef=_mm256_loadu_pd(head.data()+size_t(j)*c+k);_mm256_storeu_pd(out+k,_mm256_add_pd(score,_mm256_mul_pd(value,coef)));}}
#endif
   for(;k<c;++k)out[k]+=v*head[size_t(j)*c+k];
  }
  for(uint32_t k=0;k<c;++k){out[k]+=bias[k];require(std::isfinite(out[k]),"nonfinite class score");}
 }
#if defined(BP_PACKET) && defined(__AVX2__)
 void packet(const uint16_t*x,uint32_t first,uint32_t count,uint64_t*out)const{
  if(narrow&&count==8){auto sum=_mm256_setzero_si256();
   const uint16_t*cp=packet_centers.data()+size_t(first/8)*d*8;
   const uint16_t*wp=uniform?nullptr:packet_weights.data()+size_t(first/8)*d*8;
   for(uint32_t f=0;f<d;++f){
    auto center=_mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(cp+f*8)));
    auto delta=_mm256_sub_epi32(_mm256_set1_epi32(x[f]),center);auto term=_mm256_mullo_epi32(delta,delta);
    if(!uniform)term=_mm256_mullo_epi32(term,_mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(wp+f*8))));
    sum=_mm256_add_epi32(sum,term);
   }
   uint32_t values[8];_mm256_storeu_si256(reinterpret_cast<__m256i*>(values),sum);
   for(uint32_t i=0;i<8;++i)out[i]=uint64_t(values[i])*(uniform?units:1);
  }else for(uint32_t i=0;i<count;++i)out[i]=distance(x,first+i,false);
 }
 double value(uint64_t S,int mode)const{
  require(S<=bound,"packet signature outside bound");
  return mode==2?std::exp(-alpha*double(S)):high[S>>bits]*low[S&mask];
 }
 void score_packet(const uint8_t*raw,int mode,double*out,uint16_t*x)const{
  for(uint32_t f=0;f<d;++f)x[f]=uint16_t(raw[f]*quarter);std::fill(out,out+c,0.);
  for(uint32_t first=0;first<p;first+=8){uint32_t count=std::min(8u,p-first);uint64_t S[8];packet(x,first,count,S);
   for(uint32_t i=0;i<count;++i){double v=value(S[i],mode);uint32_t k=0;auto kernel=_mm256_set1_pd(v);
    for(;k+4<=c;k+=4)_mm256_storeu_pd(out+k,_mm256_add_pd(_mm256_loadu_pd(out+k),_mm256_mul_pd(kernel,_mm256_loadu_pd(head.data()+size_t(first+i)*c+k))));
    for(;k<c;++k)out[k]+=v*head[size_t(first+i)*c+k];
   }
  }
  for(uint32_t k=0;k<c;++k){out[k]+=bias[k];require(std::isfinite(out[k]),"nonfinite class score");}
 }
#if defined(BP_REGISTERS)
 template<int C>void score_registers(const uint8_t*raw,int mode,double*out,uint16_t*x)const{
  constexpr int V=C/4,T=C%4;__m256d sums[V];double tails[T?T:1]={};
  for(int k=0;k<V;++k)sums[k]=_mm256_setzero_pd();
  for(uint32_t f=0;f<d;++f)x[f]=uint16_t(raw[f]*quarter);
  for(uint32_t first=0;first<p;first+=8){uint32_t count=std::min(8u,p-first);uint64_t S[8];packet(x,first,count,S);
   for(uint32_t i=0;i<count;++i){double v=value(S[i],mode);auto kernel=_mm256_set1_pd(v);const double*h=head.data()+size_t(first+i)*C;
#pragma GCC unroll 8
    for(int k=0;k<V;++k)sums[k]=_mm256_add_pd(sums[k],_mm256_mul_pd(kernel,_mm256_loadu_pd(h+4*k)));
    for(int k=0;k<T;++k)tails[k]+=v*h[4*V+k];
   }
  }
  for(int k=0;k<V;++k)_mm256_storeu_pd(out+4*k,sums[k]);for(int k=0;k<T;++k)out[4*V+k]=tails[k];
  for(int k=0;k<C;++k){out[k]+=bias[k];require(std::isfinite(out[k]),"nonfinite class score");}
 }
#endif
#endif
 void score(const uint8_t*raw,int mode,double*out,uint16_t*x)const{
#if defined(BP_PACKET) && defined(__AVX2__)
  if(mode!=1){
#if defined(BP_REGISTERS)
   if(c==6){score_registers<6>(raw,mode,out,x);return;}
   if(c==10){score_registers<10>(raw,mode,out,x);return;}
   if(c==26){score_registers<26>(raw,mode,out,x);return;}
#endif
   score_packet(raw,mode,out,x);return;
  }
#endif
  score_original(raw,mode,out,x);
 }
 uint64_t storage()const{
  uint64_t n=sizeof(*this)+2ull*(centers.capacity()+weights.capacity())+8ull*(head.capacity()+bias.capacity()+high.capacity()+low.capacity());
#if defined(BP_PACKET) && defined(__AVX2__)
  n+=2ull*(packet_centers.capacity()+packet_weights.capacity());
#endif
  return n;
 }
};
}
extern "C" {
const char* bp_error(){return bp::error.c_str();}
int bp_abi(){return 1;}
void* bp_create(const unsigned char*x,uint64_t n){void*out=nullptr;bp::protect([&]{out=new bp::Model(x,n);});return out;}
void bp_destroy(void*p){delete static_cast<bp::Model*>(p);}
int bp_info(void*p,uint64_t*out,int n){return bp::protect([&]{bp::require(p&&out&&n==8,"invalid info");auto&m=*static_cast<bp::Model*>(p);uint64_t v[]={m.d,m.p,m.c,m.high.size()+m.low.size(),m.bound+1,m.storage(),uint64_t(m.uniform),uint64_t(m.narrow)};std::copy(v,v+8,out);});}
int bp_run(void*p,const uint8_t*x,int rows,int d,int mode,int*out){return bp::protect([&]{bp::require(p,"closed prototype model");auto&m=*static_cast<bp::Model*>(p);bp::environment();
 bp::require(rows>=0&&rows<=65536&&d==int(m.d)&&uint64_t(rows)*d<=8000000&&mode>=0&&mode<=2&&(!rows||(x&&out)),"invalid inference shape");
 for(size_t i=0;i<size_t(rows)*d;++i)bp::require(x[i]<=m.maximum,"input outside integer domain");
 std::vector<uint16_t> q(m.d);std::vector<double> s(m.c);
 for(int i=0;i<rows;++i){m.score(x+size_t(i)*d,mode,s.data(),q.data());out[i]=int(std::max_element(s.begin(),s.end())-s.begin());}
});}
int bp_scores(void*p,const uint8_t*x,int rows,int d,int mode,double*out,uint64_t cells){return bp::protect([&]{bp::require(p,"closed prototype model");auto&m=*static_cast<bp::Model*>(p);bp::environment();
 bp::require(rows>=0&&rows<=65536&&d==int(m.d)&&uint64_t(rows)*d<=8000000&&cells==uint64_t(rows)*m.c&&cells<=8000000&&mode>=0&&mode<=2&&(!rows||(x&&out)),"invalid score shape");
 for(size_t i=0;i<size_t(rows)*d;++i)bp::require(x[i]<=m.maximum,"input outside integer domain");std::vector<uint16_t> q(m.d);
 for(int i=0;i<rows;++i)m.score(x+size_t(i)*d,mode,out+size_t(i)*m.c,q.data());
});}
}
