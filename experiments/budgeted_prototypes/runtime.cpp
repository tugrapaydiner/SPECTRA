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
 void score(const uint8_t*raw,int mode,double*out,uint16_t*x)const{
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
 uint64_t storage()const{return sizeof(*this)+2ull*(centers.capacity()+weights.capacity())+8ull*(head.capacity()+bias.capacity()+high.capacity()+low.capacity());}
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
