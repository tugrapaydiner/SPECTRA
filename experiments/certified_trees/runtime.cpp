// Compact source-bound oblivious-tree decisions. No training or calibrated threshold.
// Numeric contract is verified offline against the original binary64 source.
#include <algorithm>
#include <array>
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
namespace st2 {
thread_local std::string error;
void require(bool x,const char*msg){if(!x)throw std::invalid_argument(msg);}
template<class F> int protect(F f){try{f();return 0;}catch(const std::exception&e){error=e.what();return 1;}catch(...){error="unknown tree runtime failure";return 1;}}
void environment(){
 require(std::fegetround()==FE_TONEAREST,"round-to-nearest required");
#if defined(__SSE__)
 require((_mm_getcsr()&0xe040)==0,"SIMD nearest and gradual underflow required");
#endif
}
uint32_t crc32(const unsigned char*p,size_t n){uint32_t c=~0u;for(size_t i=0;i<n;++i){c^=p[i];for(int j=0;j<8;++j)c=(c>>1)^(0xedb88320u&uint32_t(-int(c&1)));}return ~c;}
struct Predicate{uint16_t feature;int16_t threshold;};
struct Tree{uint32_t offset,split,depth;};
struct Model {
 uint32_t d,c,maximum,t,np,ns,n,bits,flags;bool quantized;uint64_t packed_bytes=0;
 std::array<unsigned char,32> source;
 std::vector<Predicate> predicates;std::vector<Tree> trees;std::vector<uint16_t> splits;
 std::vector<int8_t> small;std::vector<int16_t> medium;std::vector<double> full,bias;
 std::vector<int32_t> qb,suffix_low,suffix_high;
 std::vector<int64_t> low,high,pair;
 double scale=1.;static constexpr int64_t fine=int64_t(1)<<20;
 Model(const unsigned char*raw,uint64_t bytes){
  environment();uint16_t endian=1;
  require(*reinterpret_cast<unsigned char*>(&endian)==1&&sizeof(double)==8&&std::numeric_limits<double>::is_iec559,"little-endian IEEE binary64 required");
  require(raw&&bytes>=120&&bytes<=64ull*1024*1024,"tree model byte cap");
  quantized=!std::memcmp(raw,"SPTCQ001",8);require(quantized||!std::memcmp(raw,"SPTCF001",8),"unknown tree model magic");
  auto u32=[&](size_t k){uint32_t v;std::memcpy(&v,raw+8+4*k,4);return v;};
  d=u32(0);c=u32(1);maximum=u32(2);t=u32(3);np=u32(4);ns=u32(5);n=u32(6);bits=u32(7);flags=u32(9);
  uint32_t payload=u32(10),crc=u32(11);
  require(d>=1&&d<=256&&c>=2&&c<=64&&maximum>=1&&maximum<=255&&t>=1&&t<=4096&&np<=49152&&ns<=49152&&n>=c&&n<=500000,"tree geometry cap");
  require(u32(8)==20&&flags<=1&&((quantized&&(bits==8||bits==16))||(!quantized&&bits==64&&flags==0)),"invalid numerical format");
  uint64_t prefix=quantized?uint64_t(c)*20+(flags?uint64_t(c)*c*8:0):8ull*(c+1);
  uint64_t expected=prefix+4ull*np+12ull*t+2ull*ns+uint64_t(n)*(bits/8);
  require(payload==expected&&bytes==120+expected&&crc32(raw+120,payload)==crc,"tree inventory/CRC mismatch");
  packed_bytes=bytes;std::copy(raw+56,raw+88,source.begin());
  size_t off=120;auto take=[&](void*dst,size_t count){require(off<=bytes&&count<=bytes-off,"tree read bounds");if(count)std::memcpy(dst,raw+off,count);off+=count;};
  int32_t limit=bits==8?127:32767;
  if(quantized){
   qb.resize(c);low.resize(c);high.resize(c);take(qb.data(),c*4);take(low.data(),c*8);take(high.data(),c*8);
   int64_t envelope=int64_t(t+1)*limit*fine;
   for(uint32_t j=0;j<c;++j){require(qb[j]>=-limit&&qb[j]<=limit,"quantized bias range");require(low[j]<=high[j]&&low[j]>-(int64_t(1)<<61)+envelope&&high[j]<(int64_t(1)<<61)-envelope,"certificate interval envelope");}
   if(flags){pair.resize(size_t(c)*c);take(pair.data(),pair.size()*8);for(uint32_t i=0;i<c;++i)for(uint32_t j=0;j<c;++j){int64_t v=pair[size_t(i)*c+j];require(v>-(int64_t(1)<<61)&&v<(int64_t(1)<<61)&&(i!=j||v==0),"pairwise certificate range");}}
  }else{
   take(&scale,8);bias.resize(c);take(bias.data(),c*8);require(std::isfinite(scale)&&scale>0,"positive finite source scale required");
   for(double b:bias)require(std::isfinite(b),"nonfinite source bias");
  }
  predicates.resize(np);for(auto&p:predicates){take(&p.feature,2);take(&p.threshold,2);require(p.feature<d&&p.threshold>=-1&&p.threshold<=int(maximum),"invalid predicate");}
  trees.resize(t);uint64_t leaf_offset=0,split_offset=0;
  for(auto&tree:trees){take(&tree.offset,4);take(&tree.split,4);take(&tree.depth,4);
   require(tree.depth<=12&&tree.offset==leaf_offset&&tree.split==split_offset,"invalid tree descriptor order");
   leaf_offset+=(uint64_t(1)<<tree.depth)*c;split_offset+=tree.depth;
   require(leaf_offset<=n&&split_offset<=ns,"tree descriptor bounds");
  }
  require(leaf_offset==n&&split_offset==ns,"tree coverage mismatch");
  splits.resize(ns);take(splits.data(),ns*2);for(auto p:splits)require(p<np,"predicate index out of range");
  if(quantized){
   if(bits==8){small.resize(n);take(small.data(),n);for(auto v:small)require(v>=-127,"quantizer excludes negative minimum");}
   else {medium.resize(n);take(medium.data(),n*2);for(auto v:medium)require(v>=-32767,"quantizer excludes negative minimum");}
   suffix_low.resize(size_t(t+1)*c,0);suffix_high.resize(size_t(t+1)*c,0);
   for(int k=int(t)-1;k>=0;--k){auto&tree=trees[size_t(k)];uint32_t leaves=1u<<tree.depth;
    for(uint32_t j=0;j<c;++j){int32_t lo=limit,hi=-limit;
     for(uint32_t l=0;l<leaves;++l){int32_t v=qvalue(tree.offset+l*c+j);lo=std::min(lo,v);hi=std::max(hi,v);}
     suffix_low[size_t(k)*c+j]=lo+suffix_low[size_t(k+1)*c+j];suffix_high[size_t(k)*c+j]=hi+suffix_high[size_t(k+1)*c+j];
    }
   }
  }else{
   full.resize(n);take(full.data(),n*8);for(double v:full)require(std::isfinite(v),"nonfinite source leaf");
   // Conservative source no-overflow envelope, checked before any request.
   long double cap=std::ldexp(1.L,900);
   for(uint32_t j=0;j<c;++j){long double sum=0.;for(const auto&tree:trees){long double m=0.;for(uint32_t l=0;l<(1u<<tree.depth);++l)m=std::max(m,std::abs((long double)full[tree.offset+l*c+j]));sum+=m;}
    require(sum<cap&&sum*scale+std::abs((long double)bias[j])<cap,"source magnitude exceeds admitted envelope");}
  }
  require(off==bytes,"unused tree model bytes");
 }
 int32_t qvalue(uint32_t i)const{return bits==8?int32_t(small[i]):int32_t(medium[i]);}
 void predicates_for(const uint8_t*x,uint8_t*out)const{for(uint32_t i=0;i<np;++i)out[i]=int(x[predicates[i].feature])>predicates[i].threshold;}
 uint32_t leaf(const Tree&tree,const uint8_t*pred)const{uint32_t k=0;for(uint32_t b=0;b<tree.depth;++b)k|=uint32_t(pred[splits[tree.split+b]])<<b;return tree.offset+k*c;}
 void add_leaf(uint32_t offset,int32_t*sum,bool scalar)const{
  uint32_t j=0;
#if defined(__AVX2__)
  if(!scalar){
   for(;j+8<=c;j+=8){__m256i values;
    if(bits==8)values=_mm256_cvtepi8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(small.data()+offset+j)));
    else values=_mm256_cvtepi16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(medium.data()+offset+j)));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(sum+j),_mm256_add_epi32(values,_mm256_loadu_si256(reinterpret_cast<const __m256i*>(sum+j))));
   }
  }
#endif
  for(;j<c;++j)sum[j]+=qvalue(offset+j);
 }
 int settle(const int32_t*sum,uint32_t processed,bool use_pair)const{
  int best=0;int64_t best_lower=std::numeric_limits<int64_t>::min();
  for(uint32_t j=0;j<c;++j){int64_t lower=(int64_t(sum[j])+suffix_low[size_t(processed)*c+j])*fine+low[j];if(lower>best_lower){best=int(j);best_lower=lower;}}
  bool ok=true;for(uint32_t j=0;j<c;++j){if(int(j)==best)continue;int64_t upper=(int64_t(sum[j])+suffix_high[size_t(processed)*c+j])*fine+high[j];if(best_lower<upper||(best_lower==upper&&best>int(j))){ok=false;break;}}
  if(ok)return best;
  if(processed==t&&use_pair&&flags){
   // Match the reference priority: descending approximate score, then low index.
   uint32_t order[64];for(uint32_t j=0;j<c;++j)order[j]=j;
   std::sort(order,order+c,[&](uint32_t a,uint32_t b){return sum[a]!=sum[b]?sum[a]>sum[b]:a<b;});
   for(uint32_t k=0;k<c;++k){uint32_t i=order[k];bool accepted=true;
    for(uint32_t j=0;j<c;++j){if(i==j)continue;int64_t delta=(int64_t(sum[i])-sum[j])*fine;int64_t bound=pair[size_t(i)*c+j];if(delta<bound||(delta==bound&&i>j)){accepted=false;break;}}
    if(accepted)return int(i);
   }
  }
  return -1;
 }
 int run_quant(const uint8_t*x,uint32_t checkpoint,bool pairwise,bool scalar,uint8_t*pred,uint32_t&processed,int&approximate)const{
  alignas(32) int32_t sums[64];std::copy(qb.begin(),qb.end(),sums);processed=0;approximate=-1;
  int winner=checkpoint?settle(sums,0,pairwise):-1;
  if(winner<0){predicates_for(x,pred);
   for(uint32_t k=0;k<t;++k){add_leaf(leaf(trees[k],pred),sums,scalar);processed=k+1;
    if(processed==t||(checkpoint&&processed%checkpoint==0)){winner=settle(sums,processed,pairwise);if(winner>=0)break;}
   }
  }
  if(processed==t)approximate=int(std::max_element(sums,sums+c)-sums);
  return winner;
 }

#if defined(ST_TILES)
 template<typename Q,int CLASSES> void quant_tile(const uint8_t*x,int rows,bool use_pair,int*out,uint32_t*steps,int*approx)const{
  const Q*bank;
  if constexpr(sizeof(Q)==1)bank=reinterpret_cast<const Q*>(small.data());
  else bank=reinterpret_cast<const Q*>(medium.data());
  constexpr int TILE=32;
  std::vector<uint8_t>pred(std::max(size_t(1),size_t(std::min(rows,TILE))*np));
  alignas(32) int32_t sums[TILE][CLASSES];
  for(int first=0;first<rows;first+=TILE){int count=std::min(TILE,rows-first);
   for(int r=0;r<count;++r){std::copy(qb.begin(),qb.end(),sums[r]);predicates_for(x+size_t(first+r)*d,pred.data()+size_t(r)*np);}
   for(const auto&tree:trees){const uint16_t*ids=ns?splits.data()+tree.split:nullptr;
    for(int r=0;r<count;++r){const uint8_t*b=pred.data()+size_t(r)*np;uint32_t index=0;
     if(tree.depth==6)index=uint32_t(b[ids[0]])|(uint32_t(b[ids[1]])<<1)|(uint32_t(b[ids[2]])<<2)|(uint32_t(b[ids[3]])<<3)|(uint32_t(b[ids[4]])<<4)|(uint32_t(b[ids[5]])<<5);
     else for(uint32_t j=0;j<tree.depth;++j)index|=uint32_t(b[ids[j]])<<j;
     const Q*v=bank+tree.offset+index*CLASSES;int j=0;
#if defined(__AVX2__)
     for(;j+8<=CLASSES;j+=8){__m256i a;
      if constexpr(sizeof(Q)==1)a=_mm256_cvtepi8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(v+j)));
      else a=_mm256_cvtepi16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(v+j)));
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(sums[r]+j),_mm256_add_epi32(a,_mm256_loadu_si256(reinterpret_cast<const __m256i*>(sums[r]+j))));
     }
     if(j+4<=CLASSES){__m128i a;
      if constexpr(sizeof(Q)==1){int32_t values;std::memcpy(&values,v+j,4);a=_mm_cvtepi8_epi32(_mm_cvtsi32_si128(values));}
      else a=_mm_cvtepi16_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(v+j)));
      _mm_storeu_si128(reinterpret_cast<__m128i*>(sums[r]+j),_mm_add_epi32(a,_mm_loadu_si128(reinterpret_cast<const __m128i*>(sums[r]+j))));j+=4;
     }
#endif
     for(;j<CLASSES;++j)sums[r][j]+=v[j];
    }
   }
   for(int r=0;r<count;++r){out[first+r]=settle(sums[r],t,use_pair);steps[first+r]=t;approx[first+r]=int(std::max_element(sums[r],sums[r]+CLASSES)-sums[r]);}
  }
 }
 template<typename Q> bool dispatch_tile(const uint8_t*x,int rows,bool use_pair,int*out,uint32_t*steps,int*approx)const{
  switch(c){
#define ST_CASE(C) case C:quant_tile<Q,C>(x,rows,use_pair,out,steps,approx);return true
   ST_CASE(2);ST_CASE(3);ST_CASE(4);ST_CASE(6);ST_CASE(8);ST_CASE(10);ST_CASE(16);ST_CASE(26);ST_CASE(32);ST_CASE(64);
#undef ST_CASE
   default:return false;
  }
 }
#endif
 int run_full(const uint8_t*x,uint8_t*pred,double*scores)const{
  predicates_for(x,pred);std::fill(scores,scores+c,0.);
  for(const auto&tree:trees){uint32_t off=leaf(tree,pred);for(uint32_t j=0;j<c;++j)scores[j]+=full[off+j];}
  for(uint32_t j=0;j<c;++j){scores[j]=scale*scores[j]+bias[j];require(std::isfinite(scores[j]),"nonfinite source output");}
  return int(std::max_element(scores,scores+c)-scores);
 }
 uint64_t storage()const{return sizeof(*this)+sizeof(Predicate)*predicates.capacity()+sizeof(Tree)*trees.capacity()+2ull*splits.capacity()+small.capacity()+2ull*medium.capacity()+8ull*(full.capacity()+bias.capacity()+low.capacity()+high.capacity()+pair.capacity())+4ull*(qb.capacity()+suffix_low.capacity()+suffix_high.capacity());}
};
void inputs(const Model&m,const uint8_t*x,int rows,int d){require(rows>=0&&rows<=65536&&d==int(m.d)&&uint64_t(rows)*m.d<=8000000&&(!rows||x),"input shape/cap");for(size_t k=0;k<size_t(rows)*m.d;++k)require(x[k]<=m.maximum,"input outside exact integer domain");}
}
extern "C" {
int st_abi(){return 1;}
int st_layout(){
#if defined(ST_TILES)
 return 32;
#else
 return 0;
#endif
}
const char*st_error(){return st2::error.c_str();}
void*st_create(const unsigned char*p,uint64_t n){void*out=nullptr;st2::protect([&]{out=new st2::Model(p,n);});return out;}
void st_destroy(void*p){delete static_cast<st2::Model*>(p);}
int st_info(void*p,uint64_t*out,int count){return st2::protect([&]{st2::require(p&&out&&count==9,"invalid info buffer");auto&m=*static_cast<st2::Model*>(p);uint64_t a[]={m.d,m.c,m.t,m.np,m.n,m.bits,m.packed_bytes,m.storage(),m.flags};std::copy(a,a+9,out);});}
int st_run(void*p,const uint8_t*x,int rows,int d,int checkpoint,int pairwise,int scalar,int*out,uint32_t*steps,int*approx){return st2::protect([&]{
 st2::require(p,"closed tree model");auto&m=*static_cast<st2::Model*>(p);st2::environment();st2::inputs(m,x,rows,d);
 st2::require(checkpoint>=0&&checkpoint<=4096&&(pairwise==0||pairwise==1)&&(scalar==0||scalar==1)&&(!rows||(out&&steps&&approx)),"invalid prediction options");

#if defined(ST_TILES)
 if(m.quantized&&checkpoint==0&&!scalar){bool done=m.bits==8?m.dispatch_tile<int8_t>(x,rows,pairwise,out,steps,approx):m.dispatch_tile<int16_t>(x,rows,pairwise,out,steps,approx);if(done)return;}
#endif
 std::vector<uint8_t> pred(m.np);double scores[64];
 for(int r=0;r<rows;++r){if(m.quantized)out[r]=m.run_quant(x+size_t(r)*d,uint32_t(checkpoint),pairwise,scalar,pred.data(),steps[r],approx[r]);
 else {out[r]=m.run_full(x+size_t(r)*d,pred.data(),scores);steps[r]=m.t;approx[r]=out[r];}}
});}
int st_scores(void*p,const uint8_t*x,int rows,int d,double*out,uint64_t cells){return st2::protect([&]{st2::require(p,"closed tree model");auto&m=*static_cast<st2::Model*>(p);st2::environment();st2::inputs(m,x,rows,d);st2::require(!m.quantized&&cells==uint64_t(rows)*m.c&&cells<=8000000&&(!rows||out),"full score geometry");std::vector<uint8_t>pred(m.np);for(int r=0;r<rows;++r)m.run_full(x+size_t(r)*d,pred.data(),out+size_t(r)*m.c);});}
int st_hybrid(void*fast,void*slow,const uint8_t*x,int rows,int d,int checkpoint,int*out,uint32_t*steps,uint8_t*fallback){return st2::protect([&]{
 st2::require(fast&&slow,"closed hybrid model");auto&a=*static_cast<st2::Model*>(fast);auto&b=*static_cast<st2::Model*>(slow);st2::environment();st2::inputs(a,x,rows,d);
 st2::require(a.quantized&&a.d==b.d&&a.c==b.c&&a.maximum==b.maximum&&a.source==b.source&&checkpoint>=0&&checkpoint<=4096&&(!rows||(out&&steps&&fallback)),"incompatible hybrid source/options");
 std::vector<uint8_t>pa(a.np),pb(b.np);double scores[64];
 std::vector<int>approx(rows);
 if(st_run(fast,x,rows,d,checkpoint,1,0,out,steps,approx.data()))throw std::invalid_argument(st2::error);
 for(int r=0;r<rows;++r){int approximate=-1;const auto*q=x+size_t(r)*d;int result=out[r];fallback[r]=uint8_t(result<0);
 if(result<0){if(b.quantized){uint32_t second=0;result=b.run_quant(q,checkpoint,true,false,pb.data(),second,approximate);steps[r]+=second;}else {result=b.run_full(q,pb.data(),scores);steps[r]+=b.t;}}
 out[r]=result;}
});}
}
