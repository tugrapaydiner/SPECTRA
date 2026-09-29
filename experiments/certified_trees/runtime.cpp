// Experimental packed certificates. Integer decisions certify a stated source
// arithmetic contract, NOT true labels or unverified model provenance.
#include <algorithm>
#include <array>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
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
namespace tc {
thread_local std::string error;
void need(bool yes,const char*text){if(!yes)throw std::invalid_argument(text);}
template<class F>int protect(F f){try{f();return 0;}catch(const std::exception&e){error=e.what();return 1;}catch(...){error="unknown native failure";return 1;}}
void environment(){
 need(std::fegetround()==FE_TONEAREST,"round-to-nearest required");
#if defined(__SSE__)
 need((_mm_getcsr()&0xe040)==0,"SIMD nearest and gradual underflow required");
#endif
}
uint32_t crc32(const uint8_t*p,size_t n){uint32_t c=~0u;for(size_t i=0;i<n;++i){c^=p[i];for(int j=0;j<8;++j)c=(c>>1)^(0xedb88320u&uint32_t(-int(c&1)));}return ~c;}
struct Reader{
 const uint8_t*raw;size_t length,offset=0;
 template<class T>T value(){need(offset<=length&&sizeof(T)<=length-offset,"truncated packed value");T x;std::memcpy(&x,raw+offset,sizeof(T));offset+=sizeof(T);return x;}
 template<class T>void array(std::vector<T>&x,size_t n){need(n<=length/sizeof(T)&&offset<=length&&n*sizeof(T)<=length-offset,"truncated packed array");x.resize(n);if(n)std::memcpy(x.data(),raw+offset,n*sizeof(T));offset+=n*sizeof(T);}
};
struct Predicate{uint16_t feature;int16_t cutoff;};
struct Tree{uint32_t split,depth,leaf;};
struct Compact{
 uint32_t d,D,c,t,bits,finebits,flags;int32_t exponent;int64_t fine;
 std::array<uint8_t,32> source,oracle;
 std::vector<Predicate> predicates;std::vector<Tree> trees;std::vector<uint16_t> refs;
 std::vector<int32_t> bias,suffixlo,suffixhi;
 std::vector<int64_t> lo,hi,pairs;
 std::vector<int8_t> q8;std::vector<int16_t> q16;
 Compact(const uint8_t*raw,size_t n){
  const uint16_t one=1;need(*reinterpret_cast<const uint8_t*>(&one)==1,"little-endian host required");
  need(raw&&n>=124&&n<=64ull*1024*1024,"packed model byte cap");need(!std::memcmp(raw,"SPCERT02",8),"unknown packed magic");
  Reader r{raw,n,8};d=r.value<uint32_t>();D=r.value<uint32_t>();c=r.value<uint32_t>();t=r.value<uint32_t>();bits=r.value<uint32_t>();exponent=r.value<int32_t>();
  finebits=r.value<uint32_t>();flags=r.value<uint32_t>();uint32_t np=r.value<uint32_t>(),nr=r.value<uint32_t>(),nl=r.value<uint32_t>(),payload=r.value<uint32_t>(),crc=r.value<uint32_t>();
  std::memcpy(source.data(),raw+60,32);std::memcpy(oracle.data(),raw+92,32);r.offset=124;
  need(d>=1&&d<=256&&D>=1&&D<=255&&c>=2&&c<=64&&t>=1&&t<=4096,"packed geometry");
  need((bits==8||bits==16)&&exponent>=-900&&exponent<=900&&finebits==20&&!(flags&~3u)&&(!(flags&1)||c==2),"packed arithmetic policy");
  need(np<=65535&&nr<=12*t&&nl>=t&&uint64_t(nl)*c<=500000,"packed inventory cap");
  uint64_t expected=4ull*np+12ull*t+2ull*nr+20ull*c+uint64_t(bits/8)*nl*c+((flags&2)?8ull*c*c:0);
  need(payload==expected&&n==124+expected&&crc32(raw+124,n-124)==crc,"packed bytes/CRC mismatch");
  predicates.resize(np);for(auto&p:predicates){p.feature=r.value<uint16_t>();p.cutoff=r.value<int16_t>();need(p.feature<d&&p.cutoff>=-1&&p.cutoff<=int(D),"predicate out of domain");}
  trees.resize(t);uint32_t split=0,leaf=0;
  for(auto&tr:trees){tr.split=r.value<uint32_t>();tr.depth=r.value<uint32_t>();tr.leaf=r.value<uint32_t>();
   need(tr.depth<=12&&tr.split==split&&tr.leaf==leaf,"noncanonical tree offsets");split+=tr.depth;leaf+=1u<<tr.depth;}
  need(split==nr&&leaf==nl,"tree inventory mismatch");
  r.array(refs,nr);for(auto v:refs)need(v<np,"unknown predicate");
  r.array(bias,c);r.array(lo,c);r.array(hi,c);int32_t limit=bits==8?127:32767;fine=int64_t(1)<<finebits;
  int64_t envelope=int64_t(t+1)*limit*fine;
  for(uint32_t j=0;j<c;++j){need(bias[j]>=-limit&&bias[j]<=limit,"bias range");
   need(lo[j]<=hi[j]&&lo[j]>-(int64_t(1)<<61)+envelope&&hi[j]<(int64_t(1)<<61)-envelope,"unsafe certificate bound");}
  if(bits==8){q8.reserve(size_t(nl)*c+8);r.array(q8,size_t(nl)*c);for(auto v:q8)need(v>=-limit&&v<=limit,"leaf range");}
  else{q16.reserve(size_t(nl)*c+8);r.array(q16,size_t(nl)*c);for(auto v:q16)need(v>=-limit&&v<=limit,"leaf range");}
  need(bias[0]==0,"reference contrast must be zero");
  for(uint32_t k=0;k<nl;++k)need(value(k,0)==0,"reference leaf contrast must be zero");
  if(flags&2){r.array(pairs,size_t(c)*c);for(auto v:pairs)need(v>-(int64_t(1)<<61)&&v<(int64_t(1)<<61),"unsafe pair bound");}
  need(r.offset==n,"trailing packed data");
#if defined(TC_FAST_END) && defined(__AVX2__)
  // Only trailing vector lanes are unused; no per-leaf padded bank is needed.
  // Eight guard elements make every last-chunk load bounded even on the last leaf.
  if(bits==8)q8.resize(q8.size()+8,0);else q16.resize(q16.size()+8,0);
#endif
  suffixlo.resize(size_t(t+1)*c,0);suffixhi.resize(size_t(t+1)*c,0);
  for(uint32_t k=t;k-->0;){auto&tr=trees[k];
   for(uint32_t j=0;j<c;++j){int32_t lower=limit,upper=-limit;
    for(uint32_t a=0;a<(1u<<tr.depth);++a){int32_t v=value(tr.leaf+a,j);lower=std::min(lower,v);upper=std::max(upper,v);}
    suffixlo[size_t(k)*c+j]=suffixlo[size_t(k+1)*c+j]+lower;suffixhi[size_t(k)*c+j]=suffixhi[size_t(k+1)*c+j]+upper;
   }
  }
 }
 int32_t value(uint32_t row,uint32_t cls)const{return bits==8?int32_t(q8[size_t(row)*c+cls]):int32_t(q16[size_t(row)*c+cls]);}
 uint16_t route(const uint8_t*q,uint32_t k)const{auto&tr=trees[k];uint16_t id=0;for(uint32_t b=0;b<tr.depth;++b){auto&p=predicates[refs[tr.split+b]];id|=uint16_t(int(q[p.feature])>p.cutoff)<<b;}return id;}
 void add(int32_t*s,uint32_t row,bool scalar)const{
  uint32_t j=0;
#if defined(__AVX2__)
  if(!scalar)for(;j+8<=c;j+=8){__m256i v;
   if(bits==8)v=_mm256_cvtepi8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(q8.data()+size_t(row)*c+j)));
   else v=_mm256_cvtepi16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(q16.data()+size_t(row)*c+j)));
   _mm256_storeu_si256(reinterpret_cast<__m256i*>(s+j),_mm256_add_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(s+j)),v));}
#endif
  for(;j<c;++j)s[j]+=value(row,j);
 }
 int settle(const int32_t*s,uint32_t done)const{
  uint32_t winner=0;int64_t best=(int64_t(s[0])+suffixlo[size_t(done)*c])*fine+lo[0];
  for(uint32_t j=1;j<c;++j){int64_t b=(int64_t(s[j])+suffixlo[size_t(done)*c+j])*fine+lo[j];if(b>best){best=b;winner=j;}}
  bool ok=true;
  for(uint32_t j=0;j<c;++j)if(j!=winner){int64_t upper=(int64_t(s[j])+suffixhi[size_t(done)*c+j])*fine+hi[j];if(best<upper||(best==upper&&winner>j)){ok=false;break;}}
  if(ok)return int(winner);
  if(done==t&&!pairs.empty()){
   for(uint32_t k=0;k<c;++k){ok=true;for(uint32_t j=0;j<c;++j)if(k!=j){int64_t gap=(int64_t(s[k])-s[j])*fine,bound=pairs[size_t(k)*c+j];if(gap<bound||(gap==bound&&k>j)){ok=false;break;}}if(ok)return int(k);}
  }
  return -1;
 }
#if defined(TC_FAST_END) && defined(__AVX2__)
 template<int N,int B> int end_registers(const uint16_t*route,uint32_t count,uint32_t row)const{
  int32_t initial[64]={};std::copy(bias.begin(),bias.end(),initial);
  __m256i sums[N];
#pragma GCC unroll 8
  for(int v=0;v<N;++v)sums[v]=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(initial+v*8));
  for(uint32_t k=0;k<t;++k){size_t index=size_t(trees[k].leaf+route[size_t(k)*count+row])*c;
#pragma GCC unroll 8
   for(int v=0;v<N;++v){__m256i values;
    if constexpr(B==8)values=_mm256_cvtepi8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(q8.data()+index+v*8)));
    else values=_mm256_cvtepi16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(q16.data()+index+v*8)));
    sums[v]=_mm256_add_epi32(sums[v],values);
   }
  }
  int32_t final[64];for(int v=0;v<N;++v)_mm256_storeu_si256(reinterpret_cast<__m256i*>(final+v*8),sums[v]);
  return settle(final,t);
 }
 int fast_end(const uint16_t*routes,uint32_t count,uint32_t row)const{
#define TC_CASE(N) case N:return bits==8?end_registers<N,8>(routes,count,row):end_registers<N,16>(routes,count,row)
  switch((c+7)/8){TC_CASE(1);TC_CASE(2);TC_CASE(3);TC_CASE(4);TC_CASE(5);TC_CASE(6);TC_CASE(7);TC_CASE(8);}
#undef TC_CASE
  throw std::invalid_argument("unsupported register width");
 }
#endif
 uint64_t storage()const{return sizeof(*this)+4ull*predicates.capacity()+12ull*trees.capacity()+2ull*refs.capacity()+4ull*(bias.capacity()+suffixlo.capacity()+suffixhi.capacity())+8ull*(lo.capacity()+hi.capacity()+pairs.capacity())+q8.capacity()+2ull*q16.capacity();}
};
struct Official{
 void* library=nullptr;void* model=nullptr;
 void(*destroy)(void*)=nullptr;bool(*calc)(void*,size_t,const float**,size_t,double*,size_t)=nullptr;
 const char*(*error_string)()=nullptr;uint32_t d,c,dimensions;
 template<class T>T symbol(const char*name){void*p=dlsym(library,name);need(p!=nullptr,name);return reinterpret_cast<T>(p);}
 Official(const char*path,const void*raw,size_t n,uint32_t width,uint32_t classes):d(width),c(classes){
  need(path&&raw&&n>0&&n<=64ull*1024*1024,"official source/library required");
  library=dlopen(path,RTLD_NOW|RTLD_LOCAL);need(library!=nullptr,"cannot load official CatBoost library");
  try{
   auto create=symbol<void*(*)()>("ModelCalcerCreate");destroy=symbol<void(*)(void*)>("ModelCalcerDelete");
   auto load=symbol<bool(*)(void*,const void*,size_t)>("LoadFullModelFromBuffer");error_string=symbol<const char*(*)()>("GetErrorString");
   auto dims=symbol<size_t(*)(void*)>("GetDimensionsCount");auto floats=symbol<size_t(*)(void*)>("GetFloatFeaturesCount");auto cats=symbol<size_t(*)(void*)>("GetCatFeaturesCount");
   calc=symbol<bool(*)(void*,size_t,const float**,size_t,double*,size_t)>("CalcModelPredictionFlat");model=create();need(model!=nullptr,"official allocation failed");
   if(!load(model,raw,n))throw std::runtime_error(error_string());dimensions=uint32_t(dims(model));
   need(floats(model)<=d&&cats(model)==0&&(dimensions==c||(dimensions==1&&c==2)),"official/compact shape mismatch");
  }catch(...){if(model&&destroy)destroy(model);dlclose(library);library=nullptr;model=nullptr;throw;}
 }
 ~Official(){if(model)destroy(model);if(library)dlclose(library);}
 void predict(const uint8_t*q,const std::vector<uint32_t>&rows,std::vector<int32_t>&out)const{
  if(rows.empty())return;std::vector<float>x(rows.size()*d);std::vector<const float*>ptr(rows.size());std::vector<double>scores(rows.size()*dimensions);
  for(size_t i=0;i<rows.size();++i){for(uint32_t j=0;j<d;++j)x[i*d+j]=float(q[size_t(rows[i])*d+j]);ptr[i]=x.data()+i*d;}
  if(!calc(model,rows.size(),ptr.data(),d,scores.data(),scores.size()))throw std::runtime_error(error_string());
  for(size_t i=0;i<rows.size();++i){const double*s=scores.data()+i*dimensions;for(uint32_t j=0;j<dimensions;++j)need(std::isfinite(s[j]),"nonfinite official prediction");
   out[rows[i]]=dimensions==1?int(s[0]>0):int(std::max_element(s,s+dimensions)-s);}
 }
};
struct Pipeline{
 std::unique_ptr<Compact> first,second;std::unique_ptr<Official> official;uint32_t d,D,c;
 Pipeline(const uint8_t*a,size_t na,const uint8_t*b,size_t nb,const char*lib,const uint8_t*cbm,size_t nc,uint32_t width,uint32_t max,uint32_t cls):d(width),D(max),c(cls){
  environment();need(d>=1&&d<=256&&D>=1&&D<=255&&c>=2&&c<=64,"pipeline domain");
  if(na){first=std::make_unique<Compact>(a,na);need(first->d==d&&first->D==D&&first->c==c,"first compact shape");}
  if(nb){need(bool(first),"second without first");second=std::make_unique<Compact>(b,nb);auto&x=*first;auto&y=*second;
   need(x.d==y.d&&x.D==y.D&&x.c==y.c&&x.t==y.t&&x.source==y.source&&x.flags%2==y.flags%2,"refinement original binding");
   need(x.bits==8&&y.bits==16&&x.predicates.size()==y.predicates.size()&&x.refs==y.refs,"refinement precision/routing mismatch");
   for(size_t i=0;i<x.predicates.size();++i)need(x.predicates[i].feature==y.predicates[i].feature&&x.predicates[i].cutoff==y.predicates[i].cutoff,"refinement predicates mismatch");
   for(size_t i=0;i<x.t;++i)need(x.trees[i].depth==y.trees[i].depth&&x.trees[i].split==y.trees[i].split&&x.trees[i].leaf==y.trees[i].leaf,"refinement tree mismatch");
  }
  if(nc)official=std::make_unique<Official>(lib,cbm,nc,d,c);need(first||official,"empty pipeline");
 }
 // stats: first certificates, second certificates, official rows, unresolved,
 // first leaf vectors, second leaf vectors, routed trees, logical compact bytes.
 void run(const uint8_t*q,uint32_t rows,int mode,uint32_t checkpoint,bool refine,bool fallback,std::vector<int32_t>&out,std::vector<uint32_t>&used,uint64_t*stats)const{
  need(mode==0||mode==1,"mode must be scalar0 or tiled1");out.assign(rows,-1);used.assign(rows,0);std::fill(stats,stats+8,0);
  if(!first){need(bool(official),"no official model");std::vector<uint32_t>indices(rows);for(uint32_t i=0;i<rows;++i)indices[i]=i;official->predict(q,indices,out);stats[2]=rows;return;}
  auto&m=*first;constexpr uint32_t TILE=32;
  for(uint32_t start=0;start<rows;start+=TILE){uint32_t count=std::min(TILE,rows-start);std::vector<uint16_t>route(size_t(count)*m.t,UINT16_MAX);
   if(mode==1){
#if defined(TC_VECTOR_ROUTING) && defined(__AVX2__)
    constexpr uint32_t pitch=32;std::vector<uint8_t>columns(size_t(d)*pitch,0),pred(m.predicates.size()*pitch);
    for(uint32_t f=0;f<d;++f)for(uint32_t r=0;r<count;++r)columns[size_t(f)*pitch+r]=q[size_t(start+r)*d+f];
    for(size_t k=0;k<m.predicates.size();++k){auto&p=m.predicates[k];__m256i bits;
     if(p.cutoff<0)bits=_mm256_set1_epi8(1);
     else if(p.cutoff>=int(D))bits=_mm256_setzero_si256();
     else {auto x=_mm256_xor_si256(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(columns.data()+size_t(p.feature)*pitch)),_mm256_set1_epi8(char(0x80)));
      auto threshold=_mm256_set1_epi8(char(p.cutoff^0x80));bits=_mm256_and_si256(_mm256_cmpgt_epi8(x,threshold),_mm256_set1_epi8(1));}
     _mm256_storeu_si256(reinterpret_cast<__m256i*>(pred.data()+k*pitch),bits);
    }
    for(uint32_t k=0;k<m.t;++k){auto&tr=m.trees[k];
     if(tr.depth<=8){auto leaves=_mm256_setzero_si256();
      for(uint32_t b=0;b<tr.depth;++b){auto bits=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(pred.data()+size_t(m.refs[tr.split+b])*pitch));leaves=_mm256_or_si256(leaves,_mm256_sll_epi16(bits,_mm_cvtsi32_si128(int(b))));}
      uint8_t bytes[32];_mm256_storeu_si256(reinterpret_cast<__m256i*>(bytes),leaves);
      for(uint32_t r=0;r<count;++r)route[size_t(k)*count+r]=bytes[r];
     }else for(uint32_t r=0;r<count;++r){uint16_t leaf=0;for(uint32_t b=0;b<tr.depth;++b)leaf|=uint16_t(pred[size_t(m.refs[tr.split+b])*pitch+r])<<b;route[size_t(k)*count+r]=leaf;}
    }
#else
    std::vector<uint8_t>pred(m.predicates.size()*count);
    for(size_t k=0;k<m.predicates.size();++k){auto&p=m.predicates[k];for(uint32_t r=0;r<count;++r)pred[k*count+r]=int(q[size_t(start+r)*d+p.feature])>p.cutoff;}
    for(uint32_t k=0;k<m.t;++k){auto&tr=m.trees[k];for(uint32_t r=0;r<count;++r){uint16_t leaf=0;for(uint32_t b=0;b<tr.depth;++b)leaf|=uint16_t(pred[size_t(m.refs[tr.split+b])*count+r])<<b;route[size_t(k)*count+r]=leaf;}}
#endif
    stats[6]+=uint64_t(count)*m.t;
   }
   auto evaluate=[&](const Compact&cm,bool next){
#if defined(TC_FAST_END) && defined(__AVX2__)
    if(mode==1&&checkpoint==0){
     for(uint32_t r=0;r<count;++r)if(out[start+r]<0){int win=cm.fast_end(route.data(),count,r);out[start+r]=win;used[start+r]+=cm.t;stats[next?5:4]+=cm.t;if(win>=0)stats[next?1:0]++;}
     return;
    }
#endif
    std::vector<int32_t>s(size_t(count)*c);std::vector<uint8_t>active(count,0);
    for(uint32_t r=0;r<count;++r)if(out[start+r]<0){std::copy(cm.bias.begin(),cm.bias.end(),s.begin()+size_t(r)*c);active[r]=1;
     if(checkpoint){int win=cm.settle(s.data()+size_t(r)*c,0);if(win>=0){out[start+r]=win;active[r]=0;stats[next?1:0]++;}}}
    for(uint32_t k=0;k<cm.t;++k){bool any=false;
     for(uint32_t r=0;r<count;++r)if(active[r]){any=true;auto&leaf=route[size_t(k)*count+r];
      if(leaf==UINT16_MAX){leaf=m.route(q+size_t(start+r)*d,k);stats[6]++;}
      cm.add(s.data()+size_t(r)*c,cm.trees[k].leaf+leaf,mode==0);stats[next?5:4]++;used[start+r]++;
      if(k+1==cm.t||(checkpoint&&(k+1)%checkpoint==0)){int win=cm.settle(s.data()+size_t(r)*c,k+1);if(win>=0){out[start+r]=win;active[r]=0;stats[next?1:0]++;}}
     }if(!any)break;
    }
   };
   evaluate(m,false);if(refine){need(bool(second),"second precision not loaded");evaluate(*second,true);}
  }
  std::vector<uint32_t>ambiguous;for(uint32_t i=0;i<rows;++i)if(out[i]<0)ambiguous.push_back(i);
  if(fallback){need(bool(official),"official fallback not loaded");official->predict(q,ambiguous,out);stats[2]=ambiguous.size();}
  else stats[3]=ambiguous.size();
  stats[7]=first->storage()+(second?second->storage():0);
 }
};
}
extern "C"{
int tc_abi(){return 2;}const char*tc_error(){return tc::error.c_str();}
void* tc_create(const uint8_t*a,uint64_t na,const uint8_t*b,uint64_t nb,const char*lib,const uint8_t*cbm,uint64_t nc,uint32_t d,uint32_t D,uint32_t c){void*p=nullptr;tc::protect([&]{p=new tc::Pipeline(a,size_t(na),b,size_t(nb),lib,cbm,size_t(nc),d,D,c);});return p;}
void tc_destroy(void*p){delete static_cast<tc::Pipeline*>(p);}
int tc_run(void*p,const uint8_t*q,int rows,int d,int mode,int checkpoint,int refine,int fallback,int32_t*out,uint32_t*used,uint64_t*stats,int nstats){return tc::protect([&]{
 tc::need(p,"closed tree pipeline");auto&m=*static_cast<tc::Pipeline*>(p);tc::environment();
 tc::need(rows>=0&&rows<=65536&&d==int(m.d)&&uint64_t(rows)*d<=8000000&&checkpoint>=0&&(refine==0||refine==1)&&(fallback==0||fallback==1)&&stats&&nstats==8&&(!rows||(q&&out&&used)),"invalid tree batch");
 for(size_t k=0;k<size_t(rows)*d;++k)tc::need(q[k]<=m.D,"input outside declared integer domain");
 // All caller outputs are committed only after all checks and fallback calls succeed.
 std::vector<int32_t>result;std::vector<uint32_t>work;uint64_t totals[8];m.run(q,uint32_t(rows),mode,uint32_t(checkpoint),refine,fallback,result,work,totals);
 std::copy(result.begin(),result.end(),out);std::copy(work.begin(),work.end(),used);std::copy(totals,totals+8,stats);
});}
int tc_info(void*p,uint64_t*out,int n){return tc::protect([&]{tc::need(p&&out&&n==8,"invalid info");auto&m=*static_cast<tc::Pipeline*>(p);
 uint64_t v[]={m.d,m.D,m.c,m.first?m.first->t:0,m.first?m.first->storage():0,m.second?m.second->storage():0,uint64_t(bool(m.official)),uint64_t(bool(m.second))};std::copy(v,v+8,out);});}
}
