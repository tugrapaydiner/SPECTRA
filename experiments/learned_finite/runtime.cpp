// Experimental learned finite metric: same serialized table for training/deployment.
// Integer distance and ordered FP64 sums. No exponential or float metric at inference.
#ifndef LF_BASE
#error Define LF_BASE to the existing SPECTRA runtime.cpp
#endif
#include LF_BASE
namespace lf {
constexpr size_t HEADER=36;
inline uint32_t popcount(uint64_t x) {
#if defined(__POPCNT__)
 return __builtin_popcountll(x);
#else
 x-=(x>>1)&0x5555555555555555ULL;
 x=(x&0x3333333333333333ULL)+((x>>2)&0x3333333333333333ULL);
 x=(x+(x>>4))&0x0f0f0f0f0f0f0f0fULL;
 return uint32_t((x*0x0101010101010101ULL)>>56);
#endif
}
struct Engine {
 int d,c,n,words,signature;uint32_t max_distance;
 std::vector<uint8_t> weights;
 std::vector<uint64_t> supports,masks,prototypes,query,epochs;
 std::vector<int> planes;
 std::vector<double> table,kernels,priority;
 std::vector<lo::Pair> pairs;
 lo::Votes votes;lo::Work work;uint64_t serial=0,rounds=0;int mode=0;
 static uint32_t u32(const unsigned char* p){uint32_t n;std::memcpy(&n,p,4);return n;}
 static int classes(const unsigned char* raw,size_t size){
  lo::require(raw&&size>=HEADER&&size<=lo::MAX_BYTES,"model byte bound");
  lo::require((std::memcmp(raw,"SPLFK001",8)==0||std::memcmp(raw,"SPLFK002",8)==0),"model magic");
  uint32_t c=u32(raw+12);lo::require(c>=2&&c<=128,"class bound");return int(c);
 }
 Engine(const unsigned char* raw,size_t size):votes(classes(raw,size)) {
  const uint16_t one=1;
  lo::require(*reinterpret_cast<const uint8_t*>(&one)==1&&sizeof(double)==8&&std::numeric_limits<double>::is_iec559,"little-endian IEEE binary64 required");
  size_t header=std::memcmp(raw,"SPLFK001",8)==0?HEADER:HEADER+4;
  lo::require(size>=header,"truncated header");signature=header==HEADER?0:int(u32(raw+36));
  lo::require(signature==0||signature==1,"unknown signature");
  d=int(u32(raw+8));c=int(u32(raw+12));n=int(u32(raw+16));
  uint32_t nt=u32(raw+20),meta=u32(raw+24),payload=u32(raw+28),crc=u32(raw+32);
  lo::require(d>=1&&d<=4096&&n>=c&&n<=100000&&uint64_t(n)*d<=8000000,"model geometry");
  lo::require(meta>=1&&meta<=65536&&nt>=2&&nt<=uint32_t(7*d+1),"table or label geometry");
  words=(d+63)/64;
  uint64_t expected=uint64_t(meta)+d+4*c+8ull*n*words+8ull*(nt+uint64_t(c-1)*n+uint64_t(c)*(c-1)/2);
  lo::require(payload==expected&&size==header+expected&&lo::crc32(raw+header,payload)==crc,"model inventory or CRC");
  size_t off=header+meta;
  auto take=[&](void* dest,size_t count){lo::require(count<=size-off,"model overrun");std::memcpy(dest,raw+off,count);off+=count;};
  weights.resize(d);take(weights.data(),d);max_distance=0;
  for(auto w:weights){lo::require(w<=7,"invalid integer weight");max_distance+=w;}
  lo::require(nt==max_distance+1,"table/weight mismatch");
  std::vector<uint32_t> counts(c),starts(c+1,0);take(counts.data(),4*c);
  for(int i=0;i<c;++i){lo::require(counts[i]>0&&counts[i]<=uint32_t(n),"invalid support count");starts[i+1]=starts[i]+counts[i];}
  lo::require(starts[c]==uint32_t(n),"support sum");
  supports.resize(size_t(n)*words);take(supports.data(),8*supports.size());
  if(d%64)for(int i=0;i<n;++i)lo::require(!(supports[size_t(i)*words+words-1]>>(d%64)),"nonzero bit padding");
  table.resize(nt);take(table.data(),8*nt);
  for(double v:table)lo::require(std::isfinite(v)&&std::abs(v)<=0x1p20,"invalid table value");
  std::vector<double> dense(size_t(c-1)*n),bias(size_t(c)*(c-1)/2);
  take(dense.data(),8*dense.size());take(bias.data(),8*bias.size());
  for(auto* a:{&dense,&bias})for(double v:*a)lo::require(std::isfinite(v),"nonfinite coefficient");
  int index=0;
  for(int i=0;i<c;++i)for(int j=i+1;j<c;++j){
   lo::Pair pair;pair.bias=bias[index++];double bound=std::abs(pair.bias),max_kernel=0;for(double v:table)max_kernel=std::max(max_kernel,std::abs(v));
   for(auto half:{std::pair<int,int>{i,j-1},{j,i}})for(uint32_t k=starts[half.first];k<starts[half.first+1];++k){
    double value=dense[size_t(half.second)*n+k];
    if(value!=0){pair.ids.push_back(k);pair.values.push_back(value);bound+=std::abs(value)*max_kernel;}
   }
   lo::require(std::isfinite(bound)&&bound<std::numeric_limits<double>::max()/4,"unsafe coefficient bound");
   pair.ids.shrink_to_fit();pair.values.shrink_to_fit();pairs.push_back(std::move(pair));
  }
  masks.assign(size_t(3)*words,0);query.assign(words,0);prototypes.assign(size_t(c)*words,0);
  for(int plane=0;plane<3;++plane){bool active=false;
   for(int f=0;f<d;++f)if(weights[f]&(1<<plane)){masks[size_t(plane)*words+f/64]|=uint64_t(1)<<(f%64);active=true;}
   if(active)planes.push_back(plane);
  }
  for(int i=0;i<c;++i)for(int f=0;f<d;++f){uint32_t count=0;
   for(uint32_t k=starts[i];k<starts[i+1];++k)count+=(supports[size_t(k)*words+f/64]>>(f%64))&1;
   if(2*count>counts[i])prototypes[size_t(i)*words+f/64]|=uint64_t(1)<<(f%64);
  }
  kernels.resize(n);epochs.assign(n,0);priority.assign(c,0);
 }
 uint32_t distance(const uint64_t* other)const{
  uint32_t total=0;
  if(mode==2){
   for(int f=0;f<d;++f)if(((signature?(query[f/64]&other[f/64]):(query[f/64]^other[f/64]))>>(f%64))&1)total+=weights[f];
  }else{
   for(int plane:planes)for(int k=0;k<words;++k)
    total+=popcount((signature?(query[k]&other[k]):(query[k]^other[k]))&masks[size_t(plane)*words+k])<<plane;
  }
  return total;
 }
 void begin(const uint8_t* x,int selected_mode){
  mode=selected_mode;work=lo::Work{};work.classes=c;work.supports=n;rounds=0;votes.reset(work);
  std::fill(query.begin(),query.end(),0);
  for(int f=0;f<d;++f)if(x[f])query[f/64]|=uint64_t(1)<<(f%64);
  if(++serial==0){std::fill(epochs.begin(),epochs.end(),0);serial=1;}
 }
 double score(size_t index){
  const auto& p=pairs[index];double sum=0;
  for(size_t t=0;t<p.ids.size();++t){auto id=p.ids[t];
   if(epochs[id]!=serial){kernels[id]=table[distance(supports.data()+size_t(id)*words)];epochs[id]=serial;++work.kernels;}
   sum+=p.values[t]*kernels[id];
  }
  work.terms+=p.ids.size();sum+=p.bias;lo::require(std::isfinite(sum),"nonfinite score");return sum;
 }
 int edge(int i,int j){
  if(i>j)std::swap(i,j);size_t idx=size_t(i)*(2*c-i-1)/2+j-i-1;
  double value=score(idx);return c==2?(value>=0?1:0):(value>0?i:j);
 }
 int run(const uint8_t* x,int selected_mode){
  begin(x,selected_mode);
  if(c==2){int found=edge(0,1);votes.add(0,1,found);return found;}
  for(int i=0;i<c;++i)priority[i]=(signature?-1.:1.)*distance(prototypes.data()+size_t(i)*words);
  auto oracle=[&](int i,int j){return edge(i,j);};int result;
  if(mode==1)result=votes.run(oracle,priority,1);
  else result=et::beretta(votes,oracle,priority,5,rounds);
  lo::require(votes.certificate()==result,"vote certificate mismatch");return result;
 }
 uint64_t storage()const{
  uint64_t total=sizeof(*this)+weights.capacity()+8*(supports.capacity()+masks.capacity()+prototypes.capacity()+query.capacity()+epochs.capacity()+table.capacity()+kernels.capacity()+priority.capacity());
  total+=4*planes.capacity()+sizeof(lo::Pair)*pairs.capacity()+votes.known.capacity()+4*(votes.low.capacity()+votes.remain.capacity());
  for(const auto& p:pairs)total+=4*p.ids.capacity()+8*p.values.capacity();return total;
 }
};
}
extern "C" {
int lf_abi(){return 1;}
void* lf_create(const unsigned char* raw,uint64_t size){void* out=nullptr;lo::protect([&]{out=new lf::Engine(raw,size);});return out;}
void lf_destroy(void* p){delete static_cast<lf::Engine*>(p);}
int lf_info(void* p,uint64_t* out,int count){return lo::protect([&]{
 lo::require(p&&out&&count==6,"invalid info");const auto& e=*static_cast<lf::Engine*>(p);
 uint64_t values[]={uint64_t(e.d),uint64_t(e.c),uint64_t(e.n),e.table.size(),e.storage(),e.planes.size()};std::copy(values,values+6,out);
});}
int lf_run(void* p,const uint8_t* inputs,int rows,int features,int mode,int* out,uint64_t* stats,int count){return lo::protect([&]{
 lo::require(p,"null engine");auto& e=*static_cast<lf::Engine*>(p);
 lo::require(rows>=0&&rows<=65536&&features==e.d&&uint64_t(rows)*features<=8000000&&mode>=0&&mode<=2&&stats&&count==3&&(!rows||(inputs&&out)),"invalid request geometry");
 lo::require(std::fegetround()==FE_TONEAREST,"round-to-nearest required");
 for(size_t i=0;i<size_t(rows)*features;++i)lo::require(inputs[i]<=1,"nonbinary input");
 std::fill(stats,stats+3,0);
 for(int r=0;r<rows;++r){out[r]=e.run(inputs+size_t(r)*features,mode);stats[0]+=e.work.kernels;stats[1]+=e.work.pairs;stats[2]+=e.work.terms;}
});}
int lf_margins(void* p,const uint8_t* inputs,int rows,int features,double* out,uint64_t count){return lo::protect([&]{
 lo::require(p,"null engine");auto& e=*static_cast<lf::Engine*>(p);
 lo::require(rows>=0&&rows<=65536&&features==e.d&&count==uint64_t(rows)*e.pairs.size()&&count<=8000000&&(!rows||(inputs&&out)),"invalid probe geometry");
 lo::require(std::fegetround()==FE_TONEAREST,"round-to-nearest required");
 for(size_t i=0;i<size_t(rows)*features;++i)lo::require(inputs[i]<=1,"nonbinary input");
 for(int r=0;r<rows;++r){e.begin(inputs+size_t(r)*features,0);for(size_t j=0;j<e.pairs.size();++j)*out++=e.score(j);}
});}
}
